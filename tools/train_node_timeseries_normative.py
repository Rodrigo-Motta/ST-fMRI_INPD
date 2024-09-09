#
# Created on Tue Jul 18 2024
#
# by Rodrigo M. Cabral-Carvalho

# Adapted from Simon Dahan @SD3004
#
# Copyright (c) 2024 MeTrICS Lab
#

import os
import argparse
import yaml
import sys

sys.path.append('../')
sys.path.append('./')

import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error, r2_score
from utils.train_utils import pred_sets

from scipy.stats import pearsonr

#from torch.utils.tensorboard import SummaryWriter

from models.MSG3D.model import Model

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.cuda.empty_cache()


import torch
import torch.nn as nn

class CustomLossWithUniformity(nn.Module):
    def __init__(self, lambda_reg=0.3, num_bins=10, range_min=7, range_max=15):
        super(CustomLossWithUniformity, self).__init__()
        self.lambda_reg = lambda_reg
        self.num_bins = num_bins
        self.range_min = range_min
        self.range_max = range_max

    def forward(self, y_true, y_pred):
        # Mean Squared Error (MSE) term
        mse = torch.mean((y_true - y_pred) ** 2)
        
        # Create histogram of predictions
        hist = torch.histc(y_pred, bins=self.num_bins, min=self.range_min, max=self.range_max)
        
        # Calculate the target uniform distribution
        target_uniform = torch.full((self.num_bins,), 1.0 / self.num_bins, dtype=torch.float32)
        
        # Normalize histogram
        hist_normalized = hist / torch.sum(hist)
        
        # Calculate uniformity penalty using KL Divergence
        uniformity_penalty = torch.sum(hist_normalized * torch.log(hist_normalized / target_uniform + 1e-8))
        
        # Combine MSE and uniformity penalty
        total_loss = mse + self.lambda_reg * uniformity_penalty
        return total_loss
    
class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.9):
        """
        Initialize the quantile loss function.
        
        Args:
        - quantile (float): The quantile to use for the loss calculation. Must be between 0 and 1.
        """
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantile loss.

        Args:
        - y_pred (torch.Tensor): Predicted values.
        - y_true (torch.Tensor): True values.

        Returns:
        - torch.Tensor: The quantile loss.
        """
        errors = y_true - y_pred
        loss = torch.max(
            (self.quantile - 1) * errors,
            self.quantile * errors
        )
        return loss.mean()
    
# Define a custom loss function with penalties for left and right tails
class CustomMSELoss(nn.Module):
    def __init__(self, left_percentile=15, right_percentile=85, left_penalty=6, right_penalty=6, variance_penalty_weight=1):
        super(CustomMSELoss, self).__init__()
        self.left_percentile = left_percentile
        self.right_percentile = right_percentile
        self.left_penalty = left_penalty
        self.right_penalty = right_penalty
        self.variance_penalty_weight = variance_penalty_weight

    def forward(self, y_pred, y_true):
        # Calculate the residuals
        residuals = y_true - y_pred
        
        # Convert tensors to CPU for percentile calculations
        y_true_np = y_true.detach().cpu().numpy()
        
        # Calculate the percentiles
        left_threshold = torch.tensor(np.percentile(y_true_np, self.left_percentile)).to(y_true.device)
        right_threshold = torch.tensor(np.percentile(y_true_np, self.right_percentile)).to(y_true.device)
        
        # Apply penalties: scale residuals based on whether they're in the left or right tail
        left_mask = y_true < left_threshold
        right_mask = y_true > right_threshold
        
        # Apply penalties to the residuals
        penalties = torch.ones_like(y_true)
        penalties[left_mask] *= self.left_penalty
        penalties[right_mask] *= self.right_penalty
        
        # Compute the scaled MSE
        mse_loss = torch.mean(penalties * residuals ** 2)
        
        # Variance penalty to encourage a wider distribution
        variance_penalty = torch.var(y_pred)
        
        # Combine MSE loss with variance penalty
        loss = mse_loss - self.variance_penalty_weight * variance_penalty
        
        return loss
    

def train(args):

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print(device)   

    networks = args.networks

    num_epochs = args.epochs + 1
    TS = args.TS

    batch_size_training = args.bs 
    batch_size_testing = 32
    dropout = args.dropout

    num_scales_gcn = args.gcn_scales
    num_scales_g3d = args.g3d_scales

    LR = args.LR

    W_list = args.windows

    data_path = os.path.join(args.data_path, 'node_timeseries/node_timeseries')
    parcel_path = args.parcel_path

    hparams = {}
    hparams['bs'] = args.bs
    hparams['nodes'] = args.nodes
    hparams['epochs'] = args.epochs + 1
    hparams['windows'] = args.windows
    hparams['g3d_scales'] = args.gcn_scales
    hparams['gcn_scales'] = args.g3d_scales
    hparams['dropout'] = args.dropout
    hparams['LR'] = args.LR
    hparams['optim'] = args.optim

    if args.regression:
        criterion = CustomMSELoss() #CustomLossWithUniformity() #TrendRemovalLoss() #CustomMSELoss() #nn.MSELoss() #nn.L1Loss()
    else:
        criterion = nn.BCELoss()

    print('')
    print('#' * 30)
    print('Logging training information')
    print('#' * 30)
    print('')

    results = []

    for network in networks:
        for ws in W_list:

            ##############################
            ######      LOGGING     ######
            ##############################

            # creating folders for logging.
            folder_to_save_model = './logs/MS-G3D/'
            os.makedirs(folder_to_save_model, exist_ok=True)
            print('Creating folder: {}'.format(folder_to_save_model))

            folder_to_save_model = os.path.join(folder_to_save_model, 'ROI_{}'.format(network))
            os.makedirs(folder_to_save_model, exist_ok=True)
            print('Creating folder: {}'.format(folder_to_save_model))

            folder_to_save_model = os.path.join(folder_to_save_model, 'ws_{}'.format(ws))
            os.makedirs(folder_to_save_model, exist_ok=True)
            print('Creating folder: {}'.format(folder_to_save_model))

            date = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

            folder_to_save_model = os.path.join(folder_to_save_model, date)
            os.makedirs(folder_to_save_model, exist_ok=True)
            print('Creating folder: {}'.format(folder_to_save_model))

            with open(os.path.join(folder_to_save_model, 'hparams.yaml'), 'w') as file:
                yaml.dump(hparams, file)

            ##############################
            ######     TRAINING     ######
            ##############################

            print('')
            print('#' * 30)
            print('Starting training')
            print('#' * 30)
            print('')

            print('-' * 80)
            print("Window Size {}".format(ws))
            print('-' * 80)

            print('-' * 80)
            print("Window Size {}".format(ws))
            print('-' * 80)

            best_test_auc_curr = 0
            best_test_r2_curr = -100000
            best_test_epoch_curr = 0
            best_val_loss = 10000

            train_data = np.load(os.path.join(data_path, 'train_data' + '.npy'))
            train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2], 1)
            train_label = np.load(os.path.join(data_path, 'train_label' + '.npy'))/12
            train_subjects = np.load(os.path.join(data_path, 'train_subjects' + '.npy'))

            validation_data = np.load(os.path.join(data_path, 'validation_data' + '.npy'))
            validation_data = validation_data.reshape(validation_data.shape[0], 1, validation_data.shape[1], validation_data.shape[2], 1)
            test_data = np.load(os.path.join(data_path, 'test_data' + '.npy'))
            test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2], 1)
            adj_matrix = np.load(os.path.join(data_path, 'adj_matrix.npy'))

            adj_matrix = adj_matrix - np.eye(len(adj_matrix), dtype=adj_matrix.dtype)

            if networks != ' ':# Load the parcels data
                parcels = pd.read_excel(parcel_path)
                
                if args.remove_network == True:
                    print('#'*30)
                    print('Removing Network {}'.format(network))
                    print('#'*30)

                    # Remove network from training
                    indices_to_remove = parcels[parcels['Community'] == network].index

                    # Filter out these indices from the data
                    remaining_indices = parcels.index.difference(indices_to_remove)

                    # Update the data structures by excluding the indices to be removed
                    train_data = train_data[:, :, :, remaining_indices, :]
                    test_data = test_data[:, :, :, remaining_indices, :]
                    validation_data = validation_data[:, :, :, remaining_indices, :]
                    adj_matrix = adj_matrix[np.ix_(remaining_indices, remaining_indices)]
                else:
                    print('#'*30)
                    print('Selecting Network {}'.format(network))
                    print('#'*30)

                    # Select network for training
                    communities_to_filter = [network]
                    filtered_indices = parcels[parcels['Community'].isin(communities_to_filter)].index
                    train_data = train_data[:, :, :, filtered_indices, :]
                    test_data = test_data[:, :, :, filtered_indices, :]
                    validation_data = validation_data[:, :, :, filtered_indices, :]
                    adj_matrix = adj_matrix[np.ix_(filtered_indices, filtered_indices)]

            if args.ensemble_networks != ' ':
                print('#'*30)
                print('Selecting Ensemble of Networks {}'.format(args.ensemble_networks))
                print('#'*30)

                parcels = pd.read_excel(parcel_path)
                communities_to_filter = args.ensemble_networks
                filtered_indices = parcels[parcels['Community'].isin(communities_to_filter)].index
                train_data = train_data[:, :, :, filtered_indices, :]
                test_data = test_data[:, :, :, filtered_indices, :]
                adj_matrix = adj_matrix[np.ix_(filtered_indices, filtered_indices)]

            test_label = np.load(os.path.join(data_path, 'test_label' + '.npy'))/12
            validation_label = np.load(os.path.join(data_path, 'validation_label' + '.npy'))/12
            test_subjects = np.load(os.path.join(data_path, 'test_subjects' + '.npy'))
            validation_subjects = np.load(os.path.join(data_path, 'validation_subjects' + '.npy'))


            print(train_data.shape)
            ROI_nodes = train_data.shape[3]

            net = Model(num_class=1,
                        num_nodes=ROI_nodes,
                        num_person=1,
                        num_gcn_scales=num_scales_gcn,
                        num_g3d_scales=num_scales_g3d,
                        adj_matrix=adj_matrix,
                        dropout=dropout)

            net.to(device)

            if args.optim == 'SGD':
                optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
            else:
                optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.001)

            for epoch in range(num_epochs):  # number of mini-batches
                net.train()

                idx_batch = np.random.permutation(int(train_data.shape[0]))
                idx_batch = idx_batch[:int(batch_size_training)]

                train_data_batch = np.zeros((batch_size_training, 1, ws, ROI_nodes, 1))
                train_label_batch = train_label[idx_batch]

                for i in range(batch_size_training):
                    r1 = random.randint(0, train_data.shape[2] - ws)
                    train_data_batch[i] = train_data[idx_batch[i], :, r1:r1 + ws, :, :]

                train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
                train_label_batch_dev = torch.from_numpy(train_label_batch).float().to(device)

                optimizer.zero_grad()
                outputs = net(train_data_batch_dev)[0]

                if args.regression:
                    loss = criterion(outputs.squeeze(), train_label_batch_dev)
                    correlation = pearsonr(outputs.data.cpu().numpy().reshape(-1),train_label_batch_dev.cpu())[0]
                else:
                    outputs = torch.sigmoid(outputs)

                loss = criterion(outputs.squeeze(), train_label_batch_dev)
                print(loss)
                #print(loss)
                outputs = outputs.data.cpu().numpy() > 0.5
                train_acc = sum(outputs[:, 0] == train_label_batch) / train_label_batch.shape[0]
                loss.backward()
                optimizer.step()

                epoch_val = args.epochs_val
                if epoch % epoch_val == 0:
                    print('')
                    print('Epoch', epoch)

                    net.eval()

                    ######################### Evaluate on the validation set ##############################
                    validation_prediction = np.zeros((validation_data.shape[0],))
                    voter = np.zeros((validation_data.shape[0],))

                    # Initialize an array to accumulate hidden layer outputs
                    validation_hidden_accumulation = None

                    # Initialize variable to accumulate validation loss
                    total_val_loss = 0

                    for v in range(TS):
                        idx = np.random.permutation(int(validation_data.shape[0]))

                        for k in range(int(validation_data.shape[0] / batch_size_testing)):
                            idx_batch = idx[int(batch_size_testing * k):int(batch_size_testing * (k + 1))]

                            validation_data_batch = np.zeros((batch_size_testing, 1, ws, ROI_nodes, 1))
                            for i in range(batch_size_testing):
                                r1 = random.randint(0, validation_data.shape[2] - ws)
                                validation_data_batch[i] = validation_data[idx_batch[i], :, r1:r1 + ws, :, :]

                            validation_data_batch_dev = torch.from_numpy(validation_data_batch).float().to(device)

                            # Get both the output and the hidden layer
                            outputs, layer = net(validation_data_batch_dev)

                            if args.regression:
                                outputs = outputs.squeeze().data.cpu().numpy()
                                layer = layer.squeeze().data.cpu().numpy()  # Convert hidden layer to numpy
                            else:
                                outputs = torch.sigmoid(outputs).squeeze().data.cpu().numpy()
                                layer = layer.squeeze().data.cpu().numpy()  # Convert hidden layer to numpy

                            # Compute the loss for the current batch
                            true_labels = torch.from_numpy(validation_label[idx_batch]).float().to(device)
                            outputs_tensor = torch.from_numpy(outputs).float().to(device)

                            # Calculate the batch loss
                            batch_loss = criterion(outputs_tensor, true_labels)
                            total_val_loss += batch_loss.item()

                            # Initialize the hidden layer accumulation array dynamically if not already initialized
                            if validation_hidden_accumulation is None:
                                hidden_layer_dim = layer.shape[1]  # Get the dimensionality of the hidden layer
                                validation_hidden_accumulation = np.zeros((validation_data.shape[0], hidden_layer_dim))

                            # Accumulate predictions
                            validation_prediction[idx_batch] += outputs
                            voter[idx_batch] += 1

                            # Accumulate hidden layer outputs for the validation set
                            validation_hidden_accumulation[idx_batch] += layer  # Accumulate the hidden layer outputs

                    # Average the accumulated hidden layer outputs
                    validation_hidden_avg = validation_hidden_accumulation / voter[:, None]  # Use broadcasting to divide by voter

                    # Convert the averaged hidden layer outputs to a DataFrame for saving
                    validation_hidden_df = pd.DataFrame({
                        'Subject': validation_subjects,
                        'Hidden Layer Output': [validation_hidden_avg[i].tolist() for i in range(validation_hidden_avg.shape[0])],  # Convert each vector to a list
                        'Set': 'Validation',
                        'Network': network,
                        'Ensemble_networks': '_'.join(args.ensemble_networks),
                    })

                    # Finalize validation predictions
                    validation_prediction = validation_prediction / voter

                    # Calculate the average validation loss
                    val_loss = total_val_loss / (TS * (validation_data.shape[0] / batch_size_testing))
                    print(f'Validation Loss: {val_loss}')

                    ######################### Evaluate on the test set ##############################
                    test_prediction = np.zeros((test_data.shape[0],))
                    voter = np.zeros((test_data.shape[0],))

                    # Initialize a list to store hidden layer outputs for the test set
                    test_hidden_accumulation = None

                    for v in range(TS):
                        idx = np.random.permutation(int(test_data.shape[0]))

                        for k in range(int(test_data.shape[0] / batch_size_testing)):
                            idx_batch = idx[int(batch_size_testing * k):int(batch_size_testing * (k + 1))]

                            test_data_batch = np.zeros((batch_size_testing, 1, ws, ROI_nodes, 1))
                            for i in range(batch_size_testing):
                                r1 = random.randint(0, test_data.shape[2] - ws)
                                test_data_batch[i] = test_data[idx_batch[i], :, r1:r1 + ws, :, :]

                            test_data_batch_dev = torch.from_numpy(test_data_batch).float().to(device)

                            # Get both the output and the hidden layer
                            outputs, layer = net(test_data_batch_dev)

                            if args.regression:
                                outputs = outputs.squeeze().data.cpu().numpy()
                                layer = layer.squeeze().data.cpu().numpy()
                            else:
                                outputs = torch.sigmoid(outputs).squeeze().data.cpu().numpy()
                                layer = layer.squeeze().data.cpu().numpy()

                            # Initialize the hidden layer accumulation array dynamically if not already initialized
                            if test_hidden_accumulation is None:
                                hidden_layer_dim = layer.shape[1]  # Get the dimensionality of the hidden layer
                                test_hidden_accumulation = np.zeros((test_data.shape[0], hidden_layer_dim))

                            test_prediction[idx_batch] += outputs
                            voter[idx_batch] += 1

                            # Accumulate hidden layer outputs for the test set
                            test_hidden_accumulation[idx_batch] += layer  # Accumulate the hidden layer outputs

                    # Average the accumulated hidden layer outputs
                    test_hidden_avg = test_hidden_accumulation / voter[:, None]  # Use broadcasting to divide by voter

                    # Convert the averaged hidden layer outputs to a DataFrame for saving
                    test_hidden_df = pd.DataFrame({
                        'Subject': test_subjects,
                        'Hidden Layer Output': [test_hidden_avg[i].tolist() for i in range(test_hidden_avg.shape[0])],  # Convert each vector to a list
                        'Set': 'Test',
                        'Network': network,
                        'Ensemble_networks': '_'.join(args.ensemble_networks),
                    })

                    # Finalize test predictions
                    test_prediction = test_prediction / voter

                    ####################### Predictions for training set #######################
                    train_prediction = np.zeros((train_data.shape[0],))
                    voter = np.zeros((train_data.shape[0],))

                    # Initialize an array to accumulate hidden layer outputs for the training set
                    train_hidden_accumulation = None 

                    for v in range(TS):
                        idx = np.random.permutation(int(train_data.shape[0]))

                        for k in range(int(train_data.shape[0] / batch_size_training)):
                            idx_batch = idx[int(batch_size_training * k):int(batch_size_training * (k + 1))]

                            train_data_batch = np.zeros((batch_size_training, 1, ws, ROI_nodes, 1))
                            for i in range(batch_size_training):
                                r1 = random.randint(0, train_data.shape[2] - ws)
                                train_data_batch[i] = train_data[idx_batch[i], :, r1:r1 + ws, :, :]

                            train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)

                            # Get both the output and the hidden layer
                            outputs, layer = net(train_data_batch_dev)

                            if args.regression:
                                outputs = outputs.squeeze().data.cpu().numpy()
                                layer = layer.squeeze().data.cpu().numpy()
                            else:
                                outputs = torch.sigmoid(outputs).squeeze().data.cpu().numpy()
                                layer = layer.squeeze().data.cpu().numpy()

                            # Initialize the hidden layer accumulation array dynamically if not already initialized
                            if train_hidden_accumulation is None:
                                hidden_layer_dim = layer.shape[1]  # Get the dimensionality of the hidden layer
                                train_hidden_accumulation = np.zeros((train_data.shape[0], hidden_layer_dim))


                            train_prediction[idx_batch] += outputs
                            voter[idx_batch] += 1

                            # Accumulate hidden layer outputs for the training set
                            train_hidden_accumulation[idx_batch] += layer  # Accumulate the hidden layer outputs

                    # Average the accumulated hidden layer outputs
                    train_hidden_avg = train_hidden_accumulation / voter[:, None]  # Use broadcasting to divide by voter

                    # Convert the averaged hidden layer outputs to a DataFrame for saving
                    train_hidden_df = pd.DataFrame({
                        'Subject': train_subjects,  # Assuming sequential subject IDs
                        'Hidden Layer Output': [train_hidden_avg[i].tolist() for i in range(train_hidden_avg.shape[0])],  # Convert each vector to a list
                        'Set': 'Train',
                        'Network': network,
                        'Ensemble_networks': '_'.join(args.ensemble_networks),
                    })

                    # Finalize train predictions
                    train_prediction = train_prediction / voter


                    if args.regression:
                        validation_mae = mean_absolute_error(validation_label, validation_prediction)
                        validation_r2 = r2_score(validation_label, validation_prediction)

                        print('Validation MAE:', validation_mae)
                        print('Validation R2:', validation_r2) 
                        print('Validation residuals', r2_score(validation_label, validation_prediction - validation_label))

                        if val_loss < best_val_loss:
                            best_val_loss = total_val_loss
                            best_test_r2_curr = validation_r2
                            best_test_epoch_curr = epoch
                            best_validation_prediction = validation_prediction
                            best_test_prediction = test_prediction
                            # ------ TRAIN/VAL/TEST PREDICTIONS
                            predictions_df = pred_sets(train_label, train_subjects, train_prediction, validation_label, validation_subjects,
                            validation_prediction, test_subjects, test_label, test_prediction, network, args.ensemble_networks)
                            hidden_df = pd.concat([train_hidden_df, validation_hidden_df, test_hidden_df])
                            print('Saving model for best validation performance')
                            torch.save(net.state_dict(), os.path.join(folder_to_save_model, 'checkpoint_best_val.pth'))
                    else:
                        validation_auc = roc_auc_score(validation_label, validation_prediction)
                        print('Validation AUC:', validation_auc)
                        if validation_auc > best_test_auc_curr:
                            best_test_auc_curr = validation_auc
                            best_test_epoch_curr = epoch
                            best_validation_prediction = validation_prediction
                            best_test_prediction = test_prediction
                            # ------ TRAIN/VAL/TEST PREDICTIONS
                            predictions_df = pred_sets(train_label, train_subjects, train_prediction, validation_label, validation_subjects,
                            validation_prediction, test_subjects, test_label, test_prediction, network, args.ensemble_networks)

                            print('Saving model for best validation performance')
                            torch.save(net.state_dict(), os.path.join(folder_to_save_model, 'checkpoint_best_val.pth'))


                    if args.regression:
                        test_mae = mean_absolute_error(test_label, best_test_prediction)
                        #print('Test MAE:', test_mae)
                    else:
                        test_auc = roc_auc_score(test_label, best_test_prediction)
                        #print('Test AUC:', test_auc)

            if args.regression:
                print("Best MAE for window {} and {} at epoch = {}".format(ws, best_test_r2_curr, best_test_epoch_curr))
                results.append([network, '_'.join(args.ensemble_networks), ws, best_test_r2_curr, best_test_epoch_curr])
            else:
                print("Best AUC for window {} and {} at epoch = {}".format(ws, best_test_auc_curr, best_test_epoch_curr))
                results.append([network, '_'.join(args.ensemble_networks), ws, best_test_auc_curr, best_test_epoch_curr])

        if args.regression:
            # Create a DataFrame and save it to a CSV file
            results_df = pd.DataFrame(results, columns=['Network', 'Ensemble Network', 'Window Size', 'Best R2', 'Epoch'])
        else:
            results_df = pd.DataFrame(results, columns=['Network', 'Ensemble Network', 'Window Size', 'Best AUC', 'Epoch'])

        if networks != ' ':
            results_df.to_csv('normative_results_{}_networks.csv'.format(ws), index=False)
            print("Results saved to normative_results_{}_networks.csv".format(ws))
            predictions_df.to_csv('normative_predictions_{}_networks.csv'.format(ws), index=False)
            print("Predictions saved to normative_predictions_{}_networks.csv".format(ws))
            hidden_df.to_csv('normative_hidden_{}_networks.csv'.format(ws), index=False)
            print("Hidden saved to normative_predictions_{}_networks.csv".format(ws))


        if args.ensemble_networks != ' ':
            results_df.to_csv('normative_results_{}_ensemble_networks.csv'.format(ws), index=False)
            print("Results saved to normative_results_{}_ensemble_networks.csv".format(ws))
            predictions_df.to_csv('normative_predictions_{}_ensemble_networks.csv'.format(ws), index=False)
            print("Predictions saved to normative_predictions_{}_ensemble_networks.csv".format(ws))
            hidden_df.to_csv('normative_hidden_{}_networks.csv'.format(ws), index=False)
            print("Hidden saved to normative_predictions_{}_networks.csv".format(ws))

        if networks == ' ' and args.ensemble_networks == ' ':
            results_df.to_csv('normative_results_{}_333.csv'.format(ws), index=False)
            print("Results saved to normative_results_{}_333.csv".format(ws))
            predictions_df.to_csv('normative_predictions_{}_333.csv'.format(ws), index=False)
            print("Predictions saved to normative_predictions_{}_333.csv".format(ws))
            hidden_df.to_csv('normative_hidden_{}_networks.csv'.format(ws), index=False)
            print("Hidden saved to normative_predictions_{}_networks.csv".format(ws))

    torch.cuda.empty_cache()

if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser(description='MS-G3D')

    parser.add_argument(
        '--nodes',
        type=int,
        required=True,
        help='the number of ICA nodes')

    parser.add_argument(
        '--bs',
        type=int,
        required=True,
        help='batch size training')

    parser.add_argument(
        '--epochs',
        type=int,
        required=True,
        help='number of iterations')

    parser.add_argument(
        '--gpu',
        type=int,
        required=True,
        help='gpu to use')

    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        default='',
        help='path where the data is stored')

    parser.add_argument(
        '--parcel_path',
        type=str,
        required=True,
        default='',
        help='path where the parcel info is stored')

    parser.add_argument(
        '--networks',
        metavar='S',
        type=str,
        required=False,
        default=' ',
        nargs='+',
        help='A list of networks for training')
    
    parser.add_argument(
        '--remove_network',
        metavar='S',
        type=bool,
        required=False,
        default=False,
        help='Remove network from traning')
    
    parser.add_argument(
        '--ensemble_networks',
        metavar='S',
        type=str,
        required=False,
        default=' ',
        nargs='+',
        help='A list of networks for training')
        
    parser.add_argument(
        '--regression',
        metavar='S',
        type=bool,
        required=False,
        default=False,
        help='task (classification or regression)')

    parser.add_argument(
        '--windows',
        type=int,
        required=True,
        nargs='+',
        help='windows')

    parser.add_argument(
        '--dropout',
        type=float,
        required=False,
        default=0.0,
        help='windows')

    parser.add_argument(
        '--gcn_scales',
        type=int,
        required=False,
        default=8,
        help='windows')

    parser.add_argument(
        '--g3d_scales',
        type=int,
        required=False,
        default=8,
        help='windows')

    parser.add_argument(
        '--LR',
        type=float,
        required=False,
        default=0.001,
        help='windows')

    parser.add_argument(
        '--optim',
        type=str,
        required=False,
        default='Adam',
        help='windows')

    parser.add_argument(
        '--TS',
        type=int,
        required=False,
        default=64,
        help='number of voters per test subject')

    parser.add_argument(
        '--epochs_val',
        type=int,
        required=False,
        default=100,
        help='number of iterations')

    args = parser.parse_args()
    # Call training
    train(args)