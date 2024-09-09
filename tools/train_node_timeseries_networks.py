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

from scipy.stats import pearsonr

#from torch.utils.tensorboard import SummaryWriter

from models.MSG3D.model import Model

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.cuda.empty_cache()

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
    hparams['epochs'] = args.epochs + 1
    hparams['windows'] = args.windows
    hparams['g3d_scales'] = args.gcn_scales
    hparams['gcn_scales'] = args.g3d_scales
    hparams['dropout'] = args.dropout
    hparams['LR'] = args.LR
    hparams['optim'] = args.optim

    if args.regression:
        criterion = nn.L1Loss() #nn.MSELoss()
    else:
        criterion = nn.BCELoss()

    print('')
    print('#' * 30)
    print('Logging training information')
    print('#' * 30)
    print('')

    results = []
    predictions_df = pd.DataFrame(columns=['Subject', 'Real Label','Prediction', 'Network', 'Ensemble_networks', 'fold'])

    for network in networks:
        for ws in W_list:
            testing_fold = []

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
            for fold in range(1, 6):
                print('-' * 80)
                print("Window Size {}, Fold {}".format(ws, fold))
                print('-' * 80)

                best_test_auc_curr_fold = 0
                best_test_mae_curr_fold = 100000
                best_test_epoch_curr_fold = 0

                train_data = np.load(os.path.join(data_path, 'train_data_' + str(fold) + '.npy'))
                train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2], 1)
                train_label = np.load(os.path.join(data_path, 'train_label_' + str(fold) + '.npy'))
                test_data = np.load(os.path.join(data_path, 'test_data_' + str(fold) + '.npy'))
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

                test_label = np.load(os.path.join(data_path, 'test_label_' + str(fold) + '.npy'))
                test_subjects = np.load(os.path.join(data_path, 'test_subjects_' + str(fold) + '.npy'))

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
                    #print(loss)
                    outputs = outputs.data.cpu().numpy() > 0.5
                    train_acc = sum(outputs[:, 0] == train_label_batch) / train_label_batch.shape[0]
                    loss.backward()
                    optimizer.step()

                    epoch_val = args.epochs_val
                    if epoch % epoch_val == 0:
                        net.eval()
                        idx_batch = np.random.permutation(int(test_data.shape[0]))
                        idx_batch = idx_batch[:int(batch_size_testing)]

                        test_label_batch = test_label[idx_batch]
                        prediction = np.zeros((test_data.shape[0],))
                        voter = np.zeros((test_data.shape[0],))
                        if args.regression:
                            losses = 0

                        for v in range(TS):
                            idx = np.random.permutation(int(test_data.shape[0]))

                            for k in range(int(test_data.shape[0] / batch_size_testing)):
                                idx_batch = idx[int(batch_size_testing * k):int(batch_size_testing * (k + 1))]

                                test_data_batch = np.zeros((batch_size_testing, 1, ws, ROI_nodes, 1))
                                for i in range(batch_size_testing):
                                    r1 = random.randint(0, test_data.shape[2] - ws)
                                    test_data_batch[i] = test_data[idx_batch[i], :, r1:r1 + ws, :, :]

                                test_data_batch_dev = torch.from_numpy(test_data_batch).float().to(device)

                                outputs = net(test_data_batch_dev)[0]
                                if args.regression:
                                    losses+= nn.functional.mse_loss(outputs.squeeze(), torch.from_numpy(test_label[idx_batch]).float().to(device)).detach().cpu().numpy()

                                outputs = outputs.data.cpu().numpy()

                                prediction[idx_batch] = prediction[idx_batch] + outputs[:, 0]
                                voter[idx_batch] = voter[idx_batch] + 1

                        prediction = prediction / voter

                        if args.regression:
                            test_mae = mean_absolute_error(prediction,test_label)
                            print(r2_score(prediction,test_label))
                            print('MAE:', test_mae)
                            print('[%d] testing batch MAE %f' % (epoch + 1, test_mae))
                            if test_mae < best_test_mae_curr_fold:
                                best_test_mae_curr_fold = test_mae
                                best_test_epoch_curr_fold = epoch
                                best_prediction = prediction
                                print('saving model')
                                torch.save(net.state_dict(), os.path.join(folder_to_save_model,'checkpoint.pth'))
                        else:
                            test_auc = roc_auc_score(test_label, prediction)
                            print('AUC:', test_auc)
                            print('[%d] testing batch AUC %f' % (epoch + 1, test_auc))
        
                            if test_auc > best_test_auc_curr_fold:
                                best_test_auc_curr_fold = test_auc
                                best_test_epoch_curr_fold = epoch
                                best_prediction = prediction
                                print('saving model')
                                torch.save(net.state_dict(), os.path.join(folder_to_save_model, 'checkpoint.pth'))

                # Save the predictions along with subject numbers and real labels
                fold_predictions_df = pd.DataFrame({
                    'Subject': test_subjects,
                    'Real Label': test_label,
                    'Prediction': best_prediction,
                    'Network' : network,
                    'Ensemble_networks' : '_'.join(args.ensemble_networks),
                    'fold' : fold
                 })
                predictions_df = pd.concat([predictions_df, fold_predictions_df])

                if args.regression:
                    print("Best r2 for window {} and fold {} = {} at epoch = {}".format(ws, fold, best_test_mae_curr_fold, best_test_epoch_curr_fold))
                    results.append([network, '_'.join(args.ensemble_networks), ws, fold, best_test_mae_curr_fold, best_test_epoch_curr_fold])

                else:
                    print("Best accuracy for window {} and fold {} = {} at epoch = {}".format(ws, fold, best_test_auc_curr_fold, best_test_epoch_curr_fold))
                    results.append([network, '_'.join(args.ensemble_networks), ws, fold, best_test_auc_curr_fold, best_test_epoch_curr_fold])

        if args.regression:
            # Create a DataFrame and save it to a CSV file
            results_df = pd.DataFrame(results, columns=['Network','Ensemble Network', 'Window Size', 'Fold', 'Best MAE', 'Epoch'])
        else:
            results_df = pd.DataFrame(results, columns=['Network','Ensemble Network', 'Window Size', 'Fold', 'Best AUC', 'Epoch'])

        if networks != ' ':
            # Create a DataFrame and save it to a CSV file
            results_df.to_csv('training_results_{}_networks.csv'.format(ws), index=False)
            print("Results saved to training_results.csv")
            # Save the predictions with real labels and subject numbers
            predictions_df.to_csv('predictions_{}_networks.csv'.format(ws), index=False)
            print("Predictions saved to predictions.csv")

        if args.ensemble_networks != ' ':
            # Create a DataFrame and save it to a CSV file
            results_df.to_csv('training_results_{}_ensemble_networks.csv'.format(ws), index=False)
            print("Results saved to training_results.csv")
            # Save the predictions with real labels and subject numbers
            predictions_df.to_csv('predictions_{}_ensemble_networks.csv'.format(ws), index=False)
            print("Predictions saved to predictions.csv")

        if networks == ' ' and args.ensemble_networks == ' ':
            # Create a DataFrame and save it to a CSV file
            results_df.to_csv('training_results_{}_333.csv'.format(ws), index=False)
            print("Results saved to training_333.csv")
            # Save the predictions with real labels and subject numbers
            predictions_df.to_csv('predictions_{}_333.csv'.format(ws), index=False)
            print("Predictions saved to predictions.csv")

    torch.cuda.empty_cache()

if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser(description='MS-G3D')

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