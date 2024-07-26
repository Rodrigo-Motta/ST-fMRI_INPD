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


from scipy.stats import pearsonr

#from torch.utils.tensorboard import SummaryWriter

from models.MSG3D.model import Model

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()


def train(args):

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    #device = torch.device("mps:{}".format(args.gpu) if torch.backends.mps.is_available() else "cpu")
    print(device)   

    #ROI_nodes = args.nodes
    network = args.network
    num_epochs = args.epochs + 1
    TS = args.TS

    batch_size_training = args.bs 
    batch_size_testing = 32
    dropout = args.dropout

    num_scales_gcn = args.gcn_scales
    num_scales_g3d = args.g3d_scales

    LR = args.LR

    W_list = args.windows

    data_path = os.path.join(args.data_path,'node_timeseries/node_timeseries')
    parcel_path = args.parcel_path


    hparams = {}
    hparams['bs'] = args.bs
    hparams['nodes'] = args.nodes
    hparams['epochs'] = args.epochs + 1
    hparams['windows'] = args.windows
    hparams['windows'] = args.TS
    hparams['g3d_scales'] = args.gcn_scales
    hparams['gcn_scales'] = args.g3d_scales
    hparams['dropout'] = args.dropout
    hparams['LR'] = args.LR
    hparams['optim'] = args.optim

    

    criterion = nn.BCELoss()

    print('')
    print('#'*30)
    print('Logging training information')
    print('#'*30)
    print('')



    results = []

    for ws in W_list:
        
        testing_fold = []
        
        ##############################
        ######      LOGGING     ######
        ##############################

        # creating folders for logging. 

        folder_to_save_model = './logs/MS-G3D/'
        os.makedirs(folder_to_save_model, exist_ok=True)
        print('Creating folder: {}'.format(folder_to_save_model))

        # folder ICAs
        folder_to_save_model = os.path.join(folder_to_save_model, 'ROI_{}'.format(network))
        os.makedirs(folder_to_save_model, exist_ok=True)
        print('Creating folder: {}'.format(folder_to_save_model))
        
        # folder ws
        folder_to_save_model = os.path.join(folder_to_save_model, 'ws_{}'.format(ws))
        os.makedirs(folder_to_save_model, exist_ok=True)
        print('Creating folder: {}'.format(folder_to_save_model))

        date = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

        # folder time
        folder_to_save_model = os.path.join(folder_to_save_model, date)
        os.makedirs(folder_to_save_model, exist_ok=True)
        print('Creating folder: {}'.format(folder_to_save_model))

        # saving hparams
        with open(os.path.join(folder_to_save_model, 'hparams.yaml'), 'w') as file:
            yaml.dump(hparams, file)

        # starting tensorboard logging
        #writer = SummaryWriter(log_dir=folder_to_save_model)
        print('Network', network)

        ##############################
        ######     TRAINING     ######
        ##############################

        print('')
        print('#'*30)
        print('Starting training')
        print('#'*30)
        print('')

        print('-'*80)
        print("Window Size {}".format(ws))
        print('-'*80)
        for fold in range(1, 6):
            print('-'*80)
            print("Window Size {}, Fold {}".format(ws, fold))
            print('-'*80)
            
            best_test_acc_curr_fold = 0
            best_test_epoch_curr_fold = 0
            best_test_corr_curr_fold =0
            
            train_data = np.load(os.path.join(data_path,'train_data_'+str(fold)+'.npy'))
            train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2], 1)
            train_label = np.load(os.path.join(data_path,'train_label_'+str(fold)+'.npy'))
            test_data = np.load(os.path.join(data_path,'test_data_'+str(fold)+'.npy'))
            test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2], 1)
            adj_matrix = np.load(os.path.join(data_path,'adj_matrix.npy'))

            adj_matrix = adj_matrix - np.eye(len(adj_matrix), dtype=adj_matrix.dtype)
            
            if network != '':
                # Read parcels info
                parcels = pd.read_excel(parcel_path)

                # Specify the communities to filter by
                communities_to_filter = [network]

                # Identify the indices of the parcels belonging to the specified communities
                filtered_indices = parcels[parcels['Community'].isin(communities_to_filter)].index

                # Filter the arr array using these indices
                train_data = train_data[:, :, :, filtered_indices, :]
                test_data = test_data[:, :, :, filtered_indices, :]
                adj_matrix = adj_matrix[np.ix_(filtered_indices, filtered_indices)]

            test_label = np.load(os.path.join(data_path,'test_label_'+str(fold)+'.npy'))

            print(train_data.shape)
            print(adj_matrix.shape)
            ROI_nodes = train_data.shape[3]

            net = Model(num_class = 1,
                    num_nodes = ROI_nodes,
                    num_person = 1,
                    num_gcn_scales = num_scales_gcn,
                    num_g3d_scales = num_scales_g3d,
                    adj_matrix = adj_matrix,
                    dropout= dropout)
        
            net.to(device)

            if args.optim == 'SGD':
                optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
            else:
                optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.001)

            

            for epoch in range(num_epochs):  # number of mini-batches
                # select a random sub-set of subjects
                idx_batch = np.random.permutation(int(train_data.shape[0]))
                idx_batch = idx_batch[:int(batch_size_training)]
        
                # construct a mini-batch by sampling a window W for each subject
                train_data_batch = np.zeros((batch_size_training, 1, ws, ROI_nodes, 1))
                train_label_batch = train_label[idx_batch]

                for i in range(batch_size_training):
                    r1 = random.randint(0, train_data.shape[2] - ws)
                    train_data_batch[i] = train_data[idx_batch[i], :, r1:r1 + ws, :, :]

                train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
                train_label_batch_dev = torch.from_numpy(train_label_batch).float().to(device)

                # forward + backward + optimize
                optimizer.zero_grad()

                outputs = net(train_data_batch_dev)

                outputs = torch.sigmoid(outputs)
                #print(outputs)


                #if torch.any(train_data_batch_dev < 0) or torch.any(train_data_batch_dev > 1):
                #    print(f"Invalid input values found in training data: {train_data_batch_dev}")
                # if torch.any(train_label_batch_dev < 0) or torch.any(train_label_batch_dev > 1):
                #     print(f"Invalid target values found in training data label: {train_label_batch_dev}")

                loss = criterion(outputs.squeeze(), train_label_batch_dev)
                #print(loss)

                #writer.add_scalar('loss/train_{}'.format(fold), loss.item(), epoch+1)
                outputs = outputs.data.cpu().numpy() > 0.5
                train_acc = sum(outputs[:, 0] == train_label_batch) / train_label_batch.shape[0]
                #writer.add_scalar('accuracy/train_{}'.format(fold), train_acc, epoch+1)

                loss.backward()
                optimizer.step()

                ##############################
                ######    VALIDATION    ######
                ##############################


                # validate on test subjects by voting
                epoch_val = args.epochs_val
                if epoch % epoch_val == 0:  # print every K mini-batches
                    idx_batch = np.random.permutation(int(test_data.shape[0]))
                    idx_batch = idx_batch[:int(batch_size_testing)]

                    test_label_batch = test_label[idx_batch]
                    prediction = np.zeros((test_data.shape[0],))
                    voter = np.zeros((test_data.shape[0],))
                    # if args.fluid:
                    #     losses = 0
                    for v in range(TS):
                        idx = np.random.permutation(int(test_data.shape[0]))

                        # testing also performed batch by batch (otherwise it produces error)
                        for k in range(int(test_data.shape[0] / batch_size_testing)):
                            idx_batch = idx[int(batch_size_testing * k):int(batch_size_testing * (k + 1))]

                            # construct random sub-sequences from a batch of test subjects
                            test_data_batch = np.zeros((batch_size_testing, 1, ws, ROI_nodes, 1))
                            for i in range(batch_size_testing):
                                r1 = random.randint(0, test_data.shape[2] - ws)
                                test_data_batch[i] = test_data[idx_batch[i], :, r1:r1 + ws, :, :]

                            test_data_batch_dev = torch.from_numpy(test_data_batch).float().to(device)
                        
                            outputs = net(test_data_batch_dev)
                            outputs = outputs.data.cpu().numpy()

                            prediction[idx_batch] = prediction[idx_batch] + outputs[:, 0]
                            voter[idx_batch] = voter[idx_batch] + 1

                    prediction = prediction / voter

                    # Assuming `prediction` contains your model's output probabilities and `test_label` contains the true labels
                    # Calculate AUC
                    test_auc = roc_auc_score(test_label, prediction)
                    print('AUC:', test_auc)
                    print('[%d] testing batch AUC %f' % (epoch + 1, test_auc))

                    # Logging AUC to the writer instead of accuracy
                    # writer.add_scalar('auc/test_{}'.format(fold), test_auc, epoch+1)

                    # Save model if current AUC is the best
                    if test_auc > best_test_acc_curr_fold:
                        best_test_acc_curr_fold = test_auc
                        best_test_epoch_curr_fold = epoch
                        print('saving model')
                        torch.save(net.state_dict(), os.path.join(folder_to_save_model, 'checkpoint.pth'))

            print("Best accuracy for window {} and fold {} = {} at epoch = {}".format(ws, fold, best_test_acc_curr_fold, best_test_epoch_curr_fold))
            testing_fold.append(best_test_acc_curr_fold)
            results.append([network, ws, fold, best_test_auc_curr_fold, best_test_epoch_curr_fold])

    # Create a DataFrame and save it to a CSV file
    results_df = pd.DataFrame(results, columns=['Network', 'Window Size', 'Fold', 'Best AUC', 'Epoch'])
    results_df.to_csv('training_results_{}_333.csv'.format(ws), index=False)
    print("Results saved to training_results.csv")
            #writer.add_scalar('accuracy_best/test'.format(fold), best_test_acc_curr_fold, fold)
        


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
                        '--network',
                        type=str,
                        required=False,
                        default='',
                        help='which network to select')
                        
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