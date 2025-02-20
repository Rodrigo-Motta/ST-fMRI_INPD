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
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# If needed, append paths (though consider adjusting your project structure):
sys.path.append('../')
sys.path.append('./')

from models.MSG3D.model import Model

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.cuda.empty_cache()

def filter_data(args, network, train_data, test_data, adj_matrix, parcel_path):
    """
    Filters or removes specific networks (ROIs) from the data/adjacency matrix 
    based on the user's arguments.
    """
    # Load parcels data only once you need to filter or remove networks
    parcels = pd.read_excel(parcel_path)

    # If user wants to remove a single network
    if args.remove_network and network.strip() != '':
        print('#' * 30)
        print(f'Removing Network {network}')
        print('#' * 30)
        idx_remove = parcels[parcels['Community'] == network].index
        remain_idx = parcels.index.difference(idx_remove)

        train_data = train_data[:, :, :, remain_idx, :]
        test_data = test_data[:, :, :, remain_idx, :]
        adj_matrix = adj_matrix[np.ix_(remain_idx, remain_idx)]
        return train_data, test_data, adj_matrix

    # If user wants to select a single network
    if (not args.remove_network) and network.strip() != '':
        print('#' * 30)
        print(f'Selecting Network {network}')
        print('#' * 30)
        idx_select = parcels[parcels['Community'] == network].index

        train_data = train_data[:, :, :, idx_select, :]
        test_data  = test_data[:, :, :, idx_select, :]
        adj_matrix = adj_matrix[np.ix_(idx_select, idx_select)]
        return train_data, test_data, adj_matrix

    # If user wants an ensemble of networks
    if args.ensemble_networks and args.ensemble_networks[0].strip() != '':
        print('#' * 30)
        print(f'Selecting Ensemble of Networks {args.ensemble_networks}')
        print('#' * 30)
        idx_select = parcels[parcels['Community'].isin(args.ensemble_networks)].index

        train_data = train_data[:, :, :, idx_select, :]
        test_data  = test_data[:, :, :, idx_select, :]
        adj_matrix = adj_matrix[np.ix_(idx_select, idx_select)]
        return train_data, test_data, adj_matrix

    # If no network-based filtering:
    return train_data, test_data, adj_matrix

def create_output_folder(network, ws):
    """
    Creates and returns the output folder for logging and model checkpoints.
    """
    base_path = os.path.join(
        './logs/MS-G3D/',
        f'ROI_{network.strip()}',
        f'ws_{ws}',
        datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    )
    os.makedirs(base_path, exist_ok=True)
    print(f'Creating folder: {base_path}')
    return base_path

def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Hyperparameters and settings
    networks = args.networks
    num_epochs = args.epochs + 1
    TS = args.TS
    batch_size_training = args.bs 
    batch_size_testing  = 64
    dropout = args.dropout
    num_scales_gcn = args.gcn_scales
    num_scales_g3d = args.g3d_scales
    LR = args.LR
    W_list = args.windows

    data_path = os.path.join(args.data_path, 'node_timeseries/node_timeseries')
    parcel_path = args.parcel_path

    # Consolidate hyperparameters for logging
    hparams = {
        'bs': args.bs,
        'epochs': args.epochs + 1,
        'windows': args.windows,
        'g3d_scales': args.g3d_scales,
        'gcn_scales': args.gcn_scales,
        'dropout': args.dropout,
        'LR': args.LR,
        'optim': args.optim
    }

    # Choose criterion based on task
    criterion = nn.MSELoss() if args.regression else nn.BCELoss()

    print('\n' + '#' * 30)
    print('Logging training information')
    print('#' * 30 + '\n')

    results = []
    predictions_df = pd.DataFrame(
        columns=[
            'Subject', 'Real Label', 'Prediction', 
            'Network', 'Ensemble_networks', 'fold'
        ]
    )

    # Main loops over networks and window sizes
    for network in networks:
        for ws in W_list:
            # Create a folder for each (network, window) combo
            folder_to_save_model = create_output_folder(network, ws)
            # Save hyperparameters
            with open(os.path.join(folder_to_save_model, 'hparams.yaml'), 'w') as file:
                yaml.dump(hparams, file)

            print('\n' + '#' * 30)
            print('Starting training')
            print('#' * 30 + '\n')

            print('-' * 80)
            print(f"Window Size {ws}")
            print('-' * 80)

            # 5-fold cross-validation
            for fold in range(1, 6):
                print('-' * 80)
                print(f"Window Size {ws}, Fold {fold}")
                print('-' * 80)

                # Track best metrics
                best_test_auc_curr_fold = 0
                best_test_mae_curr_fold = float('inf')
                best_test_epoch_curr_fold = 0
                best_prediction = None

                # --- Load Data ---
                train_data = np.load(os.path.join(data_path, f'train_data_{fold}.npy'))
                train_label = np.load(os.path.join(data_path, f'train_label_{fold}.npy'))

                test_data = np.load(os.path.join(data_path, f'test_data_{fold}.npy'))
                test_label = np.load(os.path.join(data_path, f'test_label_{fold}.npy'))
                test_subjects = np.load(os.path.join(data_path, f'test_subjects_{fold}.npy'))

                # Reshape data to [N, C, T, V, M] as needed by MSG3D
                train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2], 1)
                test_data  = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2], 1)

                adj_matrix = np.load(os.path.join(data_path, 'adj_matrix.npy'))
                # Remove self-loops
                adj_matrix = adj_matrix - np.eye(len(adj_matrix), dtype=adj_matrix.dtype)

                # Filter data according to networks / ensemble
                train_data, test_data, adj_matrix = filter_data(
                    args, network, train_data, test_data, adj_matrix, parcel_path
                )

                print(f"Train data shape: {train_data.shape}")
                ROI_nodes = train_data.shape[3]

                # --- Build Model ---
                net = Model(
                    num_class=1,
                    num_nodes=ROI_nodes,
                    num_person=1,
                    num_gcn_scales=num_scales_gcn,
                    num_g3d_scales=num_scales_g3d,
                    adj_matrix=adj_matrix,
                    dropout=dropout
                ).to(device)

                # --- Optimizer ---
                if args.optim.lower() == 'sgd':
                    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
                else:
                    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.001)

                # --- Training Loop ---
                for epoch in range(num_epochs):
                    net.train()
                    # Randomly select a batch for training
                    idx_batch = np.random.permutation(train_data.shape[0])[:batch_size_training]

                    # Create a batch with random window slices
                    train_data_batch = np.zeros((batch_size_training, 1, ws, ROI_nodes, 1))
                    for i in range(batch_size_training):
                        r1 = random.randint(0, train_data.shape[2] - ws)
                        train_data_batch[i] = train_data[idx_batch[i], :, r1 : r1 + ws, :, :]

                    train_label_batch = train_label[idx_batch]

                    # Move to device
                    train_data_batch_dev  = torch.from_numpy(train_data_batch).float().to(device)
                    train_label_batch_dev = torch.from_numpy(train_label_batch).float().to(device)

                    # Forward + loss + backward
                    optimizer.zero_grad()
                    outputs = net(train_data_batch_dev)[0]

                    if args.regression:
                        loss = criterion(outputs.squeeze(), train_label_batch_dev)
                        correlation = pearsonr(
                            outputs.data.cpu().numpy().reshape(-1),
                            train_label_batch_dev.cpu()
                        )[0]
                    else:
                        outputs = torch.sigmoid(outputs)
                        loss = criterion(outputs.squeeze(), train_label_batch_dev)

                    # Compute simple accuracy (for classification)
                    if not args.regression:
                        binary_pred = (outputs.data.cpu().numpy() > 0.5).astype(int)
                        train_acc = (binary_pred[:, 0] == train_label_batch).mean()

                    loss.backward()
                    optimizer.step()

                    # --- Validation step ---
                    # Evaluate on the testing set every "args.epochs_val" epochs
                    if epoch % args.epochs_val == 0:
                        net.eval()
                        # We do multiple "votes" (TS) for each sample
                        prediction = np.zeros((test_data.shape[0],))
                        voter = np.zeros((test_data.shape[0],))

                        # For regression, track MSE across votes
                        total_mse = 0.0 if args.regression else None

                        idx = np.random.permutation(test_data.shape[0])
                        # Each subject gets TS random slices
                        for v in range(TS):
                            idx = np.random.permutation(test_data.shape[0])
                            for k in range(test_data.shape[0] // batch_size_testing):
                                batch_indices = idx[k * batch_size_testing : (k + 1) * batch_size_testing]
                                test_data_batch = np.zeros((batch_size_testing, 1, ws, ROI_nodes, 1))

                                for i in range(batch_size_testing):
                                    r1 = random.randint(0, test_data.shape[2] - ws)
                                    test_data_batch[i] = test_data[batch_indices[i], :, r1 : r1 + ws, :, :]

                                test_data_batch_dev = torch.from_numpy(test_data_batch).float().to(device)
                                out = net(test_data_batch_dev)[0]

                                if args.regression:
                                    # Accumulate MSE for monitoring
                                    total_mse += nn.functional.mse_loss(
                                        out.squeeze(), 
                                        torch.from_numpy(test_label[batch_indices]).float().to(device)
                                    ).item()

                                out = out.data.cpu().numpy()  # shape: (batch_size, 1)
                                prediction[batch_indices] += out[:, 0]
                                voter[batch_indices] += 1

                        prediction /= voter

                        # --- Compute performance metrics ---
                        if args.regression:
                            test_mae = mean_absolute_error(prediction, test_label)
                            print(f'R^2: {r2_score(prediction, test_label):.4f}')
                            print(f'MAE: {test_mae:.4f}')
                            print(f'[Epoch {epoch+1}] testing batch MAE {test_mae:.4f}')

                            if test_mae < best_test_mae_curr_fold:
                                best_test_mae_curr_fold = test_mae
                                best_test_epoch_curr_fold = epoch
                                best_prediction = prediction.copy()
                                print('Saving model (best so far)...')
                                torch.save(net.state_dict(), os.path.join(folder_to_save_model, 'checkpoint.pth'))
                        else:
                            test_auc = roc_auc_score(test_label, prediction)
                            print(f'AUC: {test_auc:.4f}')
                            print(f'[Epoch {epoch+1}] testing batch AUC {test_auc:.4f}')

                            if test_auc > best_test_auc_curr_fold:
                                best_test_auc_curr_fold = test_auc
                                best_test_epoch_curr_fold = epoch
                                best_prediction = prediction.copy()
                                print('Saving model (best so far)...')
                                torch.save(net.state_dict(), os.path.join(folder_to_save_model, 'checkpoint.pth'))

                # --- After training completes for this fold ---
                fold_predictions_df = pd.DataFrame({
                    'Subject': test_subjects,
                    'Real Label': test_label,
                    'Prediction': best_prediction,
                    'Network': network,
                    'Ensemble_networks': '_'.join(args.ensemble_networks),
                    'fold': fold
                })
                predictions_df = pd.concat([predictions_df, fold_predictions_df], ignore_index=True)

                if args.regression:
                    print(f"Best MAE for window {ws} fold {fold} = {best_test_mae_curr_fold} "
                          f"at epoch {best_test_epoch_curr_fold}")
                    results.append([
                        network, '_'.join(args.ensemble_networks), 
                        ws, fold, best_test_mae_curr_fold, best_test_epoch_curr_fold
                    ])
                else:
                    print(f"Best AUC for window {ws} fold {fold} = {best_test_auc_curr_fold} "
                          f"at epoch {best_test_epoch_curr_fold}")
                    results.append([
                        network, '_'.join(args.ensemble_networks), 
                        ws, fold, best_test_auc_curr_fold, best_test_epoch_curr_fold
                    ])

        # --- After finishing all folds for a given network ---
        # Prepare results DataFrame
        if args.regression:
            results_df = pd.DataFrame(results, columns=[
                'Network', 'Ensemble Network', 'Window Size', 
                'Fold', 'Best MAE', 'Epoch'
            ])
        else:
            results_df = pd.DataFrame(results, columns=[
                'Network', 'Ensemble Network', 'Window Size', 
                'Fold', 'Best AUC', 'Epoch'
            ])

        # Decide on file naming
        # (You might want to refine this logic to avoid overwriting if multiple networks are used)
        if network.strip() != '' and args.ensemble_networks[0].strip() == '':
            suffix = f'{ws}_networks'
        elif args.ensemble_networks[0].strip() != '':
            suffix = f'{ws}_ensemble_networks'
        else:
            suffix = f'{ws}_333'

        # Save results and predictions
        results_file = f'training_results_{suffix}.csv'
        preds_file   = f'predictions_{suffix}.csv'

        results_df.to_csv(results_file, index=False)
        predictions_df.to_csv(preds_file, index=False)

        print(f"Results saved to {results_file}")
        print(f"Predictions saved to {preds_file}")

    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MS-G3D')

    parser.add_argument('--bs', type=int, required=True, help='Batch size for training')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--gpu', type=int, required=True, help='GPU device ID')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data')
    parser.add_argument('--parcel_path', type=str, required=True, help='Path to the parcel info')
    
    parser.add_argument('--networks', metavar='S', type=str, nargs='+', default=[' '],
                        help='List of networks for training (or " " if none)')
    parser.add_argument('--remove_network', action='store_true',
                        help='Whether to remove the network from training instead of selecting it')

    parser.add_argument('--ensemble_networks', metavar='S', type=str, nargs='+', default=[' '],
                        help='List of networks for ensemble training (or " " if none)')

    parser.add_argument('--regression', action='store_true',
                        help='Use regression (MSELoss) instead of classification (BCELoss)?')

    parser.add_argument('--windows', type=int, nargs='+', required=True, help='List of window sizes')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--gcn_scales', type=int, default=8, help='Number of GCN scales')
    parser.add_argument('--g3d_scales', type=int, default=8, help='Number of G3D scales')
    parser.add_argument('--LR', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optim', type=str, default='Adam', help='Optimizer (SGD or Adam)')
    parser.add_argument('--TS', type=int, default=64, help='Number of voting iterations per test sample')
    parser.add_argument('--epochs_val', type=int, default=100, help='Validation interval')

    args = parser.parse_args()
    train(args)