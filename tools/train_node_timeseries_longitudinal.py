#
# Created on Tue Jul 18 2024
#
# by Rodrigo M. Cabral-Carvalho
# Adapted from Simon Dahan @SD3004
# Updated for single-pass (train/val/test) with new preprocessing
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

##############################################################################
# Utility: Filter or remove specific networks from data/adj_matrix
##############################################################################
def filter_data(args, network, data, adj_matrix, parcel_path):
    """
    Filters or removes specific networks (ROIs) from data/adjacency matrix 
    based on the user's arguments (remove_network, ensemble_networks, etc.).
    
    data is assumed shape [N, 1, T, V, 1].
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
        data = data[:, :, :, remain_idx, :]
        adj_matrix = adj_matrix[np.ix_(remain_idx, remain_idx)]
        return data, adj_matrix

    # If user wants to select a single network
    if (not args.remove_network) and network.strip() != '':
        print('#' * 30)
        print(f'Selecting Network {network}')
        print('#' * 30)
        idx_select = parcels[parcels['Community'] == network].index
        data = data[:, :, :, idx_select, :]
        adj_matrix = adj_matrix[np.ix_(idx_select, idx_select)]
        return data, adj_matrix

    # If user wants an ensemble of networks
    if args.ensemble_networks and args.ensemble_networks[0].strip() != '':
        print('#' * 30)
        print(f'Selecting Ensemble of Networks {args.ensemble_networks}')
        print('#' * 30)
        idx_select = parcels[parcels['Community'].isin(args.ensemble_networks)].index
        data = data[:, :, :, idx_select, :]
        adj_matrix = adj_matrix[np.ix_(idx_select, idx_select)]
        return data, adj_matrix

    # If no network-based filtering:
    return data, adj_matrix

##############################################################################
# Utility: Create output folder
##############################################################################
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

##############################################################################
# Main training function
##############################################################################
def train(args):
    """
    Single-pass training: 
      - Load train, val, test data 
      - Train for a fixed number of epochs 
      - Use val to find best epoch 
      - Evaluate on test with best checkpoint
    """
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Hyperparameters and settings
    networks = args.networks  # e.g. ["Vis", "Aud"] or [" "] if none
    num_epochs = args.epochs + 1
    TS = args.TS
    batch_size_training = args.bs
    batch_size_eval = 64  # for val/test
    dropout = args.dropout
    num_scales_gcn = args.gcn_scales
    num_scales_g3d = args.g3d_scales
    LR = args.LR
    W_list = args.windows  # list of window sizes to try
    data_path = os.path.join(args.data_path, 'node_timeseries', 'node_timeseries')
    parcel_path = args.parcel_path

    # Consolidate hyperparameters for logging
    hparams = {
        'batch_size': args.bs,
        'epochs': args.epochs,
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

    # Results containers
    results = []
    predictions_df = pd.DataFrame(columns=['Subject', 'Real Label', 'Prediction', 'Network', 
                                           'Ensemble_networks', 'SetType'])  # 'SetType': "Val" or "Test"

    # -----------------------------
    # Load the data from new preprocessing
    # -----------------------------
    # For shape reference: (N, T, ROI_nodes) from your script
    train_data = np.load(os.path.join(data_path, 'train_data_wave1.npy'))  # shape: (N_train, T, V)
    train_label = np.load(os.path.join(data_path, 'train_label_wave1.npy'))
    val_data   = np.load(os.path.join(data_path, 'val_data_wave1.npy'))
    val_label  = np.load(os.path.join(data_path, 'val_label_wave1.npy'))
    test_data  = np.load(os.path.join(data_path, 'test_data_wave2.npy'))
    test_label = np.load(os.path.join(data_path, 'test_label_wave2.npy'))

    # If you also saved subject IDs, load them as well:
    try:
        train_subjects = np.load(os.path.join(data_path, 'train_subjects_wave1.npy'))
        val_subjects   = np.load(os.path.join(data_path, 'val_subjects_wave1.npy'))
        test_subjects  = np.load(os.path.join(data_path, 'test_subjects_wave2.npy'))
    except FileNotFoundError:
        # If you didn't save subject arrays, skip
        train_subjects = np.arange(len(train_data))
        val_subjects   = np.arange(len(val_data))
        test_subjects  = np.arange(len(test_data))

    # Reshape data to [N, C=1, T, V, M=1] as needed by MSG3D
    train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2], 1)
    val_data   = val_data.reshape(val_data.shape[0], 1, val_data.shape[1], val_data.shape[2], 1)
    test_data  = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2], 1)

    # Load adjacency and remove self-loops
    adj_matrix = np.load(os.path.join(data_path, 'adj_matrix.npy'))
    adj_matrix = adj_matrix - np.eye(len(adj_matrix), dtype=adj_matrix.dtype)

    # -----------------------------
    # Outer loops: networks, windows
    # -----------------------------
    for network in networks:
        for ws in W_list:
            # Create a folder for each (network, window) combo
            folder_to_save_model = create_output_folder(network, ws)
            # Save hyperparameters
            with open(os.path.join(folder_to_save_model, 'hparams.yaml'), 'w') as file:
                yaml.dump(hparams, file)

            print('\n' + '#' * 30)
            print(f'Starting training: network={network}, window={ws}')
            print('#' * 30 + '\n')

            # 1) Filter data for the specified network or ensemble
            #    We do that for train/val/test, and the adjacency
            #    so all of them keep consistent ROI dimension
            filtered_train_data, filtered_adj = filter_data(
                args, network, train_data, adj_matrix, parcel_path
            )
            filtered_val_data, filtered_adj   = filter_data(
                args, network, val_data, adj_matrix, parcel_path
            )
            filtered_test_data, filtered_adj  = filter_data(
                args, network, test_data, adj_matrix, parcel_path
            )

            # The final adjacency after all filtering
            # (Note: the repeated calls to filter_data won't double-filter 
            #  unless you have conflicting arguments. Usually you'd just pass
            #  the same arguments, so it won't do anything the second time 
            #  if "network" or "ensemble_networks" didn't change.)
            # If you prefer to filter train_data, val_data, test_data in a single pass,
            # you can combine them first or call the function carefully. 
            # For simplicity we do sequential calls here.

            # Check shape
            print(f"Train data shape: {filtered_train_data.shape}")
            print(f"Val   data shape: {filtered_val_data.shape}")
            print(f"Test  data shape: {filtered_test_data.shape}")

            ROI_nodes = filtered_train_data.shape[3]

            # 2) Build the MSG3D Model
            net = Model(
                num_class=1,
                num_nodes=ROI_nodes,
                num_person=1,
                num_gcn_scales=num_scales_gcn,
                num_g3d_scales=num_scales_g3d,
                adj_matrix=filtered_adj,
                dropout=dropout
            ).to(device)

            # 3) Choose optimizer
            if args.optim.lower() == 'sgd':
                optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
            else:
                optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.001)

            # Track best epoch via validation
            best_val_metric = float('inf') if args.regression else 0.0
            best_val_epoch = 0
            best_model_path = os.path.join(folder_to_save_model, 'checkpoint_best.pth')

            # 4) Training Loop
            for epoch in range(num_epochs):
                net.train()
                # Randomly select a batch for training
                idx_batch = np.random.permutation(filtered_train_data.shape[0])[:batch_size_training]

                # Build a random-window batch
                train_data_batch = np.zeros((batch_size_training, 1, ws, ROI_nodes, 1))
                for i in range(batch_size_training):
                    r1 = random.randint(0, filtered_train_data.shape[2] - ws)
                    train_data_batch[i] = filtered_train_data[idx_batch[i], :, r1 : r1 + ws, :, :]

                train_label_batch = train_label[idx_batch]

                # Move to device
                train_data_batch_dev  = torch.from_numpy(train_data_batch).float().to(device)
                train_label_batch_dev = torch.from_numpy(train_label_batch).float().to(device)

                optimizer.zero_grad()
                outputs = net(train_data_batch_dev)[0]  # shape: (bs, 1)

                if args.regression:
                    loss = criterion(outputs.squeeze(), train_label_batch_dev)
                else:
                    outputs_sig = torch.sigmoid(outputs)
                    loss = criterion(outputs_sig.squeeze(), train_label_batch_dev)

                loss.backward()
                optimizer.step()

                # 5) Validation step (periodic)
                if epoch % args.epochs_val == 0:
                    net.eval()
                    # Let's do multiple "votes" for val, just like test
                    val_pred = np.zeros((filtered_val_data.shape[0],))
                    voter_count = np.zeros((filtered_val_data.shape[0],))

                    # We'll do TS random slices for each subject
                    for v in range(TS):
                        idx_perm = np.random.permutation(filtered_val_data.shape[0])
                        for k in range(filtered_val_data.shape[0] // batch_size_eval):
                            batch_indices = idx_perm[k * batch_size_eval : (k + 1) * batch_size_eval]
                            val_data_batch = np.zeros((batch_size_eval, 1, ws, ROI_nodes, 1))
                            for i in range(batch_size_eval):
                                r1 = random.randint(0, filtered_val_data.shape[2] - ws)
                                val_data_batch[i] = filtered_val_data[batch_indices[i], :, r1 : r1 + ws, :, :]

                            val_data_batch_dev = torch.from_numpy(val_data_batch).float().to(device)
                            out = net(val_data_batch_dev)[0].data.cpu().numpy().reshape(-1)
                            val_pred[batch_indices] += out
                            voter_count[batch_indices] += 1

                    val_pred /= voter_count

                    # Evaluate performance
                    if args.regression:
                        val_mae = mean_absolute_error(val_label, val_pred)
                        print(f"[Epoch {epoch}] Val MAE: {val_mae:.4f}")

                        # Update best checkpoint
                        if val_mae < best_val_metric:
                            best_val_metric = val_mae
                            best_val_epoch = epoch
                            torch.save(net.state_dict(), best_model_path)
                            print("  Best val so far; saved model.")
                    else:
                        val_auc = roc_auc_score(val_label, val_pred)
                        print(f"[Epoch {epoch}] Val AUC: {val_auc:.4f}")

                        if val_auc > best_val_metric:
                            best_val_metric = val_auc
                            best_val_epoch = epoch
                            torch.save(net.state_dict(), best_model_path)
                            print("  Best val so far; saved model.")

            print(f"\nTraining complete. Best epoch was {best_val_epoch} with "
                  f"{('MAE' if args.regression else 'AUC')}={best_val_metric:.4f}\n")

            # 6) Final Test with best checkpoint
            net.load_state_dict(torch.load(best_model_path))
            net.eval()

            # Multi-vote test
            test_pred = np.zeros((filtered_test_data.shape[0],))
            voter_count = np.zeros((filtered_test_data.shape[0],))

            for v in range(TS):
                idx_perm = np.random.permutation(filtered_test_data.shape[0])
                for k in range(filtered_test_data.shape[0] // batch_size_eval):
                    batch_indices = idx_perm[k * batch_size_eval : (k + 1) * batch_size_eval]
                    test_data_batch = np.zeros((batch_size_eval, 1, ws, ROI_nodes, 1))

                    for i in range(batch_size_eval):
                        r1 = random.randint(0, filtered_test_data.shape[2] - ws)
                        test_data_batch[i] = filtered_test_data[batch_indices[i], :, r1 : r1 + ws, :, :]

                    test_data_batch_dev = torch.from_numpy(test_data_batch).float().to(device)
                    out = net(test_data_batch_dev)[0].data.cpu().numpy().reshape(-1)
                    test_pred[batch_indices] += out
                    voter_count[batch_indices] += 1

            test_pred /= voter_count

            # Test performance
            if args.regression:
                test_mae = mean_absolute_error(test_label, test_pred)
                test_r2 = r2_score(test_label, test_pred)
                print(f"Test results -> MAE: {test_mae:.4f}, R^2: {test_r2:.4f}")
                results.append([network, '_'.join(args.ensemble_networks), ws, test_mae, test_r2])
            else:
                test_auc = roc_auc_score(test_label, test_pred)
                print(f"Test results -> AUC: {test_auc:.4f}")
                results.append([network, '_'.join(args.ensemble_networks), ws, test_auc])

            # Store predictions for final analysis
            test_df = pd.DataFrame({
                'Subject': test_subjects,
                'Real Label': test_label,
                'Prediction': test_pred,
                'Network': network,
                'Ensemble_networks': '_'.join(args.ensemble_networks),
                'SetType': 'Test'
            })
            predictions_df = pd.concat([predictions_df, test_df], ignore_index=True)

    # -----------------------------
    # Save final results
    # -----------------------------
    if args.regression:
        results_df = pd.DataFrame(results, columns=[
            'Network', 'Ensemble Networks', 'Window Size', 'Test MAE', 'Test R2'
        ])
    else:
        results_df = pd.DataFrame(results, columns=[
            'Network', 'Ensemble Networks', 'Window Size', 'Test AUC'
        ])

    # Decide a naming scheme for your results
    results_file = 'training_results_single_pass.csv'
    preds_file   = 'predictions_single_pass.csv'
    results_df.to_csv(results_file, index=False)
    predictions_df.to_csv(preds_file, index=False)

    print(f"\nAll done. Results saved to {results_file} and {preds_file}\n")
    torch.cuda.empty_cache()

##############################################################################
# Entry point
##############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MS-G3D (Single-pass)')

    # Training arguments
    parser.add_argument('--bs', type=int, required=True, help='Batch size for training')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--gpu', type=int, required=True, help='GPU device ID')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to folder containing node_timeseries/node_timeseries')
    parser.add_argument('--parcel_path', type=str, required=True, help='Path to the parcel info (Excel file)')

    # Network (ROI) selection arguments
    parser.add_argument('--networks', metavar='S', type=str, nargs='+', default=[' '],
                        help='List of networks for training (or " " if none). Example: --networks "Vis" "Aud"')
    parser.add_argument('--remove_network', action='store_true',
                        help='If set, remove the specified network instead of selecting it.')
    parser.add_argument('--ensemble_networks', metavar='S', type=str, nargs='+', default=[' '],
                        help='List of networks for ensemble training (or " " if none)')

    # Regression/classification
    parser.add_argument('--regression', action='store_true',
                        help='Use regression (MSE/MAE) instead of classification (BCE/AUC)?')

    # Window slicing & model hyperparams
    parser.add_argument('--windows', type=int, nargs='+', required=True, help='List of window sizes')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--gcn_scales', type=int, default=8, help='Number of GCN scales')
    parser.add_argument('--g3d_scales', type=int, default=8, help='Number of G3D scales')
    parser.add_argument('--LR', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optim', type=str, default='Adam', help='Optimizer (SGD or Adam)')

    # Voting
    parser.add_argument('--TS', type=int, default=64, help='Number of voting iterations per val/test sample')

    # Validation interval
    parser.add_argument('--epochs_val', type=int, default=100, 
                        help='Perform validation every X epochs (for best checkpoint)')

    args = parser.parse_args()
    train(args)