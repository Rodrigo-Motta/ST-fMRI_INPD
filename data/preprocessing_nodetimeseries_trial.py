import os
import argparse

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

def compute_adjacency_matrix(data):
    """
    Computes the adjacency matrix for a given node timeseries data.
    Args:
        data: Node timeseries data of shape (T, ROI_nodes).
    Returns:
        Adjacency matrix of shape (ROI_nodes, ROI_nodes).
    """
    n_regions = data.shape[1]
    A = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(i, n_regions):
            if i == j:
                A[i][j] = 1
            else:
                A[i][j] = abs(np.corrcoef(data[:, i], data[:, j])[0][1])
                A[j][i] = A[i][j]
    A = np.nan_to_num(A)  # Replace NaN with 0
    A = np.tanh(A)
    return A

def main(args):
    """
    Main function to preprocess timeseries data and compute adjacency matrices.

    Args:
        args: Parsed command-line arguments containing:
            - filename: Path to the file with the list of subject IDs (.txt or .csv).
            - ICA: Number of Independent Component Analysis (ICA) nodes.
            - data_path: Root directory path where node timeseries data files are stored.
            - output_folder: Path to the directory where results will be saved.
    """
     
    ### loading subject list
    print('')
    print('#'*30)
    print('Starting: preprocessing script')
    print('#'*30)
    print('')

    if args.filename.find('txt') != -1:
        ids = np.loadtxt(args.filename)
        nb_subject = ids.shape[0]
    elif args.filename.find('csv') != -1: 
        ids = pd.read_csv(args.filename)
        ## -----------
        print(ids.shape)
        ids = ids.dropna(subset=[args.label]).reset_index()
        print(ids.shape)
        ids.gender.replace(1.0, 0, inplace=True)
        ids.gender.replace(2.0, 1, inplace=True)
        ## --------
        ids['TOTAL_DAWBA'] = ids['TOTAL_DAWBA'].apply(lambda x: 0 if x == 0.0 else 1)
        nb_subject = ids.shape[0]
    else:
        raise TypeError('filetype not implemented')
    
    data_path = str(args.data_path)
    
    T = 110
    ROI_nodes = 333
    data = np.zeros((nb_subject, 1, T, ROI_nodes, 1))
    label = np.zeros((nb_subject,))

    print('')
    print('Number of subjects: {}'.format(nb_subject))
    print('Temporal length sequence: {}'.format(T))
    print('')

    ### loading node timeseries

    print('')
    print('#'*30)
    print('Loading: node timeseries')
    print('#'*30)
    print('')

    idx = 0
    non_used = []
    adj_matrices = []
    
    for i in range(nb_subject):

        # Original subjects files
        #subject_string = format(int(ids[i,0]),'06d')

        ## ---------
        subject_string = str(int(ids.loc[i,'subject']))
        ## --------
        filename = '/INPD/GordonConnBOLD/GordonConnBOLD-'+subject_string+'.txt'
        filepath = str(data_path) + filename
        if os.path.exists(filepath):
            full_sequence = np.loadtxt(filepath)[4:]

            if full_sequence.shape[0] < T:
                print('sequence too short :{}'.format(filepath))
                continue

            full_sequence = np.transpose(full_sequence[:T,:])

            z_sequence = stats.zscore(full_sequence, axis=1)

            z_sequence = np.nan_to_num(z_sequence)

            if np.sum(np.isnan(z_sequence)) != 0:
                print('contains nan')
                continue
            
            if idx ==0:
                data_all = z_sequence
            else:
                data_all = np.concatenate((data_all, z_sequence), axis=1)
            
            data[idx,0,:,:,0] = np.transpose(z_sequence)
            # Original subjects file
            #label[idx] = ids[i,1]

            ## --------
            label[idx] = ids.loc[i, args.label]
            ## --------

            # Compute adjacency matrix for the current subject
            adj_matrix = compute_adjacency_matrix(z_sequence.T)
            adj_matrices.append(adj_matrix)

            idx = idx + 1
            if idx % 100 == 0:
                print('subject preprocessed: {}'.format(idx))
            
        else:
            non_used.append(subject_string)
    
    ### keep only subjects
    data = data[:idx,0,:,:,0]
    label = label[:idx]

    print('')
    print('Number of subjects loaded: {}'.format(idx))
    print('Data shape: {}'.format(data.shape))
    print('')

    df = pd.DataFrame(data=np.array(non_used))

    try:
        os.mkdir(args.output_folder)
        print('Creating folder: {}'.format(args.output_folder))
    except OSError:
        print('folder already exist: {}'.format(args.output_folder))

    df.to_csv(os.path.join(args.output_folder,'ids_not_used.csv'))

    print('')
    print('#'*30)
    print('Computing: Adjacency Matrix')
    print('#'*30)
    print('')
           
    # Average the adjacency matrices
    avg_adj_matrix = np.arctanh(np.mean(adj_matrices, axis=0))

    print('')
    print('saving adjacency matrix')
    print('')

    try:        
        os.mkdir(os.path.join(args.output_folder,'node_timeseries'))  
        print('Creating folder: {}'.format(os.path.join(args.output_folder,'node_timeseries')))
    except OSError:
        print('folder already exist: {}'.format(os.path.join(args.output_folder,'node_timeseries')))


    try:        
        os.mkdir(os.path.join(args.output_folder,'node_timeseries/node_timeseries'))  
        print('Creating folder: {}'.format(os.path.join(args.output_folder,'node_timeseries/node_timeseries/')))
    except OSError:
        print('folder already exist: {}'.format(os.path.join(args.output_folder,'node_timeseries/node_timeseries/')))

    np.save(os.path.join(args.output_folder,'node_timeseries/node_timeseries/adj_matrix.npy'), avg_adj_matrix)

    # split train/test and save data
    
    skf = StratifiedKFold(n_splits=5,shuffle=True)
    fold = 1
    for train_idx, test_idx in skf.split(data, label):
        train_data = data[train_idx]
        train_label = label[train_idx]
        test_data = data[test_idx]
        test_label = label[test_idx]

        filename = os.path.join(args.output_folder,'node_timeseries/node_timeseries', 'train_data_'+str(fold)+'.npy')
        np.save(filename,train_data)
        filename = os.path.join(args.output_folder,'node_timeseries/node_timeseries','train_label_'+str(fold)+'.npy')
        np.save(filename,train_label)
        filename = os.path.join(args.output_folder,'node_timeseries/node_timeseries','test_data_'+str(fold)+'.npy')
        np.save(filename,test_data)
        filename = os.path.join(args.output_folder,'node_timeseries/node_timeseries','test_label_'+str(fold)+'.npy')
        np.save(filename,test_label)

        fold = fold + 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing script.")
    parser.add_argument('--filename', required=True, help='Path to the subject ID file (txt or csv).')
    parser.add_argument('--label', required=True, type=str, help='Label name.')
    parser.add_argument('--data_path', required=True, help='Base directory of node timeseries data.')
    parser.add_argument('--output_folder', required=True, help='Directory to save the output data.')

    args = parser.parse_args()
    main(args)