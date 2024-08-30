import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats

def compute_adjacency_matrix(data):
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
        print(ids.shape)
        ids = ids.dropna(subset=[args.label]).reset_index()
        print(ids.shape)
        ids.gender.replace(1.0, 0, inplace=True)
        ids.gender.replace(2.0, 1, inplace=True)
        if not args.regression:
            ids['TOTAL_DAWBA'] = ids['TOTAL_DAWBA'].apply(lambda x: 0 if x == 0.0 else 1)
        nb_subject = ids.shape[0]
    else:
        raise TypeError('filetype not implemented')
    
    data_path = str(args.data_path)
    
    T = 176
    ROI_nodes = 333
    data = np.zeros((nb_subject, 1, T, ROI_nodes, 1))
    label = np.zeros((nb_subject,))

    print('')
    print('Number of subjects: {}'.format(nb_subject))
    print('Temporal length sequence: {}'.format(T))
    print('')

    print('')
    print('#'*30)
    print('Loading: node timeseries')
    print('#'*30)
    print('')

    idx = 0
    non_used = []
    adj_matrices = []
    
    for i in range(nb_subject):
        subject_string = str(int(ids.loc[i,'subject']))
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
            label[idx] = ids.loc[i, args.label]

            adj_matrix = compute_adjacency_matrix(z_sequence.T)
            adj_matrices.append(adj_matrix)

            idx = idx + 1
            if idx % 100 == 0:
                print('subject preprocessed: {}'.format(idx))
            
        else:
            non_used.append(subject_string)
        
     # Keep only successfully processed subjects
    data = data[:idx,0,:,:,0]
    label = label[:idx]

    # Align `ids` DataFrame indices with the processed data
    ids = ids.loc[ids.index[:idx]].reset_index(drop=True)

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

    # split train/test based on TOTAL_DAWBA condition
    train_idx = ids[ids['TOTAL_DAWBA'] == 0].index
    test_idx = ids[ids['TOTAL_DAWBA'] != 0].index

    # Select 50 subjects for validation set where dawba == 0
    validation_idx = train_idx[:70]
    remaining_train_idx = train_idx[70:]

    validation_data = data[validation_idx]
    validation_label = label[validation_idx]
    remaining_train_data = data[remaining_train_idx]
    remaining_train_label = label[remaining_train_idx]
    test_data = data[test_idx]
    test_label = label[test_idx]

    filename = os.path.join(args.output_folder, 'node_timeseries', 'node_timeseries', 'validation_data.npy')
    np.save(filename, validation_data)
    filename = os.path.join(args.output_folder, 'node_timeseries', 'node_timeseries', 'validation_label.npy')
    np.save(filename, validation_label)

    filename = os.path.join(args.output_folder, 'node_timeseries', 'node_timeseries', 'train_data.npy')
    np.save(filename, remaining_train_data)
    filename = os.path.join(args.output_folder, 'node_timeseries', 'node_timeseries', 'train_label.npy')
    np.save(filename, remaining_train_label)
    filename = os.path.join(args.output_folder, 'node_timeseries', 'node_timeseries', 'test_data.npy')
    np.save(filename, test_data)
    filename = os.path.join(args.output_folder, 'node_timeseries', 'node_timeseries', 'test_label.npy')
    np.save(filename, test_label)

    # Save subject IDs
    train_subjects = ids.loc[remaining_train_idx, 'subject'].values
    validation_subjects = ids.loc[validation_idx, 'subject'].values
    test_subjects = ids.loc[test_idx, 'subject'].values
    filename = os.path.join(args.output_folder, 'node_timeseries', 'node_timeseries', 'train_subjects.npy')
    np.save(filename, train_subjects)
    filename = os.path.join(args.output_folder, 'node_timeseries', 'node_timeseries', 'validation_subjects.npy')
    np.save(filename, validation_subjects)
    filename = os.path.join(args.output_folder, 'node_timeseries', 'node_timeseries', 'test_subjects.npy')
    np.save(filename, test_subjects)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing script.")
    parser.add_argument('--filename', required=True, help='Path to the subject ID file (txt or csv).')
    parser.add_argument('--label', required=True, type=str, help='Label name.')
    parser.add_argument('--regression', required=False, default=False, metavar='S', type=bool, help='task (classification or regression).')
    parser.add_argument('--data_path', required=True, help='Base directory of node timeseries data.')
    parser.add_argument('--output_folder', required=True, help='Directory to save the output data.')

    args = parser.parse_args()
    main(args)