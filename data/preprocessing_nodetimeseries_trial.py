import os
import argparse

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

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
                A[i][j] = 0
            else:
                A[i][j] = (np.corrcoef(data[:, i], data[:, j])[0][1]) #abs
                A[j][i] = A[i][j]
    A = np.nan_to_num(A)  # Replace NaN with 0
    A = np.arctanh(A)
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

    Variables:
        ids: List of subject IDs loaded from the specified file.
        nb_subject: Total number of subjects derived from the shape of `ids`.
        data_path: Root directory path where node timeseries data files are stored.
        T: Temporal length sequence.
        data: Numpy array initialized to zeros with shape (nb_subject, 1, T, 333, 1) to store preprocessed timeseries data for each subject.
        label: Numpy array initialized to zeros with shape (nb_subject,) to store labels corresponding to each subject.
        non_used: List to keep track of subject IDs not used due to data issues.
        subject_string: Formatted string representing the subject ID for constructing file paths.
        filename: Specific filename for each subject's node timeseries data.
        filepath: Full path to the node timeseries data file for each subject.
        full_sequence: Timeseries data for a subject loaded from the file.
        z_sequence: Z-score normalized timeseries data for a subject.
        data_all: Concatenated array of z-score normalized timeseries data for all subjects.
        A: Adjacency matrix computed based on the correlation of node timeseries data.
        train_data: Subset of `data` used for training obtained from the StratifiedKFold split.
        train_label: Subset of `label` corresponding to `train_data`.
        test_data: Subset of `data` used for testing obtained from the StratifiedKFold split.
        test_label: Subset of `label` corresponding to `test_data`.
        idx: Index counter used to track the number of successfully processed subjects.
        df: Pandas DataFrame containing the list of subject IDs not used.
        skf: Instance of `StratifiedKFold` from `scikit-learn` used to split the data into training and testing sets.
        fold: Counter to track the fold number in the StratifiedKFold split.
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
        ids = ids.dropna(subset=[args.label]).reset_index()
        ids.gender.replace(1.0, 0, inplace=True)
        ids.gender.replace(2.0, 1, inplace=True)
        ## --------
        if args.regression == False:
            ids['TOTAL_DAWBA'] = ids['TOTAL_DAWBA'].apply(lambda x: 0 if x == 0.0 else 1)
            ids['dcany2010'] = ids['dcany2010'].apply(lambda x: 0 if x == 0.0 else 1)

            
        nb_subject = ids.shape[0]
    else:
        raise TypeError('filetype not implemented')
    
    data_path = str(args.data_path)
    
    T = 176 #110

    if args.parcel == 'Gordon':
        ROI_nodes = 333
    if args.parcel == 'Schaefer':
        ROI_nodes = 300    
    
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
    used = []
    adj_matrices = []
    
    print(args.parcel)
    for i in range(nb_subject):

        # Original subjects files
        #subject_string = format(int(ids[i,0]),'06d')

        ## ---------
        subject_string = str(int(ids.loc[i,'subject']))
        ## --------

        # Gordon Parcellation
        if args.parcel == 'Gordon':
            filename = '/GordonConnBOLD-'+subject_string+'.txt'
            
        # Schaefer Parcellation
        if args.parcel == 'Schaefer':
            filename = '/Schaefer_fMRIPREP_BOLD-'+subject_string+'.txt'
            
        filepath = str(data_path) + filename
        if os.path.exists(filepath):

            if args.parcel == 'Gordon':
                full_sequence = np.loadtxt(filepath)[4:]
            if args.parcel == 'Schaefer':
                full_sequence = np.loadtxt(filepath)[:]
            

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
            label[idx] = ids[ids.subject == int(subject_string)][args.label].values
            ## --------

            # Compute adjacency matrix for the current subject
            adj_matrix = compute_adjacency_matrix(z_sequence.T)
            adj_matrices.append(adj_matrix)

            idx = idx + 1
            used.append(int(subject_string))

            if idx % 100 == 0:
                print('subject preprocessed: {}'.format(idx))
            
        else:
            non_used.append(subject_string)
    
    used = np.array(used)
    ### keep only subjects
    data = data[:idx,0,:,:,0]
    label = label[:idx]

    ids['subject'] = ids['subject'].astype(str)
    ids = ids[ids.subject.isin(used)]
    ids['subject'] = pd.Categorical(ids['subject'], categories=used, ordered=True)
    ids = ids.sort_values('subject').reset_index(drop=True)

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
    avg_adj_matrix = np.tanh(np.mean(adj_matrices, axis=0))

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
    
    if args.regression:
        skf = KFold(n_splits=5, shuffle=True, random_state=42)
    else:
        skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)

    fold = 1
    for train_idx, test_idx in skf.split(data, label):
        train_data = data[train_idx]
        train_label = label[train_idx]
        test_data = data[test_idx]
        test_label = label[test_idx]

        filename = os.path.join(args.output_folder, 'node_timeseries', 'node_timeseries', 'train_data_'+str(fold)+'.npy')
        np.save(filename, train_data)
        filename = os.path.join(args.output_folder, 'node_timeseries', 'node_timeseries', 'train_label_'+str(fold)+'.npy')
        np.save(filename, train_label)
        filename = os.path.join(args.output_folder, 'node_timeseries', 'node_timeseries', 'test_data_'+str(fold)+'.npy')
        np.save(filename, test_data)
        filename = os.path.join(args.output_folder, 'node_timeseries', 'node_timeseries', 'test_label_'+str(fold)+'.npy')
        np.save(filename, test_label)

        # Save subject IDs for each fold
        train_subjects = used[train_idx] #ids.loc[train_idx, 'subject'].values
        test_subjects = used[test_idx] #ids.loc[test_idx, 'subject'].values
        filename = os.path.join(args.output_folder, 'node_timeseries', 'node_timeseries', 'train_subjects_'+str(fold)+'.npy')
        np.save(filename, train_subjects)
        filename = os.path.join(args.output_folder, 'node_timeseries', 'node_timeseries', 'test_subjects_'+str(fold)+'.npy')
        np.save(filename, test_subjects)

        fold += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing script.")
    parser.add_argument('--filename', required=True, help='Path to the subject ID file (txt or csv).')
    parser.add_argument('--label', required=True, type=str, help='Label name.')
    parser.add_argument('--regression', required=False, default=False, metavar='S',type=bool, help='task (classification or regression).')
    parser.add_argument('--data_path', required=True, help='Base directory of node timeseries data.')
    parser.add_argument('--output_folder', required=True, help='Directory to save the output data.')
    parser.add_argument('--parcel', required=True, type=str, help='Parcellation name.')

    args = parser.parse_args()
    main(args)