import os
import argparse

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit, ShuffleSplit

def compute_adjacency_matrix(data):
    """
    Computes the adjacency matrix (correlation between each pair of ROIs)
    for a given node timeseries data.

    Args:
        data (np.array): Node timeseries data of shape (T, ROI_nodes).
                         Each column is the timeseries of a given ROI.

    Returns:
        Adjacency matrix of shape (ROI_nodes, ROI_nodes),
        with Fisher z-transformed correlation values.
    """
    n_regions = data.shape[1]
    A = np.zeros((n_regions, n_regions))

    for i in range(n_regions):
        for j in range(i, n_regions):
            if i == j:
                A[i, j] = 0
            else:
                # Compute correlation between ROI i and ROI j
                corr_val = np.corrcoef(data[:, i], data[:, j])[0, 1]
                A[i, j] = corr_val
                A[j, i] = corr_val

    # Replace NaNs with 0
    A = np.nan_to_num(A)
    # Fisher z-transformation
    A = np.arctanh(A)
    return A

def main(args):
    """
    Main function to preprocess timeseries data and store correlation
    coefficients (adjacency matrices) as the node features.

    Provides two split modes:
      1) default_5fold (original 5-fold logic)
      2) start_split   (split by subject string '1' vs '2')
    """
    print('\n' + '#' * 30)
    print('Starting: preprocessing script')
    print('#' * 30 + '\n')

    # ----------------------------------------------------------------------
    # 1) Load subject IDs and any necessary labels
    # ----------------------------------------------------------------------
    if args.filename.lower().endswith('.txt'):
        ids = np.loadtxt(args.filename)
        nb_subject = ids.shape[0]
        # If .txt has shape [subject_id, label], handle accordingly
        # (Adjust code as needed if using a .txt with multiple columns)
        # For simplicity, assume single column or handle as you do normally.
        raise NotImplementedError("Example code for .txt not implemented. Use CSV or adapt this block.")
    elif args.filename.lower().endswith('.csv'):
        ids = pd.read_csv(args.filename)
        # Example cleaning steps / label adjustments
        ids = ids.dropna(subset=[args.label]).reset_index(drop=True)
        # Example: recode 'gender' or convert labels if needed
        if 'gender' in ids.columns:
            ids.gender.replace(1.0, 0, inplace=True)
            ids.gender.replace(2.0, 1, inplace=True)

        # Example for binary classification
        if not args.regression:
            if 'TOTAL_DAWBA' in ids.columns:
                ids['TOTAL_DAWBA'] = ids['TOTAL_DAWBA'].apply(lambda x: 0 if x == 0.0 else 1)
            if 'dcany2010' in ids.columns:
                ids['dcany2010'] = ids['dcany2010'].apply(lambda x: 0 if x == 0.0 else 1)

        nb_subject = ids.shape[0]
    else:
        raise TypeError('Input file must be .txt or .csv')

    data_path = str(args.data_path)
    print(f'\nNumber of subjects in file: {nb_subject}')

    # Adjust for your dataset's known time length
    T = 176

    # Number of ROIs depending on your parcellation
    if args.parcel == 'Gordon':
        ROI_nodes = 333
    elif args.parcel == 'Schaefer':
        ROI_nodes = 300
    else:
        raise ValueError("Unknown parcellation. Choose 'Gordon' or 'Schaefer'.")

    # ----------------------------------------------------------------------
    # 2) Allocate storage for adjacency matrices (correlation features)
    # ----------------------------------------------------------------------
    # we'll store adjacency matrices of shape (nb_subject, 1, ROI_nodes, ROI_nodes, 1).
    data = np.zeros((nb_subject, 1, ROI_nodes, ROI_nodes, 1), dtype=np.float32)
    label = np.zeros((nb_subject,), dtype=np.float32)

    # For checking how many subjects were actually used
    idx = 0
    non_used = []
    used = []

    # Keep individual adjacency matrices for each subject to compute the average
    adj_matrices = []

    print('\n' + '#' * 30)
    print('Loading and computing adjacency from node timeseries')
    print('#' * 30)

    # Example: read a file of subject IDs that you specifically want to consider
    caras = np.loadtxt('CARAS12.txt', dtype=str)

    for i in range(nb_subject):
        # ------------------------------------------------------------------
        # Retrieve subject ID (edit this part according to your file format)
        # ------------------------------------------------------------------
        subject_string = str(int(ids.loc[i, 'subject']))

        # Example check. If you want to specifically handle only subjects starting with '1':
        # but in *loading* we might still want all subjects.
        # You can adapt the below if needed.
        # if subject_string in caras:

        # Build filename/path
        if args.parcel == 'Gordon':
            filename = f'GordonConnBOLD-{subject_string}.txt'
            skip_initial_frames = 4
        else:  # 'Schaefer'
            filename = f'Schaefer_fMRIPREP_BOLD-{subject_string}.txt'
            skip_initial_frames = 0

        filepath = os.path.join(data_path, filename)

        if os.path.exists(filepath):
            # Load timeseries
            full_sequence = np.loadtxt(filepath)

            # Possibly skip frames if using Gordon
            if skip_initial_frames > 0:
                full_sequence = full_sequence[skip_initial_frames:]

            # Check length
            if full_sequence.shape[0] < T:
                print(f'Sequence too short: {filepath}')
                non_used.append(subject_string)
                continue

            # Shape: (ROI_nodes, T)
            full_sequence = full_sequence[:T, :].T

            # z-score each ROI's timeseries
            z_sequence = stats.zscore(full_sequence, axis=1)
            z_sequence = np.nan_to_num(z_sequence)

            # Final check for NaNs
            if np.isnan(z_sequence).any():
                print(f'Contains NaN: {filepath}')
                non_used.append(subject_string)
                continue

            # ------------------------------------------------------------------
            # Compute adjacency
            # ------------------------------------------------------------------
            adj_matrix = compute_adjacency_matrix(z_sequence.T)  # shape: (ROI_nodes, ROI_nodes)
            adj_matrices.append(adj_matrix)

            # Store adjacency in data
            data[idx, 0, :, :, 0] = adj_matrix

            # Save label (replace with your correct label extraction)
            label_val = ids.loc[i, args.label]
            label[idx] = label_val

            used.append(subject_string)
            idx += 1

            if idx % 100 == 0:
                print(f'{idx} subjects processed so far...')
        else:
            non_used.append(subject_string)

    # ----------------------------------------------------------------------
    # 3) Final shape and filtering out unused subjects
    # ----------------------------------------------------------------------
    data = data[:idx]
    label = label[:idx]

    used = np.array(used, dtype=str)

    # Filter the 'ids' dataframe to keep only used subjects
    ids['subject'] = ids['subject'].astype(str)
    ids = ids[ids['subject'].isin(used)].copy()
    ids = ids.sort_values(by='subject').reset_index(drop=True)

    print(f'\nNumber of subjects successfully loaded: {idx}')
    print(f'Data shape (adjacency): {data.shape}\n')

    # Save IDs not used
    df_non_used = pd.DataFrame(non_used, columns=['unused_subjects'])

    # Create output folder(s) if needed
    try:
        os.makedirs(args.output_folder, exist_ok=True)
        print(f"Output folder: {args.output_folder} (created or already exists)")
    except OSError as e:
        print(f"Error creating directory {args.output_folder}: {e}")

    df_non_used.to_csv(os.path.join(args.output_folder, 'ids_not_used.csv'), index=False)

    # ----------------------------------------------------------------------
    # 4) (Optional) Compute average adjacency across subjects
    # ----------------------------------------------------------------------
    print('\n' + '#' * 30)
    print('Computing average adjacency matrix (optional)')
    print('#' * 30)

    if len(adj_matrices) > 0:
        # Mean of Fisher z-values, then apply tanh to go back to correlation scale (optional)
        avg_adj_matrix = np.tanh(np.mean(adj_matrices, axis=0))
    else:
        avg_adj_matrix = np.zeros((ROI_nodes, ROI_nodes))

    print('Saving average adjacency matrix...')
    try:
        os.makedirs(os.path.join(args.output_folder, 'node_FC', 'node_FC'), exist_ok=True)
    except OSError as e:
        print(f"Error creating output subfolders: {e}")

    np.save(
        os.path.join(args.output_folder, 'node_FC', 'node_FC', 'adj_matrix.npy'),
        avg_adj_matrix
    )

    # ----------------------------------------------------------------------
    # 5) Split data, depending on mode
    # ----------------------------------------------------------------------
    print('\n' + '#' * 30)
    print(f'Splitting data (mode: {args.split_mode})')
    print('#' * 30, '\n')

    outdir = os.path.join(args.output_folder, 'node_FC', 'node_FC')
    os.makedirs(outdir, exist_ok=True)

    if args.split_mode == 'default_5fold':
        # -------------------------------
        # Original 5-Fold logic
        # -------------------------------
        if args.regression:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            split_iterator = kf.split(data)
        else:
            # Classification => stratified
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            split_iterator = skf.split(data, label)

        fold = 1
        for train_idx, test_idx in split_iterator:
            train_data = data[train_idx]
            train_label = label[train_idx]
            test_data = data[test_idx]
            test_label = label[test_idx]

            # Save the splits
            np.save(os.path.join(outdir, f'train_data_{fold}.npy'), train_data)
            np.save(os.path.join(outdir, f'train_label_{fold}.npy'), train_label)
            np.save(os.path.join(outdir, f'test_data_{fold}.npy'), test_data)
            np.save(os.path.join(outdir, f'test_label_{fold}.npy'), test_label)

            # Save subject IDs for each fold
            train_subjects = used[train_idx]
            test_subjects = used[test_idx]
            np.save(os.path.join(outdir, f'train_subjects_{fold}.npy'), train_subjects)
            np.save(os.path.join(outdir, f'test_subjects_{fold}.npy'), test_subjects)

            print(f'Fold {fold} | train size: {len(train_idx)}, test size: {len(test_idx)}')
            fold += 1

    elif args.split_mode == 'start_split':
        # -----------------------------------------------------------
        # New: 
        #   - test = subjects whose string starts with "2"
        #   - train+val = subjects whose string starts with "1"
        #   - then split those "1" subjects as 80% train, 20% val
        # -----------------------------------------------------------
        used_str = used.tolist()
        used_array = np.array(used_str)

        # Indices for subjects that start with '1' or '2'
        idx_s1 = [i for i, s in enumerate(used_str) if s.startswith('1')]
        idx_s2 = [i for i, s in enumerate(used_str) if s.startswith('2')]

        data_s1 = data[idx_s1]
        label_s1 = label[idx_s1]
        data_s2 = data[idx_s2]
        label_s2 = label[idx_s2]

        # We'll put everything that starts with '2' into the "test" set
        test_data = data_s2
        test_label = label_s2
        test_subjects = used_array[idx_s2]

        # Now split the "1" group into 80/20 for train/val
        if len(data_s1) == 0:
            raise ValueError("No subjects found that start with '1'. Cannot create train/val split.")
        
        if not args.regression:
            # classification => stratified
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            s1_split_iter = sss.split(data_s1, label_s1)
        else:
            # regression => standard shuffle
            sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            s1_split_iter = sss.split(data_s1)

        for train_index_s1, val_index_s1 in s1_split_iter:
            train_data = data_s1[train_index_s1]
            train_label = label_s1[train_index_s1]
            val_data = data_s1[val_index_s1]
            val_label = label_s1[val_index_s1]

            train_subjects = used_array[idx_s1][train_index_s1]
            val_subjects = used_array[idx_s1][val_index_s1]

        # Save to .npy
        np.save(os.path.join(outdir, 'train_data.npy'), train_data)
        np.save(os.path.join(outdir, 'train_label.npy'), train_label)
        np.save(os.path.join(outdir, 'val_data.npy'), val_data)
        np.save(os.path.join(outdir, 'val_label.npy'), val_label)
        np.save(os.path.join(outdir, 'test_data.npy'), test_data)
        np.save(os.path.join(outdir, 'test_label.npy'), test_label)

        np.save(os.path.join(outdir, 'train_subjects.npy'), train_subjects)
        np.save(os.path.join(outdir, 'val_subjects.npy'), val_subjects)
        np.save(os.path.join(outdir, 'test_subjects.npy'), test_subjects)

        print(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")

    print('\nPreprocessing complete!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing script.")
    parser.add_argument('--filename', required=True, help='Path to the subject ID file (txt or csv).')
    parser.add_argument('--label', required=True, type=str, help='Name of the label/target column.')
    parser.add_argument('--regression', action='store_true', help='Use regression (MSELoss) instead of classification (BCELoss)?')
    parser.add_argument('--data_path', required=True, help='Base directory of node timeseries data.')
    parser.add_argument('--output_folder', required=True, help='Directory to save the output data.')
    parser.add_argument('--parcel', required=True, type=str, help='Parcellation name (Gordon or Schaefer).')

    # New argument to control splitting mode
    parser.add_argument('--split_mode', required=True, type=str, default='default_5fold',
                        choices=['default_5fold', 'start_split'],
                        help='Splitting mode: "default_5fold" for original 5-fold, '
                             '"start_split" for train/val (start="1") + test (start="2").')

    args = parser.parse_args()
    main(args)
