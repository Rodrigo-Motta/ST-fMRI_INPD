import os
import argparse

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.model_selection import train_test_split

##############################################################################
# Helper function to compute adjacency matrix
##############################################################################
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
                A[i][j] = abs(np.corrcoef(data[:, i], data[:, j])[0][1])
                A[j][i] = A[i][j]
    A = np.nan_to_num(A)  # Replace NaN with 0
    A = np.arctanh(A)
    return A

##############################################################################
# Helper function to load timeseries data for a list of subjects
##############################################################################
def load_timeseries_and_labels(
    subject_info_df, 
    data_path, 
    label_name,
    parcel='Gordon', 
    T=176, 
    regression=False
):
    """
    Loads the timeseries data for each subject in `subject_info_df` from `data_path`,
    applies z-score, stores data and labels in numpy arrays.

    Args:
        subject_info_df: pd.DataFrame with columns ['subject', label_name, ...]
        data_path: folder path where subject timeseries are found
        parcel: 'Gordon' or 'Schaefer'
        T: number of timepoints to keep
        label_name: name of the label column
        regression: whether the task is regression or classification (for any special handling)
    
    Returns:
        data: np.array of shape (N, T, ROI_nodes)
        label: np.array of shape (N,)
        used_subjects: list of subject IDs actually loaded
        adj_matrices: list of adjacency matrices (one per subject)
    """
    # Decide number of parcellation nodes
    if parcel == 'Gordon':
        ROI_nodes = 333
    elif parcel == 'Schaefer':
        ROI_nodes = 300
    else:
        raise ValueError("Unknown parcel argument")

    nb_subjects = subject_info_df.shape[0]
    # Prepare storage
    data = np.zeros((nb_subjects, T, ROI_nodes))
    labels = np.zeros((nb_subjects,))
    used_subjects = []
    non_used = []
    adj_matrices = []

    idx = 0
    for i in range(nb_subjects):
        subj_id_str = str(int(subject_info_df.loc[i, 'subject']))

        # Construct the filename for the parcellation chosen
        if parcel == 'Gordon':
            filename = f"/GordonConnBOLD-{subj_id_str}.txt"
        else:  # 'Schaefer'
            filename = f"/Schaefer_fMRIPREP_BOLD-{subj_id_str}.txt"

        full_path = os.path.join(data_path, filename.lstrip('/'))  # handle leading '/'

        if os.path.exists(full_path):
            # Load timeseries
            if parcel == 'Gordon':
                full_sequence = np.loadtxt(full_path)[4:]  # Skip first 4 frames if desired
            else:
                full_sequence = np.loadtxt(full_path)      # Schaefer uses entire timeseries

            # Ensure length is enough
            if full_sequence.shape[0] < T:
                print(f"sequence too short: {full_path}")
                non_used.append(subj_id_str)
                continue

            # Transpose to shape (ROI_nodes, T)
            full_sequence = np.transpose(full_sequence[:T, :])

            # z-score each node's timeseries
            z_sequence = stats.zscore(full_sequence, axis=1)
            z_sequence = np.nan_to_num(z_sequence)  # fill any NaN with 0

            # Skip if it still contains any unexpected NaN
            if np.sum(np.isnan(z_sequence)) != 0:
                print(f"contains nan in subject: {subj_id_str}")
                non_used.append(subj_id_str)
                continue

            # Store in final data array
            data[idx, :, :] = np.transpose(z_sequence)  # shape (T, ROI_nodes)

            # Grab label
            labels[idx] = subject_info_df.loc[i, label_name]

            # Compute adjacency matrix for the current subject, store
            A = compute_adjacency_matrix(z_sequence.T)  # shape (T, ROI_nodes) -> transpose = (T, ROI_nodes)
            adj_matrices.append(A)

            used_subjects.append(subj_id_str)
            idx += 1
        else:
            # File not found
            print(f"File does not exist for subject {subj_id_str}: {full_path}")
            non_used.append(subj_id_str)

    # Trim to only the loaded subjects
    data = data[:idx, :, :]
    labels = labels[:idx]
    used_subjects = np.array(used_subjects)

    return data, labels, used_subjects, adj_matrices, non_used

##############################################################################
# Main
##############################################################################
def main(args):
    """
    Script to:
      1) Load train/val subject list (CSV or TXT) from path `'asdas'` 
      2) Load test subject list (CSV or TXT) from path `'bscd'`
      3) Create a single train/val split (instead of cross-validation)
      4) Save train, val, and test sets + adjacency matrix
    """
    print("\n" + "#"*30)
    print("Starting: preprocessing script (Train/Val/Test)")
    print("#"*30 + "\n")

    # -------------------------------
    # 1) LOAD TRAIN/VAL SUBJECTS
    # -------------------------------
    if args.train_val_filename.endswith(".txt"):
        ids_train_val = np.loadtxt(args.train_val_filename)
        # If .txt, we assume first column is subject, second is label
        # Adapt as needed
        # For uniformity, convert to a dataframe:
        df_train_val = pd.DataFrame(ids_train_val, columns=["subject", args.label_train])
    elif args.train_val_filename.endswith(".csv"):
        df_train_val = pd.read_csv(args.train_val_filename)
        df_train_val = df_train_val.dropna(subset=[args.label_train]).reset_index(drop=True)
    else:
        raise TypeError("train_val_filename filetype not implemented")

    # For classification example (as in your original code)
    if not args.regression:
        # Example: Binarize label
        df_train_val["TOTAL_DAWBA"] = df_train_val["TOTAL_DAWBA"].apply(lambda x: 0 if x == 0 else 1)
        df_train_val["dcany2010"] = df_train_val["dcany2010"].apply(lambda x: 0 if x == 0 else 1)
        df_train_val["dcFUPany"] = df_train_val["dcFUPany"].apply(lambda x: 0 if x == 0 else 1)


    print(f"[Train/Val] Number of subjects in file: {len(df_train_val)}")

    # -------------------------------
    # 2) LOAD TEST SUBJECTS
    # -------------------------------
    if args.test_filename.endswith(".txt"):
        ids_test = np.loadtxt(args.test_filename)
        df_test = pd.DataFrame(ids_test, columns=["subject", args.label_test])
    elif args.test_filename.endswith(".csv"):
        df_test = pd.read_csv(args.test_filename)
        df_test = df_test.dropna(subset=[args.label_test]).reset_index(drop=True)
    else:
        raise TypeError("test_filename filetype not implemented")

    if not args.regression:
        df_test["TOTAL_DAWBA"] = df_test["TOTAL_DAWBA"].apply(lambda x: 0 if x == 0 else 1)

    print(f"[Test] Number of subjects in file: {len(df_test)}")

    # -------------------------------
    # 3) READ PARCELLATION TIME-SERIES FROM 'asdas' (train/val)
    # -------------------------------
    print("\n" + "#"*30)
    print("Loading node timeseries: train/val data")
    print("#"*30 + "\n")

    trainval_data, trainval_label, trainval_subjects, trainval_adj_list, non_used_trainval = load_timeseries_and_labels(
        subject_info_df=df_train_val,
        data_path=args.train_val_path,
        label_name=args.label_train,

        parcel=args.parcel,
        T=176,
        regression=args.regression
    )

    print(f"[Train/Val] Loaded {trainval_data.shape[0]} subjects successfully.")
    print(f"[Train/Val] Data shape: {trainval_data.shape}")  # (N, T, ROI_nodes)

    # -------------------------------
    # 4) READ PARCELLATION TIME-SERIES FROM 'bscd' (test)
    # -------------------------------
    print("\n" + "#"*30)
    print("Loading node timeseries: test data")
    print("#"*30 + "\n")

    test_data, test_label, test_subjects, test_adj_list, non_used_test = load_timeseries_and_labels(
        subject_info_df=df_test,
        data_path=args.test_path,
        label_name=args.label_test,
        parcel=args.parcel,
        T=176,
        regression=args.regression
    )

    print(f"[Test] Loaded {test_data.shape[0]} subjects successfully.")
    print(f"[Test] Data shape: {test_data.shape}")  # (N, T, ROI_nodes)

    # -------------------------------
    # 5) Create adjacency from train/val (or entire data)
    #    Typically, we only use the training set for "group-based" adjacency,
    #    but you can adapt if you want to combine train+test.
    # -------------------------------
    print("\n" + "#"*30)
    print("Computing adjacency matrix (from train/val)")
    print("#"*30 + "\n")

    # Average adjacency across all train/val subjects
    # If you'd rather do it only from the *training split* (below),
    # then move this after the train/val split.
    avg_adj_matrix = np.tanh(np.mean(trainval_adj_list, axis=0))  # shape (ROI_nodes, ROI_nodes)

    # -------------------------------
    # 6) TRAIN/VAL SPLIT (80/20 or user-defined)
    # -------------------------------
    print("\nSplitting into train/val sets ...\n")
    if args.regression:
        stratify_flag = None
    else:
        stratify_flag = trainval_label

    train_idx, val_idx = train_test_split(
        np.arange(len(trainval_data)),
        test_size=0.2,             # or your chosen fraction
        random_state=42,
        stratify=stratify_flag
    )

    train_data = trainval_data[train_idx]
    train_label = trainval_label[train_idx]
    train_subjects_split = trainval_subjects[train_idx]

    val_data = trainval_data[val_idx]
    val_label = trainval_label[val_idx]
    val_subjects_split = trainval_subjects[val_idx]

    print(f"Train set size: {train_data.shape[0]}")
    print(f"Val set size:   {val_data.shape[0]}")
    print(f"Test set size:  {test_data.shape[0]}")

    # -------------------------------
    # 7) Save everything
    # -------------------------------
    outdir = args.output_folder
    try:
        os.makedirs(os.path.join(outdir, "node_timeseries", "node_timeseries"), exist_ok=True)
        print(f"Created output folder structure under: {outdir}")
    except OSError:
        pass

    # Save adjacency
    np.save(os.path.join(outdir, "node_timeseries", "node_timeseries", "adj_matrix.npy"), avg_adj_matrix)

    # Save train
    np.save(os.path.join(outdir, "node_timeseries", "node_timeseries", "train_data_wave1.npy"), train_data)
    np.save(os.path.join(outdir, "node_timeseries", "node_timeseries", "train_label_wave1.npy"), train_label)
    np.save(os.path.join(outdir, "node_timeseries", "node_timeseries", "train_subjects_wave1.npy"), train_subjects_split)

    # Save val
    np.save(os.path.join(outdir, "node_timeseries", "node_timeseries", "val_data_wave1.npy"), val_data)
    np.save(os.path.join(outdir, "node_timeseries", "node_timeseries", "val_label_wave1.npy"), val_label)
    np.save(os.path.join(outdir, "node_timeseries", "node_timeseries", "val_subjects_wave1.npy"), val_subjects_split)

    # Save test
    np.save(os.path.join(outdir, "node_timeseries", "node_timeseries", "test_data_wave2.npy"), test_data)
    np.save(os.path.join(outdir, "node_timeseries", "node_timeseries", "test_label_wave2.npy"), test_label)
    np.save(os.path.join(outdir, "node_timeseries", "node_timeseries", "test_subjects_wave2.npy"), test_subjects)

    # Save the IDs that were not used due to file missing / short sequence
    all_non_used = {
        "train_val_non_used": non_used_trainval,
        "test_non_used": non_used_test
    }
    df_non_used = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_non_used.items()]))
    df_non_used.to_csv(os.path.join(outdir, "node_timeseries", "ids_not_used.csv"), index=False)

    print("\nAll done!")

##############################################################################
# Entry point
##############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing script (Train/Val/Test).")

    # Filenames for subject IDs
    parser.add_argument('--train_val_filename', required=True, 
                        help='Path to the train/val subject ID file (txt or csv).')
    parser.add_argument('--test_filename', required=True, 
                        help='Path to the test subject ID file (txt or csv).')

    # Data paths for the actual time-series
    parser.add_argument('--train_val_path', required=True, 
                        help='Directory containing train/val timeseries data (e.g., "asdas").')
    parser.add_argument('--test_path', required=True, 
                        help='Directory containing test timeseries data (e.g., "bscd").')

    # Label info
    parser.add_argument('--label_train', required=True, type=str, 
                        help='Name of the label column.')
    
    parser.add_argument('--label_test', required=True, type=str, 
                        help='Name of the label column.')
    
    parser.add_argument('--regression', required=False, default=False, action='store_true',
                        help='If set, treat problem as regression (disables label binarization).')

    # Output
    parser.add_argument('--output_folder', required=True, 
                        help='Directory to save the output data.')

    # Parcellation
    parser.add_argument('--parcel', required=True, type=str, 
                        help='Parcellation name: "Gordon" or "Schaefer".')

    args = parser.parse_args()
    main(args)