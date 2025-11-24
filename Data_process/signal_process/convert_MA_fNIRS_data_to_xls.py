import numpy as np
import scipy.io as scio
import os
import pandas as pd


def MA_subject_data(path, sub):
    """
    Load data and labels for a single subject
    Args:
        path: Root path of the dataset
        sub: Subject number
    """
    sub_path = os.path.join(path, str(sub))
    # Read labels
    label_path = os.path.join(sub_path, f"{sub}_desc.mat")
    signal_label = np.array(scio.loadmat(label_path)['label']).squeeze()
    # Convert labels to 0/1
    signal_label = np.where(signal_label == 1, 0, np.where(signal_label == 2, 1, signal_label))

    # Read data
    oxy_data_list = []
    deoxy_data_list = []
    for wins in range(9, 19):  # Time windows 9-18, total 10 windows
        oxy_file = os.path.join(sub_path, f"{wins}_oxy.mat")
        deoxy_file = os.path.join(sub_path, f"{wins}_deoxy.mat")
        oxy = np.array(scio.loadmat(oxy_file)['signal'])
        deoxy = np.array(scio.loadmat(deoxy_file)['signal'])
        oxy_data_list.append(oxy)
        deoxy_data_list.append(deoxy)

    oxy_data = np.array(oxy_data_list)  # Shape: (10, 31, 36, 60)
    deoxy_data = np.array(deoxy_data_list)
    return oxy_data, deoxy_data, signal_label


def save_subject_data(oxy_data, deoxy_data, signal_label, sub, output_root):
    """
    Save data and labels for a single subject to the specified directory
    Args:
        oxy_data: Oxyhemoglobin data (10, 31, 36, 60)
        deoxy_data: Deoxyhemoglobin data (10, 31, 36, 60)
        signal_label: Label array (60,)
        sub: Subject number
        output_root: Output root directory
    """
    sub_output_dir = os.path.join(output_root, f"sub_{sub:02d}")
    os.makedirs(sub_output_dir, exist_ok=True)  # Create subject-specific folder

    # Save data files (10 time windows Excel files)
    for time_window in range(10):
        current_oxy = oxy_data[time_window]
        current_deoxy = deoxy_data[time_window]
        num_sheets = current_oxy.shape[2]  # 60 tasks

        file_name = os.path.join(sub_output_dir, f"time_window_{time_window + 1}.xls")
        with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
            for sheet_idx in range(num_sheets):
                oxy_sheet = current_oxy[:, :, sheet_idx]  # (31, 36)
                deoxy_sheet = current_deoxy[:, :, sheet_idx]  # (31, 36)
                combined = np.hstack((oxy_sheet, deoxy_sheet))  # (31, 72)

                df = pd.DataFrame(combined)
                sheet_name = f"task_{sheet_idx + 1}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Saved subject {sub} time window {time_window + 1} to {file_name}")

    # Save label file
    label_df = pd.DataFrame(signal_label, columns=["label"])
    label_file = os.path.join(sub_output_dir, f"sub_{sub:02d}_label.xlsx")
    label_df.to_excel(label_file, index=False)
    print(f"Saved subject {sub} labels to {label_file}")


def process_all_subjects(data_root, output_root):
    """
    Process all subjects (1-29)
    """
    os.makedirs(output_root, exist_ok=True)  # Create total output directory
    for sub in range(1, 30):  # Iterate through subjects 1-29
        print(f"Processing subject {sub}...")
        try:
            oxy_data, deoxy_data, signal_label = MA_subject_data(data_root, sub)
            save_subject_data(oxy_data, deoxy_data, signal_label, sub, output_root)
            print(f"Subject {sub} processing completed\n")
        except Exception as e:
            print(f"Error processing subject {sub}: {str(e)}")


if __name__ == "__main__":
    # Configure paths
    data_root = r'Data_process\MA_fNIRS_data'  # Original data root directory
    output_root = r'DATA\MA_xlsdata'  # Output root directory

    process_all_subjects(data_root, output_root)