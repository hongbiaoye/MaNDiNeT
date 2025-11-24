import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration
BASE_DIR = Path(r'Data_process\WGraw')  # Subject root directory
OUTPUT_ROOT = Path(r'DATA\WGXLS')  # Output root directory
SUBJECT_TEMPLATE = "VP{:03d}-NIRS"  # Subject folder template (VP001-NIRS to VP026-NIRS)
NUM_SUBJECTS = 26  # Number of subjects


# ---------------------- Core conversion logic ----------------------
def convert_subject(subject_dir, output_dir):
    cnt_file = subject_dir / "cnt_wg.mat"
    mrk_file = subject_dir / "mrk_wg.mat"

    if not (cnt_file.exists() and mrk_file.exists()):
        print(f"Warning: {subject_dir.name} is missing required files, skipping.")
        return

    try:
        # Parse cnt_wg.mat
        cnt_var = sio.loadmat(str(cnt_file))['cnt_wg'][0, 0]
        oxy_tuple = cnt_var['oxy'][0, 0]
        deoxy_tuple = cnt_var['deoxy'][0, 0]

        # Locate the signal matrix (index 5 based on prior inspection)
        oxy_signal = oxy_tuple[5]
        deoxy_signal = deoxy_tuple[5]
        channel_labels = oxy_tuple[4][0].tolist()  # Channel labels

        combined_channels = [f"{ch}_oxy" for ch in channel_labels] + [f"{ch}_deoxy" for ch in channel_labels]
        combined_signal = np.hstack((oxy_signal, deoxy_signal))

        # Parse mrk_wg.mat (timestamps divided by 100)
        mrk_var = sio.loadmat(str(mrk_file))['mrk_wg'][0, 0]
        raw_timestamps = mrk_var[0][0].flatten()
        timestamps = (raw_timestamps / 100).astype(int)
        markers = mrk_var[1][0, :].astype(int).flatten()

        # Build task intervals
        task_intervals = []
        for i in range(len(timestamps)):
            start = max(0, timestamps[i])
            end = timestamps[i + 1] if i < len(timestamps) - 1 else combined_signal.shape[0]
            end = min(combined_signal.shape[0], end)
            if end > start:
                task_intervals.append((start, end))

        # Save files
        output_dir.mkdir(exist_ok=True, parents=True)
        save_signal_to_excel(combined_channels, combined_signal, task_intervals, output_dir)
        save_labels_to_excel(markers, output_dir)
        print(f"Finished processing: {subject_dir.name}")

    except Exception as e:
        print(f"Error while processing {subject_dir.name}: {str(e)}")


# ---------------------- Save signal data to Excel ----------------------
def save_signal_to_excel(channels, signal, intervals, output_dir):
    signal_path = output_dir / "signal_data.xlsx"
    with pd.ExcelWriter(str(signal_path)) as writer:
        for idx, (start, end) in enumerate(intervals, 1):
            task_data = signal[start:end, :]
            if task_data.size > 0:
                df = pd.DataFrame(task_data, columns=channels)
                df.to_excel(writer, sheet_name=f"Task_{idx}", index=False)


# ---------------------- Save label data to Excel ----------------------
def save_labels_to_excel(markers, output_dir):
    label_path = output_dir / "labels.xlsx"
    label_df = pd.DataFrame(markers, columns=["label"])
    label_df.to_excel(str(label_path), index=False)


# ---------------------- Batch process all subjects ----------------------
def process_all_subjects():
    OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)

    for subject_num in range(1, NUM_SUBJECTS + 1):
        subject_name = SUBJECT_TEMPLATE.format(subject_num)
        subject_dir = BASE_DIR / subject_name
        output_subdir = OUTPUT_ROOT / subject_name  # Same-named subdirectory

        print(f"Processing subject: {subject_name}")
        convert_subject(subject_dir, output_subdir)


if __name__ == "__main__":
    process_all_subjects()
    print("\nBatch processing finished!")