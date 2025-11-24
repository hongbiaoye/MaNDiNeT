# MaNDi-Net: Data Preprocessing Guide

This document provides detailed instructions on how to obtain and preprocess three fNIRS datasets for **MaNDi-Net** (Mamba-Transformer Deep Integration Network): MA (Mental Arithmetic), UFFT (Unilateral Finger and Foot Tapping), and WG (Word Generation).

## Dataset Acquisition

### 1. MA Dataset (Mental Arithmetic)
- **Download Link**: [https://doc.ml.tu-berlin.de/hBCI/](https://doc.ml.tu-berlin.de/hBCI/)
- **Description**: EEG-NIRS hybrid brain-computer interface data from 29 subjects, including mental arithmetic and resting state tasks
- **Data Format**: MATLAB format (.mat files)

### 2. UFFT Dataset (Unilateral Finger and Foot Tapping)
- **Download Link**: [https://figshare.com/articles/dataset/Open_accessfNIRS_dataset_for_classification_of_the_unilateral_finger_and_foottapping/9783755/1](https://figshare.com/articles/dataset/Open_accessfNIRS_dataset_for_classification_of_the_unilateral_finger_and_foottapping/9783755/1)
- **Description**: fNIRS dataset for classification of unilateral finger and foot tapping, containing 30 subjects
- **Data Format**: MATLAB format (.mat files)

### 3. WG Dataset (Word Generation)
- **Download Link**: [https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/](https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/)
- **Description**: Cognitive task data with simultaneous EEG and NIRS recordings from 26 subjects, including word generation tasks
- **Data Format**: MATLAB format (.mat files)

## Preprocessing Steps

### MA Dataset Preprocessing

The MA dataset requires two preprocessing steps:

#### Step 1: MATLAB Format Conversion
Use MATLAB script to convert raw data to intermediate format:

```matlab
% Script path: Data_process/signal_process/MA_fNIRS_to_mat.m
```

**Execution Method**:
1. Ensure BBCI toolbox is installed ([https://github.com/bbci/bbci_public](https://github.com/bbci/bbci_public))
2. Open and run `Data_process/signal_process/MA_fNIRS_to_mat.m` in MATLAB
3. The script will:
   - Load raw data from 29 subjects
   - Perform Beer-Lambert law conversion (MBLL)
   - Apply 0.01-0.1 Hz bandpass filtering
   - Separate oxyhemoglobin (HbO) and deoxyhemoglobin (HbR)
   - Segment data using sliding time windows (3-second window, 1-second step)
   - Save processed data to `Data_process/MA_fNIRS_data/` directory

**Output Directory Structure**:
```
Data_process/MA_fNIRS_data/
├── 1/
│   ├── 1_deoxy.mat
│   ├── 1_oxy.mat
│   ├── 2_deoxy.mat
│   ├── 2_oxy.mat
│   ├── ...
│   └── 1_desc.mat (label file)
├── 2/
└── ...
```

#### Step 2: Python Data Segmentation and Format Conversion
Use Python script to convert MATLAB data to Excel format:

```bash
# Script path: Data_process/signal_process/convert_MA_fNIRS_data_to_xls.py
python Data_process/signal_process/convert_MA_fNIRS_data_to_xls.py
```

**Script Functions**:
- Read MATLAB files generated in Step 1
- Extract data from time windows 9-18 (10 windows total)
- Organize each subject's data by time window
- Save each time window as an Excel file with multiple task worksheets
- Save label file in Excel format

**Output Directory Structure**:
```
Data_process/MA_xlsdata/
├── sub_01/
│   ├── time_window_1.xls
│   ├── time_window_2.xls
│   ├── ...
│   └── sub_01_label.xlsx
├── sub_02/
└── ...
```

### UFFT Dataset Preprocessing

The UFFT dataset requires only one preprocessing step:

#### MATLAB Preprocessing Script
Use MATLAB script to complete all preprocessing steps:

```matlab
% Script path: Data_process/signal_process/UFFT_fNIRS_to_xls.m
```

**Execution Method**:
1. Ensure BBCI toolbox is installed
2. Open and run `Data_process/signal_process/UFFT_fNIRS_to_xls.m` in MATLAB
3. The script will:
   - Load raw data from 30 subjects
   - Apply 0.01-0.1 Hz bandpass filtering (3rd-order Butterworth filter)
   - Segment data (-1 to 25 seconds relative to task onset)
   - Perform baseline correction (-1 to 0 seconds as reference interval)
   - Save 75 task data files per subject as Excel files

**Output Directory Structure**:
```
Data_process/UFFT_data/
├── 1/
│   ├── 1.xls (task 1)
│   ├── 2.xls (task 2)
│   ├── ...
│   └── 1_desc.xls (label file)
├── 2/
└── ...
```

### WG Dataset Preprocessing

The WG dataset uses Python script for preprocessing:

#### Python Preprocessing Script
```bash
# Script path: Data_process/signal_process/convertWG_fNIRS_data_to_xls.py
python Data_process/signal_process/convertWG_fNIRS_data_to_xls.py
```

**Script Functions**:
- Read MATLAB data files (`cnt_wg.mat` and `mrk_wg.mat`) from 26 subjects
- Extract oxyhemoglobin and deoxyhemoglobin signals
- Build task intervals based on marker timestamps
- Organize each subject's data by task as Excel files
- Save label information

**Input Directory Structure**:
```
Data_process/wgnirs/
├── VP001-NIRS/
│   ├── cnt_wg.mat
│   └── mrk_wg.mat
├── VP002-NIRS/
└── ...
```

**Output Directory Structure**:
```
Data_process/WGXLS/
├── VP001-NIRS/
│   ├── signal_data.xlsx (contains multiple task worksheets)
│   └── labels.xlsx
├── VP002-NIRS/
└── ...
```

## Preprocessing Notes

1. **System Environment**: The preprocessing scripts have been developed and tested on Linux (Ubuntu 22.04.5 LTS). While the Python scripts may be compatible with other operating systems, the MATLAB preprocessing components require a Linux environment for optimal compatibility with the BBCI toolbox.
2. **BBCI Toolbox Dependency**: Preprocessing of MA and UFFT datasets requires BBCI toolbox. Ensure it is properly installed and configured.
3. **Path Configuration**: All paths in scripts are relative paths. Ensure execution from project root directory.
4. **Data Integrity**: Before preprocessing, check that all raw data files are complete and all subject data has been downloaded.
5. **Memory Requirements**: Large-scale data processing may require significant memory. Consider batch processing or using high-performance computing resources.
6. **MATLAB Version**: MATLAB R2018b or higher is recommended for compatibility.

## Post-Preprocessing Data Format

After preprocessing, all datasets will generate Excel files in a unified format:
- **Signal Data**: Each task/time window saved as an independent Excel worksheet or file
- **Label Data**: Saved separately as Excel files containing task category labels
- **Data Organization**: Directory structure organized by subject number

After preprocessing is complete, the data is ready for MaNDi-Net model training.

