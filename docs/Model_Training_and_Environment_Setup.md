# MaNDi-Net: Model Training and Environment Setup Guide

This document provides instructions on model training, testing, and environment setup for **MaNDi-Net** (Mamba-Transformer Deep Integration Network), a hybrid architecture for fNIRS signal classification.

## Model Architecture

MaNDi-Net uses a hybrid Mamba-Transformer architecture (`HybridMambaTransformer`) for fNIRS signal classification. The model combines Mamba's state space model with Transformer's attention mechanism.

For detailed architecture information, see `docs/Model_Architecture_Reference.md`.

### Quick Configuration Reference

| Dataset | window_feature_dim | num_pairs | num_classes |
|---------|-------------------|-----------|-------------|
| MA | 2340 | 36 | 2 |
| UFFT | 580 | 20 | 3 |
| WG | 828 | 36 | 2 |

All datasets use `aggregate_mamba_internal=True` and `use_permute=True`.

## Training Scripts

### K-fold Cross-Validation
- `MI_Kfold.py` - UFFT dataset (3-class)
- `MA_Kfold.py` - MA dataset (2-class)
- `WG_Kfold.py` - WG dataset (2-class)

### Leave-One-Subject-Out (LOSO)
- `MILOSO.py` - UFFT dataset
- `MALOSO.py` - MA dataset
- `WGLOSO.py` - WG dataset

### Usage

```bash
# K-fold examples
python MA_Kfold.py
python MI_Kfold.py
python WG_Kfold.py

# LOSO examples
python MALOSO.py
python MILOSO.py
python WGLOSO.py
```

### Training Parameters

- Learning Rate: 0.001
- Optimizer: Adam
- Batch Size: 32
- Epochs: 600
- Scheduler: StepLR (decay 0.1 every 10 epochs)

### Output

- Logs: `logs/training_log.txt`
- Metrics: `logs/subject_metrics/`
- Models: Saved automatically (best validation accuracy)
  - K-fold: `sub_{subject_id:02d}_fold_{fold_idx}_best_acc.pth`
  - LOSO: `sub_{subject_id:02d}_best_acc.pth` (WG: `sub_VP{subject_id:03d}_best_acc.pth`)

## Testing Scripts

Test scripts are located in `LOSO_Test/`:
- `MA_test.py`
- `MI_test.py`
- `WG_test.py`

### Usage

```bash
python LOSO_Test/MA_test.py
python LOSO_Test/MI_test.py
python LOSO_Test/WG_test.py
```

### Important Notes

1. Update `DATA_ROOT` and `MODEL_SAVE_DIR` in test scripts
2. Ensure configuration matches training script exactly
3. Model file naming:
   - MA/MI: `sub_{subject_id:02d}_best_acc.pth`
   - WG: `sub_VP{subject_id:03d}_best_acc.pth`

## Environment Setup

### System Requirements

This implementation has been developed and tested on the following computational environment:

- **Operating System**: Linux (Ubuntu 22.04.5 LTS)
- **Graphics Processing Unit**: NVIDIA GeForce RTX 4090 (recommended)
- **Python**: 3.10
- **CUDA**: 11.8
- **PyTorch**: 2.1.2

**Note**: The RTX 4090 GPU is recommended for optimal performance due to its substantial computational capacity (24GB VRAM) and high memory bandwidth, which are essential for efficiently training the hybrid Mamba-Transformer architecture. While the model can be executed on other GPU configurations, the RTX 4090 provides the most reliable performance for the experiments described in this work.

### Installation

```bash
# 1. Create conda environment
conda create -n mandinet python=3.10
conda activate mandinet

# 2. Install PyTorch
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Download Mamba SSM wheel files
# Create Mamba_ssm directory if it doesn't exist
mkdir -p Mamba_ssm
cd Mamba_ssm

# Download causal_conv1d wheel file (required dependency)
# Source: https://github.com/Dao-AILab/causal-conv1d/releases
# File to download: causal_conv1d-1.5.0.post8+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# Instructions:
#   1. Navigate to https://github.com/Dao-AILab/causal-conv1d/releases
#   2. Find the release containing version 1.5.0.post8
#   3. Download the file: causal_conv1d-1.5.0.post8+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
#   4. Save it to the Mamba_ssm/ directory

# Download mamba_ssm wheel file
# Source: https://github.com/state-spaces/mamba/releases
# File to download: mamba_ssm-2.2.4+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# Instructions:
#   1. Navigate to https://github.com/state-spaces/mamba/releases
#   2. Find the release containing version 2.2.4
#   3. Download the file: mamba_ssm-2.2.4+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
#   4. Save it to the Mamba_ssm/ directory

# 4. Install Mamba SSM (from local wheel files)
pip install causal_conv1d-1.5.0.post8+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-2.2.4+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
cd ..

# 5. Install other dependencies
pip install numpy==1.26.4 pandas==2.2.3 scipy==1.13.1 scikit-learn==1.6.1
pip install nolds==0.6.1 antropy==0.1.9
pip install einops==0.8.1 tqdm==4.66.5 openpyxl==3.1.5 xlrd==2.0.1
```

**Note**: Mamba SSM must be installed from local wheel files in `Mamba_ssm/` directory.

### Downloading Mamba SSM Wheel Files

Before installing Mamba SSM, you need to download the required wheel files from their respective GitHub release pages:

1. **causal_conv1d wheel file**:
   - **Source**: [causal-conv1d releases](https://github.com/Dao-AILab/causal-conv1d/releases)
   - **Required file**: `causal_conv1d-1.5.0.post8+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
   - **Instructions**:
     - Navigate to the [causal-conv1d releases page](https://github.com/Dao-AILab/causal-conv1d/releases)
     - Locate the release containing version `1.5.0.post8`
     - Download the wheel file matching your system configuration: `causal_conv1d-1.5.0.post8+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
     - Save the file to the `Mamba_ssm/` directory in your project root

2. **mamba_ssm wheel file**:
   - **Source**: [mamba releases](https://github.com/state-spaces/mamba/releases)
   - **Required file**: `mamba_ssm-2.2.4+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
   - **Instructions**:
     - Navigate to the [mamba releases page](https://github.com/state-spaces/mamba/releases)
     - Locate the release containing version `2.2.4`
     - Download the wheel file matching your system configuration: `mamba_ssm-2.2.4+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
     - Save the file to the `Mamba_ssm/` directory in your project root

**Important**: Ensure both wheel files are placed in the `Mamba_ssm/` directory before proceeding with the installation step (step 4 in the Installation section above).

### Verify Installation

```python
import torch
from mamba_ssm import Mamba
from model import HybridMambaTransformer
print("Installation successful!")
```

## Common Issues

1. **CUDA Mismatch**: Check CUDA version matches PyTorch version. Ensure NVIDIA drivers are properly installed and compatible with CUDA 11.8.
2. **GPU Memory Error**: If encountering out-of-memory errors, reduce batch size. The recommended RTX 4090 (24GB VRAM) should accommodate the default batch size of 32, but lower-end GPUs may require batch size reduction.
3. **Mamba SSM Error**: Ensure installed from local wheel files in `Mamba_ssm/` directory.
4. **Path Error**: Verify `DATA_ROOT` paths in scripts match your system configuration.
5. **Model Loading Error**: Check model file naming and architecture match between training and testing scripts.
6. **Operating System Compatibility**: This implementation is optimized for Linux (Ubuntu 22.04.5 LTS). While the code may run on other Linux distributions, Ubuntu 22.04.5 LTS is the recommended environment.

## Project Structure

```
MaNDi-Net/
├── model.py              # Model definition
├── *_Kfold.py           # K-fold training scripts
├── *LOSO.py             # LOSO training scripts
├── LOSO_Test/           # Test scripts
├── Mamba_ssm/           # Mamba SSM wheel files
└── logs/                # Training logs
```

## Quick Start

1. **Setup Environment**: Follow installation steps above
2. **Prepare Data**: See `docs/Data_Preprocessing_Guide.md` for detailed preprocessing instructions
3. **Train MaNDi-Net Model**: Run appropriate training script (see Training Scripts section)
4. **Test MaNDi-Net Model**: Run test script after training (see Testing Scripts section)

## Additional Documentation

- **Model Architecture Details**: `docs/Model_Architecture_Reference.md` - Detailed technical reference
- **Data Preprocessing**: `docs/Data_Preprocessing_Guide.md` - Complete preprocessing guide

---

**Note**: Update paths in scripts to match your system configuration.
