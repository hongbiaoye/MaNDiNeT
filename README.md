# ManDI-Net

Lightweight overview of the repository layout and key entry points for
working with the ManDI-Net fNIRS decoding project.

## Directory Overview

- `DATA/` – curated datasets in XLS/XLSX format (MA, WG, UFFT subsets).
- `Data_process/` – raw recordings, MATLAB/Python converters, and the
  `signal_process` toolbox for preprocessing and feature extraction.
- `docs/` – project notes, training instructions, and model references
  (`Model_Training_and_Environment_Setup.md`, etc.).
- `LOSO_Test/` – leave-one-subject-out evaluation scripts for MA, MI,
  and WG cohorts.
- `logs/`, `logs2/` – training/evaluation logs plus saved metrics.
- `Mamba_ssm/` – local wheels and configs required for installing the
  Mamba SSM dependency.
- `save_model/` – checkpoint directories (`MA_models`, `MI_models`,
  `Wg_models`) containing the best saved weights.

## Training & Evaluation Scripts

- `MA_Kfold.py`, `MI_Kfold.py`, `WG_Kfold.py` – k-fold training entry
  points for each dataset variant.
- `MALOSO.py`, `MILOSO.py`, `WGLOSO.py` – LOSO training pipelines.
- `LOSO_Test/*.py` – testing utilities matching the LOSO-trained models.
- `model.py` – central network architecture definition.

## Getting Started

1. Install dependencies listed in `requirements.txt` (see the notes
   inside for CUDA/PyTorch and Mamba SSM setup specifics).
2. Follow `docs/Model_Training_and_Environment_Setup.md` for detailed
   environment configuration.
3. Run the appropriate training script (k-fold or LOSO) to reproduce or
   fine-tune results.
