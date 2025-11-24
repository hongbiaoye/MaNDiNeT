import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import os
from nolds import lyap_r
from scipy.signal import hilbert
import warnings
from mamba_ssm import Mamba
from model import HybridMambaTransformer  

warnings.filterwarnings("ignore")

# Dataset configuration (new channel pair parameters, variable names compatible with original version)
DATA_ROOT = r'DATA\WGXLS'
NUM_CLASSES = 2  # Binary classification
TIME_WINDOWS = [3, 4, 5]  # Use windows 1, 2, 3 (1-based indexing, original comment error, actually takes first 3 windows)

# ========== Model Configuration Parameters ==========
MODEL_CONFIG = {
    'num_windows': len(TIME_WINDOWS),
    'd_model': 16,
    'nhead': 4,
    'window_feature_dim': 828,  # WG dataset: 72*10 + 72 + 36 = 828
    'num_pairs': 36,  # WG dataset: 36 channel pairs
    'aggregate_mamba_internal': True,  # WG_Kfold aggregates internally
    'use_permute': True,  # WG_Kfold uses permute
    'num_classes': NUM_CLASSES  # Binary classification
}
# =====================================================
WINDOW_LENGTH = 10          # 10 time point window
NUM_CHANNEL_PAIRS = 36      # First 36 + last 36 channels form 36 pairs (key parameter)
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, "subject_metrics"), exist_ok=True)
GLOBAL_LOG = os.path.join(LOG_DIR, "training_log.txt")

# Feature calculation functions (completely consistent with original version, 36 pairs of PLV)
def calculate_lyapunov(signal):
    try:
        return lyap_r(signal, emb_dim=3, delay=1)
    except:
        return 0.0

def calculate_plv(signal1, signal2):
    analytic1 = hilbert(signal1)
    analytic2 = hilbert(signal2)
    phase_diff = np.angle(analytic1) - np.angle(analytic2)
    return np.abs(np.mean(np.exp(1j * phase_diff)))

def extract_window_features(window_data):
    channel_data = window_data.reshape(72, WINDOW_LENGTH)  # 72 channels × 10 time points
    lyap_features = [calculate_lyapunov(channel_data[ch]) for ch in range(72)]  # 72 Lyapunov exponents
    plv_features = [calculate_plv(channel_data[ch], channel_data[ch + 36]) for ch in range(36)]  # 36 pairs of PLV
    return np.concatenate([window_data.flatten(), lyap_features, plv_features])  # 828 dimensions per window

# Dataset class (core modification: output 36 pairs of channel signals, variable name strictly maintains raw_signals)
class AFNIRSDataset(Dataset):
    def __init__(self, subject_id, time_windows=TIME_WINDOWS):
        super().__init__()
        self.subject_id = subject_id
        self.time_windows = time_windows
        self.subject_dir = os.path.join(DATA_ROOT, f"VP{subject_id:03d}-NIRS")
        self.data_path = os.path.join(self.subject_dir, "signal_data.xlsx")
        self.label_path = os.path.join(self.subject_dir, "labels.xlsx")
        self.labels = self._load_labels()
        self.features, self.raw_signals = self._process_data()  # Keep raw_signals variable name unchanged

    def _load_labels(self):
        label_df = pd.read_excel(self.label_path, header=0)
        label_data = label_df.iloc[0:, 0] 
        if len(label_data) != 60:
            raise ValueError(f"Label count must be 60, got {len(label_data)}")
        numeric_labels = label_data.astype(int).tolist()
        for label in numeric_labels:
            if label not in {0, 1}:
                raise ValueError(f"Invalid label value: {label}, must be 0 or 1")
        return numeric_labels

    def _process_data(self):
        xls = pd.ExcelFile(self.data_path)
        task_sheets = [s for s in xls.sheet_names if s.startswith('Task_')]
        task_sheets.sort(key=lambda x: int(x.split('_')[1]))
        if len(task_sheets) != 60:
            raise ValueError(f"Signal data must contain 60 Task sheets, got {len(task_sheets)}")
        
        features_list = []
        raw_signals_list = []  # Keep variable name, store 36 pairs of channel signals
        
        for sheet_name in task_sheets:
            data = xls.parse(sheet_name).values[200:200+10*WINDOW_LENGTH]
            windows = [data[i*WINDOW_LENGTH:(i+1)*WINDOW_LENGTH] for i in range(10)]
            selected_windows = [windows[w-1] for w in self.time_windows]  # 1-based to 0-based
            
            task_features = []
            task_raw_signals = []  # Store 36 pairs of channel signals for 3 windows of current task
            
            for win in selected_windows:
                channel_data = win.T  # Transpose to (72,10) (channels × time points)
                # Split first 36 and last 36 channels to form 36 pairs (ch, ch+36)
                pair_data = np.stack([channel_data[:36], channel_data[36:]], axis=1)  # [36 pairs, 2 channels, 10 time points]
                task_raw_signals.append(pair_data)  # Channel pair signals for single window (36,2,10)
                task_features.append(extract_window_features(win.flatten()))  # Extract 828-dimensional features
            
            # Concatenate features from 3 windows (3×828=2484 dimensions)
            features_list.append(np.concatenate(task_features))
            # Channel pair signal shape: [3 windows, 36 pairs, 2 channels, 10 time points]
            raw_signals_list.append(np.stack(task_raw_signals))
        
        # Feature standardization
        features = np.array(features_list)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(np.array(raw_signals_list), dtype=torch.float32)  # Keep raw_signals variable name
        )

    def __getitem__(self, idx):
        return (self.features[idx], self.raw_signals[idx]), self.labels[idx]  # Return value format unchanged


# Training function (variable names and logic completely consistent with original version, only model internal implementation differs)
def train_subject(subject_id, device, batch_size=5, epochs=600):
    dataset = AFNIRSDataset(subject_id)
    X = (dataset.features.numpy(), dataset.raw_signals.numpy())  # Keep raw_signals variable name
    y = np.array(dataset.labels)
    kf = KFold(5, shuffle=True, random_state=42)
    subject_metrics = []

    with open(GLOBAL_LOG, "a") as f_global:
        f_global.write(f"===== Processing subject VP{subject_id:03d} =====\n")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X[0], y)):
        train_set = torch.utils.data.Subset(dataset, train_idx)
        test_set = torch.utils.data.Subset(dataset, test_idx)
        train_loader = DataLoader(train_set, batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size, shuffle=False)

        # Use unified model, configuration parameters defined at the top of the script
        model = HybridMambaTransformer(**MODEL_CONFIG).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=150)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_metrics = {}

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for (features, signals), labels in train_loader:
                features, signals, labels = features.to(device), signals.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model((features, signals))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for (features, signals), labels in test_loader:
                    features, signals = features.to(device), signals.to(device)
                    outputs = model((features, signals))
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            pre = precision_score(all_labels, all_preds, average='binary', zero_division=0)
            rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
            kap = cohen_kappa_score(all_labels, all_preds)

            if acc > best_acc:
                best_acc = acc
                best_metrics = {
                    "acc": acc,
                    "pre": pre,
                    "rec": rec,
                    "f1": f1,
                    "kap": kap
                }

        subject_metrics.append(best_metrics)
        print(f"Subject VP{subject_id:03d} Fold {fold + 1} Best Metrics | "
              f"ACC: {best_metrics['acc']:.4f} | "
              f"Precision: {best_metrics['pre']:.4f} | "
              f"Recall: {best_metrics['rec']:.4f} | "
              f"F1: {best_metrics['f1']:.4f} | "
              f"Kappa: {best_metrics['kap']:.4f}")

        metrics_path = os.path.join(
            LOG_DIR, "subject_metrics",
            f"sub_VP{subject_id:03d}_fold_{fold + 1}_metrics.txt"
        )
        with open(metrics_path, "w") as f:
            f.write(f"Subject VP{subject_id:03d} Fold {fold + 1} Best Metrics:\n")
            for k, v in best_metrics.items():
                f.write(f"{k}: {v:.4f}\n")

        with open(GLOBAL_LOG, "a") as f_global:
            f_global.write(f"Subject VP{subject_id:03d} Fold {fold + 1} ACC: {best_metrics['acc']:.4f}\n")

    return subject_metrics

# Main function (completely maintains original version, no variable name modifications)
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    with open(GLOBAL_LOG, "w") as f:
        f.write(f"Window length {WINDOW_LENGTH} - Binary classification complete training script (36 pairs of cross-channel PLV)\n")

    all_subject_metrics = []
    for subject_id in range(1, 27):
        print(f"\n===== Starting training for subject VP{subject_id:03d} =====")
        try:
            subject_metrics = train_subject(subject_id, device)
            all_subject_metrics.extend(subject_metrics)
            print(f"Subject VP{subject_id:03d} training completed")
        except Exception as e:
            with open(GLOBAL_LOG, "a") as f_global:
                f_global.write(f"Subject VP{subject_id:03d} training error: {str(e)}\n")
            print(f"Subject VP{subject_id:03d} training error: {str(e)}")

    metrics = ["acc", "pre", "rec", "f1", "kap"]
    metric_data = {m: [] for m in metrics}
    for metric in all_subject_metrics:
        for m in metrics:
            metric_data[m].append(metric[m])

    print("\n================= Binary Classification Overall Evaluation Metrics (Mean ± Std) ==================")
    with open(GLOBAL_LOG, "a") as f_global:
        f_global.write("\nBinary classification overall evaluation metrics:\n")
        for m in metrics:
            data = np.array(metric_data[m])
            mean, std = np.mean(data), np.std(data)
            if m == "kap":
                print(f"Kappa coefficient: {mean:.4f} ± {std:.4f}")
                f_global.write(f"Kappa: {mean:.4f} ± {std:.4f}\n")
            else:
                print(f"{m.upper()}: {mean*100:.2f}% ± {std*100:.2f}%")
                f_global.write(f"{m.upper()}: {mean*100:.2f}% ± {std*100:.2f}%\n")

    print("\nTraining completed, all logs saved to", LOG_DIR)

if __name__ == "__main__":
    main()