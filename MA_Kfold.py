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
from mamba_ssm import Mamba  # Requires mamba-ssm library
from model import HybridMambaTransformer  # Import unified model

warnings.filterwarnings("ignore")

# Dataset configuration
DATA_ROOT = r'DATA\MA_xlsdata'
NUM_CLASSES = 2  # Binary classification
TIME_WINDOWS = [8, 9, 10]  # Use windows 8, 9, 10 (1-based indexing)

# ========== Model Configuration Parameters ==========
MODEL_CONFIG = {
    'num_windows': len(TIME_WINDOWS),
    'd_model': 16,
    'nhead': 4,
    'window_feature_dim': 2340,  # MA dataset: 72*31 + 72 + 36 = 2340
    'num_pairs': 36,  # MA dataset: 36 channel pairs
    'aggregate_mamba_internal': True,  # MA_Kfold aggregates internally
    'use_permute': True,  # MA_Kfold uses permute
    'num_classes': NUM_CLASSES  # Binary classification
}
# =====================================================
WINDOW_LENGTH = 31          # Each window has 31 time points
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, "subject_metrics"), exist_ok=True)
GLOBAL_LOG = os.path.join(LOG_DIR, "training_log.txt")

# Feature calculation functions
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
    channel_data = window_data.reshape(72, WINDOW_LENGTH)  # 72 channels × 31 time points
    lyap_features = [calculate_lyapunov(channel_data[ch]) for ch in range(72)]
    plv_features = [calculate_plv(channel_data[ch], channel_data[ch + 36]) for ch in range(36)]  # 36 pairs of PLV
    return np.concatenate([window_data.flatten(), lyap_features, plv_features])  # 2340 dimensions per window


# Dataset class optimized for channel pairs
class AFNIRSDataset(Dataset):
    def __init__(self, subject_id, time_windows=TIME_WINDOWS):
        super().__init__()
        self.subject_id = subject_id
        self.time_windows = time_windows
        self.label_path = os.path.join(DATA_ROOT, f"sub_{subject_id:02d}", f"sub_{subject_id:02d}_label.xlsx")
        self.window_files = [f"time_window_{w}.xls" for w in self.time_windows]  # 3 window files
        self.labels = self._load_labels()
        self.features, self.pair_signals = self._process_data()  # Output channel pair signals (not raw signals)

    def _load_labels(self):
        label_df = pd.read_excel(self.label_path, header=0)
        valid_labels = label_df.values[0:, 0].tolist()
        if len(valid_labels) != 60:
            raise ValueError(f"Label count should be 60, got {len(valid_labels)}")
        return valid_labels

    def _process_data(self):
        features_list = []
        pair_signals_list = []  # Store 36 pairs of channel signals (2 channels per pair)
        
        for task_idx in range(60):
            task_features = []
            task_pair_signals = []  # Store channel pair signals for 3 windows of current task
            
            for win_idx, window_file in enumerate(self.window_files):
                window_path = os.path.join(DATA_ROOT, f"sub_{self.subject_id:02d}", window_file)
                xls = pd.ExcelFile(window_path)
                sheet_name = xls.sheet_names[task_idx]
                data = xls.parse(sheet_name).values  # Raw data shape: (31,72) (time points × channels)
                channel_data = data.T  # Transpose to (72,31) (channels × time points)
                
                # Channel pair division: pair first 36 channels with last 36 channels (0-based: ch and ch+36)
                pair_data = channel_data.reshape(36, 2, WINDOW_LENGTH)  # 36 pairs × 2 channels × 31 time points
                task_pair_signals.append(pair_data)  # Channel pair signals for single window
                
                # Feature extraction (consistent with original logic)
                task_features.append(extract_window_features(data.flatten()))  # 2340 dimensions per window
            
            # Concatenate features from 3 windows (3×2340=7020 dimensions)
            features_list.append(np.concatenate(task_features))
            # Convert raw signals to channel pair signals: [3 windows × 36 pairs × 2 channels × 31 time points]
            pair_signals_list.append(np.stack(task_pair_signals))  # Shape: [60 tasks, 3 windows, 36 pairs, 2 channels, 31 time points]
        
        # Feature standardization
        features = np.array(features_list)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        return (
            torch.tensor(features, dtype=torch.float32),  # Shape: [60, 7020]
            torch.tensor(np.array(pair_signals_list), dtype=torch.float32)  # Shape: [60, 3, 36, 2, 31]
        )

    def __getitem__(self, idx):
        return (self.features[idx], self.pair_signals[idx]), self.labels[idx]  # Return channel pair signals


# Training function
def train_subject(subject_id, device, batch_size=5, epochs=600):
    dataset = AFNIRSDataset(subject_id)
    X = (dataset.features.numpy(), dataset.pair_signals.numpy())  # Changed to channel pair signals
    y = np.array(dataset.labels)
    kf = KFold(5, shuffle=True, random_state=42)
    subject_metrics = []

    with open(GLOBAL_LOG, "a") as f_global:
        f_global.write(f"===== Processing subject {subject_id:02d} =====\n")

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
            for (features, signals), labels in train_loader:
                features, signals, labels = features.to(device), signals.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model((features, signals))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for (features, signals), labels in test_loader:
                    features, signals = features.to(device), signals.to(device)
                    outputs = model((features, signals))
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            scheduler.step(acc)

            if acc > best_acc:
                best_acc = acc
                best_metrics = {
                    "acc": acc,
                    "pre": precision_score(all_labels, all_preds, average='macro', zero_division=0),
                    "rec": recall_score(all_labels, all_preds, average='macro', zero_division=0),
                    "f1": f1_score(all_labels, all_preds, average='macro', zero_division=0),
                    "kap": cohen_kappa_score(all_labels, all_preds)
                }

        subject_metrics.append(best_metrics)
        print(f"[Subject {subject_id:02d} Fold {fold+1}] Best ACC: {best_acc:.4f} | F1: {best_metrics['f1']:.4f} | KAP: {best_metrics['kap']:.4f}")
        with open(GLOBAL_LOG, "a") as f_global:
            f_global.write(f"Subject {subject_id:02d} Fold {fold+1}: ACC={best_acc:.4f}, PRE={best_metrics['pre']:.4f}, REC={best_metrics['rec']:.4f}, F1={best_metrics['f1']:.4f}, KAP={best_metrics['kap']:.4f}\n")

    return subject_metrics


# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    with open(GLOBAL_LOG, "w") as f:
        f.write(f"Window length {WINDOW_LENGTH} - Channel pair optimized Mamba-Transformer model training log\n")

    all_subject_metrics = []
    for subject_id in range(1, 30):  # 29 subjects
        print(f"\n===== Starting training for subject {subject_id:02d} =====")
        try:
            subject_metrics = train_subject(subject_id, device)
            all_subject_metrics.extend(subject_metrics)
            print(f"Subject {subject_id:02d} training completed (5 folds)")
        except Exception as e:
            with open(GLOBAL_LOG, "a") as f_global:
                f_global.write(f"Subject {subject_id:02d} training error: {str(e)}\n")
            print(f"Subject {subject_id:02d} training error: {str(e)}")

    # Overall metrics calculation
    metrics = ["acc", "pre", "rec", "f1", "kap"]
    metric_data = {m: [] for m in metrics}
    for metrics_dict in all_subject_metrics:
        for m in metrics:
            metric_data[m].append(metrics_dict[m])

    print("\n================= Overall Evaluation Metrics (Mean ± Std) ==================")
    with open(GLOBAL_LOG, "a") as f_global:
        f_global.write("\nOverall evaluation metrics:\n")
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