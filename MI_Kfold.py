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
from model import HybridMambaTransformer  # Import unified model

warnings.filterwarnings("ignore")

# Dataset configuration (new channel pair parameters, variable names compatible with original version)
DATA_ROOT = r'DATA\UFFT_data'
NUM_CLASSES = 3  # Three-class classification
TIME_WINDOWS = [8, 9, 10]  # Use windows 8, 9, 10 (1-based indexing)

# ========== Model Configuration Parameters ==========
MODEL_CONFIG = {
    'num_windows': len(TIME_WINDOWS),
    'd_model': 16,
    'nhead': 4,
    'window_feature_dim': 580,  # MI dataset: 40*13 + 40 + 20 = 580
    'num_pairs': 20,  # MI dataset: 20 channel pairs
    'aggregate_mamba_internal': True,  # MI_Kfold aggregates internally
    'use_permute': True,  # MI_Kfold uses permute
    'num_classes': NUM_CLASSES  # Three-class classification
}
# =====================================================
WINDOW_LENGTH = 13          # Each window has 13 time points
NUM_CHANNEL_PAIRS = 20      # First 20 + last 20 channels form 20 pairs (new variable, maintain naming convention)
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, "subject_metrics"), exist_ok=True)
GLOBAL_LOG = os.path.join(LOG_DIR, "training_log.txt")

# Feature calculation functions (completely consistent with original version, ensure PLV calculation corresponds to 20 pairs)
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
    channel_data = window_data.reshape(40, WINDOW_LENGTH)  # 40 channels × 13 time points
    lyap_features = [calculate_lyapunov(channel_data[ch]) for ch in range(40)]  # 40 Lyapunov exponents
    # 20 pairs of channel pair PLV (first 20 paired with last 20 channels: ch and ch+20)
    plv_features = [calculate_plv(channel_data[ch], channel_data[ch + 20]) for ch in range(20)]  # 20 pairs of PLV
    # Total feature dimension: 40*13 (raw) + 40 (Lyapunov) + 20 (PLV) = 580 dimensions per window
    return np.concatenate([window_data.flatten(), lyap_features, plv_features])

# Dataset class (core modification: output channel pair signals, variable names consistent with original version)
class AFNIRSDataset(Dataset):
    def __init__(self, subject_id, time_windows=TIME_WINDOWS):
        super().__init__()
        self.subject_id = subject_id
        self.time_windows = time_windows  # Window indices used (1-based)
        self.data_path = os.path.join(DATA_ROOT, str(subject_id), f"{subject_id}.xls")
        self.label_path = os.path.join(DATA_ROOT, str(subject_id), f"{subject_id}_desc.xls")
        self.labels = self._load_labels()
        self.features, self.raw_signals = self._process_data()  # Keep variable name raw_signals (actually channel pair signals)
        # Validate label count matches data sample count
        if len(self.labels) != len(self.features):
            raise ValueError(f"Label count ({len(self.labels)}) does not match data sample count ({len(self.features)}) for subject {subject_id}")

    def _load_labels(self):
        label_df = pd.read_excel(self.label_path, header=None)
        labels = label_df.values.flatten().tolist()
        # Convert to 0-based indexing, ensuring all values are valid integers
        converted_labels = []
        for idx, l in enumerate(labels):
            # Handle NaN/None
            if pd.isna(l) or l is None:
                raise ValueError(f"Found NaN/None label at index {idx} in subject {self.subject_id}")
            # Convert to int (handles float values like 1.0, 2.0, etc.)
            try:
                l_val = float(l)  # First convert to float to handle both int and float strings
                l_int = int(l_val)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid label value '{l}' (type: {type(l)}) at index {idx} in subject {self.subject_id}")
            # Validate range before conversion
            if l_int < 1 or l_int > NUM_CLASSES:
                raise ValueError(f"Label value {l_int} at index {idx} out of range [1, {NUM_CLASSES}] in subject {self.subject_id}")
            converted_labels.append(l_int - 1)  # Convert to 0-based: [0, NUM_CLASSES-1]
        return converted_labels

    def _process_data(self):
        xls = pd.ExcelFile(self.data_path)
        features_list = []
        raw_signals_list = []  # Keep variable name raw_signals, actually stores channel pair signals
        
        for sheet_idx in range(len(xls.sheet_names)):
            data = xls.parse(sheet_idx).values[20:20+10*WINDOW_LENGTH]  # Extract 10 windows
            windows = [data[i*WINDOW_LENGTH:(i+1)*WINDOW_LENGTH] for i in range(10)]
            selected_windows = [windows[w-1] for w in self.time_windows]  # Select windows 8/9/10 (0-based indexing)
            
            task_features = []
            task_raw_signals = []  # Store channel pair signals for 3 windows of current task
            
            for win in selected_windows:
                channel_data = win.T  # Transpose to (40,13) (channels × time points)
                # Split first 20 and last 20 channels to form 20 pairs (ch, ch+20)
                pair_data = np.stack([channel_data[:20], channel_data[20:]], axis=1)  # [20 pairs, 2 channels, 13 time points]
                task_raw_signals.append(pair_data)  # Channel pair signals for single window (20,2,13)
                task_features.append(extract_window_features(win.flatten()))  # Extract handcrafted features
            
            # Concatenate features from 3 windows (3×580=1740 dimensions)
            features_list.append(np.concatenate(task_features))
            # Channel pair signal shape: [3 windows, 20 pairs, 2 channels, 13 time points]
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
        # Return label as Python int (DataLoader will handle conversion to tensor)
        # This matches the original implementation behavior
        return (self.features[idx], self.raw_signals[idx]), self.labels[idx]  # Keep return value format consistent

# Training function (only model class modified, all other variable names and logic completely consistent with original version)
def train_subject(subject_id, device, batch_size=5, epochs=600):
    dataset = AFNIRSDataset(subject_id)
    X = (dataset.features.numpy(), dataset.raw_signals.numpy())  # Keep raw_signals variable name
    y = np.array(dataset.labels)
    kf = KFold(5, shuffle=True, random_state=42)
    subject_metrics = []

    with open(GLOBAL_LOG, "a") as f_global:
        f_global.write(f"===== Processing subject {subject_id} =====\n")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X[0], y)):
        train_set = torch.utils.data.Subset(dataset, train_idx)
        test_set = torch.utils.data.Subset(dataset, test_idx)
        train_loader = DataLoader(train_set, batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size, shuffle=False)

        # Initialize model (keep class name HybridMambaTransformer)
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
                # Ensure labels are long type (required by CrossEntropyLoss)
                labels = labels.long()
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
        print(f"Subject {subject_id} Fold {fold + 1} Best Metrics | "
              f"ACC: {best_metrics['acc']:.4f} | "
              f"Precision: {best_metrics['pre']:.4f} | "
              f"Recall: {best_metrics['rec']:.4f} | "
              f"F1: {best_metrics['f1']:.4f} | "
              f"Kappa: {best_metrics['kap']:.4f}")

        metrics_path = os.path.join(
            LOG_DIR, "subject_metrics",
            f"sub_{subject_id:02d}_fold_{fold + 1}_metrics.txt"
        )
        with open(metrics_path, "w") as f:
            f.write(f"Subject {subject_id} Fold {fold + 1} Best Metrics:\n")
            for k, v in best_metrics.items():
                f.write(f"{k}: {v:.4f}\n")

        with open(GLOBAL_LOG, "a") as f_global:
            f_global.write(f"Subject {subject_id} Fold {fold + 1} ACC: {best_metrics['acc']:.4f}\n")

    return subject_metrics

# Main function (completely maintains original version, no variable name modifications)
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    with open(GLOBAL_LOG, "w") as f:
        f.write(f"Window length {WINDOW_LENGTH} - Hybrid Mamba-Transformer model training log\n")

    all_subject_metrics = []
    for subject_id in range(1, 31):
        print(f"\n===== Starting training for subject {subject_id} =====")
        try:
            subject_metrics = train_subject(subject_id, device)
            all_subject_metrics.extend(subject_metrics)
            print(f"Subject {subject_id} training completed")
        except Exception as e:
            with open(GLOBAL_LOG, "a") as f_global:
                f_global.write(f"Subject {subject_id} training error: {str(e)}\n")
            print(f"Subject {subject_id} training error: {str(e)}")

    metrics = ["acc", "pre", "rec", "f1", "kap"]
    metric_data = {m: [] for m in metrics}
    for metric in all_subject_metrics:
        for m in metrics:
            metric_data[m].append(metric[m])

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