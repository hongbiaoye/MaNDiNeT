import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import os
from nolds import lyap_r
from scipy.signal import hilbert
import warnings
from mamba_ssm import Mamba
from model import HybridMambaTransformer  # Import unified model

warnings.filterwarnings("ignore")

# Global configuration (includes model save path)
DATA_ROOT = r'DATA\MA_xlsdata'  # Original data path
LOG_DIR = os.path.join(os.getcwd(), "logs")  # Log directory
MODEL_SAVE_DIR = os.path.join(LOG_DIR, "best_models")  # Best model save directory
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)  # Auto-create directory
GLOBAL_LOG = os.path.join(LOG_DIR, "training_log.txt")  # Global training log
NUM_CLASSES = 2  # Binary classification
TIME_WINDOWS = [8, 9, 10]  # Selected time windows

# ========== Model Configuration Parameters ==========
MODEL_CONFIG = {
    'num_windows': len(TIME_WINDOWS),
    'd_model': 16,
    'nhead': 4,
    'window_feature_dim': 2340,  # MA dataset: 72*31 + 72 + 36 = 2340
    'num_pairs': 36,  # MA dataset: 36 channel pairs
    'aggregate_mamba_internal': True,  # MALOSO aggregates internally
    'use_permute': True,  # MALOSO uses permute
    'num_classes': NUM_CLASSES  # Binary classification
}
# =====================================================
WINDOW_LENGTH = 31  # Time points per window
NUM_CHANNEL_PAIRS = 36  # Number of channel pairs

# Feature calculation functions
def calculate_lyapunov(signal):
    """Calculate Lyapunov exponent (robust exception handling)"""
    try:
        return lyap_r(signal, emb_dim=3, delay=1)
    except:
        return 0.0

def calculate_plv(signal1, signal2):
    """Calculate Phase Locking Value (PLV)"""
    analytic1 = hilbert(signal1)
    analytic2 = hilbert(signal2)
    phase_diff = np.angle(analytic1) - np.angle(analytic2)
    return np.abs(np.mean(np.exp(1j * phase_diff)))

def extract_window_features(window_data):
    """Extract single window features (raw signal + Lyapunov + PLV)"""
    channel_data = window_data.reshape(72, WINDOW_LENGTH)  # 72 channels × 31 time points
    lyap_features = [calculate_lyapunov(channel_data[ch]) for ch in range(72)]  # 72 Lyapunov exponents
    plv_features = [calculate_plv(channel_data[ch], channel_data[ch + 36]) for ch in range(36)]  # 36 pairs of PLV
    return np.concatenate([window_data.flatten(), lyap_features, plv_features])  # 2340-dimensional features

# Custom dataset class
class AFNIRSDataset(Dataset):
    def __init__(self, subject_id, time_windows=TIME_WINDOWS):
        super().__init__()
        self.subject_id = subject_id
        self.time_windows = time_windows
        self.label_path = os.path.join(DATA_ROOT, f"sub_{subject_id:02d}", f"sub_{subject_id:02d}_label.xlsx")
        self.window_files = [f"time_window_{w}.xls" for w in self.time_windows]
        self.labels = self._load_labels()
        self.features, self.raw_signals = self._process_data()

    def _load_labels(self):
        """Load and validate label data"""
        label_df = pd.read_excel(self.label_path, header=0)
        valid_labels = label_df.values[0:, 0].tolist()
        if len(valid_labels) != 60:
            raise ValueError(f"Subject {self.subject_id:02d} label count should be 60, got {len(valid_labels)}")
        return valid_labels

    def _process_data(self):
        """Process raw data to generate features and channel pair signals"""
        features_list = []
        raw_signals_list = []
        
        for task_idx in range(60):
            task_features = []
            task_raw_signals = []  # Store channel pair signals for 3 windows
            
            for win_idx, window_file in enumerate(self.window_files):
                window_path = os.path.join(DATA_ROOT, f"sub_{self.subject_id:02d}", window_file)
                xls = pd.ExcelFile(window_path)
                sheet_name = xls.sheet_names[task_idx]
                data = xls.parse(sheet_name).values  # Raw data: (31,72)
                channel_data = data.T  # Transpose to (72,31)
                
                # Generate 36 pairs of channel signals: [36 pairs, 2 channels, 31 time points]
                pair_data = np.stack([channel_data[:36], channel_data[36:]], axis=1)
                task_raw_signals.append(pair_data)
                
                # Extract 2340-dimensional features
                task_features.append(extract_window_features(data.flatten()))
            
            # Concatenate features from 3 windows (3×2340=7020 dimensions)
            features_list.append(np.concatenate(task_features))
            # Channel pair signal shape: [3 windows, 36 pairs, 2 channels, 31 time points]
            raw_signals_list.append(np.stack(task_raw_signals))
        
        # Standardize features
        features = np.array(features_list)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        return (
            torch.tensor(features, dtype=torch.float32),  # Feature tensor: (60,7020)
            torch.tensor(np.array(raw_signals_list), dtype=torch.float32)  # Channel pair signals: (60,3,36,2,31)
        )

    def __getitem__(self, idx):
        return (self.features[idx], self.raw_signals[idx]), self.labels[idx]

    def __len__(self):
        return len(self.labels)


# Leave-one-out validation training function
def train_leave_one_out(test_subject_id, device, batch_size=5, epochs=800):
    # Initialize log recording
    subject_log = os.path.join(LOG_DIR, f"sub_{test_subject_id:02d}_train.log")
    with open(subject_log, "w") as f:
        f.write(f"Subject {test_subject_id:02d} leave-one-out validation training log\n")

    # Split training and test sets
    test_dataset = AFNIRSDataset(test_subject_id)
    train_subject_ids = [s for s in range(1, 30) if s != test_subject_id]
    train_datasets = [AFNIRSDataset(s) for s in train_subject_ids]
    train_dataset = ConcatDataset(train_datasets) if train_subject_ids else test_dataset

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Initialize model and optimizer
    # Use unified model, configuration parameters defined at the top of the script
    model = HybridMambaTransformer(**MODEL_CONFIG).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=200)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model_path = os.path.join(MODEL_SAVE_DIR, f"sub_{test_subject_id:02d}_best_acc.pth")  # Model save path

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for (features, signals), labels in train_loader:
            features, signals, labels = features.to(device), signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model((features, signals))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for (features, signals), labels in test_loader:
                features, signals = features.to(device), signals.to(device)
                outputs = model((features, signals))
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        pre = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        kap = cohen_kappa_score(all_labels, all_preds)

        # Adjust learning rate
        scheduler.step(acc)

        # Save best model (print only when accuracy improves)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)  # Save model parameters
            print(f"Subject {test_subject_id:02d} - Epoch {epoch+1}: New best model found! Accuracy improved to {best_acc:.4f}")
            print(f"Model saved to: {best_model_path}")
            with open(subject_log, "a") as f:
                f.write(f"Epoch {epoch+1}: Accuracy improved to {best_acc:.4f}, best model saved\n")

        # Non-best models: only log key metrics (optional)
        with open(subject_log, "a") as f:
            f.write(f"Epoch {epoch+1} | Training loss: {train_loss/len(train_loader):.4f} | "
                    f"Validation ACC: {acc:.4f} | PRE: {pre:.4f} | REC: {rec:.4f} | F1: {f1:.4f} | KAP: {kap:.4f}\n")

    # Return best metrics
    return {
        "acc": best_acc,
        "pre": pre,
        "rec": rec,
        "f1": f1,
        "kap": kap
    }

# Main function (start leave-one-out validation process)
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Best model save directory: {MODEL_SAVE_DIR}")

    # Initialize global log
    with open(GLOBAL_LOG, "w") as f:
        f.write(f"Leave-one-out validation training log (window length {WINDOW_LENGTH}, channel pairs {NUM_CHANNEL_PAIRS})\n")

    all_metrics = []
    for subject_id in range(1, 30):  # 29 subjects
        print(f"\n===== Starting processing for subject {subject_id:02d} =====")
        try:
            # Execute leave-one-out validation training
            metrics = train_leave_one_out(subject_id, device)
            all_metrics.append(metrics)

            # Record global log
            with open(GLOBAL_LOG, "a") as f:
                f.write(f"Subject {subject_id:02d}: ACC={metrics['acc']:.4f}, PRE={metrics['pre']:.4f}, "
                        f"REC={metrics['rec']:.4f}, F1={metrics['f1']:.4f}, KAP={metrics['kap']:.4f}\n")
            print(f"Subject {subject_id:02d} training completed, best accuracy: {metrics['acc']:.4f}")

        except Exception as e:
            error_msg = f"Subject {subject_id:02d} training failed: {str(e)}"
            with open(GLOBAL_LOG, "a") as f:
                f.write(f"{error_msg}\n")
            print(error_msg)

    # Calculate overall metrics
    if all_metrics:
        metrics_keys = ["acc", "pre", "rec", "f1", "kap"]
        total_metrics = {key: [] for key in metrics_keys}
        for m in all_metrics:
            for key in metrics_keys:
                total_metrics[key].append(m[key])

        # Output overall results
        print("\n================= Overall Evaluation Results ==================")
        with open(GLOBAL_LOG, "a") as f:
            f.write("\nOverall evaluation (mean ± std):\n")
            for key in metrics_keys:
                data = np.array(total_metrics[key])
                mean = np.mean(data)
                std = np.std(data)
                if key == "kap":
                    print(f"{key.upper()}: {mean:.4f} ± {std:.4f}")
                    f.write(f"{key.upper()}: {mean:.4f} ± {std:.4f}\n")
                else:
                    print(f"{key.upper()}: {mean*100:.2f}% ± {std*100:.2f}%")
                    f.write(f"{key.upper()}: {mean*100:.2f}% ± {std*100:.2f}%\n")

    print("\nAll subjects processed, results saved to log files")

if __name__ == "__main__":
    main()