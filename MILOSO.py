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

# Global configuration (new model save path)
DATA_ROOT = r'DATA\UFFT_data'
LOG_DIR = os.path.join(os.getcwd(), "logs")
MODEL_SAVE_DIR = os.path.join(LOG_DIR, "MIbest_models")  # New: model save directory
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)  # Auto-create model save directory
os.makedirs(os.path.join(LOG_DIR, "subject_metrics"), exist_ok=True)
GLOBAL_LOG = os.path.join(LOG_DIR, "training_log.txt")

# Experiment parameters (three-class classification adjustment)
NUM_CLASSES = 3  # Three-class classification
TIME_WINDOWS = [8, 9, 10]  # Window indices used (1-based)

# ========== Model Configuration Parameters ==========
MODEL_CONFIG = {
    'num_windows': len(TIME_WINDOWS),
    'd_model': 16,
    'nhead': 4,
    'window_feature_dim': 580,  # MI dataset: 40*13 + 40 + 20 = 580
    'num_pairs': 20,  # MI dataset: 20 channel pairs
    'aggregate_mamba_internal': True,  # MILOSO aggregates internally
    'use_permute': True,  # MILOSO uses permute
    'num_classes': NUM_CLASSES  # Three-class classification
}
# =====================================================
WINDOW_LENGTH = 13          # Each window has 13 time points
NUM_CHANNEL_PAIRS = 20      # First 20 + last 20 channels form 20 pairs

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
    channel_data = window_data.reshape(40, WINDOW_LENGTH)  # 40 channels × 13 time points
    lyap_features = [calculate_lyapunov(channel_data[ch]) for ch in range(40)]  # 40 Lyapunov exponents
    plv_features = [calculate_plv(channel_data[ch], channel_data[ch + 20]) for ch in range(20)]  # 20 pairs of PLV
    return np.concatenate([window_data.flatten(), lyap_features, plv_features])  # 580 dimensions per window

# Custom dataset class
class AFNIRSDataset(Dataset):
    def __init__(self, subject_id, time_windows=TIME_WINDOWS):
        super().__init__()
        self.subject_id = subject_id
        self.time_windows = time_windows
        self.data_path = os.path.join(DATA_ROOT, str(subject_id), f"{subject_id}.xls")
        self.label_path = os.path.join(DATA_ROOT, str(subject_id), f"{subject_id}_desc.xls")
        self.labels = self._load_labels()
        self.features, self.raw_signals = self._process_data()

    def _load_labels(self):
        label_df = pd.read_excel(self.label_path, header=None)
        labels = label_df.values.flatten().tolist()
        return [l - 1 for l in labels]  # Convert to 0-based indexing

    def _process_data(self):
        xls = pd.ExcelFile(self.data_path)
        features_list = []
        raw_signals_list = []
        
        for sheet_idx in range(len(xls.sheet_names)):
            data = xls.parse(sheet_idx).values[30:30+10*WINDOW_LENGTH]
            windows = [data[i*WINDOW_LENGTH:(i+1)*WINDOW_LENGTH] for i in range(10)]
            selected_windows = [windows[w-1] for w in self.time_windows]
            
            task_features = []
            task_raw_signals = []
            
            for win in selected_windows:
                channel_data = win.T  # (40,13)
                pair_data = np.stack([channel_data[:20], channel_data[20:]], axis=1)  # (20,2,13)
                task_raw_signals.append(pair_data)
                task_features.append(extract_window_features(win.flatten()))  # 580 dimensions
            
            features_list.append(np.concatenate(task_features))  # 3×580=1740 dimensions
            raw_signals_list.append(np.stack(task_raw_signals))  # (3,20,2,13)
        
        # Standardize features
        features = np.array(features_list)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        return (
            torch.tensor(features, dtype=torch.float32),  # (N,1740)
            torch.tensor(np.array(raw_signals_list), dtype=torch.float32)  # (N,3,20,2,13)
        )

    def __getitem__(self, idx):
        return (self.features[idx], self.raw_signals[idx]), self.labels[idx]

    def __len__(self):
        return len(self.labels)

# Mamba module specialized for channel pairs

# Training function (new model save logic)
def train_leave_one_subject_out(test_subject, device, batch_size=5, epochs=800):
    # Define model save path (named by subject ID)
    best_model_path = os.path.join(MODEL_SAVE_DIR, f"sub_{test_subject:02d}_best_acc.pth")
    
    # Prepare data
    train_subjects = [sub for sub in range(1, 31) if sub != test_subject]
    train_datasets = [AFNIRSDataset(sub) for sub in train_subjects]
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = AFNIRSDataset(test_subject)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=False)

    # Initialize model
    # Use unified model, configuration parameters defined at the top of the script
    model = HybridMambaTransformer(**MODEL_CONFIG).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=200)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_metrics = {}

    for epoch in range(epochs):
        # Training phase
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

        # Validation phase
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

        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        scheduler.step(acc)

        # Update best metrics and save model
        if acc > best_acc:
            best_acc = acc
            best_metrics = {
                "acc": acc,
                "pre": precision_score(all_labels, all_preds, average='macro', zero_division=0),
                "rec": recall_score(all_labels, all_preds, average='macro', zero_division=0),
                "f1": f1_score(all_labels, all_preds, average='macro', zero_division=0),
                "kap": cohen_kappa_score(all_labels, all_preds)
            }
            # Save current best model
            torch.save(model.state_dict(), best_model_path)
            print(f"Subject {test_subject:02d} Epoch {epoch+1}: Accuracy improved to {acc:.4f}, saved to {best_model_path}")

    # Record metrics
    print(f"Test subject {test_subject} Best Metrics | "
          f"ACC: {best_metrics['acc']:.4f} | "
          f"Precision: {best_metrics['pre']:.4f} | "
          f"Recall: {best_metrics['rec']:.4f} | "
          f"F1: {best_metrics['f1']:.4f} | "
          f"Kappa: {best_metrics['kap']:.4f}")

    metrics_path = os.path.join(LOG_DIR, "subject_metrics", f"sub_{test_subject:02d}_loo_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Leave-one-out validation subject {test_subject} Best Metrics:\n")
        for k, v in best_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    with open(GLOBAL_LOG, "a") as f_global:
        f_global.write(f"Test subject {test_subject} ACC: {best_metrics['acc']:.4f}\n")

    return best_metrics

# Main function (new model save path prompt)
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Best models will be saved to: {MODEL_SAVE_DIR}")  # Prompt model save path
    
    with open(GLOBAL_LOG, "w") as f:
        f.write(f"Window length {WINDOW_LENGTH} - Leave-one-subject-out validation training log (20 pairs of cross-channel PLV)\n")

    all_subject_metrics = []
    for test_subject in range(1, 31):
        print(f"\n===== Starting leave-one-out validation: Test subject {test_subject} =====")
        try:
            subject_metrics = train_leave_one_subject_out(test_subject, device)
            all_subject_metrics.append(subject_metrics)
            print(f"Subject {test_subject} validation completed")
        except Exception as e:
            with open(GLOBAL_LOG, "a") as f_global:
                f_global.write(f"Subject {test_subject} validation error: {str(e)}\n")
            print(f"Subject {test_subject} validation error: {str(e)}")

    # Calculate overall metrics
    metrics = ["acc", "pre", "rec", "f1", "kap"]
    metric_data = {m: [] for m in metrics}
    for metric in all_subject_metrics:
        for m in metrics:
            metric_data[m].append(metric[m])

    print("\n================= Overall Evaluation Metrics (Mean ± Std) ==================")
    with open(GLOBAL_LOG, "a") as f_global:
        f_global.write("\nLeave-one-out validation overall evaluation metrics:\n")
        for m in metrics:
            data = np.array(metric_data[m])
            mean, std = np.mean(data), np.std(data)
            if m == "kap":
                print(f"Kappa coefficient: {mean:.4f} ± {std:.4f}")
                f_global.write(f"Kappa: {mean:.4f} ± {std:.4f}\n")
            else:
                print(f"{m.upper()}: {mean*100:.2f}% ± {std*100:.2f}%")
                f_global.write(f"{m.upper()}: {mean*100:.2f}% ± {std*100:.2f}%\n")

    print("\nValidation completed, all logs saved to", LOG_DIR)

if __name__ == "__main__":
    main()