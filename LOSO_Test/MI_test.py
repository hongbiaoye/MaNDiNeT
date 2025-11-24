import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import os
from nolds import lyap_r
from scipy.signal import hilbert
from mamba_ssm import Mamba

# Global configuration (must match training script exactly)
DATA_ROOT = r'DATA\UFFT_data'  # Data root path
MODEL_SAVE_DIR = r'save_model\MI_models'  # Model save directory (user-specified)
NUM_CLASSES = 3  # Three-class classification (must match training)
TIME_WINDOWS = [8, 9, 10]  # Time windows (1-based index)
WINDOW_LENGTH = 13  # Window length (13 time points)
NUM_CHANNEL_PAIRS = 20  # Number of channel pairs (20 pairs)
BATCH_SIZE = 30  # Test batch size (recommended to match training)

# Feature calculation functions (must match training script exactly)
def calculate_lyapunov(signal):
    """Calculate Lyapunov exponent (robust handling of exceptions)"""
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
    channel_data = window_data.reshape(40, WINDOW_LENGTH)  # 40 channels × 13 time points
    lyap_features = [calculate_lyapunov(channel_data[ch]) for ch in range(40)]  # 40 Lyapunov exponents
    plv_features = [calculate_plv(channel_data[ch], channel_data[ch + 20]) for ch in range(20)]  # 20 PLV pairs
    return np.concatenate([window_data.flatten(), lyap_features, plv_features])  # 580-dimensional features

# Dataset class (must match training script exactly)
class AFNIRSDataset(Dataset):
    def __init__(self, subject_id):
        super().__init__()
        self.subject_id = subject_id
        self.data_path = os.path.join(DATA_ROOT, str(subject_id), f"{subject_id}.xls")
        self.label_path = os.path.join(DATA_ROOT, str(subject_id), f"{subject_id}_desc.xls")
        self.labels = self._load_labels()
        self.features, self.raw_signals = self._process_data()

    def _load_labels(self):
        """Load labels and convert to 0-based index"""
        label_df = pd.read_excel(self.label_path, header=None)
        labels = label_df.values.flatten().tolist()
        return [l - 1 for l in labels]  # Convert to 0/1/2 classes

    def _process_data(self):
        """Process raw data to generate features and channel pair signals"""
        xls = pd.ExcelFile(self.data_path)
        features_list = []
        raw_signals_list = []
        
        for sheet_idx in range(len(xls.sheet_names)):
            data = xls.parse(sheet_idx).values[30:30+10*WINDOW_LENGTH]
            windows = [data[i*WINDOW_LENGTH:(i+1)*WINDOW_LENGTH] for i in range(10)]
            selected_windows = [windows[w-1] for w in TIME_WINDOWS]  # Select 3 windows
            
            task_features = []
            task_raw_signals = []
            
            for win in selected_windows:
                channel_data = win.T  # (40,13)
                pair_data = np.stack([channel_data[:20], channel_data[20:]], axis=1)  # (20,2,13)
                task_raw_signals.append(pair_data)
                task_features.append(extract_window_features(win.flatten()))  # 580-dimensional features
            
            features_list.append(np.concatenate(task_features))  # 3×580=1740 dimensions
            raw_signals_list.append(np.stack(task_raw_signals))  # (3,20,2,13)
        
        # Normalize features (consistent with training logic)
        features = np.array(features_list)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)  # Independent normalization per subject (or use training scaler)
        
        return (
            torch.tensor(features, dtype=torch.float32),  # Feature tensor: (N,1740)
            torch.tensor(np.array(raw_signals_list), dtype=torch.float32)  # Channel pair signals: (N,3,20,2,13)
        )

    def __getitem__(self, idx):
        return (self.features[idx], self.raw_signals[idx]), self.labels[idx]

    def __len__(self):
        return len(self.labels)

# Model architecture (must match training script exactly)
class PairwiseMamba(nn.Module):
    def __init__(self, d_model=16, d_state=8, d_conv=2):
        super().__init__()
        self.d_model = d_model
        self.num_pairs = NUM_CHANNEL_PAIRS  # 20 channel pairs
        
        self.mamba = Mamba(
            d_model=2,  # Input dimension per pair: 2
            d_state=d_state,
            d_conv=d_conv,
            expand=2
        )
        
        self.channel_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, raw_signals):
        B, num_windows, num_pairs, ch_per_pair, T = raw_signals.shape
        mamba_outputs = []
        for win_idx in range(num_windows):
            win_signals = raw_signals[:, win_idx].permute(0, 1, 3, 2).reshape(-1, T, ch_per_pair)
            mamba_out = self.mamba(win_signals)
            win_feat = mamba_out.mean(dim=1)
            projected = self.channel_proj(win_feat).view(B, num_pairs, self.d_model)
            mamba_outputs.append(projected)
        return torch.stack(mamba_outputs, dim=1).mean(dim=2)  # (B,3,d_model)

class HybridMambaTransformer(nn.Module):
    def __init__(self, num_windows=len(TIME_WINDOWS), d_model=16, nhead=4):
        super().__init__()
        self.num_windows = num_windows  # 3 windows
        self.d_model = d_model
        
        self.mamba = PairwiseMamba(d_model=d_model)
        self.window_proj = nn.Sequential(
            nn.Linear(580, d_model),  # Single window feature dimension: 580
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2*d_model,  # Concatenated dimension: 2×d_model
                nhead=nhead,
                dim_feedforward=512,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=2
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(num_windows * 2 * d_model, 256),  # Input dimension: 3×2×d_model
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES)  # Three-class output
        )

    def forward(self, x):
        features, raw_signals = x
        B = features.shape[0]
        window_features = features.view(B, self.num_windows, 580)  # (B,3,580)
        window_proj = self.window_proj(window_features)  # (B,3,d_model)
        mamba_proj = self.mamba(raw_signals)  # (B,3,d_model)
        fused = torch.cat([window_proj, mamba_proj], dim=-1)  # (B,3,2d_model)
        trans_out = self.transformer(fused)  # (B,3,2d_model)
        flattened = trans_out.reshape(B, -1)  # (B, 3×2d_model)
        return self.classifier(flattened)

# Single subject test function
def test_single_subject(subject_id, device):
    # 1. Load test dataset
    test_dataset = AFNIRSDataset(subject_id)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load pre-trained model
    model_path = os.path.join(MODEL_SAVE_DIR, f"sub_{subject_id:02d}_best_acc.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = HybridMambaTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. Inference and collect results
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for (features, signals), labels in test_loader:
            features, signals = features.to(device), signals.to(device)
            outputs = model((features, signals))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels)
    
    # 4. Calculate evaluation metrics (three-class, macro average)
    acc = accuracy_score(all_labels, all_preds)
    pre = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    kap = cohen_kappa_score(all_labels, all_preds, labels=[0, 1, 2])  # Explicitly specify class order
    
    print(f"Subject {subject_id:02d} test results | "
          f"ACC: {acc:.4f} | "
          f"Precision(macro): {pre:.4f} | "
          f"Recall(macro): {rec:.4f} | "
          f"F1(macro): {f1:.4f} | "
          f"Kappa: {kap:.4f}")
    
    return {
        "subject_id": subject_id,
        "acc": acc,
        "pre": pre,
        "rec": rec,
        "f1": f1,
        "kap": kap
    }

# Main function (batch test all subjects)
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Starting model testing, model path: {MODEL_SAVE_DIR}")
    
    # Subject ID range (must match training, assuming subjects 1-30)
    subject_ids = range(1, 31)
    all_results = []
    
    for subject_id in subject_ids:
        print(f"\n===== Testing subject {subject_id:02d} =====")
        try:
            result = test_single_subject(subject_id, device)
            all_results.append(result)
        except Exception as e:
            print(f"Subject {subject_id:02d} test failed: {str(e)}")
    
    # Aggregate overall metrics (optional)
    if all_results:
        metrics = ["acc", "pre", "rec", "f1", "kap"]
        total_metrics = {m: [] for m in metrics}
        for res in all_results:
            for m in metrics:
                total_metrics[m].append(res[m])
        
        print("\n================= Overall Test Metrics (Mean ± Std) ==================")
        for m in metrics:
            data = np.array(total_metrics[m])
            mean = np.mean(data)
            std = np.std(data)
            if m == "kap":
                print(f"{m.upper()}: {mean:.4f} ± {std:.4f}")
            else:
                print(f"{m.upper()}: {mean*100:.2f}% ± {std*100:.2f}%")

if __name__ == "__main__":
    main()