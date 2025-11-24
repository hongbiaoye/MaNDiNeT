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
DATA_ROOT = r'DATA\WGXLS'  # Data root path
MODEL_SAVE_DIR = r'save_model\Wg_models'  # Model save directory (with sub_ prefix)
NUM_CLASSES = 2                                   # Binary classification
TIME_WINDOWS = [7, 8, 9]                          # Time windows (1-based index)
WINDOW_LENGTH = 10                                # Window length
NUM_CHANNEL_PAIRS = 36                            # Number of channel pairs
BATCH_SIZE = 5                                    # Test batch size

# Feature calculation functions (must match training script)
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
    channel_data = window_data.reshape(72, WINDOW_LENGTH)
    lyap_features = [calculate_lyapunov(channel_data[ch]) for ch in range(72)]
    plv_features = [calculate_plv(channel_data[ch], channel_data[ch + 36]) for ch in range(36)]
    return np.concatenate([window_data.flatten(), lyap_features, plv_features])

# Dataset class (must match training script)
class AFNIRSDataset(Dataset):
    def __init__(self, subject_id):
        super().__init__()
        self.subject_id = subject_id
        self.subject_dir = os.path.join(DATA_ROOT, f"VP{subject_id:03d}-NIRS")
        self.data_path = os.path.join(self.subject_dir, "signal_data.xlsx")
        self.label_path = os.path.join(self.subject_dir, "labels.xlsx")
        self.labels = self._load_labels()
        self.features, self.raw_signals = self._process_data()

    def _load_labels(self):
        label_df = pd.read_excel(self.label_path, header=0)
        label_data = label_df.iloc[:, 0]
        if len(label_data) != 60:
            raise ValueError(f"Subject VP{self.subject_id:03d} has incorrect number of labels")
        numeric_labels = label_data.astype(int).tolist()
        for label in numeric_labels:
            if label not in {0, 1}:
                raise ValueError(f"Subject VP{self.subject_id:03d} contains invalid label {label}")
        return numeric_labels

    def _process_data(self):
        xls = pd.ExcelFile(self.data_path)
        task_sheets = [s for s in xls.sheet_names if s.startswith('Task_')]
        task_sheets.sort(key=lambda x: int(x.split('_')[1]))
        if len(task_sheets) != 60:
            raise ValueError(f"Subject VP{self.subject_id:03d} signal data missing")
        
        features_list = []
        raw_signals_list = []
        for sheet_name in task_sheets:
            data = xls.parse(sheet_name).values[20:20+10*WINDOW_LENGTH]
            windows = [data[i*WINDOW_LENGTH:(i+1)*WINDOW_LENGTH] for i in range(10)]
            selected_windows = [windows[w-1] for w in TIME_WINDOWS]
            
            task_features = []
            task_raw_signals = []
            for win in selected_windows:
                channel_data = win.T
                pair_data = np.stack([channel_data[:36], channel_data[36:]], axis=1)
                task_raw_signals.append(pair_data)
                task_features.append(extract_window_features(win.flatten()))
            
            features_list.append(np.concatenate(task_features))
            raw_signals_list.append(np.stack(task_raw_signals))
        
        features = np.array(features_list)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)        
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(np.array(raw_signals_list), dtype=torch.float32)
        )

    def __getitem__(self, idx):
        return (self.features[idx], self.raw_signals[idx]), self.labels[idx]

    def __len__(self):
        return len(self.labels)

# Model architecture (must match training script)
class PairwiseMamba(nn.Module):
    def __init__(self, d_model=16, d_state=8, d_conv=2):
        super().__init__()
        self.d_model = d_model
        self.num_pairs = NUM_CHANNEL_PAIRS
        self.mamba = Mamba(d_model=2, d_state=d_state, d_conv=d_conv, expand=2)
        self.channel_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

    def forward(self, raw_signals):
        B, num_windows, num_pairs, ch_per_pair, T = raw_signals.shape
        mamba_outputs = []
        for win_idx in range(num_windows):
            win_signals = raw_signals[:, win_idx].permute(0, 1, 3, 2)
            flat_input = win_signals.reshape(-1, T, ch_per_pair)
            mamba_out = self.mamba(flat_input)
            win_feat = mamba_out.mean(dim=1)
            projected = self.channel_proj(win_feat).view(B, num_pairs, self.d_model)
            mamba_outputs.append(projected)
        return torch.stack(mamba_outputs, dim=1).mean(dim=2)

class HybridMambaTransformer(nn.Module):
    def __init__(self, num_windows=3, d_model=16, nhead=4):
        super().__init__()
        self.num_windows = num_windows
        self.d_model = d_model
        self.mamba = PairwiseMamba(d_model=d_model)
        self.window_proj = nn.Sequential(
            nn.Linear(828, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2*d_model,
                nhead=nhead,
                dim_feedforward=512,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=2
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model*2 * num_windows, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        features, raw_signals = x
        B = features.shape[0]
        window_features = features.view(B, self.num_windows, 828)
        window_proj = self.window_proj(window_features)
        mamba_proj = self.mamba(raw_signals)
        fused = torch.cat([window_proj, mamba_proj], dim=-1)
        trans_out = self.transformer(fused)
        flattened = trans_out.reshape(B, -1)
        return self.classifier(flattened)

# Single subject test function (corrected filename prefix to sub_)
def test_single_subject(subject_id, device):
    test_dataset = AFNIRSDataset(subject_id)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Match training script filename format: sub_VP001_best_model.pth
    model_filename = f"sub_VP{subject_id:03d}_best_acc.pth"
    model_path = os.path.join(MODEL_SAVE_DIR, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = HybridMambaTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for (features, signals), labels in test_loader:
            features, signals = features.to(device), signals.to(device)
            outputs = model((features, signals))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    pre = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    kap = cohen_kappa_score(all_labels, all_preds)
    
    print(f"Subject VP{subject_id:03d} test results | "
          f"ACC: {acc:.4f} | "
          f"Precision: {pre:.4f} | "
          f"Recall: {rec:.4f} | "
          f"F1: {f1:.4f} | "
          f"Kappa: {kap:.4f}")
    
    return {
        "subject_id": subject_id,
        "acc": acc,
        "pre": pre,
        "rec": rec,
        "f1": f1,
        "kap": kap
    }

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model loading path: {MODEL_SAVE_DIR}")
    
    subject_ids = range(1, 27)  # Assuming 26 subjects
    all_results = []
    
    for subject_id in subject_ids:
        print(f"\n===== Testing subject VP{subject_id:03d} =====")
        try:
            result = test_single_subject(subject_id, device)
            all_results.append(result)
        except Exception as e:
            print(f"Subject VP{subject_id:03d} test failed: {str(e)}")
    
    if all_results:
        metrics = ["acc", "pre", "rec", "f1", "kap"]
        total_metrics = {m: [] for m in metrics}
        for res in all_results:
            for m in metrics:
                total_metrics[m].append(res[m])
        
        print("\n================= Overall Test Metrics ==================")
        for m in metrics:
            data = np.array(total_metrics[m])
            mean = np.mean(data)
            std = np.std(data)
            print(f"{m.upper()}: {mean*100:.2f}% ± {std*100:.2f}%")

if __name__ == "__main__":
    main()