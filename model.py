"""
Unified model definition file
Shared HybridMambaTransformer model for all scripts
"""
import torch
import torch.nn as nn
from mamba_ssm import Mamba


class PairwiseMamba(nn.Module):
    """
    Mamba module specialized for processing channel pair signals
    
    Args:
        d_model: Model dimension
        d_state: Mamba state dimension
        d_conv: Mamba convolution kernel dimension
        num_pairs: Number of channel pairs
        aggregate_internal: Whether to aggregate channel pairs internally (True: returns [B,3,d_model], False: returns [B,3,num_pairs,d_model])
    """
    def __init__(self, d_model=16, d_state=8, d_conv=2, num_pairs=36, aggregate_internal=True, use_permute=True):
        super().__init__()
        self.d_model = d_model
        self.num_pairs = num_pairs
        self.aggregate_internal = aggregate_internal
        self.use_permute = use_permute
        
        # Parameter-shared Mamba instance (processes 2D input for each channel pair)
        self.mamba = Mamba(
            d_model=2,  # Input dimension for each channel pair is 2 (ch1, ch2)
            d_state=d_state,
            d_conv=d_conv,
            expand=2
        )
        
        # Channel pair feature aggregation: 2D output → d_model dimension
        self.channel_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, raw_signals):
        """
        Args:
            raw_signals: [B, num_windows, num_pairs, 2, T] or [B, num_windows, num_pairs, ch_per_pair, T]
        Returns:
            If aggregate_internal=True: [B, num_windows, d_model]
            If aggregate_internal=False: [B, num_windows, num_pairs, d_model]
        """
        B, num_windows, num_pairs, ch_per_pair, T = raw_signals.shape
        
        mamba_outputs = []
        for win_idx in range(num_windows):
            # Extract channel pair signals for current window
            win_signals = raw_signals[:, win_idx]  # [B, num_pairs, ch_per_pair, T]
            
            # Decide whether to use permute to adjust dimension order based on configuration
            if self.use_permute:
                # Adjust to [B, num_pairs, T, ch_per_pair] format
                win_signals = win_signals.permute(0, 1, 3, 2)
            
            # Flatten for Mamba input: [B*num_pairs, T, ch_per_pair]
            flat_input = win_signals.reshape(-1, T, ch_per_pair)
            
            # Mamba processes temporal features
            mamba_out = self.mamba(flat_input)  # [B*num_pairs, T, 2]
            
            # Average aggregation along time dimension
            win_feat = mamba_out.mean(dim=1)  # [B*num_pairs, 2]
            
            # Project to d_model dimension and restore shape
            projected = self.channel_proj(win_feat).view(B, num_pairs, self.d_model)  # [B, num_pairs, d_model]
            mamba_outputs.append(projected)
        
        # Merge features from all windows
        stacked = torch.stack(mamba_outputs, dim=1)  # [B, num_windows, num_pairs, d_model]
        
        # Decide whether to aggregate internally based on configuration
        if self.aggregate_internal:
            return stacked.mean(dim=2)  # [B, num_windows, d_model]
        else:
            return stacked  # [B, num_windows, num_pairs, d_model]


class HybridMambaTransformer(nn.Module):
    """
    Hybrid Mamba-Transformer model
    
    Args:
        num_windows: Number of time windows
        d_model: Model dimension
        nhead: Number of Transformer attention heads
        window_feature_dim: Window feature dimension (varies by dataset: 2340, 580, 828, etc.)
        num_pairs: Number of channel pairs (varies by dataset: 36, 20, etc.)
        aggregate_mamba_internal: Whether PairwiseMamba aggregates channel pairs internally
        use_permute: Whether PairwiseMamba uses permute to adjust dimension order
        num_classes: Number of classes (default 2, MI dataset uses 3)
    """
    def __init__(self, num_windows=3, d_model=16, nhead=4, window_feature_dim=2340, 
                 num_pairs=36, aggregate_mamba_internal=True, use_permute=True, num_classes=2):
        super().__init__()
        self.num_windows = num_windows
        self.d_model = d_model
        self.aggregate_mamba_internal = aggregate_mamba_internal
        
        # Mamba module specialized for channel pairs
        self.mamba = PairwiseMamba(
            d_model=d_model,
            num_pairs=num_pairs,
            aggregate_internal=aggregate_mamba_internal,
            use_permute=use_permute
        )
        
        # Window feature projection
        self.window_proj = nn.Sequential(
            nn.Linear(window_feature_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Transformer processes inter-window relationships
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2*d_model,  # Concatenated dimension (window features + Mamba features)
                nhead=nhead,
                dim_feedforward=512,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=2
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_windows * 2 * d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)  # Configurable number of classes
        )

    def forward(self, x):
        """
        Args:
            x: (features, raw_signals) tuple
                - features: [B, total_features] window features
                - raw_signals: [B, num_windows, num_pairs, 2, T] raw signals
        Returns:
            [B, num_classes] classification output
        """
        features, raw_signals = x
        B = features.shape[0]
        
        # Window feature processing: [B, total_features] → [B, num_windows, window_feature_dim] → [B, num_windows, d_model]
        window_feature_dim = self.window_proj[0].in_features
        window_features = features.view(B, self.num_windows, window_feature_dim)
        window_proj = self.window_proj(window_features)
        
        # Mamba channel pair feature processing
        mamba_output = self.mamba(raw_signals)
        
        # If Mamba doesn't aggregate internally, need to aggregate in forward
        if not self.aggregate_mamba_internal:
            # mamba_output: [B, num_windows, num_pairs, d_model] → [B, num_windows, d_model]
            mamba_feat = mamba_output.mean(dim=2)
        else:
            # mamba_output: [B, num_windows, d_model]
            mamba_feat = mamba_output
        
        # Feature fusion: [B, num_windows, d_model] + [B, num_windows, d_model] → [B, num_windows, 2*d_model]
        fused_features = torch.cat([window_proj, mamba_feat], dim=-1)
        
        # Transformer processes inter-window relationships
        trans_out = self.transformer(fused_features)
        
        # Flatten and classify
        flattened = trans_out.reshape(B, -1)
        return self.classifier(flattened)

