# MaNDi-Net: Model Architecture Reference

This document provides detailed technical reference for the **MaNDi-Net** (Mamba-Transformer Deep Integration Network) model architecture, including component descriptions, parameter configurations, and implementation details.

## Model Architecture Overview

### HybridMambaTransformer

MaNDi-Net uses a hybrid Mamba-Transformer architecture (`HybridMambaTransformer`) for fNIRS signal classification. The model combines Mamba's state space model (SSM) with Transformer's attention mechanism, effectively processing temporal signals and inter-window relationships.

## Model Components

### 1. PairwiseMamba Module

The PairwiseMamba module is specifically designed to process channel pair signals.

**Key Features**:
- Uses parameter-shared Mamba instances to process 2D inputs for each channel pair
- Supports internal or external aggregation modes
- Configurable permute option to adjust dimension order

**Architecture Details**:
- Processes input of shape `[B, num_windows, num_pairs, ch_per_pair, T]`
- Each channel pair is processed by a shared Mamba instance
- Output shape depends on `aggregate_mamba_internal` parameter:
  - `True`: `[B, num_windows, d_model]` (aggregated internally)
  - `False`: `[B, num_windows, num_pairs, d_model]` (aggregated externally)

**Permute Option**:
- When `use_permute=True`: Adjusts dimension order from `[B, num_pairs, ch, T]` to `[B, num_pairs, T, ch]`
- When `use_permute=False`: Maintains original dimension order

### 2. Window Feature Projection Layer

Projects high-dimensional window features to model dimension (d_model).

**Components**:
- Linear projection layer
- ReLU activation
- Dropout regularization

**Function**: Reduces feature dimensionality from `window_feature_dim` to `d_model` for efficient processing.

### 3. Transformer Encoder

Processes temporal relationships between windows.

**Architecture**:
- Multi-head attention mechanism
- 2-layer encoder structure
- Positional encoding for window sequences

**Function**: Captures long-range dependencies and temporal patterns across windows.

### 4. Classifier

Fully connected layer structure for final classification.

**Features**:
- Configurable number of classes (2 or 3)
- Outputs class probabilities

## Model Parameter Configurations

Different datasets use different model configurations:

### MA Dataset Configuration
- `window_feature_dim`: 2340 (72*31 + 72 + 36)
- `num_pairs`: 36
- `num_classes`: 2
- `aggregate_mamba_internal`: True
- `use_permute`: True
- `d_model`: 128 (default)
- `nhead`: 4 (default)
- `num_layers`: 2 (default)

### UFFT Dataset Configuration
- `window_feature_dim`: 580 (40*13 + 40 + 20)
- `num_pairs`: 20
- `num_classes`: 3
- `aggregate_mamba_internal`: True
- `use_permute`: True
- `d_model`: 128 (default)
- `nhead`: 4 (default)
- `num_layers`: 2 (default)

### WG Dataset Configuration
- `window_feature_dim`: 828 (72*10 + 72 + 36)
- `num_pairs`: 36
- `num_classes`: 2
- `aggregate_mamba_internal`: True
- `use_permute`: True
- `d_model`: 128 (default)
- `nhead`: 4 (default)
- `num_layers`: 2 (default)

## Parameter Explanations

### aggregate_mamba_internal
- **True**: Mamba module aggregates channel pair features internally, directly outputs `[B, num_windows, d_model]`
- **False**: Mamba module does not aggregate, outputs `[B, num_windows, num_pairs, d_model]`, aggregation happens in main model forward

### use_permute
- **True**: Use permute to adjust dimension order `[B, num_pairs, ch, T] → [B, num_pairs, T, ch]`
- **False**: Keep original dimension order, process directly

### window_feature_dim
Total feature dimension extracted from each time window, calculated as:
- `num_channels * time_points + num_channels + num_pairs`
- Includes temporal features, channel-level features, and pair-level features

### num_pairs
Number of channel pairs used for pairwise feature extraction.

### d_model
Model dimension for internal representations. Default: 128

### nhead
Number of attention heads in Transformer encoder. Default: 4

### num_layers
Number of Transformer encoder layers. Default: 2

## Model Forward Pass

1. **Input**: Raw signals `[B, num_windows, num_pairs, ch_per_pair, T]`
2. **PairwiseMamba**: Process channel pairs → `[B, num_windows, d_model]` or `[B, num_windows, num_pairs, d_model]`
3. **Feature Projection**: Project window features → `[B, num_windows, d_model]`
4. **Transformer Encoder**: Process window sequences → `[B, num_windows, d_model]`
5. **Pooling**: Aggregate window features → `[B, d_model]`
6. **Classifier**: Final classification → `[B, num_classes]`

## Implementation Details

### Model File
- Location: `model.py`
- Main class: `HybridMambaTransformer`
- Supporting class: `PairwiseMamba`

### Key Design Decisions
1. **Parameter Sharing**: Mamba instances share parameters across channel pairs for efficiency
2. **Hybrid Architecture**: Combines Mamba's efficiency with Transformer's expressiveness
3. **Flexible Aggregation**: Supports both internal and external aggregation modes
4. **Dimension Flexibility**: Permute option allows adaptation to different data formats

## Performance Considerations

1. **Memory Usage**: Internal aggregation reduces memory footprint
2. **Computational Efficiency**: Parameter sharing reduces model size
3. **Scalability**: Architecture supports varying numbers of channel pairs and windows

## References

For MaNDi-Net training and usage instructions, see:
- `docs/Model_Training_and_Environment_Setup.md` - Quick start guide
- Training scripts: `*_Kfold.py` and `*LOSO.py`

