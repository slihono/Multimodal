"""
Phase 4 - Multimodal Data Fusion: Modality Encoders
======================================================
One encoder per data source, each projecting into its own embedding
dimension per the project spec:
  - Vision Encoder            -> 1024-d
  - Physiological Encoder     -> 256-d
  - Environmental Encoder     -> 32-d
  - OBD-II Encoder            -> 16-d

These feed the cross-attention Fusion Transformer in fusion_transformer.py.

Requires: torch
    pip install torch --break-system-packages
"""

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """Consumes a rolling window of Phase 1/3 vision features (EAR,
    PERCLOS, yawn score, head-nod angle, distraction one-hot, emotion
    one-hot, rPPG HR) over T timesteps and produces a single 1024-d
    embedding via a small temporal transformer.

    In a full deployment this would consume raw CNN feature maps from
    the vision backbone directly; here we consume the already-extracted
    scalar features (VisionFrame) to keep the fusion layer decoupled
    from whichever vision backbone Phase 1/3 uses.
    """
    FEATURE_DIM = 16  # ear, perclos, yawn, head_nod, 5x distraction 1-hot, 6x emotion 1-hot, rppg_hr

    def __init__(self, embed_dim: int = 1024, num_layers: int = 2, nhead: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(self.FEATURE_DIM, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 2, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, FEATURE_DIM) -> (batch, embed_dim)"""
        h = self.input_proj(x)
        h = self.temporal_encoder(h)
        h = h.transpose(1, 2)          # (batch, embed_dim, T)
        pooled = self.pool(h).squeeze(-1)
        return pooled


class PhysioEncoder(nn.Module):
    """Consumes a window of Phase 2 physio samples (HR, SpO2, GSR,
    accel xyz, gyro xyz = 8 features) -> 256-d embedding. A 1D-CNN is
    used here since biosignals benefit from local temporal convolution
    (capturing pulse waveform shape) more than pure attention at this
    embedding size."""
    FEATURE_DIM = 9

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(self.FEATURE_DIM, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, FEATURE_DIM) -> (batch, embed_dim)"""
        h = x.transpose(1, 2)  # (batch, FEATURE_DIM, T)
        h = self.conv(h).squeeze(-1)
        return self.proj(h)


class EnvironmentEncoder(nn.Module):
    """Cabin environment: temp, humidity, CO2, VOC, alcohol, lux (6
    features) -> 32-d embedding. Simple MLP -- these signals are slow
    moving and don't need temporal modeling at the fusion stage."""
    FEATURE_DIM = 6

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, FEATURE_DIM) -> (batch, embed_dim)"""
        return self.mlp(x)


class OBDEncoder(nn.Module):
    """Vehicle bus: speed, rpm, throttle, brake, steering angle (5
    features) -> 16-d embedding."""
    FEATURE_DIM = 5

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, FEATURE_DIM) -> (batch, embed_dim)"""
        return self.mlp(x)
