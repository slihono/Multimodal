"""
Phase 4 - Multimodal Data Fusion: Cross-Attention Fusion Transformer
=======================================================================
Combines the four modality embeddings (Vision 1024-d, Physio 256-d, Env
32-d, OBD 16-d) via a cross-attention Transformer, then predicts the
four Phase 4 model outputs:
  - Drowsiness score        (0-1)
  - Health anomaly score    (0-1)
  - Driver distraction score(0-1)
  - Time-to-event prediction (seconds, regression, only meaningful
    when risk is elevated -- see the "claim discipline" note in the
    project overview: this is framed as an operational risk estimate,
    not a medical prediction).

Requires: torch
    pip install torch --break-system-packages
"""

import torch
import torch.nn as nn

from encoders import VisionEncoder, PhysioEncoder, EnvironmentEncoder, OBDEncoder

FUSION_DIM = 1024  # common space all modalities are projected into before cross-attention


class ModalityFusionTransformer(nn.Module):
    def __init__(self, fusion_dim: int = FUSION_DIM, num_fusion_layers: int = 4, nhead: int = 8):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=1024)
        self.physio_encoder = PhysioEncoder(embed_dim=256)
        self.env_encoder = EnvironmentEncoder(embed_dim=32)
        self.obd_encoder = OBDEncoder(embed_dim=16)

        # Project every modality embedding into the shared fusion_dim so
        # they can attend to each other as tokens in one sequence.
        self.vision_to_fusion = nn.Linear(1024, fusion_dim)
        self.physio_to_fusion = nn.Linear(256, fusion_dim)
        self.env_to_fusion = nn.Linear(32, fusion_dim)
        self.obd_to_fusion = nn.Linear(16, fusion_dim)

        # Learned modality-type embeddings (like BERT's segment embeddings)
        # so the model can tell which token came from which sensor.
        self.modality_embed = nn.Parameter(torch.randn(4, fusion_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=nhead, dim_feedforward=fusion_dim * 2,
            batch_first=True, dropout=0.1,
        )
        self.cross_attention_fusion = nn.TransformerEncoder(encoder_layer, num_layers=num_fusion_layers)

        # CLS-style pooling token that attends over all four modality tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)

        # Prediction heads
        self.drowsiness_head = nn.Sequential(nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.health_anomaly_head = nn.Sequential(nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.distraction_head = nn.Sequential(nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.time_to_event_head = nn.Sequential(nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, vision_x, physio_x, env_x, obd_x):
        """
        vision_x : (batch, T_vis, VisionEncoder.FEATURE_DIM)
        physio_x : (batch, T_phys, PhysioEncoder.FEATURE_DIM)
        env_x    : (batch, EnvironmentEncoder.FEATURE_DIM)
        obd_x    : (batch, OBDEncoder.FEATURE_DIM)
        """
        batch = vision_x.shape[0]

        v = self.vision_to_fusion(self.vision_encoder(vision_x))       # (B, F)
        p = self.physio_to_fusion(self.physio_encoder(physio_x))       # (B, F)
        e = self.env_to_fusion(self.env_encoder(env_x))                # (B, F)
        o = self.obd_to_fusion(self.obd_encoder(obd_x))                # (B, F)

        tokens = torch.stack([v, p, e, o], dim=1)                       # (B, 4, F)
        tokens = tokens + self.modality_embed.unsqueeze(0)              # add modality-type info

        cls = self.cls_token.expand(batch, -1, -1)                      # (B, 1, F)
        sequence = torch.cat([cls, tokens], dim=1)                      # (B, 5, F)

        fused = self.cross_attention_fusion(sequence)
        pooled = fused[:, 0, :]  # take the CLS token's output representation

        drowsiness = torch.sigmoid(self.drowsiness_head(pooled)).squeeze(-1)
        health_anomaly = torch.sigmoid(self.health_anomaly_head(pooled)).squeeze(-1)
        distraction = torch.sigmoid(self.distraction_head(pooled)).squeeze(-1)
        time_to_event = torch.relu(self.time_to_event_head(pooled)).squeeze(-1)  # seconds, >= 0

        return {
            "drowsiness_score": drowsiness,
            "health_anomaly_score": health_anomaly,
            "distraction_score": distraction,
            "time_to_event_s": time_to_event,
        }


if __name__ == "__main__":
    # Smoke test with random tensors of the right shapes.
    model = ModalityFusionTransformer()
    batch, T_vis, T_phys = 2, 90, 200  # e.g. 3s @ 30fps vision, 8s @ 25Hz physio
    vision_x = torch.randn(batch, T_vis, VisionEncoder.FEATURE_DIM)
    physio_x = torch.randn(batch, T_phys, PhysioEncoder.FEATURE_DIM)
    env_x = torch.randn(batch, EnvironmentEncoder.FEATURE_DIM)
    obd_x = torch.randn(batch, OBDEncoder.FEATURE_DIM)

    out = model(vision_x, physio_x, env_x, obd_x)
    for k, v in out.items():
        print(f"{k}: shape={tuple(v.shape)} values={v.detach().numpy()}")
