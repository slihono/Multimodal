"""
Phase 4 - Training loop for the Modality Fusion Transformer.

This is a reference implementation: it trains on a SyntheticDrivingDataset
(random-but-shaped data) so you can verify the pipeline end-to-end before
your Phase 7 real-world dataset is collected. Swap SyntheticDrivingDataset
for a Dataset that loads your logged Phase 1-3 vision features, Phase 2
physio samples, Phase 6 env/OBD logs, and ground-truth labels
(drowsiness/anomaly/distraction annotated by observers, or KSS
self-report scores collected during Phase 7 validation drives).

Requires: torch
    pip install torch --break-system-packages
"""

import torch
from torch.utils.data import Dataset, DataLoader

from encoders import VisionEncoder, PhysioEncoder, EnvironmentEncoder, OBDEncoder
from fusion_transformer import ModalityFusionTransformer


class SyntheticDrivingDataset(Dataset):
    """Placeholder dataset with the correct tensor shapes for every
    modality plus the four regression/classification targets."""

    def __init__(self, n_samples: int = 512, t_vis: int = 90, t_phys: int = 200):
        self.n = n_samples
        self.t_vis = t_vis
        self.t_phys = t_phys

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        vision_x = torch.randn(self.t_vis, VisionEncoder.FEATURE_DIM)
        physio_x = torch.randn(self.t_phys, PhysioEncoder.FEATURE_DIM)
        env_x = torch.randn(EnvironmentEncoder.FEATURE_DIM)
        obd_x = torch.randn(OBDEncoder.FEATURE_DIM)

        targets = {
            "drowsiness_score": torch.rand(()),
            "health_anomaly_score": torch.rand(()) * 0.3,   # rare event, skewed low
            "distraction_score": torch.rand(()),
            "time_to_event_s": torch.rand(()) * 60.0,
        }
        return vision_x, physio_x, env_x, obd_x, targets


def train(epochs: int = 5, batch_size: int = 8, lr: float = 1e-4, device: str = "cpu"):
    dataset = SyntheticDrivingDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ModalityFusionTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = torch.nn.BCELoss()
    mse = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for vision_x, physio_x, env_x, obd_x, targets in loader:
            vision_x, physio_x = vision_x.to(device), physio_x.to(device)
            env_x, obd_x = env_x.to(device), obd_x.to(device)

            optimizer.zero_grad()
            out = model(vision_x, physio_x, env_x, obd_x)

            loss = (
                bce(out["drowsiness_score"], targets["drowsiness_score"].to(device))
                + bce(out["health_anomaly_score"], targets["health_anomaly_score"].to(device))
                + bce(out["distraction_score"], targets["distraction_score"].to(device))
                + 0.01 * mse(out["time_to_event_s"], targets["time_to_event_s"].to(device))
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[epoch {epoch+1}/{epochs}] loss={total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "fusion_model.pt")
    print("[*] Saved fusion_model.pt")
    return model


if __name__ == "__main__":
    train(epochs=3)  # short smoke-test run; raise epochs for real training
