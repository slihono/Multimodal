"""
Phase 3 - Advanced Computer Vision: Emotion Recognition
==========================================================
EfficientNet-B0/B3 backbone fine-tuned for facial emotion recognition
(anger, anxiety, neutral, etc.) -- used for road-rage / stress detection.

Requires: torch, torchvision, opencv-python, pillow
    pip install torch torchvision opencv-python pillow --break-system-packages

Note: this ships an EfficientNet-B0 classifier head with random
initialization for the emotion classes. Train it on a facial-emotion
dataset (e.g. FER2013, AffectNet, RAF-DB) before production use --
`train_stub()` below shows the training loop structure to plug your
dataset into.
"""

import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

sys.path.append("..")
from common.schemas import now  # noqa: E402

EMOTIONS = ["neutral", "happy", "sad", "angry", "anxious", "tired"]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class EmotionRecognizer(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", num_classes: int = len(EMOTIONS)):
        super().__init__()
        if backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        elif backbone == "efficientnet_b3":
            self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unknown backbone {backbone}")

    def forward(self, x):
        return self.backbone(x)


class EmotionPipeline:
    def __init__(self, weights_path: str = None, backbone: str = "efficientnet_b0", device: str = "cpu"):
        self.device = device
        self.model = EmotionRecognizer(backbone=backbone).to(device)
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model.eval()

    @torch.no_grad()
    def predict(self, face_crop_bgr: np.ndarray):
        """face_crop_bgr: cropped face region from the vision pipeline (Phase 1)."""
        rgb = face_crop_bgr[:, :, ::-1]
        pil_img = Image.fromarray(rgb)
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs))
        return EMOTIONS[idx], float(probs[idx])


def train_stub(train_loader, val_loader, epochs: int = 20, lr: float = 3e-4,
                device: str = "cpu", out_path: str = "emotion_model.pt"):
    """Reference training loop. Wire in a DataLoader over your annotated
    emotion dataset (e.g. FER2013/AffectNet) to fine-tune the backbone."""
    model = EmotionRecognizer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / max(1, total)
        print(f"[epoch {epoch+1}/{epochs}] val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_path)

    print(f"[*] Best val_acc={best_val_acc:.3f}, saved to {out_path}")


if __name__ == "__main__":
    # Smoke test with a synthetic random image (no dataset required).
    pipeline = EmotionPipeline()
    fake_face = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    label, conf = pipeline.predict(fake_face)
    print(f"[*] Smoke test prediction: {label} ({conf:.2f}) t={now():.2f}")
