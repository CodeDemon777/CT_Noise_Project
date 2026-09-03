import torch
import torch.nn as nn


class NoiseCNN(nn.Module):
    """
    Model 4: NoiseCNN (LoDoPaB Noise Classifier — Vasanth).
    Detects Quantization Noise and Periodic Noise across the scan.
    Output: 3 classes:
      0 = Clean / Normal
      1 = Quantization Noise
      2 = Periodic Noise
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat)
