import torch
import torch.nn as nn

class Flow3DCNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),  # (C_in=2, T, H, W)
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  # Reduce H, W

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),  # Reduce T, H, W

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))  # Output shape: [B, 128, 1, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),              # → [B, 128]
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, seq_lengths=None):
        """
        Args:
            x: [B, T, 2, H, W]
        Returns:
            logits: [B, num_classes]
        """
        x = x.permute(0, 2, 1, 3, 4)  # → [B, C=2, T, H, W]
        x = self.encoder(x)           # → [B, 128, 1, 1, 1]
        logits = self.classifier(x)   # → [B, num_classes]
        return logits