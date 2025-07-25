import torch
import torch.nn as nn
import torch.nn.functional as F

class Flow3DCNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def conv_block(in_c, out_c, pool_kernel=(1, 2, 2), stride=1):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=pool_kernel)
            )

        self.encoder = nn.Sequential(
            conv_block(2, 32, pool_kernel=(1, 2, 2)),  # Only pool spatially
            conv_block(32, 64, pool_kernel=(1, 2, 2)),  # Only pool spatially
            conv_block(64, 128, pool_kernel=(2, 2, 2)),  # Finally reduce T, H, W
            nn.AdaptiveAvgPool3d((1, 1, 1))  # → [B, 128, 1, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),            # → [B, 128]
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.4),
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