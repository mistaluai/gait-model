import torch
import torch.nn as nn
from utils.models_helpers import ConvLSTM  # keep your existing ConvLSTMCell implementation

class FlowConvLSTMClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # More complex feature extractor for 2-channel optical flow input
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # Increased channels
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (H, W) → (H/2, W/2)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (H/2, W/2) → (H/4, W/4)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Deeper ConvLSTM stack
        self.temporal_model = nn.Sequential(
            ConvLSTM(input_channels=128, hidden_channels=128, kernel_size=3),
            ConvLSTM(input_channels=128, hidden_channels=256, kernel_size=3)
        )

        # Normalize LSTM output
        self.norm = nn.BatchNorm2d(256)

        # Pool and classify
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, seq_lengths):
        """
        Args:
            x: Tensor [B, T, 2, H, W]
            seq_lengths: Tensor [B] of sequence lengths
        Returns:
            logits: [B, num_classes]
        """
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        x = self.encoder(x)  # → (B*T, 128, H/4, W/4)

        _, C2, H2, W2 = x.shape
        x = x.view(B, T, C2, H2, W2)

        # Pass through stacked ConvLSTM layers
        h_seq = x
        for layer in self.temporal_model:
            h_seq, _ = layer(h_seq, seq_lengths)
        h_mean = torch.mean(h_seq, dim=1)               # → [B, 256, H2, W2]

        h_norm = self.norm(h_mean)
        pooled = self.pool(h_norm)                      # → [B, 256, 1, 1]
        logits = self.classifier(pooled)                # → [B, num_classes]

        return logits