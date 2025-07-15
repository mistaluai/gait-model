import torch
import torch.nn as nn
from utils.models_helpers import ConvLSTM  # keep your existing ConvLSTMCell implementation

class FlowConvLSTMClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Feature extractor for 2-channel optical flow input
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),  # Accepts 2-channel input
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (H, W) → (H/2, W/2)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # ConvLSTM layer
        self.temporal_model = ConvLSTM(input_channels=32, hidden_channels=64, kernel_size=3)

        # Normalize LSTM output
        self.norm = nn.BatchNorm2d(64)

        # Pool and classify
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
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
        x = self.encoder(x)  # → (B*T, 32, H/2, W/2)

        _, C2, H2, W2 = x.shape
        x = x.view(B, T, C2, H2, W2)

        h_seq, _ = self.temporal_model(x, seq_lengths)  # → [B, T, 64, H2, W2]
        h_mean = torch.mean(h_seq, dim=1)               # → [B, 64, H2, W2]

        h_norm = self.norm(h_mean)
        pooled = self.pool(h_norm)                      # → [B, 64, 1, 1]
        logits = self.classifier(pooled)                # → [B, num_classes]

        return logits