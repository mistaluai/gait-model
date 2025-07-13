import torch
import torch.nn as nn
from utils.models_helpers import ConvLSTM  # Assuming ConvLSTM is defined in models_helpers

# --- Final GEI ConvLSTM Classifier for small dataset ---
class GEIConvLSTMClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # CNN for feature extraction from GEIs
        # Maybe not that important but we can try it out due to small dataset 
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (H, W) → (H/2, W/2) >> Suggest to use avg pooling instead
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # ConvLSTM for temporal modeling of extracted features
        self.convlstm = ConvLSTM(input_channels=16, hidden_channels=32, kernel_size=3)

        # self.norm = nn.LayerNorm([32, 32, 32]) # Normalization of the output
        self.norm = nn.BatchNorm2d(32) # temp issue fix till we meet

        # Global average pooling to compress spatial features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final classifier head (shallow to avoid overfitting)
        self.classifier = nn.Sequential(
            nn.Flatten(),          # (B, C, 1, 1) → (B, C)
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes)
        )

    def forward(self, x, seq_lengths):
        # x: (B, T, 1, H, W) — GEI sequence

        B, T, _, H, W = x.shape

        # Step 1: Encode spatial features frame-by-frame
        x = x.view(B * T, 1, H, W)
        x = self.spatial_encoder(x)  # → (B*T, 16, H/2, W/2)

        C, H2, W2 = x.shape[1:]  # Extract for later use
        x = x.view(B, T, C, H2, W2)  # → (B, T, 16, H/2, W/2)

        # Step 2: ConvLSTM across time
        h_seq, _ = self.convlstm(x, seq_lengths)  # (B, T, 32, H/2, W/2)

        # Step 3: Mean pooling over time (temporal pooling)
        h_mean = torch.mean(h_seq, dim=1)  # (B, 32, H/2, W/2)

        # Step 4: LayerNorm (optional)
        h_norm = self.norm(h_mean)

        # Step 5: Global average pooling (spatial pooling)
        pooled = self.global_pool(h_norm)  # (B, 32, 1, 1)

        # Step 6: Classification
        out = self.classifier(pooled)  # (B, num_classes)

        return out
if __name__ == '__main__':
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    num_classes = 5
    model = GEIConvLSTMClassifier(num_classes=num_classes)
    # Dummy input: (B, T, 1, H, W)
    B, T, H, W = 4, 6, 64, 64  # Example shape
    x = torch.randn(B, T, 1, H, W)
    # Sequence lengths (for variable-length sequences, if needed by ConvLSTM)
    seq_lengths = torch.full((B,), T, dtype=torch.long)  # Assuming all have same length
    # Forward pass
    output = model(x, seq_lengths)
    # Print output shape
    print("Output shape:", output.shape)  # Expected: (B, num_classes)