import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    # analogies to LSTM cell in PyTorch
    
    def __init__(self, 
                 input_channels, # number of channels for each GEI
                 hidden_channels, # number of channels in the hidden state
                 kernel_size, # size of the convolutional kernel
                 bias=False
                 ):
        super().__init__()
        padding = kernel_size // 2 # same padding to keep the spatial dimensions

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        #core component 
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels, # concatination of the last state with the current one in across the channels dimension No flattening
            out_channels=4 * hidden_channels, # 4 gates: input, forget, output, and candidate
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self,
                x, # the current GEI image (B, C, H, W)
                h_prev, # previous hidden state (B, hidden_channels, H, W)
                c_prev # previous cell state (B, hidden_channels, H, W)
                ):

        combined = torch.cat([x, h_prev], dim=1)  # (B, C+H, H, W)
        gates = self.conv(combined)              # (B, 4H, H, W)

        i, f, o, g = torch.chunk(gates, 4, dim=1)  # Split into 4 gate tensors

        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        o = torch.sigmoid(o)  # Output gate
        g = torch.tanh(g)     # Candidate cell state (-1,1)

        c = f * c_prev + i * g  # New cell state
        h = o * torch.tanh(c)   # New hidden state

        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)

    def forward(self, input_seq, seq_lengths): # IDK how to pass the sequence lengths
        # input_seq: (B, T, C, H, W)
        B, T, C, H, W = input_seq.size()
        device = input_seq.device
        H_ch = self.cell.hidden_channels

        h, c = (torch.zeros(B, H_ch, H, W, device=device),
                torch.zeros(B, H_ch, H, W, device=device))

        outputs = []
        for t in range(T):
            xt = input_seq[:, t]
            mask = (t < seq_lengths).float().view(B, 1, 1, 1)
            h, c = self.cell(xt, h, c)
            h = h * mask  # zero out for padded steps
            outputs.append(h)

        return torch.stack(outputs, dim=1), h  # (B, T, H_ch, H, W), (B, H_ch, H, W)
