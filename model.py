# model.py
import torch
import torch.nn as nn

class PPG1DCNN(nn.Module):
    """
    Simple 1D CNN for two-channel PPG windows (channels: IR, Red).
    Outputs two scalars: hr_bpm (regression) and spo2_percent (regression).
    Input shape: (batch, 2, L)
    """
    def __init__(self, in_channels=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden*2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden*2),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden*2, hidden*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden*4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # hr and spo2
        )

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        # hr in first output, spo2 in second
        return x
