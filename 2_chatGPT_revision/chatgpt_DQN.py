# DQN.py
import torch
from torch import nn

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        """
        input_shape: (C, H, W)
        num_actions: size of discrete action space
        """
        super().__init__()
        self.input_shape = input_shape
        c, h, w = input_shape

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Compute flattened feature size automatically
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            n_flat = self.feature_extractor(dummy).view(1, -1).size(1)
        self.n_flat = n_flat

        self.fc = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        # x: (N, C, H, W)
        features = self.feature_extractor(x)
        flat = features.view(features.size(0), -1)
        q_values = self.fc(flat)
        return q_values

    def loadModel(self, filename):
        state_dict = torch.load(filename, weights_only=False)
        self.load_state_dict(state_dict)

    def saveModel(self, filename):
        torch.save(self.state_dict(), filename)

    def __repr__(self):
        s = ""
        s += f"Input shape: {self.input_shape}\n"
        s += f"Flattened feature dim: {self.n_flat}\n"
        s += f"Network:\n{self}\n"
        return s
