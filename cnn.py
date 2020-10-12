import torch
from torch import nn
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2
            ), # 16 * 48 * 48
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 16*24*24
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32*12*12
        )

        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(32, 64, 5, 1, 2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )

        self.out1 = nn.Linear(32 * 12 * 12, 144)
        self.out2 = nn.Linear(144, 7)
        # self.out1 = nn.Linear(64 * 6 * 6, 7)
        # self.out2 = nn.Linear(128, 7)

    def forward(self, x):
        # x = np.expand_dims(x, 1)
        # x = torch.tensor(x, dtype=torch.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.size(0), -1)
        # x = self.out1(x)
        x = self.out1(x)
        output = self.out2(x)
        return output
