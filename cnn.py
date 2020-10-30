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
            ), # 12 * 48 * 48
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 12*24*24
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 24*12*12
        )

        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)  # 24*12*12
        # )

        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(128, 256, 5, 1, 2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2) # 256 * 6 * 6
        # )

        self.out1 = nn.Linear(32 * 12 * 12, 144)
        self.out2 = nn.Linear(144, 7)
        # self.out3 = nn.Linear(216, 2)
        # self.out5 = nn.Linear(64, 16)
        # self.out6 = nn.Linear(16, 2)

    def forward(self, x):
        # x = np.expand_dims(x, 1)
        # x = torch.tensor(x, dtype=torch.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.out1(x)
        # x = self.out2(x)
        # x = self.out3(x)
        # x = self.out4(x)
        # x = self.out5(x)
        output = self.out2(x)
        return output
