import torch
import torch.utils.data as Data
from torch import nn

from copy import deepcopy

class TrainModel():
    def __init__(self, train_loader, learning_rate, epoch):
        self._learning_rate = learning_rate
        self._epoch = epoch
        self._train_loader = train_loader

    def training(self, model, optimizer, loss_function):
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        # loss_function = nn.CrossEntropyLoss()
        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(self._train_loader):
                output = model(x)
                loss = loss_function(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()