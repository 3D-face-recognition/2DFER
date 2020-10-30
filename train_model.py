import torch
import torch.utils.data as Data
from torch import nn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import numpy as np

from copy import deepcopy

class TrainModel():
    def __init__(self, data_provider, learning_rate, epoch):
        self._learning_rate = learning_rate
        self._epoch = epoch
        self._data_provider = data_provider

    def training(self, model, optimizer, loss_function):
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        # loss_function = nn.CrossEntropyLoss()
        accuracy_list = []
        # for epoch in range(1, self._epoch):
        while (len(accuracy_list) < 4) or \
                (abs(accuracy_list[-1][0] - accuracy_list[-2][0]) > 0.01) or \
                (abs(accuracy_list[-1][0] - accuracy_list[-3][0] > 0.01)) or \
                (accuracy_list[-1][1][0] < 0.7) or \
                (accuracy_list[-1][1][1] < 0.7):
            print('\nEpoch :', len(accuracy_list), '...')
            for step, (x, y) in enumerate(self._data_provider.train_loader):
                output = model(x)
                loss = loss_function(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            accuracy_list.append(self.__accuracy__(model))
            print('\nAccuracy :', accuracy_list[-1][0])
            for index, emotion_accuracy in enumerate(accuracy_list[-1][1]):
                print(index)
                print(' Accuracy(%s) : %f' % (self._data_provider.emotion_label(index),
                                              emotion_accuracy))
        plt.plot(range(len(accuracy_list)), [accuracies[0] for accuracies in accuracy_list])
        plt.show()


    def __accuracy__(self, model):
        test_output = model(self._data_provider.x_validation)
        pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()

        accuracy_list = [accuracy_score(y_pred=pred_y, y_true=self._data_provider.y_validation_encoded), []]
        validation_y = self._data_provider.y_validation_encoded
        for emotion in range(2):
            accuracy_list[1].append(accuracy_score(y_pred=pred_y[validation_y == emotion],
                                                         y_true=validation_y[validation_y == emotion]))
        return accuracy_list
