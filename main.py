import cv2 as cv
import os
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
import torch.onnx
import onnx

from cnn import CNN
from sklearn.preprocessing import LabelEncoder
from train_model import TrainModel
from sklearn.metrics import accuracy_score

from data_provider import DataProvider
from ck_data_provider import CK_DataProvider

LEARNING_RATE = 0.01
BATCH_SIZE = 50
EPOCH = 30

data_provider = CK_DataProvider(BATCH_SIZE, 0)

cnn = CNN()
cnn_model = TrainModel(data_provider, LEARNING_RATE, EPOCH)
cnn_model.training(cnn, torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE), nn.CrossEntropyLoss())

test_output = cnn(data_provider.x_validation)
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()

dummy_input = torch.tensor(np.expand_dims([np.random.rand(48, 48)], 1), dtype=torch.float32)
torch.onnx.export(cnn, dummy_input, 'cnn_model.onnx', verbose=True)

onnx_model = onnx.load('cnn_model.onnx')
onnx.checker.check_model(onnx_model)

print('Export cnn_model.onnx complete!')
