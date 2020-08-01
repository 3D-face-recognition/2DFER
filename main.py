import cv2 as cv
import os
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
from cnn import CNN
from sklearn.preprocessing import LabelEncoder

TRAINING_SIZE = 1000
LEARNING_RATE = 0.01
BATCH_SIZE = 50
EPOCH = 1

def get_train_images_path(path, emotion):
    images_path = os.path.join(".", path, emotion)
    return images_path

def get_data(category="train"):
    X = []
    y= []
    if category == "train":
        path = "images\\train"
    else:
        path = "images\\validation"
    for emotion in  os.listdir(os.path.join(".", path)):
        print('Processing', emotion, '......')
        images_path = get_train_images_path(path, emotion)
        for image_idx, image in enumerate(os.listdir(images_path)):
            X.append([cv.imread(os.path.join(images_path, image), cv.IMREAD_GRAYSCALE) / 255])
            y.append(emotion)
            if image_idx >= TRAINING_SIZE:
                break
    return np.array(X), np.array(y)

def training(X_train, y_train, model):
    print(X_train)
    print(y_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    train_loader = transform_dataset(X_train, y_train)
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            output = model(x)
            loss = loss_function(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def transform_dataset(X_train, y_train):
    torch_dataset = Data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    return Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

X_train, y_train = get_data("train")
print("X shape(training data): ", np.shape(X_train))
print("y shape(training data): ", np.shape(y_train))

X_validation, y_validation = get_data("validation")
print("X shape(validation data): ", np.shape(X_validation))
print("y shape(validation data): ", np.shape(y_validation))

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_validation_encoded = label_encoder.transform(y_validation)
print(y_train_encoded)

cnn = CNN()
training(X_train, y_train_encoded, cnn)
test_output = cnn(X_validation[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(y_validation_encoded[:10], 'real number')
