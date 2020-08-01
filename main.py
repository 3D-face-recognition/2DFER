# import cv2 as cv
import os
import numpy as np
import torch

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
        images_path = get_train_images_path(path, emotion)
        for image in os.listdir(images_path):
            #X.append(cv.imread(os.path.join(images_path, image), cv.IMREAD_GRAYSCALE))
            y.append(emotion)
    return X, y

X_train, y_train = get_data("train")
print("X shape(training data): ", np.shape(X_train))
print("y shape(training data): ", np.shape(y_train))

X_validation, y_validation = get_data("validation")
print("X shape(validation data): ", np.shape(X_validation))
print("y shape(validation data): ", np.shape(y_validation))

