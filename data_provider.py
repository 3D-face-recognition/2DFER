import torch
import torch.utils.data as Data
from sklearn.preprocessing import LabelEncoder

import os
import cv2 as cv
import numpy as np

class DataProvider():
    def __init__(self, batch_size, workers):
        self._batch_size = batch_size
        self._workers = workers

        label_encoder = LabelEncoder()

        self._train_data = self.__get_data__('train')
        self._y_train_encoded = label_encoder.fit_transform(self.y_train)
        self._train_dataset = \
            Data.TensorDataset(torch.from_numpy(self.x_train), torch.from_numpy(self.y_train_encoded))

        self._validation_data = self.__get_data__('validation')
        self._y_validation_encoded = label_encoder.fit_transform(self.y_validation)
        self._validation_dataset = \
            Data.TensorDataset(torch.from_numpy(self.x_validation), torch.from_numpy(self.y_validation_encoded))

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def x_train(self):
        return self._train_data[0]

    @property
    def y_train(self):
        return self._train_data[1]

    @property
    def y_train_encoded(self):
        return self._y_train_encoded

    @property
    def x_validation(self):
        return self._validation_data[0]

    @property
    def y_validation(self):
        return self._validation_data[1]

    @property
    def y_validation_encoded(self):
        return self._y_validation_encoded

    @property
    def train_loader(self):
        return Data.DataLoader(
            dataset=self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self._workers,
        )

    def __get_images_path__(self, path, emotion):
        images_path = os.path.join(".", path, emotion)
        return images_path

    def __get_data__(self, category="train"):
        print('\nProcessing', category, 'data\n')
        X = []
        y = []
        if category == "train":
            path = "images\\train"
        else:
            path = "images\\validation"
        for emotion in os.listdir(os.path.join(".", path)):
            print('Processing', emotion, '......')
            images_path = self.__get_images_path__(path, emotion)
            for image_idx, image in enumerate(os.listdir(images_path)):
                X.append([cv.imread(os.path.join(images_path, image), cv.IMREAD_GRAYSCALE) / 255])
                y.append(emotion)
        return np.array(X), np.array(y)