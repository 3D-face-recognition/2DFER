import torch
import torch.utils.data as Data
from sklearn.preprocessing import LabelEncoder

import os

class DataProvider():
    def __init__(self, batch_size, workers):
        self._batch_size = batch_size
        self._workers = workers

        self._label_encoder = LabelEncoder()

        self._train_data = self.__get_data__('train')
        # print(self.y_train)
        self._y_train_encoded = self._label_encoder.fit_transform(self.y_train)
        self._train_dataset = \
            Data.TensorDataset(self.x_train, torch.from_numpy(self.y_train_encoded))
        self._validation_data = self.__get_data__('validation')
        # print(self.y_validation)
        self._y_validation_encoded = self._label_encoder.fit_transform(self.y_validation)
        self._validation_dataset = \
            Data.TensorDataset(self.x_validation, torch.from_numpy(self.y_validation_encoded))

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
        dataLoader = Data.DataLoader(
            dataset=self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self._workers,
        )
        return dataLoader

    def emotion_label(self, index):
        return self._label_encoder.inverse_transform([index])[0]

    def __get_images_path__(self, path, emotion):
        images_path = os.path.join(".", path, emotion)
        return images_path

    def __get_data__(self, category="train"):
        pass