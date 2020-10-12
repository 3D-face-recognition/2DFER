import os
import numpy as np
import cv2 as cv
import torch

from sklearn.model_selection import train_test_split

from data_provider import DataProvider

class CK_DataProvider(DataProvider):
    def __get_data__(self, category="train"):
        print('\nProcessing', category, 'data\n')
        X = []
        y = []
        path = 'CK+48'
        for emotion in os.listdir(os.path.join(".", path)):
            print('Processing', emotion, '......')
            images_path = self.__get_images_path__(path, emotion)
            for image_idx, image in enumerate(os.listdir(images_path)):
                X.append(cv.imread(os.path.join(images_path, image), cv.IMREAD_GRAYSCALE) / 255)
                y.append(emotion)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
        if category == 'train':
            return torch.tensor(np.expand_dims(np.array(X_train), 1), dtype=torch.float32), np.array(y_train)
        else:
            return torch.tensor(np.expand_dims(np.array(X_test), 1), dtype=torch.float32), np.array(y_test)