import os
import numpy as np
import cv2 as cv
import torch

from data_provider import DataProvider

class FER2013_DataProvider(DataProvider):
    def __get_data__(self, category="train"):
        print('\nProcessing', category, 'data\n')
        X = []
        y = []
        if category == "train":
            path = "images\\train"
        else:
            path = "images\\validation"
        for emotion in os.listdir(os.path.join(".", path)):
            if not('happy' in emotion) and not('sad' in emotion):
                continue
            print('Processing', emotion, '......')
            images_path = self.__get_images_path__(path, emotion)
            for image_idx, image in enumerate(os.listdir(images_path)):
                X.append(cv.imread(os.path.join(images_path, image), cv.IMREAD_GRAYSCALE) / 255)
                y.append(emotion)
        return torch.tensor(np.expand_dims(np.array(X), 1), dtype=torch.float32), np.array(y)