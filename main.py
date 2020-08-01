import cv2 as cv
import os
import torch

# img = cv.imread('angry0.jpg', cv.IMREAD_GRAYSCALE)
# cv.imshow('Image', img)

# cv.waitKey(0)
# cv.destroyAllWindows()

# tensor = torch.tensor(img)
# print(tensor.size(0))
# print(tensor.view(tensor.size(0), -1))

tensor = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] for x in range(9)])
print(tensor)
print(tensor.size(0))
print(tensor.view(tensor.size(0), -1))

# y = torch.linspace(-1, 1, 100)
# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# print(y)
# print(x)