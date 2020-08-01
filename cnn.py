
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(
#             in_channels=1,
#             out_put_channels=16,
#             kernel_size=5,
#             stride=1,
#             padding=2
#             ), # 16 * 48 * 48
#             nn.ReLu(),
#             nn.MaxPool2d(kernel_size=2) # 16*24*24
#
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(16, 32, 5, 1, 2),
#             nn.ReLu(),
#             nn.MaxPool2d(2) # 32*12*12
#         )
#         self.output = nn.Linear(32 * 12 * 12, 7)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.size(0), -1)
#         output = self.out(x)
#         return output

# cnn = CNN()
# print(cnn)
