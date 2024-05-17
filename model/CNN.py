import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes, attack_test=False, grayscale=False, input_shape=(32, 32)):
        super(CNN, self).__init__()

        input_channel = 1 if grayscale else 3

        feature_size = (input_shape[0] // 4) ** 2

        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding='same')
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(128 * feature_size, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(len(x), -1)
        x = self.fc(x)
        return x