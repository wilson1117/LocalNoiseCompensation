import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes, attack_test=False, grayscale=False, input_shape=(32, 32)):
        super(LeNet, self).__init__()

        input_channel = 1 if grayscale else 3

        feature_size = (input_shape[0] // 4) ** 2 * 12

        act = nn.Sigmoid if attack_test else nn.ReLU
        self.body = nn.Sequential(
            nn.Conv2d(input_channel, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(feature_size, num_classes)
        )
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out