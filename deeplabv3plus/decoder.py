import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()

        self.project_low_level_features = nn.Sequential(
            nn.Conv2d(in_channels=low_level_channels, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.classifer = nn.Sequential(
            nn.Conv2d(in_channels=256+64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1, stride=1),
        )

    def forward(self, low_level_features, high_level_features):
        size = low_level_features.shape[-2:]
        low_level_features = self.project_low_level_features(low_level_features)
        high_level_features = F.interpolate(high_level_features, size=size, mode='bilinear', align_corners=True)
        x = torch.cat((high_level_features, low_level_features), dim=1)
        x = self.classifer(x)
        return x
