import torch
from torch import nn
from torch.nn import functional as F

from .resnet import ResNet
from .assp import ASPP

class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=20, aspp_dilate=[6, 12, 18], output_features_name='layer_4', low_level_features_name='layer_1', intermediacte_channels=256):
        super(DeepLabV3Plus, self).__init__()
        self.output_features_name = output_features_name
        self.low_level_features_name = low_level_features_name
        self.layer_to_channels = {  # output_stride
            'layer_0': 64,          # 1/4
            'layer_1': 256,         # 1/4
            'layer_2': 512,         # 1/8
            'layer_3': 1024,        # 1/16
            'layer_4': 2048,        # 1/32, modified to 1/16
        }

        self.resnet = ResNet(model=backbone, replace_stride_with_dilation=[False, False, True])

        self.project = nn.Sequential( 
            nn.Conv2d(self.layer_to_channels[self.low_level_features_name], intermediacte_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(intermediacte_channels // 2),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(self.layer_to_channels[self.output_features_name], intermediacte_channels, aspp_dilate)

        self.fuse = nn.Sequential(
            nn.Conv2d(3 * intermediacte_channels // 2, intermediacte_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(intermediacte_channels),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(intermediacte_channels, intermediacte_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(intermediacte_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediacte_channels, num_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, image):
        feature = self.resnet(image)
        low_level_feature = self.project(feature[self.low_level_features_name])
        output_feature = self.aspp(feature[self.output_features_name])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        fused_features = self.fuse(torch.cat([low_level_feature, output_feature], dim=1))
        fused_features = F.interpolate(fused_features, size=image.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier(fused_features)

        