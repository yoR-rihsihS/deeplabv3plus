import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet
from .assp import ASPP
from .decoder import Decoder

class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=20, output_stride=16):
        super(DeepLabV3Plus, self).__init__()
        if output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
            atrous_rates = [6, 12, 18]
        elif output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            atrous_rates = [12, 24, 36]
        else:
            raise ValueError(f"output stride {output_stride} is not supported. Choose from '8' or '16'")
        
        if backbone in ['resnet50', 'resnet101', 'resnet152']:
            low_level_channels = 256
            high_level_channels = 2048
        else:
            raise ValueError(f"backbone {backbone} is not supported. Choose from 'resnet50' or 'resnet101' or 'resnet152'")

        self.resnet = ResNet(model=backbone, replace_stride_with_dilation=replace_stride_with_dilation)
        self.aspp = ASPP(in_channels=high_level_channels, out_channels=256, atrous_rates=atrous_rates)
        self.decoder = Decoder(low_level_channels=low_level_channels, num_classes=num_classes)

    def forward(self, image):
        size = image.shape[-2:]
        low_level_features, high_level_features = self.resnet(image)
        high_level_features = self.aspp(high_level_features)
        x = self.decoder(low_level_features, high_level_features)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x