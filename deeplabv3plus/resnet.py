import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights

class ResNet(nn.Module):
    def __init__(self, model, replace_stride_with_dilation):
        super(ResNet, self).__init__()
        if model == 'resnet152':
            model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2, replace_stride_with_dilation=replace_stride_with_dilation)
        elif model == 'resnet101':
            model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2, replace_stride_with_dilation=replace_stride_with_dilation)
        else:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, replace_stride_with_dilation=replace_stride_with_dilation)

        self.layer_0 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )
        self.layer_1 = model.layer1
        self.layer_2 = model.layer2
        self.layer_3 = model.layer3
        self.layer_4 = model.layer4

    def forward(self, x):
        x = self.layer_0(x)
        low_level_features = self.layer_1(x)
        x = self.layer_2(low_level_features)
        x = self.layer_3(x)
        high_level_featues = self.layer_4(x)
        return low_level_features, high_level_featues
        