import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights

class ResNet(nn.Module):
    def __init__(self, model='resnet50', replace_stride_with_dilation=[False, False, True]):
        super(ResNet, self).__init__()
        if model == 'resnet50':
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, replace_stride_with_dilation=replace_stride_with_dilation)
        elif model == 'resnet101':
            model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2, replace_stride_with_dilation=replace_stride_with_dilation)
        elif model == 'resnet152':
            model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2, replace_stride_with_dilation=replace_stride_with_dilation)
        else:
            raise ValueError(f"{model} is not supported. Choose from 'resnet50', 'resnet101', or 'resnet152'.")

        layer_0 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )
        layer_1 = model.layer1
        layer_2 = model.layer2
        layer_3 = model.layer3
        layer_4 = model.layer4

        self.features = nn.Sequential(
            layer_0,    # 1/4, 64
            layer_1,    # 1/4, 256
            layer_2,    # 1/8, 512
            layer_3,    # 1/16, 1024
            layer_4,    # 1/32, 2048, modified to 1/16
        )

    def forward(self, image):
        outputs = {}
        x = image
        for i, layer in enumerate(self.features):
            x = layer(x)
            outputs[f"layer_{i}"] = x
        return outputs