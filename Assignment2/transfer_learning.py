from typing import cast
from torchvision import models
import torch.nn as nn


def get_pretrained_model(model_name='resnet18', num_classes=18, feature_extract=True):
    
    model = None

    if model_name == "resnet18":
        model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)

        if feature_extract:
            for parameter in model.parameters():
                parameter.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vgg16":
        model = models.vgg16(weights = models.VGG16_Weights.DEFAULT)

        if feature_extract:
            for parameter in model.features.parameters():
                parameter.requires_grad = False
            for parameter in model.classifier.parameters():
                parameter.requires_grad = False

        layer_6 = cast(nn.Linear, model.classifier[6])
        model.classifier[6] = nn.Linear(layer_6.in_features, num_classes)

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        if feature_extract:
            for parameter in model.parameters():
                parameter.requires_grad = False

        layer_1 = cast(nn.Linear, model.classifier[1])
        model.classifier[1] = nn.Linear(layer_1.in_features, num_classes)

    else:
        raise ValueError(f"Model {model_name} is unsupported.")

    return model