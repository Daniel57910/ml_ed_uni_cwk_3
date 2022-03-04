from webbrowser import get
import torch.hub as hub
import torch.nn as nn

def get_resnet_18(num_classes):
    model = hub.load('pytorch/vision:v0.10.0', 'resnet18')
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(*[
        nn.Linear(num_ftrs, num_classes),
        nn.Sigmoid()
    ])

    model.half()

    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()
    return model
