from webbrowser import get
import torch.hub as hub
import torch.nn as nn
from torch import save, load
import os
def get_resnet_18(num_classes):
    if not os.path.exists("ml_ed_uni_cwk_3/code/models/resnet_18"):
        print("Loading model from hub")
        model = hub.load('pytorch/vision:v0.10.0', 'resnet18')
    else:
        model = load("ml_ed_uni_cwk_3/code/models/resnet_18")
        print("Loading untrained model from disk")

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

