from sys import path_hooks
import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_layers_v2 import make_convolution_layer
from torchvision import models
import torch.nn.functional as F
from torchsummary import summary
import pdb

class AttModelV2(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.layer_one = make_convolution_layer(3, 64, pool_type="max", dropout_rate=0.2)

    def forward(self, x):
        out = self.layer_one(x)
        return out



model = AttModelV2(81)
print(summary(model, (3, 180, 180)))