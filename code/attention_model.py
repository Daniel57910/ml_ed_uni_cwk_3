from sys import path_hooks
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import ProjectorBlock, SpatialAttn, TemporalAttn
from basic_layers import ResNetBasicBlock
from torchvision import models
import torch.nn.functional as F
import pdb

class BaseModel(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.conv_1 = self._make_layer(3, 64, 3)
        self.res_block_1 = ResNetBasicBlock(64, 128)
        self.n_classes = n_classes

    def _make_layer(self, input_channels, out_features, kernel_size):
        return nn.Sequential(*[
            nn.Conv2d(input_channels, out_features, kernel_size),
            nn.BatchNorm2d(out_features, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        ])

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        try:
            output_1 = self.conv_1(x)
            print(f"Device of output 1 conv_1: {output_1.get_device()}")
        except:
            print("Unable to run convnet on GPU")
            raise Exception

        try:
            res_layer = self.res_block_1(output_1)
            print(f"Device of flat: {res_layer.get_device()}")
        except:
            print("Unable to run res layer on device")
            raise Exception

        try:
            flat_layer = res_layer.view(-1)
            final_activation_function = nn.Linear(
                in_features=flat_layer.shape[1],
                out_features=self.n_classes,
                bias=True
            ).to('cuda')
            final = final_activation_function(flat_layer)
            softmax_final = nn.Softmax(final)
        except:
            print("Final activation failed")
            raise Exception

        return final


model = BaseModel(27)
print(model)