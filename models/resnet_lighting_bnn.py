import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import BasicBlock
from tqdm.auto import tqdm
import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
sys.path.append("../../../../../")
sys.path.append("../../../../../../")
sys.path.append("../../../../../../../")
sys.path.append("../../../../../../../../")
from .BNNs_component import MeanFieldGaussianFeedForward, MeanFieldGaussian2DConvolution
# def _weights_init(m):
#     if isinstance(m, (MeanFieldGaussianFeedForward, MeanFieldGaussian2DConvolution)):
#         torch.nn.init.kaiming_normal_(m.weight)
class VIModule(nn.Module):
    """
    A mixin class to attach loss functions to layer. This is useful when doing variational inference with deep learning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._internalLosses = []
        self.lossScaleFactor = 1

    def addLoss(self, func):
        self._internalLosses.append(func)

    def evalLosses(self):
        t_loss = 0
        for l in self._internalLosses:
            t_loss = t_loss + l(self)
        return t_loss

    def evalAllLosses(self):
        t_loss = self.evalLosses() * self.lossScaleFactor
        for m in self.children():
            if isinstance(m, VIModule):
                t_loss = t_loss + m.evalAllLosses() * self.lossScaleFactor
        return t_loss

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResNet(VIModule):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.layers = nn.Sequential(
            MeanFieldGaussian2DConvolution(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            self._make_layer(16, num_blocks, stride=1),
            self._make_layer(32, num_blocks, stride=2),
            self._make_layer(64, num_blocks, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            MeanFieldGaussianFeedForward(64, num_classes)
        )
        # self.apply(_weights_init)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            downsample = None
            if stride != 1 or self.in_planes != planes:
                downsample = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0
                    )
                )
            layers.append(BasicBlock(self.in_planes, planes, stride, downsample))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def resnet20():
    model = ResNet(num_blocks=3)
    return model

