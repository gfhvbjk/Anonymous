import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        prior_mu = 0.0
        prior_sigma = 0.1

        self.bn_conv1 = bnn.BayesConv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False,
                                        prior_mu=prior_mu, prior_sigma=prior_sigma)
        self.bn_bn1 = nn.BatchNorm2d(planes)
        self.bn_conv2 = bnn.BayesConv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False,
                                        prior_mu=prior_mu, prior_sigma=prior_sigma)
        self.bn_bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                bnn.BayesConv2d(in_channels=in_planes, out_channels=self.expansion * planes, kernel_size=1, stride=stride, bias=False,
                                prior_mu=prior_mu, prior_sigma=prior_sigma),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.bn_conv1(x)
        out = self.bn_bn1(out)
        out = nn.ReLU()(out)
        out = self.bn_conv2(out)
        out = self.bn_bn2(out)
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

class BayesianResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(BayesianResNet, self).__init__()
        self.in_planes = 16
        prior_mu = 0.0
        prior_sigma = 0.1

        self.bn_conv1 = bnn.BayesConv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False,
                                        prior_mu=prior_mu, prior_sigma=prior_sigma)  # 输入通道数调整为1
        self.bn_bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, prior_mu=prior_mu, prior_sigma=prior_sigma)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, prior_mu=prior_mu, prior_sigma=prior_sigma)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, prior_mu=prior_mu, prior_sigma=prior_sigma)
        self.bn_linear = bnn.BayesLinear(in_features=64, out_features=num_classes, prior_mu=prior_mu, prior_sigma=prior_sigma)

    def _make_layer(self, block, planes, num_blocks, stride, prior_mu, prior_sigma):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn_conv1(x)
        out = self.bn_bn1(out)
        out = nn.ReLU()(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = nn.AvgPool2d(8)(out)
        out = out.view(out.size(0), -1)
        out = self.bn_linear(out)
        return out

def BayesianResNet20():
    return BayesianResNet(BasicBlock, [3, 3, 3])

#### Deterministic
class BasicBlockDeterministic(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockDeterministic, self).__init__()


        self.bn_conv1 = nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_bn1 = nn.BatchNorm2d(planes)
        self.bn_conv2 =  nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.bn_conv1(x)
        out = self.bn_bn1(out)
        out = nn.ReLU()(out)
        out = self.bn_conv2(out)
        out = self.bn_bn2(out)
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

class DeterministicResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(DeterministicResNet, self).__init__()
        self.in_planes = 16
        prior_mu = 0.0
        prior_sigma = 0.1

        self.bn_conv1 =  nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)  # 输入通道数调整为1
        self.bn_bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, prior_mu=prior_mu, prior_sigma=prior_sigma)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, prior_mu=prior_mu, prior_sigma=prior_sigma)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, prior_mu=prior_mu, prior_sigma=prior_sigma)
        self.bn_linear = nn.Linear(in_features=64, out_features=num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, prior_mu, prior_sigma):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn_conv1(x)
        out = self.bn_bn1(out)
        out = nn.ReLU()(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = nn.AvgPool2d(8)(out)
        out = out.view(out.size(0), -1)
        out = self.bn_linear(out)
        return out

def ResNet20():
    return DeterministicResNet(BasicBlockDeterministic, [3, 3, 3])