import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, prior_mu=0.0, prior_sigma=0.01):
        super(BasicBlock, self).__init__()
        self.conv1 = bnn.BayesConv2d(
            in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False,
            prior_mu=prior_mu, prior_sigma=prior_sigma
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = bnn.BayesConv2d(
            in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False,
            prior_mu=prior_mu, prior_sigma=prior_sigma
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                bnn.BayesConv2d(
                    in_channels=in_planes, out_channels=planes * self.expansion, kernel_size=1, stride=stride, bias=False,
                    prior_mu=prior_mu, prior_sigma=prior_sigma
                ),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut(x)
        out += shortcut
        out = F.relu(out)
        return out

# 定义 BayesianResNet 类
class BayesianResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, prior_mu=0.0, prior_sigma=0.01):
        super(BayesianResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = bnn.BayesConv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False,
            prior_mu=prior_mu, prior_sigma=prior_sigma
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(
            block, 16, num_blocks[0], stride=1, prior_mu=prior_mu, prior_sigma=prior_sigma
        )
        self.layer2 = self._make_layer(
            block, 32, num_blocks[1], stride=2, prior_mu=prior_mu, prior_sigma=prior_sigma
        )
        self.layer3 = self._make_layer(
            block, 64, num_blocks[2], stride=2, prior_mu=prior_mu, prior_sigma=prior_sigma
        )
        self.linear = bnn.BayesLinear(
            in_features=64 * block.expansion, out_features=num_classes, prior_mu=prior_mu, prior_sigma=prior_sigma
        )

    def _make_layer(self, block, planes, num_blocks, stride, prior_mu, prior_sigma):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(
                self.in_planes, planes, s, prior_mu=prior_mu, prior_sigma=prior_sigma
            ))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size(3))  # 自适应平均池化
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 定义 ResNet20 架构
def BayesianResNet20(num_classes=10, prior_mu=0.0, prior_sigma=0.1):
    return BayesianResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, prior_mu=prior_mu, prior_sigma=prior_sigma)

#### next is for the deterministic neural network

class BasicBlockDeterministic(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockDeterministic, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_planes, out_channels=planes * self.expansion, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut(x)
        out += shortcut
        out = F.relu(out)
        return out

# 定义 BayesianResNet 类
class ResNetDeterministic(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetDeterministic, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(
            block, 16, num_blocks[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, 32, num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, 64, num_blocks[2], stride=2
        )
        self.linear = nn.Linear(
            in_features=64 * block.expansion, out_features=num_classes
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(
                self.in_planes, planes, s
            ))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size(3))  # 自适应平均池化
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 定义 ResNet20 架构
def ResNet20(num_classes=10):
    return ResNetDeterministic(BasicBlockDeterministic, [3, 3, 3], num_classes=num_classes)
