import torch
import torch.nn as nn
from bnn.gmm_bayesian_components import BayesConv2d, BayesLinear

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, prior_mu=0.0, prior_sigma=0.1, **kwargs):
        super().__init__()
        self.residual_function = nn.Sequential(
            BayesConv2d(prior_mu, prior_sigma, in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            BayesConv2d(prior_mu, prior_sigma, out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                BayesConv2d(prior_mu, prior_sigma, in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False, **kwargs),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, prior_mu=0.0, prior_sigma=0.1, **kwargs):
        super().__init__()
        self.residual_function = nn.Sequential(
            BayesConv2d(prior_mu, prior_sigma, in_channels, out_channels, kernel_size=1, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            BayesConv2d(prior_mu, prior_sigma, out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            BayesConv2d(prior_mu, prior_sigma, out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                BayesConv2d(prior_mu, prior_sigma, in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False, **kwargs),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100, prior_mu=0.0, prior_sigma=0.1, **kwargs):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            BayesConv2d(prior_mu, prior_sigma, 3, 64, kernel_size=3, padding=1, bias=False, **kwargs),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, prior_mu, prior_sigma, **kwargs)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, prior_mu, prior_sigma, **kwargs)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, prior_mu, prior_sigma, **kwargs)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, prior_mu, prior_sigma, **kwargs)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = BayesLinear(prior_mu, prior_sigma, 512 * block.expansion, num_classes, **kwargs)

    def _make_layer(self, block, out_channels, num_blocks, stride, prior_mu, prior_sigma, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, prior_mu, prior_sigma, **kwargs))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

def resnet18(num_classes=10, prior_mu=0.0, prior_sigma=0.1, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, prior_mu, prior_sigma, **kwargs)

def resnet34(num_classes=100, prior_mu=0.0, prior_sigma=0.1, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, prior_mu, prior_sigma, **kwargs)

def resnet50bnn(num_classes=100, prior_mu=0.0, prior_sigma=0.1, **kwargs):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes, prior_mu, prior_sigma, **kwargs)

def resnet101(num_classes=100, prior_mu=0.0, prior_sigma=0.1, **kwargs):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes, prior_mu, prior_sigma, **kwargs)

def resnet152(num_classes=100, prior_mu=0.0, prior_sigma=0.1, **kwargs):
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes, prior_mu, prior_sigma, **kwargs)

if __name__ == '__main__':
    model = resnet50()
    # 输出模型中所有参数的名称和形状
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Shape: {param.shape}")

    # 保存模型参数
    save_path = 'checkpoints/model_with_gmm_params.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model parameters saved to {save_path}")
