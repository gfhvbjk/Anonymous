"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
import torch.nn.functional as F
import torch
import torch.nn as nn

class MeanFieldGaussian2DConvolution(nn.Module):
    ''' Convolutional Layer with Bayesian Weights (Direct Reparameterization) '''

    def __init__(self, in_channels, out_channels, kernel_size, mu_init, rho_init, prior_init, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super().__init__()

        # 保存卷积参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.bias_flag = bias

        # 初始化权重的均值和rho（用于sigma的计算）
        self.weight_mu = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size).uniform_(*mu_init))
        self.weight_rho = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size).uniform_(*rho_init))

        # 初始化偏置的均值和rho（如果有偏置）
        if self.bias_flag:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels).uniform_(*mu_init))
            self.bias_rho = nn.Parameter(torch.Tensor(out_channels).uniform_(*rho_init))

        self.normal = torch.distributions.Normal(0, 1)

        # 先验分布参数
        assert len(prior_init) == 1, "Gaussian Prior requires one value in prior initialisation"
        self.weight_prior = [0, prior_init[0]]
        if self.bias_flag:
            self.bias_prior = [0, prior_init[0]]

        # KL 效用初始化
        self.weight_kl_cost = 0
        if self.bias_flag:
            self.bias_kl_cost = 0
        self.kl_cost = 0

    def compute_kl_cost(self, p_params, q_params):
        ''' Compute closed-form KL Divergence between two Gaussians '''
        [p_mu, p_sigma] = p_params
        [q_mu, q_sigma] = q_params
        kl_cost = 0.5 * (
                    2 * torch.log(p_sigma / q_sigma) - 1 + (q_sigma / p_sigma).pow(2) + ((p_mu - p_mu) / p_sigma).pow(
                2)).sum()
        return kl_cost

    def call_kl_divergence(self):
        # if self.training or calculate_log_probs:
        w_sigma = torch.log1p(torch.exp(self.weight_rho))
        b_sigma = torch.log1p(torch.exp(self.bias_rho))
        self.weight_kl_cost = self.compute_kl_cost(self.weight_prior, [self.weight_mu, w_sigma]).sum()
        self.bias_kl_cost = self.compute_kl_cost(self.bias_prior, [self.bias_mu, b_sigma]).sum()
        self.kl_cost = self.weight_kl_cost + self.bias_kl_cost
        return self.kl_cost
    def forward(self, input, sample=False, calculate_log_probs=False):
        # 计算权重和偏置的 sigma
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        if self.bias_flag:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        # if self.training or sample:
        # 对权重进行采样
        weight_eps = self.normal.sample(self.weight_mu.size()).to(input.device)
        weight = self.weight_mu + weight_sigma * weight_eps
        # 对偏置进行采样
        if self.bias_flag:
            bias_eps = self.normal.sample(self.bias_mu.size()).to(input.device)
            bias = self.bias_mu + bias_sigma * bias_eps
        else:
            bias = None
        # 进行卷积运算
        activation = F.conv2d(input, weight, bias=bias, stride=self.stride,
                              padding=self.padding, dilation=self.dilation, groups=self.groups)
        #     # 应用 ReLU 激活函数
        #     # activation = F.relu(activation)
        # # else:
        # #     # 测试阶段使用权重和偏置的均值
        # #     activation = F.conv2d(input, self.weight_mu, bias=self.bias_mu if self.bias_flag else None,
        # #                           stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        # #     # activation = F.relu(activation)
        #
        # if self.training or calculate_log_probs:
        #     # 计算 KL 效用
        #     self.weight_kl_cost = self.compute_kl_cost(self.weight_prior, [self.weight_mu, weight_sigma])
        #     if self.bias_flag:
        #         self.bias_kl_cost = self.compute_kl_cost(self.bias_prior, [self.bias_mu, bias_sigma])
        #         self.kl_cost = self.weight_kl_cost + self.bias_kl_cost
        #     else:
        #         self.kl_cost = self.weight_kl_cost

        return activation


class MeanFieldGaussianFeedForward(nn.Module):
    ''' FC Layer with Bayesian Weights and Local Reparameterisation '''
    def __init__(self, in_features, out_features, mu_init, rho_init, prior_init, mixture_prior=False):
        super().__init__()

        self.weight_mu = nn.Parameter(torch.Tensor(in_features, out_features).uniform_(*mu_init))
        self.weight_rho = nn.Parameter(torch.Tensor(in_features, out_features).uniform_(*rho_init))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(*mu_init))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(*rho_init))
        self.normal = torch.distributions.Normal(0,1)

        assert len(prior_init)==1, "Gaussian Prior requires one value in prior initialisation"
        self.weight_prior = [0, prior_init[0]]
        self.bias_prior = [0, prior_init[0]]
        self.weight_kl_cost = 0
        self.bias_kl_cost = 0
        self.kl_cost = 0

    def compute_kl_cost(self, p_params, q_params):
        ''' Compute closed-form KL Divergence between two Gaussians '''
        [p_mu, p_sigma] = p_params
        [q_mu, q_sigma] = q_params
        kl_cost = 0.5 * (2*torch.log(p_sigma/q_sigma) - 1 + (q_sigma/p_sigma).pow(2) + ((p_mu - q_mu)/p_sigma).pow(2)).sum()
        return kl_cost
    def call_kl_divergence(self):
        # if self.training or calculate_log_probs:
        w_sigma = torch.log1p(torch.exp(self.weight_rho))
        b_sigma = torch.log1p(torch.exp(self.bias_rho))
        self.weight_kl_cost = self.compute_kl_cost(self.weight_prior, [self.weight_mu, w_sigma]).sum()
        self.bias_kl_cost = self.compute_kl_cost(self.bias_prior, [self.bias_mu, b_sigma]).sum()
        self.kl_cost = self.weight_kl_cost + self.bias_kl_cost
        return self.kl_cost
    def forward(self, input, sample=False, calculate_log_probs=False):

        w_sigma = torch.log1p(torch.exp(self.weight_rho))
        b_sigma = torch.log1p(torch.exp(self.bias_rho))
        activation_mu = torch.mm(input, self.weight_mu)
        activation_sigma = torch.sqrt(torch.mm(input.pow(2), w_sigma.pow(2)))

        w_epsilon = self.normal.sample(activation_sigma.size()).to(input.device)
        b_epsilon = self.normal.sample(b_sigma.size()).to(input.device)

        activation_w = activation_mu + activation_sigma * w_epsilon
        activation_b = self.bias_mu + b_sigma * b_epsilon

        activation =  activation_w + activation_b.unsqueeze(0).expand(input.shape[0], -1)

        # else:
        #     activation = torch.mm(input, self.weight_mu) + self.b_mu

        # if self.training or calculate_log_probs:
        #     self.weight_kl_cost = self.compute_kl_cost(self.weight_prior, [self.weight_mu, w_sigma]).sum()
        #     self.bias_kl_cost = self.compute_kl_cost(self.bias_prior, [self.bias_mu, b_sigma]).sum()
        #     self.kl_cost = self.weight_kl_cost + self.bias_kl_cost

        return activation




class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, mu_init, rho_init, prior_init, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            MeanFieldGaussian2DConvolution(in_channels, out_channels, kernel_size=3, mu_init=mu_init, rho_init=rho_init, prior_init=prior_init,stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            MeanFieldGaussian2DConvolution(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, mu_init=mu_init, rho_init=rho_init, prior_init=prior_init, padding=1, bias=True),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                MeanFieldGaussian2DConvolution(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride,mu_init=mu_init, rho_init=rho_init, prior_init=prior_init, bias=True),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, mu_init, rho_init, prior_init, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            MeanFieldGaussian2DConvolution(in_channels, out_channels, kernel_size=1, mu_init=mu_init, rho_init=rho_init, prior_init=prior_init,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            MeanFieldGaussian2DConvolution(out_channels, out_channels, stride=stride, kernel_size=3, padding=1,mu_init=mu_init, rho_init=rho_init, prior_init=prior_init, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            MeanFieldGaussian2DConvolution(out_channels, out_channels * BottleNeck.expansion, kernel_size=1,mu_init=mu_init, rho_init=rho_init, prior_init=prior_init, bias=True),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                MeanFieldGaussian2DConvolution(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1,mu_init=self.mu_init, rho_init=self.rho_init, prior_init=self.prior_init, bias=True),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNetBNN(nn.Module):

    def __init__(self, block, num_block, model_params, num_classes=10):
        super().__init__()

        self.input_shape = model_params['input_shape']
        self.classes = model_params['classes']
        self.batch_size = model_params['batch_size']
        self.hidden_units = model_params['hidden_units']
        self.mode = model_params['mode']
        self.mu_init = model_params['mu_init']
        self.rho_init = model_params['rho_init']
        self.prior_init = model_params['prior_init']
        self.mixture_prior = model_params['mixture_prior']
        self.local_reparam = model_params['local_reparam']

        self.in_channels = 64
        # in_channels, out_channels, kernel_size, mu_init, rho_init, prior_init, stride = 1, padding = 0,
        # dilation = 1, groups = 1, bias = True
        self.conv1 = nn.Sequential(
            MeanFieldGaussian2DConvolution(in_channels=3, out_channels=64, kernel_size=3, mu_init=self.mu_init, rho_init=self.rho_init, prior_init=self.prior_init, stride = 1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = MeanFieldGaussianFeedForward(512 * block.expansion, num_classes, self.mu_init, self.rho_init, self.prior_init, self.mixture_prior)
        # self.sigmoid = nn.Sigmoid()
        # self.fc2 = nn.Linear(512 * block.expansion // 2, num_classes)
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, self.mu_init, self.rho_init, self.prior_init, stride))
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
        # output = self.fc2(self.sigmoid(self.fc1(output)))

        return output

    # def log_prior(self):
    #     return self.l1.log_prior + self.l2.log_prior + self.l3.log_prior
    #
    # def log_variational_posterior(self):
    #     return self.l1.log_variational_posterior + self.l2.log_variational_posterior + self.l3.log_variational_posterior
    #
    # def kl_cost(self):
    #     return self.l1.kl_cost + self.l2.kl_cost + self.l3.kl_cost

    def log_prior(self):
        total_log_prior = 0
        for layer in self.modules():
            if isinstance(layer, (MeanFieldGaussian2DConvolution, MeanFieldGaussianFeedForward)):
                total_log_prior += layer.log_prior
        return total_log_prior

    def log_variational_posterior(self):
        total_log_variational_posterior = 0
        for layer in self.modules():
            if isinstance(layer, (MeanFieldGaussian2DConvolution, MeanFieldGaussianFeedForward)):
                total_log_variational_posterior += layer.log_variational_posterior
        return total_log_variational_posterior

    def kl_cost(self):
        total_kl_cost = 0
        for layer in self.modules():
            if isinstance(layer, (MeanFieldGaussian2DConvolution, MeanFieldGaussianFeedForward)):
                total_kl_cost += layer.kl_cost
        return total_kl_cost

    def get_nll(self, outputs, target, sigma=1.):
        if self.mode == 'regression':
            nll = -torch.distributions.Normal(outputs, sigma).log_prob(target).sum()
        elif self.mode == 'classification':
            nll = nn.CrossEntropyLoss(reduction='sum')(outputs, target)
        else:
            raise Exception("Training mode must be either 'regression' or 'classification'")
        return nll

    def sample_elbo(self, input, target, beta, samples, sigma=1.):
        ''' Sample ELBO for BNN w/o Local Reparameterisation '''
        assert self.local_reparam == False, 'sample_elbo() method returns loss for BNNs without local reparameterisation, alternatively use sample_elbo_lr()'
        log_priors = torch.zeros(samples).to(input.device)
        log_variational_posteriors = torch.zeros(samples).to(input.device)
        negative_log_likelihood = torch.zeros(1).to(input.device)

        for i in range(samples):
            output = self.forward(input)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
            negative_log_likelihood += self.get_nll(output, target, sigma)

        log_prior = beta * log_priors.mean()
        log_variational_posterior = beta * log_variational_posteriors.mean()
        negative_log_likelihood = negative_log_likelihood / samples
        loss = log_variational_posterior - log_prior + negative_log_likelihood
        return loss, log_priors.mean(), log_variational_posteriors.mean(), negative_log_likelihood

    def sample_elbo_lr(self, input, target, beta, samples, sigma=1.):
        ''' Sample ELBO for BNN w/ Local Reparameterisation '''
        assert self.local_reparam == True, 'sample_elbo_lr() method returns loss for BNNs with local reparameterisation, alternatively use sample_elbo()'
        kl_costs = torch.zeros(samples).to(input.device)
        negative_log_likelihood = torch.zeros(1).to(input.device)

        for i in range(samples):
            output = self.forward(input)
            # kl_costs[i] = self.kl_cost()
            negative_log_likelihood += self.get_nll(output, target, sigma)

        kl_cost = beta * self.kl_cost()
        negative_log_likelihood = negative_log_likelihood / samples
        loss = kl_cost + negative_log_likelihood
        return loss, kl_cost, negative_log_likelihood


def resnet18(model_params, num_classes=10):
    """ return a ResNet 18 object
    """
    return ResNetBNN(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, model_params = model_params)

def resnet34(model_params, num_classes=10):
    """ return a ResNet 34 object
    """
    return ResNetBNN(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, model_params = model_params)

def resnet50(model_params, num_classes=10):
    """ return a ResNet 50 object
    """
    return ResNetBNN(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, model_params = model_params)

def resnet101(model_params, num_classes=10):
    """ return a ResNet 101 object
    """
    return ResNetBNN(BottleNeck, [3, 4, 23, 3], num_classes=num_classes, model_params = model_params)

def resnet152(model_params, num_classes=10):
    """ return a ResNet 152 object
    """
    return ResNetBNN(BottleNeck, [3, 8, 36, 3], num_classes=num_classes, model_params = model_params)



