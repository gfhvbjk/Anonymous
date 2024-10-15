import torch
from torch.nn import Module, Parameter, Linear, Conv2d
from torch.nn.modules.utils import _single, _pair, _triple
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.stats import gamma
import torch.nn as nn
import math
from .util import transform_model,transform_layer
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
class _BayesConvNd(Module):
    r"""
    Applies Bayesian Convolution
    Arguments:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.
    .. note:: other arguments are following conv of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
    """
    __constants__ = ['prior_mu', 'prior_sigma', 'stride', 'padding', 'dilation',
                     'groups', 'bias', 'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, prior_mu, prior_sigma, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, bias_mu=0.0, prior=True):
        super(_BayesConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.prior_bias_mu = bias_mu
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = math.log(prior_sigma)
        self.bias = True

        if transposed:
            self.weight_mu = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.weight_sigma = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.register_buffer('weight_eps', None)
            print("啊啊啊啊啊啊啊啊啊")
        else:
            self.weight_mu = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.weight_sigma = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.register_buffer('weight_eps', None)

        nn.init.xavier_uniform_(self.weight_sigma.data)
        self.weight_sigma.data.mul_(0.01)

        nn.init.xavier_uniform(self.weight_mu.data)
        # self.reset_parameters()



        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_sigma = Parameter(torch.Tensor(out_channels))
            self.register_buffer('bias_eps', None)
            nn.init.uniform_(self.bias_mu.data, a=-1.0, b=1.0)
            nn.init.uniform_(self.bias_sigma.data, a=-1.0, b=1.0)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)
            self.register_buffer('bias_eps', None)

        assert torch.isfinite(self.weight_mu).all(), "weight_mu contains non-finite values!"
        assert torch.isfinite(self.weight_sigma).all(), "weight_sigma contains non-finite values!"
        # if self.bias:
        #     assert torch.isfinite(self.bias_mu).all(), "bias_mu contains non-finite values!"
        #     assert torch.isfinite(self.bias_sigma).all(), "bias_sigma contains non-finite values!"

        # self.reset_parameters(prior)

    # def reset_parameters(self):
    #     # Define 10 cluster centers
    #     num_clusters = 10
    #
    #     # Set cluster centers for mu to be more concentrated around 0
    #     cluster_centers_mu = torch.randn(num_clusters) * 0.5  # Normally distributed centers with small variance
    #
    #     # Set cluster centers for sigma to be uniformly distributed between 0 and 1
    #     cluster_centers_sigma = torch.rand(num_clusters) * 0.01
    #
    #     # Randomly choose a cluster for each output channel
    #     cluster_idx_mu = torch.randint(0, num_clusters, (self.weight_mu.shape[0], 1, 1, 1))
    #     cluster_idx_sigma = torch.randint(0, num_clusters, (self.weight_sigma.shape[0], 1, 1, 1))
    #
    #     # Expand the cluster centers to match the shape of the weights
    #     centers_mu = cluster_centers_mu[cluster_idx_mu].expand_as(self.weight_mu)
    #     centers_sigma = cluster_centers_sigma[cluster_idx_sigma].expand_as(self.weight_sigma)
    #
    #     # Initialize the weights to be clustered around the centers
    #     self.weight_mu.data = centers_mu + torch.randn_like(self.weight_mu) * 0.1
    #     self.weight_sigma.data = centers_sigma + torch.rand_like(self.weight_sigma) * 0.1
    #
    #     # Ensure weight_sigma values are in the range [0, 1]
    #     self.weight_sigma.data = torch.clamp(self.weight_sigma.data, 0.0, 1.0)
    # def nearest_power_of_two_for_each_weight_mu(self, weight, up):
    #     if up:
    #         return torch.sign(weight) * 2 ** torch.ceil(torch.log2(torch.abs(weight)))
    #     if not up:
    #         return torch.sign(weight) * 2 ** torch.floor(torch.log2(torch.abs(weight)))
    #
    # def relative_distance(self, mu, pow2):
    #     return torch.abs((mu - pow2) / mu)
    #
    # # def set_prior_to_relative_distance_to_nearest_power_of_two(self, weight):
    # #     pow2u = self.nearest_power_of_two_for_each_weight_mu(weight, up=True)
    # #     pow2d = self.nearest_power_of_two_for_each_weight_mu(weight, up=False)
    # #     rel_distu = self.relative_distance(weight, pow2u)
    # #     rel_distd = self.relative_distance(weight, pow2d)
    # #     rel_dist = (rel_distd * rel_distu)
    # #     dist = torch.clamp(0.0025 * (rel_dist / torch.quantile(rel_dist, 0.75)), min=2 ** -30, max=0.05)
    # #     return dist
    # def set_prior_to_relative_distance_to_nearest_power_of_two(self, weight):
    #     pow2u = self.nearest_power_of_two_for_each_weight_mu(weight, up=True)
    #     pow2d = self.nearest_power_of_two_for_each_weight_mu(weight, up=False)
    #     rel_distu = self.relative_distance(weight, pow2u)
    #     rel_distd = self.relative_distance(weight, pow2d)
    #     rel_dist = (rel_distd * rel_distu)
    #     dist = torch.clamp(0.0025 * (rel_dist / torch.quantile(rel_dist, 0.75)), min=2 ** -30, max=0.05)
    #     # noise = weight + torch.randn(weight.shape)
    #     return dist
    #
    # def reset_parameters2(self, prior):
    #     # Initialization method of Adv-BNN
    #     if isinstance(self.prior_mu, torch.Tensor) and torch.numel(self.prior_mu) > 1:
    #         self.weight_mu.data = self.prior_mu
    #     else:
    #         nn.init.normal_(self.weight_mu.data, mean=0.0, std=0.01)
    #
    #     self.weight_sigma = Parameter(self.set_prior_to_relative_distance_to_nearest_power_of_two(self.weight_mu.data))
    #
    #     if self.bias:
    #         self.bias_sigma = Parameter(self.set_prior_to_relative_distance_to_nearest_power_of_two(self.bias_mu.data))
    #     if not prior:
    #         self.weight_sigma.data.fill_(self.prior_sigma)
    #         if self.bias:
    #             self.bias_sigma.data.fill_(self.prior_sigma)
    #     if self.bias:
    #         if isinstance(self.prior_bias_mu, torch.Tensor) and torch.numel(self.prior_bias_mu) > 1:
    #             self.bias_mu.data = self.prior_bias_mu
    #         else:
    #             nn.init.normal_(self.bias_mu.data, mean=0.0, std=0.00001)
    #
    #     # 检查初始化后的值是否存在异常
    #     assert torch.isfinite(self.weight_mu).all(), "weight_mu contains non-finite values!"
    #     assert torch.isfinite(self.weight_sigma).all(), "weight_sigma contains non-finite values!"
    #     if self.bias:
    #         assert torch.isfinite(self.bias_mu).all(), "bias_mu contains non-finite values!"
    #         assert torch.isfinite(self.bias_sigma).all(), "bias_sigma contains non-finite values!"
    #
    # def reset_parameters2(self, prior):
    #     # Initialization method of Adv-BNN
    #     if isinstance(self.prior_mu, torch.Tensor) and torch.numel(self.prior_mu) > 1:
    #         self.weight_mu.data = self.prior_mu
    #     else:
    #         nn.init.normal_(self.weight_mu.data, mean=0.0, std=0.01)
    #
    #     self.weight_sigma = Parameter(self.set_prior_to_relative_distance_to_nearest_power_of_two(self.weight_mu.data))
    #
    #     if self.bias:
    #         self.bias_sigma = Parameter(self.set_prior_to_relative_distance_to_nearest_power_of_two(self.bias_mu.data))
    #     if not prior:
    #         self.weight_sigma.data.fill_(self.prior_sigma)
    #         if self.bias:
    #             self.bias_sigma.data.fill_(self.prior_sigma)
    #     if self.bias:
    #         if isinstance(self.prior_bias_mu, torch.Tensor) and torch.numel(self.prior_bias_mu) > 1:
    #             self.bias_mu.data = self.prior_bias_mu
    #         else:
    #             nn.init.normal_(self.bias_mu.data, mean=0.0, std=0.00001)
    #
    def set_sigmas(self):
        nn.init.xavier_uniform_(self.weight_sigma.data)
        self.weight_sigma.data.mul_(0.01)
        nn.init.xavier_uniform_(self.weight_mu.data)
        # self.weight_sigma = Parameter(self.set_prior_to_relative_distance_to_nearest_power_of_two(self.weight_mu.data))
        # if self.bias:
        #     self.bias_sigma = Parameter(self.set_prior_to_relative_distance_to_nearest_power_of_two(self.bias_mu.data))

    # def freeze(self):
    #     self.weight_eps = torch.zeros_like(self.weight_sigma)
    #     if self.bias:
    #         self.bias_eps = torch.zeros_like(self.bias_sigma)
    #
    # def unfreeze(self):
    #     self.weight_eps = None
    #     if self.bias:
    #         self.bias_eps = None
    #
    # def extra_repr(self):
    #     s = ('{prior_mu}, {prior_sigma}'
    #          ', {in_channels}, {out_channels}, kernel_size={kernel_size}'
    #          ', stride={stride}')
    #     if self.padding != (0,) * len(self.padding):
    #         s += ', padding={padding}'
    #     if self.dilation != (1,) * len(self.dilation):
    #         s += ', dilation={dilation}'
    #     if self.output_padding != (0,) * len(self.output_padding):
    #         s += ', output_padding={output_padding}'
    #     if self.groups != 1:
    #         s += ', groups={groups}'
    #     if self.bias is False:
    #         s += ', bias=False'
    #     return
    #
    def __setstate__(self, state):
        super(_BayesConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class BayesConv2d(_BayesConvNd):
    r"""
    Applies Bayesian Convolution for 2D inputs
    Arguments:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.
    .. note:: other arguments are following conv of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py

    """

    def __init__(self, prior_mu, prior_sigma, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros', bias_mu=0.0, prior=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BayesConv2d, self).__init__(
            prior_mu, prior_sigma, in_channels, out_channels, kernel_size, stride,
            padding, dilation, False, _pair(0), groups, bias, padding_mode, bias_mu, prior=prior)

    def conv2d_forward(self, input, weight):
        if self.bias:
            self.bias_sigma.eps = Variable(torch.randn_like(self.bias_sigma))
            std = torch.clamp(self.bias_sigma, min=0.0001)
            bias = self.bias_mu * torch.log(0.01+torch.exp(std))
        else:
            bias = None

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        r"""
        Overriden.
        """
        # # # if self.weight_eps is None:
        # self.weight_sigma.eps = Variable(torch.randn_like(self.weight_sigma))
        # std = torch.clamp(self.weight_sigma, min=2 ** -30)
        # weight = self.weight_mu + self.weight_sigma.eps * std

        eps = torch.randn_like(self.weight_sigma)

        if torch.isnan(self.weight_sigma).all():
            raise ValueError("self.weight_sigmag have NaNs")
        # Ensure std deviation is not too small
        std = torch.clamp(self.weight_sigma, min=0.001)
        # print(torch.max(std))
        # print("adadsasds----")
        # Calculate weight
        weight = self.weight_mu + eps * torch.log(0.01+torch.exp(std))

        # Check for NaNs
        if torch.isnan(weight).all():
            raise ValueError("Weights have NaNs")

        # print(f"weight is {weight}")
        # else:
        #     self.weight_sigma.eps = Variable(torch.zeros_like(self.weight_sigma))
        #     weight = self.weight_mu
        return self.conv2d_forward(input, weight)


# %%

class BayesLinear(Module):
    r"""
    Applies Bayesian Linear
    Arguments:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.
    .. note:: other arguments are following linear of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py

    """
    __constants__ = ['prior_mu', 'prior_sigma', 'bias', 'in_features', 'out_features']

    def __init__(self, prior_mu, prior_sigma, in_features, out_features, bias_mu=False, bias=True, prior=True):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.prior_mu = prior_mu
        self.prior_bias_mu = bias_mu
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = math.log(prior_sigma)

        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_eps', None)

        if bias is None or bias is False:
            self.bias = False
        else:
            self.bias = True

        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_sigma = Parameter(torch.Tensor(out_features))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)
            self.register_buffer('bias_eps', None)
        nn.init.xavier_uniform_(self.weight_sigma.data)
        self.weight_sigma.data.mul_(0.01)

        nn.init.xavier_uniform(self.weight_mu.data)
        # self.reset_parameters(prior)

    def nearest_power_of_two_for_each_weight_mu(self, weight, up):
        if up:
            return torch.sign(weight) * 2 ** torch.ceil(torch.log2(torch.abs(weight)))
        if not up:
            return torch.sign(weight) * 2 ** torch.floor(torch.log2(torch.abs(weight)))

    def relative_distance(self, mu, pow2):
        return torch.abs((mu - pow2) / mu)

    def set_prior_to_relative_distance_to_nearest_power_of_two(self, weight):
        pow2u = self.nearest_power_of_two_for_each_weight_mu(weight, up=True)
        pow2d = self.nearest_power_of_two_for_each_weight_mu(weight, up=False)
        rel_distu = self.relative_distance(weight, pow2u)
        rel_distd = self.relative_distance(weight, pow2d)
        rel_dist = (rel_distd * rel_distu)
        dist = torch.clamp(0.0025 * (rel_dist / torch.quantile(rel_dist, 0.75)), min=2 ** -30, max=0.05)
        return dist

    # def reset_parameters(self, prior):
    #     # Initialization method of Adv-BNN
    #     if isinstance(self.prior_mu, torch.Tensor) and torch.numel(self.prior_mu) > 1:
    #         self.weight_mu.data = self.prior_mu
    #     else:
    #         nn.init.xavier_uniform_(self.weight_mu.data)
    #
    #     self.weight_sigma = Parameter(self.set_prior_to_relative_distance_to_nearest_power_of_two(self.weight_mu.data))
    #     if self.bias:
    #         self.bias_sigma = Parameter(self.set_prior_to_relative_distance_to_nearest_power_of_two(self.bias_mu.data))
    #     if not prior:
    #         self.weight_sigma.data.fill_(self.prior_sigma)
    #         if self.bias:
    #             self.bias_sigma.data.fill_(self.prior_sigma)
    #     if self.bias:
    #         if isinstance(self.prior_bias_mu, torch.Tensor) and torch.numel(self.prior_bias_mu) > 1:
    #             self.bias_mu.data = self.prior_bias_mu
    #         else:
    #             nn.init.xavier_uniform_(self.bias_mu.data)

    def set_sigmas(self):
        nn.init.xavier_uniform_(self.weight_sigma.data)
        self.weight_sigma.data.mul_(0.01)
        nn.init.xavier_uniform_(self.weight_mu.data)
        # self.weight_sigma = Parameter(self.set_prior_to_relative_distance_to_nearest_power_of_two(self.weight_mu.data))
        # if self.bias:
        #     self.bias_sigma = Parameter(self.set_prior_to_relative_distance_to_nearest_power_of_two(self.bias_mu.data))

    def freeze(self):
        self.weight_eps = torch.zeros_like(self.weight_sigma)
        if self.bias:
            self.bias_eps = torch.zeros_like(self.bias_sigma)

    def unfreeze(self):
        self.weight_eps = None
        if self.bias:
            self.bias_eps = None

    def forward(self, input):
        r"""
        Overriden.
        """
        if self.weight_eps is None:
            self.weight_sigma.eps = Variable(torch.randn_like(self.weight_sigma))
            std = torch.clamp(self.weight_sigma, min=2 ** -30)
            weight = self.weight_mu
        else:
            self.weight_sigma.eps = Variable(torch.zeros_like(self.weight_sigma))
            weight = self.weight_mu

        if self.bias:
            if self.bias_eps is None:
                self.bias_sigma.eps = Variable(torch.randn_like(self.bias_sigma))
                std = torch.clamp(self.bias_sigma, min=2 ** -30)
                bias = self.bias_mu
            else:
                self.bias_sigma.eps = Variable(torch.zeros_like(self.bias_sigma))
                bias = self.bias_mu
        else:
            bias = None
        out = F.linear(input, weight, bias)
        return out

    def extra_repr(self):
        r"""
        Overriden.
        """
        return 'prior_mu={}, prior_sigma={}, in_features={}, out_features={}, bias={}'.format(self.prior_mu,
                                                                                              self.prior_sigma,
                                                                                              self.in_features,
                                                                                              self.out_features,
                                                                                              self.bias is not None)


class Convertor():
    def __init__(self):
        pass

    @staticmethod
    def bayes_to_orig(model, prior):
        transform_model(model, BayesConv2d, nn.Conv2d,
                        args={"in_channels": ".in_channels", "out_channels": ".out_channels",
                              "kernel_size": ".kernel_size",
                              "padding": ".padding", "bias": ".bias"
                              },
                        attrs={"weight": ".weight_mu"})

        transform_model(model, BayesLinear, nn.Linear,
                        args={"in_features": ".in_features", "out_features": ".out_features",
                              "bias": ".bias"
                              },
                        attrs={"weight": ".weight_mu"})

    @staticmethod
    def orig_to_bayes(model, prior):
        transform_model(model, Conv2d, BayesConv2d,
                        args={"prior_mu": ".weight", "prior_sigma": 0.0001, "in_channels": ".in_channels",
                              "out_channels": ".out_channels", "kernel_size": ".kernel_size",
                              "stride": ".stride", "padding": ".padding", "bias_mu": '.bias', "bias": ".bias",
                              "prior": prior,
                              })

        transform_model(model, Linear, BayesLinear,
                        args={"prior_mu": '.weight', "prior_sigma": 0.0001, "in_features": ".in_features",
                              "out_features": ".out_features", "bias_mu": '.bias', "bias": ".bias", "prior": prior,
                              }, )
        for m in model.modules():
            if isinstance(m, BayesConv2d) or isinstance(m, BayesLinear):
                m.set_sigmas()

