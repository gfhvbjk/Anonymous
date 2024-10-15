import torch
from torch.nn import Module, Parameter
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import math
import torch.nn as nn
class _BayesConvNd(Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']

    def __init__(self, prior_mu, prior_sigma, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias, padding_mode, bias_mu=0.0, prior=True):
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
        self.bias = bias

        if transposed:
            self.weight_mu = Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size))
            self.weight_sigma = Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size))
            self.register_buffer('weight_eps', None)
        else:
            self.weight_mu = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
            self.weight_sigma = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
            self.register_buffer('weight_eps', None)

        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_sigma = Parameter(torch.Tensor(out_channels))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)
            self.register_buffer('bias_eps', None)

        self.reset_parameters(prior)

    def reset_parameters(self, prior):
        nn.init.xavier_uniform_(self.weight_mu, gain=0.1)
        nn.init.xavier_uniform_(self.weight_sigma, gain=0.1)
        if self.bias:
            nn.init.normal_(self.bias_sigma, mean=0.0, std=0.0001)
            nn.init.normal_(self.bias_mu, mean=0.0, std=0.00001)

    def conv2d_forward(self, input, weight):
        if self.bias:
            bias = self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_sigma)
        else:
            bias = None

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'), weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input):
        weight = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_sigma)
        return self.conv2d_forward(input, weight)

class BayesConv2d(_BayesConvNd):
    def __init__(self, prior_mu, prior_sigma, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros', bias_mu=0.0, prior=True,
                 in_percentile_indices=None, out_percentile_indices=None, gmm_labels=None,
                 gmm_means=None, gmm_covariances_1d=None, membership_keys=None, membership_values=None, membership_lengths=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BayesConv2d, self).__init__(prior_mu, prior_sigma, in_channels, out_channels, kernel_size, stride,
                                          padding, dilation, False, _pair(0), groups, bias, padding_mode, bias_mu, prior)

        if in_percentile_indices is not None:
            self.register_buffer('in_percentile_indices', torch.tensor(in_percentile_indices, dtype=torch.long))
        if out_percentile_indices is not None:
            self.register_buffer('out_percentile_indices', torch.tensor(out_percentile_indices, dtype=torch.long))
        if gmm_labels is not None:
            self.register_buffer('gmm_labels', torch.tensor(gmm_labels, dtype=torch.long))
        if gmm_means is not None:
            self.register_parameter('gmm_means', nn.Parameter(torch.tensor(gmm_means, dtype=torch.float32)))
        if gmm_covariances_1d is not None:
            self.register_parameter('gmm_covariances_1d', nn.Parameter(torch.tensor(gmm_covariances_1d, dtype=torch.float32)))
        if membership_keys is not None:
            self.register_buffer('membership_keys', torch.tensor(membership_keys, dtype=torch.long))
        if membership_values is not None:
            self.register_buffer('membership_values', torch.tensor(membership_values, dtype=torch.long))
        if membership_lengths is not None:
            self.register_buffer('membership_lengths', torch.tensor(membership_lengths, dtype=torch.long))

class BayesLinear(Module):
    def __init__(self, prior_mu, prior_sigma, in_features, out_features, bias_mu=False, bias=True, prior=True,
                 in_percentile_indices=None, out_percentile_indices=None, gmm_labels=None,
                 gmm_means=None, gmm_covariances_1d=None, membership_keys=None, membership_values=None, membership_lengths=None):
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

        self.reset_parameters(prior)

        if in_percentile_indices is not None:
            self.register_buffer('in_percentile_indices', torch.tensor(in_percentile_indices, dtype=torch.long))
        if out_percentile_indices is not None:
            self.register_buffer('out_percentile_indices', torch.tensor(out_percentile_indices, dtype=torch.long))
        if gmm_labels is not None:
            self.register_buffer('gmm_labels', torch.tensor(gmm_labels, dtype=torch.long))
        if gmm_means is not None:
            self.register_parameter('gmm_means', nn.Parameter(torch.tensor(gmm_means, dtype=torch.float32)))
        if gmm_covariances_1d is not None:
            self.register_parameter('gmm_covariances_1d', nn.Parameter(torch.tensor(gmm_covariances_1d, dtype=torch.float32)))
        if membership_keys is not None:
            self.register_buffer('membership_keys', torch.tensor(membership_keys, dtype=torch.long))
        if membership_values is not None:
            self.register_buffer('membership_values', torch.tensor(membership_values, dtype=torch.long))
        if membership_lengths is not None:
            self.register_buffer('membership_lengths', torch.tensor(membership_lengths, dtype=torch.long))

    def reset_parameters(self, prior):
        nn.init.xavier_uniform_(self.weight_mu, gain=0.1)
        nn.init.xavier_uniform_(self.weight_sigma, gain=0.1)
        if self.bias:
            nn.init.normal_(self.bias_sigma, mean=0.0, std=0.0001)
            nn.init.normal_(self.bias_mu, mean=0.0, std=0.00001)

    def forward(self, input):
        weight = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_sigma)
        if self.bias:
            bias = self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_sigma)
        else:
            bias = None
        return F.linear(input, weight, bias)

