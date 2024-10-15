import torch
import torch.nn as nn
import json
import argparse
from torch.distributions import MultivariateNormal,Normal
import numpy as np
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, init_weights
import os
from torch.optim import Adam
from bnn.exp_bayesian_components import Convertor, BayesConv2d, BayesLinear
from utils import get_network
from conf import settings
from torch.nn.parameter import Parameter

from cifar10_utils import progress_bar, init_params
import torch.nn.functional as F
from models.resnet import VIModule

def create_multi_gaussian_distributions(means, covs):
    # 检查输入形状是否匹配
    assert means.shape[0] == covs.shape[0], "The number of means must match the number of covariance matrices"
    assert means.shape[1] == covs.shape[1], "The dimension of means must match the first dimension of covariance matrices"
    assert means.shape[1] == covs.shape[2], "The dimension of means must match the second dimension of covariance matrices"

    distributions = []
    for mean, cov in zip(means, covs):
        distributions.append(MultivariateNormal(mean, cov))
    return distributions

def sample_from_distributions(distributions):
    samples = []
    for dist in distributions:
        samples.append(dist.sample())
    return torch.stack(samples)

class MeanFieldGaussian2DConvolution(VIModule):
    """
    A Bayesian module that fits a posterior Gaussian distribution on a 2D convolution module with normal prior.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 wPriorSigma=1.,
                 bPriorSigma=1.,
                 initMeanZero=False,
                 initBiasMeanZero=False,
                 initPriorSigmaScale=0.01,
                 weights_mean=None,
                 weights_sigma=None,
                 bias_mean=None,
                 lbias_sigma=None,
                 in_percentile_indices=None,
                 out_percentile_indices=None,
                 gmm_labels=None,
                 gmm_means=None,
                 gmm_covariances_1d=None,
                 membership_keys=None,
                 distances=None,
                 membership_values=None,
                 membership_lengths=None,
                 ):

        super(MeanFieldGaussian2DConvolution, self).__init__()

        self.samples = {'out_weights': None, 'bias': None, 'sample_01': None, 'gmm_mu': None, 'gmm_sigma': None,
                        'bNoiseState': None, 'post_gmm': None}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias
        self.padding_mode = padding_mode

        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        self.register_buffer('in_percentile_indices',
                             torch.tensor(in_percentile_indices, dtype=torch.int32, device=device))
        self.register_buffer('out_percentile_indices',
                             torch.tensor(out_percentile_indices, dtype=torch.int32, device=device))
        self.register_buffer('gmm_labels', torch.tensor(gmm_labels, dtype=torch.long, device=device))
        self.register_parameter('gmm_means', nn.Parameter(torch.tensor(gmm_means, dtype=torch.float32, device=device)))
        self.register_parameter('gmm_covariances_1d',
                                nn.Parameter(torch.tensor(gmm_covariances_1d, dtype=torch.float32, device=device)))
        self.register_buffer('membership_keys', torch.tensor(membership_keys, dtype=torch.int32, device=device))
        self.register_buffer('membership_values', torch.tensor(membership_values, dtype=torch.int32, device=device))
        self.register_buffer('distances', torch.tensor(distances, dtype=torch.float32, device=device))

        self.register_buffer('membership_lengths', torch.tensor(membership_lengths, dtype=torch.int32, device=device))

        self.weights_mean_flatten = nn.Parameter(weights_mean.to(device=device).flatten(), requires_grad=True)
        self.weights_sigma_flatten = nn.Parameter(weights_sigma.to(device=device).flatten(), requires_grad=True)

        self.out_percentile_mu = nn.Parameter(self.weights_mean_flatten[self.out_percentile_indices].to(device=device))
        self.out_percentile_sigma = nn.Parameter(
            self.weights_sigma_flatten[self.out_percentile_indices].to(device=device))

        self.weight_shape = (out_channels, int(in_channels / groups), self.kernel_size[0], self.kernel_size[1])
        self.means = self.gmm_means

        self.covs = torch.tensor([
            [[cov[0], cov[-1]], [cov[-1], cov[1]]] for cov in self.gmm_covariances_1d
        ], dtype=torch.float32).to(device=device)

        self.post_distributions = MultivariateNormal(self.means, self.covs)

        self.unique_membership, self.indices_unique = torch.unique(self.membership_values.to(device), return_inverse=True, dim=0)

        self.normal_outlier = torch.distributions.Normal(
            torch.zeros(self.out_percentile_mu.shape, device=device),
            torch.ones(self.out_percentile_mu.shape, device=device))
        # out weights
        self.addLoss(lambda s: 0.5 * s.getSampledWeights().pow(2).sum() / wPriorSigma ** 2)
        self.addLoss(lambda s: -self.out_channels / 2 * np.log(2 * np.pi) - 0.5 * s.samples['sample_01'].pow(
            2).sum() - s.out_percentile_sigma.sum())

        self.addLoss(lambda s: 0.5 * self.gmm_means.pow(2).sum() / wPriorSigma ** 2)
        self.addLoss(lambda s: self.post_distributions.log_prob(self.weights_mean_flatten[self.in_percentile_indices]))

        if self.has_bias:
            self.bias_mean = nn.Parameter(bias_mean.to(device=device), requires_grad=True)
            self.lbias_sigma = nn.Parameter(lbias_sigma.to(device=device), requires_grad=True)
            self.noiseSourceBias = torch.distributions.Normal(torch.zeros(out_channels, device=device), torch.ones(out_channels, device=device))

            self.addLoss(lambda s: 0.5 * s.getSampledBias().pow(2).sum() / bPriorSigma ** 2)
            self.addLoss(lambda s: -self.out_channels / 2 * np.log(2 * np.pi) - 0.5 * s.samples['bNoiseState'].pow(
                2).sum() - self.lbias_sigma.sum())

    def sampleTransformCov(self, stochastic=True):
        samples = self.post_distributions.sample((1,)).squeeze()
        self.samples['post_gmm'] = samples
        means_1d = samples[:, 0]
        sigma_1d = torch.abs(samples[:, 1])
        distributions_1d = Normal(means_1d, sigma_1d)
        weights_sample = distributions_1d.sample((1,)).clone().detach().to(device).squeeze()

        # weights_sample = torch.tensor(distributions_1d.sample((1,)),device=device).squeeze() # number of gaussians
        # print(f"一维采样的shape {weights_sample.shape}")


        new_weights_mean_flatten = self.weights_mean_flatten.clone()
        for index, value in enumerate(self.unique_membership):

            indices_unique_member_cluster = self.indices_unique == index
            indices_unique_member_cluster = indices_unique_member_cluster.to(device=device)
            true_indices = torch.nonzero(indices_unique_member_cluster, as_tuple=False).to(device=device).flatten()
            value_unique_member = self.membership_values[indices_unique_member_cluster]

            value_unique_distance = self.distances[indices_unique_member_cluster]
            value_unique_member = value_unique_member.to(device=self.weights_mean_flatten.device)

            # print(f"value_unique_member,取出来的点的shape{value_unique_member.shape}")
            weight_sample_unique = weights_sample[value_unique_member]

            value_unique_distance = value_unique_distance.to(device=self.weights_mean_flatten.device)
            weight_sample_unique[value_unique_member == -1] = 0

            get_dot_value = value_unique_distance * weight_sample_unique
            weights_value = get_dot_value.sum(dim=1)

            new_weights_mean_flatten[true_indices] = weights_value

        self.weights_mean_flatten.data.copy_(new_weights_mean_flatten)

        self.samples['sample_01'] = self.normal_outlier.sample().to(device=self.weights_mean_flatten.device)
        self.samples["out_weights"] = self.out_percentile_mu + (
                torch.exp(self.out_percentile_sigma) * self.samples['sample_01'])

        self.weights_mean_flatten.data[self.out_percentile_indices] = self.samples["out_weights"]

        if self.has_bias:
            self.samples['bNoiseState'] = self.noiseSourceBias.sample().to(device=self.bias_mean.device)
            self.samples['bias'] = self.bias_mean + (
                torch.exp(self.lbias_sigma) * self.samples['bNoiseState'] if stochastic else 0)

    def getSampledWeights(self):
        return self.samples['out_weights']

    def getSampledBias(self):
        return self.samples['bias']

    def forward(self, x, stochastic=True):

        self.sampleTransformCov(stochastic=stochastic)
        if self.padding != 0 and self.padding != (0, 0):
            padkernel = (self.padding, self.padding, self.padding, self.padding) if isinstance(self.padding, int) else (
                self.padding[1], self.padding[1], self.padding[0], self.padding[0])
            mx = F.pad(x, padkernel, mode='constant', value=0)
        else:
            mx = x
        weight = self.weights_mean_flatten.view(self.weight_shape)
        return F.conv2d(mx,
                        weight,
                        bias=self.samples['bias'] if self.has_bias else None,
                        stride=self.stride,
                        padding=0,
                        dilation=self.dilation,
                        groups=self.groups)


class MeanFieldGaussianLinear(VIModule):
    """
    A Bayesian module that fits a posterior Gaussian distribution on a Linear module with normal prior.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 wPriorSigma=1.,
                 bPriorSigma=1.,
                 initMeanZero=False,
                 initBiasMeanZero=False,
                 initPriorSigmaScale=0.01,
                 weights_mean=None,
                 weights_sigma=None,
                 bias_mean=None,
                 lbias_sigma=None,
                 in_percentile_indices=None,
                 out_percentile_indices=None,
                 gmm_labels=None,
                 gmm_means=None,
                 gmm_covariances_1d=None,
                 membership_keys=None,
                 membership_values=None,
                 membership_lengths=None,
                 distances=None):

        super(MeanFieldGaussianLinear, self).__init__()

        self.samples = {'out_weights': None, 'bias': None, 'sample_01': None, 'gmm_mu': None, 'gmm_sigma': None,
                        'bNoiseState': None, 'post_gmm': None}
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.register_buffer('in_percentile_indices', torch.tensor(in_percentile_indices, dtype=torch.int32, device=device))
        self.register_buffer('out_percentile_indices', torch.tensor(out_percentile_indices, dtype=torch.int32, device=device))
        self.register_buffer('gmm_labels', torch.tensor(gmm_labels, dtype=torch.long, device=device))
        self.register_parameter('gmm_means', nn.Parameter(torch.tensor(gmm_means, dtype=torch.float32, device=device)))
        self.register_parameter('gmm_covariances_1d', nn.Parameter(torch.tensor(gmm_covariances_1d, dtype=torch.float32, device=device)))
        self.register_buffer('membership_keys', torch.tensor(membership_keys, dtype=torch.int32, device=device))
        self.register_buffer('membership_values', torch.tensor(membership_values, dtype=torch.int32, device=device))
        self.register_buffer('membership_lengths', torch.tensor(membership_lengths, dtype=torch.int32, device=device))
        self.register_buffer('distances', torch.tensor(distances, dtype=torch.float32, device=device))

        self.weights_mean_flatten = nn.Parameter(weights_mean.to(device=device).flatten(), requires_grad=True)
        self.weights_sigma_flatten = nn.Parameter(weights_sigma.to(device=device).flatten(), requires_grad=True)

        self.out_percentile_mu = nn.Parameter(self.weights_mean_flatten[self.out_percentile_indices].to(device=device))
        self.out_percentile_sigma = nn.Parameter(self.weights_sigma_flatten[self.out_percentile_indices].to(device=device))

        self.weight_shape = (out_features, in_features)
        self.means = self.gmm_means
        self.covs = torch.tensor([
            [[cov[0], cov[-1]], [cov[-1], cov[1]]] for cov in self.gmm_covariances_1d
        ], dtype=torch.float32).to(device=self.weights_mean_flatten.device)

        self.post_distributions = MultivariateNormal(self.means, self.covs)

        self.unique_membership, self.indices_unique = torch.unique(self.membership_values.to(device), return_inverse=True, dim=0)

        self.normal_outlier = torch.distributions.Normal(
            torch.zeros(self.out_percentile_mu.shape, device=device),
            torch.ones(self.out_percentile_mu.shape, device=device))

        self.addLoss(lambda s: 0.5 * s.getSampledWeights().pow(2).sum() / wPriorSigma ** 2)
        self.addLoss(lambda s: -1 / 2 * np.log(2 * np.pi) - 0.5 * s.samples['sample_01'].pow(2).sum() - s.out_percentile_sigma.sum())



        self.addLoss(lambda s: 0.5 * self.gmm_means.pow(2).sum() / wPriorSigma ** 2)
        self.addLoss(lambda s: self.post_distributions.log_prob(self.weights_mean_flatten[self.in_percentile_indices]))

        if self.has_bias:
            self.bias_mean = nn.Parameter(bias_mean.to(device=device), requires_grad=True)
            self.lbias_sigma = nn.Parameter(lbias_sigma.to(device=device), requires_grad=True)
            self.noiseSourceBias = torch.distributions.Normal(torch.zeros(out_features, device=device), torch.ones(out_features, device=device))

            self.addLoss(lambda s: 0.5 * s.getSampledBias().pow(2).sum() / bPriorSigma ** 2)
            self.addLoss(lambda s: -self.out_features / 2 * np.log(2 * np.pi) - 0.5 * s.samples['bNoiseState'].pow(2).sum() - self.lbias_sigma.sum())

    def sampleTransform(self, stochastic=True):
        samples = self.post_distributions.sample((1,)).squeeze()
        self.samples['post_gmm'] = samples
        means_1d = samples[:, 0]
        sigma_1d = torch.abs(samples[:, 1])
        distributions_1d = Normal(means_1d, sigma_1d)
        weights_sample = distributions_1d.sample((1,)).clone().detach().to(device).squeeze()

        new_weights_mean_flatten = self.weights_mean_flatten.clone()
        for index, value in enumerate(self.unique_membership):
            indices_unique_member_cluster = (self.indices_unique == index).to(device=device)
            true_indices = torch.nonzero(indices_unique_member_cluster, as_tuple=False).to(device=device).flatten()

            value_unique_member = self.membership_values[indices_unique_member_cluster]
            value_unique_distance = self.distances[indices_unique_member_cluster]
            value_unique_member = value_unique_member.to(device=self.weights_mean_flatten.device)
            value_unique_distance = value_unique_distance.to(device=self.weights_mean_flatten.device)

            weight_sample_unique = weights_sample[value_unique_member]

            weight_sample_unique[value_unique_member == -1] = 0
            # print(f"value_unique_distance {value_unique_distance.shape}")
            # print(f"value_unique_member {value_unique_member.shape}")
            get_dot_value = value_unique_distance * weight_sample_unique
            weights_value = get_dot_value.sum(dim=1)
            # print(f"indices_unique_member_cluster shape is {true_indices.shape}")
            # print(f"weights_value shape is {weights_value.shape}")
            new_weights_mean_flatten[true_indices] = weights_value

        self.weights_mean_flatten.data.copy_(new_weights_mean_flatten)

        self.samples['sample_01'] = self.normal_outlier.sample().to(device=self.weights_mean_flatten.device)
        self.samples["out_weights"] = self.out_percentile_mu + (
            torch.exp(self.out_percentile_sigma) * self.samples['sample_01'])

        self.weights_mean_flatten.data[self.out_percentile_indices] = self.samples["out_weights"]

        if self.has_bias:
            self.samples['bNoiseState'] = self.noiseSourceBias.sample().to(device=self.bias_mean.device)
            self.samples['bias'] = self.bias_mean + (
                torch.exp(self.lbias_sigma) * self.samples['bNoiseState'] if stochastic else 0)

    def getSampledWeights(self):
        return self.samples['out_weights']

    def getSampledBias(self):
        return self.samples['bias']

    def forward(self, x, stochastic=True):
        self.sampleTransform(stochastic=stochastic)
        weights = self.weights_mean_flatten.view(self.weight_shape)
        return nn.functional.linear(x, weights, self.samples['bias'] if self.has_bias else None)



def load_params_to_layer(params_path):
    with open(params_path, 'r') as f:
        params = json.load(f)
    return params


def convert_bayes_to_meanfield(net, module, name, params_path):
    """
    Recursively replace BayesConv2d and BayesLinear layers with MeanFieldGaussian layers after loading JSON params.
    """
    # for name, submodule in module.named_children():
    parent_module = net
    name_parts = name.split('.')

    for part in name_parts[:-1]:
        if not part.isdigit():
            parent_module = getattr(parent_module, part)
        elif part.isdigit():
            part = int(part)
            parent_module = parent_module[part]


    if isinstance(module, BayesConv2d):
        print(f"Loaded parameters and converted layer {params_path}")
            # Load JSON params to BayesConv2d layer
        params = load_params_to_layer(params_path)
        # Create a new MeanFieldGaussian2DConvolution layer with the same parameters
        new_layer = MeanFieldGaussian2DConvolution(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            initMeanZero=False,  # Or another appropriate initialization
            initBiasMeanZero=False,  # Or another appropriate initialization
            weights_mean=module.weight_mu,
            weights_sigma=module.weight_sigma,
            bias_mean=module.bias_mu,
            lbias_sigma=module.bias_sigma,
            in_percentile_indices=params['in_percentile_indices'],
            out_percentile_indices=params['out_percentile_indices'],
            gmm_labels=params['gmm_labels'],
            gmm_means=params['gmm_means'],
            gmm_covariances_1d=params['gmm_covariances_1d'],
            membership_keys=params['membership_keys'],
            distances=params['distances'],
            membership_values=params['membership_values'],
            membership_lengths=params['membership_lengths'],

        )

        # Replace the old layer with the new layer
        if not name_parts[-1].isdigit():
            setattr(parent_module, name_parts[-1], new_layer)
        elif name_parts[-1].isdigit():
            part = int(name_parts[-1])
            parent_module[part] = new_layer

    elif isinstance(module, BayesLinear):
            print(f"Loaded parameters and converted layer {params_path}")
            # Load JSON params to BayesLinear layer
            params = load_params_to_layer(params_path)
            # Create a new MeanFieldGaussianLinear layer with the same parameters
            new_layer = MeanFieldGaussianLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                initMeanZero=False,  # Or another appropriate initialization
                initBiasMeanZero=False,  # Or another appropriate initialization,
                weights_mean=module.weight_mu,
                weights_sigma=module.weight_sigma,
                bias_mean=module.bias_mu,
                lbias_sigma=module.bias_sigma,
                in_percentile_indices=params['in_percentile_indices'],
                out_percentile_indices=params['out_percentile_indices'],
                gmm_labels=params['gmm_labels'],
                gmm_means=params['gmm_means'],
                gmm_covariances_1d=params['gmm_covariances_1d'],
                membership_keys=params['membership_keys'],
                distances=params['distances'],
                membership_values=params['membership_values'],
                membership_lengths=params['membership_lengths']
            )

            if not name_parts[-1].isdigit():
                setattr(parent_module, name_parts[-1], new_layer)
            elif name_parts[-1].isdigit():
                part = int(name_parts[-1])
                parent_module[part] = new_layer




def test(cifar100_test_loader, models):
    with torch.no_grad():
        samples = torch.zeros((len(models), len(cifar100_test_loader), 100))
        total = 0
        correct = 0
        accuracy_all = []
        for i in np.arange(args.nruntests):
            model = np.random.randint(len(models))
            model = models[model]
            for batch_id, sampl in enumerate(cifar100_test_loader):
                print("\r", "\tTest run {}/{}".format(i + 1, args.nruntests), end="")
                images, labels = sampl
                images = images.cuda()
                labels = labels.cuda()
                pred = model(images)

                logprob = loss(pred, labels)
                l = images.shape[0] * logprob

                modelloss = model.evalAllLosses()
                l += modelloss

                _, predicted = torch.max(pred.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy_all.append(correct / total)
                samples[i, batch_id*images.shape[0]:(batch_id+1)*images.shape[0], :] = torch.exp(pred)

        print(f"The average accuracy is {np.mean(accuracy_all)}")
        print("")

        withinSampleMean = torch.mean(samples, dim=0)
        samplesMean = torch.mean(samples, dim=(0, 1))

        withinSampleStd = torch.sqrt(torch.mean(torch.var(samples, dim=0), dim=0))
        acrossSamplesStd = torch.std(withinSampleMean, dim=0)

        print("")
        print("Class prediction analysis:")
        print("\tMean class probabilities:")
        print(samplesMean)
        print("\tPrediction standard deviation per sample:")
        print(withinSampleStd)
        print("\tPrediction standard deviation across samples:")
        print(acrossSamplesStd)


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="resnet50", required=False, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-weights_path', type=str,
                        default="checkpoint/resnet50/exp_bnn_freeze_mu/resnet50-10-regular-0.7750999927520752.pth",
                        required=False)
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')

    parser.add_argument('--nepochs', type=int, default=20, help="The number of epochs to train for")

    parser.add_argument('--nruntests', type=int, default=20,
                        help="The number of pass to use at test time for monte-carlo uncertainty estimation")
    parser.add_argument('--numnetworks', type=int, default=8,
                        help="The number of networks to train to make an ensemble")

    args = parser.parse_args()

    # define network
    # original network -> stochastic network -> BNNs
    net = get_network(args)
    Convertor.orig_to_bayes(net, prior=True)

    if args.resume:
        weights_path = args.weights_path
        net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print("Weights loaded.")
    net = net.to(device=device)
    save_params_path = os.path.join("params", args.net)
    # bayes_layers = [[name, layer] for name, layer in net.named_modules()]
    # print(bayes_layers)
    bayes_conv_layers = [[name, layer] for name, layer in net.named_modules() if isinstance(layer, (BayesConv2d, BayesLinear))]
    # bayes_linear_layers = [[name, layer] for name, layer in net.named_modules() if isinstance(layer, BayesLinear)]
    bayes_conv_layers = bayes_conv_layers[0:5]
    # print(bayes_conv_layers)
    save_model_path = os.path.join("models", f"{args.net}_converted_model.pth")
    # if os.path.exists(save_model_path):
    #     # 加载整个模型
    #     net = torch.load(save_model_path, map_location=torch.device('cpu'))
    #     print(f"模型已从 {save_model_path} 加载")
    # else:
    # 逐层加载 JSON 参数并转换
    for i, bayes_conv_layer in enumerate(bayes_conv_layers):
        layer_idx = f'{i}'
        params_path = os.path.join(save_params_path, f'layer_{i + 1}_params.json')
        if os.path.exists(params_path):
            convert_bayes_to_meanfield(net, bayes_conv_layer[1], bayes_conv_layer[0], params_path)
    for name,module in net.named_modules():
        print(module)
    # save_model_path = os.path.join("models", f"{args.net}_converted_model.pth")
    # torch.save(net, save_model_path)

    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    # start training
    batchLen = len(cifar100_training_loader)
    digitsBatchLen = len(str(batchLen))

    models = []

    # Training or Loading
    if args.resume:
        pass
    else:

        for i in np.arange(args.numnetworks):
            print("Training model {}/{}:".format(i + 1, args.numnetworks))
            # Initialize the model
            model = net.to(device=device)  # p_mc_dropout=None will disable MC-Dropout for this bnn, as we found out it makes learning much much slower.
            loss = torch.nn.NLLLoss(reduction='mean')  # negative log likelihood will be part of the ELBO
            optimizer = Adam(model.parameters(), lr=args.lr)
            optimizer.zero_grad()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepochs)

            for n in np.arange(args.nepochs):
                total = 0
                correct = 0
                accuracy_all = []
                train_loss = 0
                for batch_id, sampl in enumerate(cifar100_training_loader):
                    images, labels = sampl
                    images = images.to(device=device)
                    labels = labels.to(device=device)

                    pred = model(images)

                    logprob = loss(pred, labels)
                    l = images.shape[0] * logprob

                    modelloss = model.evalAllLosses()
                    l += modelloss

                    optimizer.zero_grad()
                    l.backward()
                    train_loss += l.item()

                    optimizer.step()
                    _, predicted = torch.max(pred.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    accuracy_all.append(correct / total)

                    print("\r", ("\tEpoch {}/{}: Train step {" + (":0{}d".format(
                        digitsBatchLen)) + "}/{} prob = {:.4f} model = {:.4f} loss = {:.4f}          ").format(
                        n + 1, args.nepochs,
                        batch_id + 1,
                        batchLen,
                        torch.exp(-logprob.detach().cpu()).item(),
                        modelloss,
                        l.detach().cpu().item()), end="")

                scheduler.step()
                acc = np.mean(accuracy_all)
                checkpoint_path = os.path.join("BNNs", '{net}-{num}-{epoch}-{type}-{acc}.pth')

                weights_path = checkpoint_path.format(net=args.net, num = i,epoch=n, type='best', acc=acc)
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)
                best_acc = acc
            print("")
            models.append(model)
            test(cifar100_test_loader, models)
