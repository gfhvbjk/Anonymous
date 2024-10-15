import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from bnn.bayesian_components import Convertor, BayesConv2d, BayesLinear
from utils import get_network
import argparse
import numpy as np
import os
import json
import cudf
from cuml import KMeans
from networks.wide_resnet import WideResNet
def plot_kmeans_clusters_gpu(weight_mu_clipped, weight_sigma_exp, output_path):
    X_gpu = cudf.DataFrame({
        "weight_mu": weight_mu_clipped,
        "weight_sigma_exp": weight_sigma_exp
    })
    print("Shape of X_gpu:", X_gpu.shape)
    kmeans = KMeans(n_clusters=6000, random_state=0)
    kmeans.fit(X_gpu)
    labels = kmeans.labels_.values_host
    centers = kmeans.cluster_centers_.values_host

    plt.figure(figsize=(12, 8))
    for i in range(10):
        cluster_points = X_gpu.iloc[labels == i]
        plt.scatter(cluster_points['weight_mu'].values_host, cluster_points['weight_sigma_exp'].values_host,
                    label=f'Cluster {i + 1}', alpha=0.5)
    plt.title('K-means Clustering on GPU of Weights - First 10 Clusters')
    plt.xlabel('Weight Mu')
    plt.ylabel('Weight Sigma')
    plt.legend()
    plt.grid(True)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    first_fig_path = os.path.join(output_path, 'kmeans_first10clusters.png')
    plt.savefig(first_fig_path)
    plt.close()

    plt.figure(figsize=(15, 10))
    plt.scatter(X_gpu['weight_mu'].values_host, X_gpu['weight_sigma_exp'].values_host, c=labels, cmap='viridis',
                alpha=0.5, s=2)
    plt.title('K-means Clustering on GPU of Weights - All Clusters')
    plt.xlabel('Weight Mu')
    plt.ylabel('Weight Sigma')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    all_clusters_fig_path = os.path.join(output_path, 'kmeans.png')
    plt.savefig(all_clusters_fig_path)
    plt.close()

    results = {
        'labels': labels.tolist(),
        'centers': centers.tolist()
    }
    save_path = os.path.join(output_path, 'kmeans_clusters_gpu.json')
    with open(save_path, 'w') as f:
        print("start serialization")
        json.dump(results, f)
        print("end serialization")
    print(f"Results saved to {save_path}")

def plot_bayes_conv_layer_with_gmm_bootstrap(all_weights_mu, all_weights_sigma, output_path):
    weight_mu_clipped = all_weights_mu
    weight_sigma_exp = all_weights_sigma
    plot_kmeans_clusters_gpu(weight_mu_clipped, weight_sigma_exp, output_path)

def find_bayesconv2d_layers(module, bayesconv2d_layers=None):
    if bayesconv2d_layers is None:
        bayesconv2d_layers = []
    for name, submodule in module.named_children():
        if isinstance(submodule, BayesConv2d):
            bayesconv2d_layers.append(submodule)
        else:
            find_bayesconv2d_layers(submodule, bayesconv2d_layers)
    return bayesconv2d_layers

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="wrn20", required=False, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-output', type=str, default="checkpoint/wrn20/cifar100/init_gmm", help='output')
    parser.add_argument('-weights_path', type=str,
                        default="checkpoint/wrn20/cifar100/stochastic/stochastic_wrn.pth",
                        required=False)
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--layers', default=28, type=int,
                        help='total number of layers (default: 28)')
    parser.add_argument('--widen-factor', default=10, type=int,
                        help='widen factor (default: 10)')
    parser.add_argument('--droprate', default=0, type=float,
                        help='dropout probability (default: 0.0)')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='whether to use standard augmentation (default: True)')
    parser.add_argument('--name', default='WideResNet-28-10', type=str,
                        help='name of experiment')
    parser.add_argument('-dataset', type=str, default="cifar100", help='cifar10 or cifar100')
    parser.add_argument('-resume', action='store_true', default=True, help='resume training')
    args = parser.parse_args()
    net = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                     args.widen_factor, dropRate=args.droprate)
    Convertor.orig_to_bayes(net, prior=True)
    if args.resume:
        weights_path = args.weights_path
        checkpoint = torch.load(weights_path, map_location="cpu")["net"]
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 去掉 'module.' 前缀
            else:
                new_state_dict[k] = v
        net.load_state_dict(new_state_dict)
        print("Weights loaded.")
    print(net)
    bayes_conv_layers = [layer for layer in net.modules() if isinstance(layer, (BayesLinear, BayesConv2d))]
    weights_mu_concate = []
    weights_sigma_concate = []
    for i, bayes_conv_layer in enumerate(bayes_conv_layers):
        weight_mu_ = bayes_conv_layer.weight_mu.data.detach().cpu().numpy().flatten()
        weight_sigma_ = bayes_conv_layer.weight_sigma.data.detach().cpu().numpy().flatten()
        weights_mu_concate.append(weight_mu_)
        weights_sigma_concate.append(weight_sigma_)
    all_weights_mu = np.concatenate(weights_mu_concate)
    all_weights_sigma = np.concatenate(weights_sigma_concate)
    print(all_weights_mu.shape)
    plot_bayes_conv_layer_with_gmm_bootstrap(all_weights_mu, all_weights_sigma, args.output)