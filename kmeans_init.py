import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from bnn.bayesian_components import Convertor, BayesConv2d, BayesLinear
from utils import get_network
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
import os
import json
from torch.distributions.normal import Normal
import cudf
from cuml import KMeans
import numpy as np
import matplotlib.pyplot as plt

def plot_kmeans_clusters_gpu(weight_mu_clipped, weight_sigma_exp):
    # 创建 GPU DataFrame
    X_gpu = cudf.DataFrame({
        "weight_mu": weight_mu_clipped,
        "weight_sigma_exp": weight_sigma_exp
    })
    print("Shape of X_gpu:", X_gpu.shape)
    # 初始化 KMeans，设置聚类数为500，使用 GPU
    kmeans = KMeans(n_clusters=256, random_state=0)
    kmeans.fit(X_gpu)

    # 获取聚类标签和中心
    labels = kmeans.labels_.values_host
    centers = kmeans.cluster_centers_.values_host
    # 可视化前5000个点
    plt.figure(figsize=(12, 8))
    # 筛选前10个簇的数据点进行绘制
    for i in range(10):
        # 筛选属于当前簇的点
        cluster_points = X_gpu.iloc[labels == i]
        plt.scatter(cluster_points['weight_mu'].values_host, cluster_points['weight_sigma_exp'].values_host,
                    label=f'Cluster {i + 1}', alpha=0.5)

    # 突出显示前十个簇的中心点
    plt.figure(figsize=(15, 10))
    plt.scatter(X_gpu['weight_mu'].values_host, X_gpu['weight_sigma_exp'].values_host, c=labels, cmap='viridis',
                alpha=0.5, s=2)
    # plt.scatter(centers[:, 0], centers[:, 1], c='red', s=5, marker='x', label='Centers')
    plt.title('K-means Clustering on GPU of Weights - All Clusters')
    plt.xlabel('Weight Mu')
    plt.ylabel('Weight Sigma')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    if not os.path.exists('output'):
        os.makedirs('output')
    plt.savefig('output/kmeans_clusters_256.png')
    plt.close()

    # 将结果保存为 JSON
    results = {
        'labels': labels.tolist(),
        'centers': centers.tolist()
    }
    os.makedirs('output', exist_ok=True)
    with open('output/kmeans_results_gpu3_256.json', 'w') as f:
        json.dump(results, f)

    print("Results saved to 'output/kmeans_results_gpu3_256.json'.")



def plot_bayes_conv_layer_with_gmm_bootstrap(all_weights_mu, all_weights_sigma):
    # Extract the parameters
    # weight_mu = bayes_conv_layer.weight_mu.data.detach().cpu().numpy().flatten()
    # weight_sigma = bayes_conv_layer.weight_sigma.data.detach().cpu().numpy().flatten()

    # Apply transformations
    weight_mu_clipped = all_weights_mu
    weight_sigma_exp = all_weights_sigma

    # data = json.load(open('models.json'))
    # in_percentile_indices = data["in_percentile_indices"]
    # out_percentile_indices = data["out_percentile_indices"]

    # Call the function to plot GMM clusters separately and get the GMM parameters
    plot_kmeans_clusters_gpu(weight_mu_clipped, weight_sigma_exp)


    # Save parameters to a file
    # params = {
    #     'in_percentile_indices': in_percentile_indices,
    #     'out_percentile_indices': out_percentile_indices,
    #     'gmm_labels': labels_all.tolist(),
    #     'gmm_means': means_all.tolist(),
    #     'gmm_covariances_1d': covariances_1d_all.tolist(),
    #     'probability_values': probability_values_all.tolist(),
    # }
    # with open("params/resnet18.json", 'w') as f:
    #     json.dump(params, f)
    # print(f"Parameters saved to {params/resnet.json}")


def find_bayesconv2d_layers(module, bayesconv2d_layers=None):
    if bayesconv2d_layers is None:
        bayesconv2d_layers = []  # 存储找到的 BayesConv2d 层

    for name, submodule in module.named_children():
        if isinstance(submodule, BayesConv2d):
            bayesconv2d_layers.append(submodule)
        else:
            find_bayesconv2d_layers(submodule, bayesconv2d_layers)

    return bayesconv2d_layers


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="resnet18", required=False, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    # parser.add_argument('-weights_path', type=str,
    #                     default="checkpoint/resnet18/exp_bnn_resnet18_5/resnet18-99-best-0.7755999565124512.pth",
    #                     required=False)
    # parser.add_argument('-weights_path', type=str,
    #                     default="checkpoint/resnet50/exp_bnn_0.003/resnet50-31-best-0.770799994468689.pth",
    #                     required=False) 11210432
    parser.add_argument('-weights_path', type=str,
                        default="checkpoint/resnet18/exp_822/resnet18-21-best-0.7736999988555908.pth",
                        required=False)

    parser.add_argument('-resume', action='store_true', default=True, help='resume training')
    args = parser.parse_args()

    net = get_network(args)
    Convertor.orig_to_bayes(net, prior=True)

    if args.resume:
        if not os.path.exists(os.path.join("figs3", args.net)):
            os.makedirs(os.path.join("figs3", args.net))
        weights_path = args.weights_path
        net.load_state_dict(torch.load(weights_path, map_location="cpu")["net"])
        print("Weights loaded.")
    print(net)
    # Iterate over BayesConv2d layers and plot their parameters
    bayes_conv_layers = [layer for layer in net.modules() if isinstance(layer, (BayesLinear, BayesConv2d))]


    figs_path = os.path.join("figs3", args.net)
    os.makedirs(figs_path, exist_ok=True)
    save_params_path = os.path.join("params", args.net)
    os.makedirs(save_params_path, exist_ok=True)
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
    # plt.figure(figsize=(10, 6))
    # plt.scatter(all_weights_mu, all_weights_sigma, alpha=0.5)
    # plt.xlabel('Weight Mu')
    # plt.ylabel('Weight Sigma')
    # plt.title('Scatter Plot of Weight Mu vs Weight Sigma')
    # plt.grid(True)
    # plt.show()
    plot_bayes_conv_layer_with_gmm_bootstrap(all_weights_mu, all_weights_sigma)

