import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from bnn.bayesian_components import Convertor, BayesConv2d,BayesLinear
from utils import get_network
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.utils import resample
import math


def plot_gmm_clusters_bootstrap(weight_mu_clipped, weight_sigma_exp, in_percentile_region, mu_1st_percentile, mu_99th_percentile, sigma_1st_percentile, sigma_99th_percentile):
    X = np.vstack((weight_mu_clipped[in_percentile_region], weight_sigma_exp[in_percentile_region])).T

    num_parameters = X.shape[0]
    n_components_range = range(int((num_parameters)/20), int((num_parameters)/20) + 20)

    # Original GMM
    bic_scores = []
    gmm_models = []

    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, covariance_type='full').fit(X)
        bic_scores.append(gmm.bic(X))
        gmm_models.append(gmm)

    best_n_components = n_components_range[np.argmin(bic_scores)]
    best_gmm = gmm_models[np.argmin(bic_scores)]

    print(f"Optimal number of GMM components for original data: {best_n_components}")
    # Plot GMM clusters for original data
    plt.figure(figsize=(10, 6))
    labels = best_gmm.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5, label='GMM Clusters (Original)')
    plt.title('Clustering of weight_mu vs weight_sigma_exp (Original Data)')
    plt.xlabel('weight_mu')
    plt.ylabel('weight_sigma_exp')
    print("aaa")
    print(labels)
    print("aaa")

    # Plot ellipses representing the 2D Gaussian distributions for original data
    colors = plt.cm.viridis(np.linspace(0, 1, best_n_components))
    for i, (mean, covar, color) in enumerate(zip(best_gmm.means_, best_gmm.covariances_, colors)):
        eigenvalues, eigenvectors = np.linalg.eigh(covar)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])) # Use correct eigenvector components
        print(angle)
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor=color, fc='None', lw=2)
        plt.gca().add_patch(ellipse)
        # Plot a second contour at a different scale
        ellipse2 = Ellipse(xy=mean, width=width * 2, height=height * 2, angle=angle, edgecolor=color, fc='None', lw=1,linestyle='--')
        plt.gca().add_patch(ellipse2)
        # Plot center points
        plt.plot(mean[0], mean[1], 'o', color=color, markersize=2)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_bayes_conv_layer_with_gmm_bootstrap(bayes_conv_layer):
    # Extract the parameters
    weight_mu = bayes_conv_layer.weight_mu.data.detach().cpu().numpy().flatten()
    weight_sigma = bayes_conv_layer.weight_sigma.data.detach().cpu().numpy().flatten()

    # Apply transformations
    weight_mu_clipped = np.clip(weight_mu, a_min=-0.4, a_max=0.6)
    weight_sigma_exp = np.log(0.04 + np.exp(weight_sigma))

    # Calculate 1% and 99% percentiles for weight_mu and weight_sigma_exp
    mu_1st_percentile = np.percentile(weight_mu_clipped, 1)
    mu_99th_percentile = np.percentile(weight_mu_clipped, 99)
    sigma_1st_percentile = np.percentile(weight_sigma_exp, 1)
    sigma_99th_percentile = np.percentile(weight_sigma_exp, 99)

    print(f"1% percentile of weight_mu: {mu_1st_percentile}")
    print(f"99% percentile of weight_mu: {mu_99th_percentile}")
    print(f"1% percentile of weight_sigma_exp: {sigma_1st_percentile}")
    print(f"99% percentile of weight_sigma_exp: {sigma_99th_percentile}")

    # Calculate the number of points within the 1% and 99% percentiles rectangle
    in_percentile_region = ((weight_mu_clipped >= mu_1st_percentile) & (weight_mu_clipped <= mu_99th_percentile) &
                            (weight_sigma_exp >= sigma_1st_percentile) & (weight_sigma_exp <= sigma_99th_percentile))
    points_in_percentile_region = np.sum(in_percentile_region)
    total_points = len(weight_mu_clipped)

    area_ratio = points_in_percentile_region / total_points * 100

    print(f"Area ratio between 1% and 99% percentiles: {area_ratio:.2f}%")

    # Plot the parameters and the rectangle
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    # Plot all points
    axs[0].scatter(weight_mu_clipped, weight_sigma_exp, alpha=0.5, label='BayesConv2d Parameters')
    axs[0].axvline(mu_1st_percentile, color='r', linestyle=':', label='1st Percentile (weight_mu)')
    axs[0].axvline(mu_99th_percentile, color='r', linestyle='--', label='99th Percentile (weight_mu)')
    axs[0].axhline(sigma_1st_percentile, color='b', linestyle=':', label='1st Percentile (weight_sigma_exp)')
    axs[0].axhline(sigma_99th_percentile, color='b', linestyle='--', label='99th Percentile (weight_sigma_exp)')
    axs[0].set_title('Scatter Plot of weight_mu vs weight_sigma_exp (All Points)')
    axs[0].set_xlabel('weight_mu')
    axs[0].set_ylabel('weight_sigma_exp')
    axs[0].legend()
    axs[0].grid(True)

    # Plot points within the 1% and 99% percentiles rectangle
    axs[1].scatter(weight_mu_clipped[in_percentile_region], weight_sigma_exp[in_percentile_region], alpha=0.5, label='BayesConv2d Parameters (1%-99%)')
    axs[1].set_title('Scatter Plot of weight_mu vs weight_sigma_exp (1%-99% Points)')
    axs[1].set_xlabel('weight_mu')
    axs[1].set_ylabel('weight_sigma_exp')
    axs[1].legend()
    axs[1].grid(True)

    plt.show()

    # Call the function to plot GMM clusters separately
    plot_gmm_clusters_bootstrap(weight_mu_clipped, weight_sigma_exp, in_percentile_region, mu_1st_percentile, mu_99th_percentile, sigma_1st_percentile, sigma_99th_percentile)

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="resnet50", required=False, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=True, help='resume training')
    args = parser.parse_args()

    net = get_network(args)
    Convertor.orig_to_bayes(net, prior=True)

    if args.resume:
        weights_path = "checkpoints/exp_bnn_freeze_mu_0.1/resnet50-10-regular-0.7750999927520752.pth"
        net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print("Weights loaded.")


    # print(bayes_conv_layers)
    # Iterate over BayesConv2d layers and plot their parameters
    bayes_conv_layers = [layer for layer in net.modules() if isinstance(layer, BayesConv2d)]
    for i, bayes_conv_layer in enumerate(bayes_conv_layers):
        plot_bayes_conv_layer_with_gmm_bootstrap(bayes_conv_layer)
