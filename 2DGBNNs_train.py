import torch
import torch.nn as nn
import json
import copy
from pyro.infer import Predictive
import argparse
import gc
import pyro.distributions.constraints as constraints
import numpy as np
import torch.nn.functional as F
from pyro import poutine
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, init_weights
import os
# import cudf
# from cuml import KMeans
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import PyroOptim
import torch.optim.lr_scheduler as lr_scheduler
import pyro.distributions as dist
from pyro.optim import Adam, ClippedAdam, PyroLRScheduler,AdamW,SGD
from bnn.Bayes_Modules import Convertor, BayesConv2d, BayesLinear
from utils import get_network
from conf import settings
from pyro.ops.indexing import Vindex
import dill
from matplotlib.patches import Ellipse
from torch.nn.parameter import Parameter
from pyro.nn import PyroModule, PyroSample
from pyro.nn.module import to_pyro_module_
import pyro
from collections import defaultdict
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
import time
from torch.utils.data import Dataset, DataLoader, Sampler
import dill
import matplotlib.pyplot as plt
from gmm_pyro_cuda.pyro_gmm import GMMGPU
from torch.distributions import MultivariateNormal
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

from networks.wide_resnet import WideResNet
import torch.optim as optim
from bnn.exp_bayesian_components import Convertor, BayesConv2d, BayesLinear
@torch.no_grad()
def model_test_predictive(model, guide, test_loader, num_samples=30):
    predictive = Predictive(model, guide=guide, num_samples=num_samples, return_sites=("obs", "_RETURN"))

    correct = 0
    total = 0
    all_labels = []
    all_mean_preds = []
    all_outputs = []
    i = 0
    # 分批次处理测试数据
    for images, labels in test_loader:
        print(i)
        i = i+1
        images, labels = images.to(device), labels.to(device)
        sampled_models = predictive(images)  # 运行预测
        outputs = sampled_models['obs']  # 获取输出
        mean_preds = torch.mean(sampled_models['_RETURN'], dim=0)  # 取样本的平均结果作为最终输出

        mode_predictions = torch.mode(outputs, dim=0)[0]  # 取众数作为最终预测
        total += labels.size(0)
        correct += (mode_predictions == labels).sum().item()

        # 收集所有预测和标签
        all_labels.append(labels)
        all_mean_preds.append(mean_preds)
        all_outputs.append(outputs)

    # 计算总体准确率
    accuracy = 100 * correct / total
    print(f'Accuracy on CIFAR-100 test set: {accuracy}%')

    # 合并所有批次的结果
    all_labels = torch.cat(all_labels)
    all_mean_preds = torch.cat(all_mean_preds)
    all_outputs = torch.cat(all_outputs, dim=1)
    log_soft = F.log_softmax(all_mean_preds, dim=1)
    # 计算 NLL
    nll = F.nll_loss(log_soft, all_labels).item()
    print(f"NLL: {nll}")

    # 计算 ECE
    ece = compute_ece(all_mean_preds, all_labels)
    print(f"ECE: {ece}")

    model.model_cnn.train()

    return accuracy, nll, ece
def eval_training(model_cnn, epoch=0):

    start = time.time()
    model_cnn.eval()
    loss_function = nn.CrossEntropyLoss()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.to(device)
            labels = labels.to(device)

        outputs = model_cnn(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))

# ECE计算函数
def compute_ece(preds, labels, n_bins=15):
    softmaxes = F.softmax(preds, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=preds.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

def ellipse_outlier(centers, labels, weight_mu, weight_sigma):
    # device = weight_mu.device  # Ensure the device is correctly set
    data = torch.stack((weight_mu, weight_sigma), dim=1).to(device)
    print(data.device)
    print(inliers_indices.device)
    print("check")
    data_inliers = data[inliers_indices]
    data_inliers = data_inliers.to(device)
    ellipse_outdices = []
    filterd_labels_inliers = labels[inliers_indices]
    cov_matrices = torch.zeros((centers.shape[0], data.shape[1], data.shape[1]), device=device)

    for i in range(centers.shape[0]):
        cluster_mask = torch.where(filterd_labels_inliers == i)[0].to(device)
        cluster_data = data_inliers[cluster_mask]

        if cluster_data.shape[0] < 2:
            # Handle cases where there are not enough points to compute covariance
            cov_matrix = torch.eye(data.shape[1], device=device) * 1e-6
        else:
            cov_matrix = torch.cov(cluster_data.T)

        # Ensure the covariance matrix is positive definite
        cov_matrices[i] = torch.linalg.cholesky(cov_matrix + 1e-6 * torch.eye(cov_matrix.shape[0], device=device))

        # Inverse of the covariance matrix
        cov_matrix_inv = torch.inverse(cov_matrix + 1e-6 * torch.eye(cov_matrix.shape[0], device=device))

        # Mean of the current cluster
        mean = centers[i]

        # Compute Mahalanobis distance
        diff = cluster_data - mean
        ellipse_value = torch.sum(diff @ cov_matrix_inv * diff, dim=1)

        # Identify points outside the ellipse
        outside_ellipse = ellipse_value > 5.991  # Chi-squared value for 2 DOF and 95% confidence
        ellipse_outdices.extend(cluster_mask[outside_ellipse].tolist())

    print("Identified outlier points")

    # Prepare dictionaries for outlier indices
    ellipse_outdices_dict = {idx: [] for idx in ellipse_outdices}
    ellipse_outdices_value_dict = {idx: [] for idx in ellipse_outdices}

    k = 2
    n_points = len(ellipse_outdices)
    print("Processing outlier points")

    # Data for outlier points
    n_points_data = data[ellipse_outdices]  # Shape: [n_points, D]
    n_gaussians = centers.shape[0]
    ellipse_outdices = torch.tensor(ellipse_outdices, device=device)

    # Number of nearest Gaussians to consider
    num_keep = 5
    centers = centers.to(torch.float64)
    n_points_data = n_points_data.to(torch.float64)
    # Compute distances to all centers and find nearest Gaussians
    distance_n_points_centers = torch.cdist(n_points_data, centers)  # Shape: [n_points, n_gaussians]
    nearest_gaussians = torch.topk(distance_n_points_centers, k=num_keep, largest=False).indices  # Shape: [n_points, num_keep]

    # Prepare parameters for batched MultivariateNormal
    locs = centers[nearest_gaussians]  # Shape: [n_points, num_keep, D]
    scale_trils = cov_matrices[nearest_gaussians]  # Shape: [n_points, num_keep, D, D]

    # Expand data to match the shape of locs
    expanded_data = n_points_data.unsqueeze(1).expand(-1, num_keep, -1)  # Shape: [n_points, num_keep, D]

    # Flatten the batch dimensions
    flat_locs = locs.reshape(-1, locs.shape[-1])  # Shape: [n_points * num_keep, D]
    flat_scale_trils = scale_trils.reshape(-1, scale_trils.shape[-2], scale_trils.shape[-1])  # Shape: [n_points * num_keep, D, D]
    flat_data = expanded_data.reshape(-1, expanded_data.shape[-1])  # Shape: [n_points * num_keep, D]

    # Create batched MultivariateNormal
    mvn = MultivariateNormal(loc=flat_locs, scale_tril=flat_scale_trils)

    # Compute log probabilities
    flat_log_probs = mvn.log_prob(flat_data)  # Shape: [n_points * num_keep]

    # Reshape back to [n_points, num_keep]
    log_probs = flat_log_probs.view(n_points, num_keep)

    # Find top k log probabilities and their indices
    top_values, top_indices = torch.topk(log_probs, k, dim=1)  # Shape: [n_points, k]
    top_indices_gather = nearest_gaussians.gather(1, top_indices)  # Shape: [n_points, k]

    # Update dictionaries with top indices and their log probabilities
    for idx, out_index in enumerate(ellipse_outdices.tolist()):
        ellipse_outdices_dict[out_index].extend(top_indices_gather[idx].tolist())
        ellipse_outdices_value_dict[out_index].extend(top_values[idx].tolist())

    return ellipse_outdices.tolist(), ellipse_outdices_dict, ellipse_outdices_value_dict


# 在load_kmeans_result中确定哪些点需要被gmm覆盖,哪些点不需要被gmm覆盖
# 1. 根据权重值,大于0.2或者小于-0.2的,划分为outliers
# 2. 根据梯度值,大于某个阈值的,划分为outliers
# 3. 根据kmeans聚类的数量, 少于mini-sample点的直接划分为outliers
#######################
# load_kmeans_result for splitting outliers and inliers.
# 1. based on weight's value, greater than 0.2 or less than -0.2 points will be categorized into outliers
# 2. based on weight's grad, higher than threshold points will be categorized into outliers
# 3. based on kmeans, points in less than mini-sample cluster will be categorized into outliers.
# def ellipse_outlier(centers, labels, weight_mu, weight_sigma):
#     data = torch.stack((weight_mu, weight_sigma), dim=1).to(device)
#     data_inliers = data[inliers_indices]
#     data_inliers = data_inliers.to(device)
#     ellipse_outdices = []
#     filterd_labels_inliers = labels[inliers_indices]
#     for i in range(centers.shape[0]):
#         cluster_mask = torch.where(filterd_labels_inliers == i)[0].to(device)
#         cluster_data = data_inliers[cluster_mask]
#
#         cov_matrix = torch.cov(cluster_data.T)
#         cov_matrices[i] = torch.linalg.cholesky(cov_matrix)
#
#         ########## ##########
#         cov_matrix_inv = torch.linalg.inv(cov_matrix)
#
#         # 获取当前簇的均值
#         mean = centers[i]
#
#         # 计算每个点到均值的偏差矩阵
#         diff = cluster_data - mean
#
#         # 计算 (x - μ)^T Σ^{-1} (x - μ) 使用矩阵操作
#         # diff * cov_matrix_inv * diff^T 的每个对角线元素
#         ellipse_value = torch.sum(diff @ cov_matrix_inv * diff, dim=1)
#         # transformed_diff = diff @ cov_matrix_inv
#
#         # 计算 (x - μ)^T Σ^{-1} (x - μ)
#         # mahalanobis_squared = torch.dot(transformed_diff, diff)
#
#         # 判断 (x - μ)^T Σ^{-1} (x - μ) 是否小于等于 1
#         outside_ellipse = ellipse_value > 5.991
#
#         # 记录符合条件的点的索引
#         ellipse_outdices.extend(cluster_mask[outside_ellipse].tolist())
#     print("得到了ouliter的points")
#     ## 把这些点直接加入outliers
#     # ellipse_outdices = torch.from_numpy(np.array(ellipse_outdices)).to(outliers_indices.device)
#     # outliers_indices = torch.cat((outliers_indices, ellipse_outdices))
#     # print(len(ellipse_outdices))
#     #### 重新分配label或者新增label
#     ellipse_outdices_dict = {}
#     ellipse_outdices_value_dict = {}
#     for idx, value in enumerate(ellipse_outdices):
#         ellipse_outdices_dict[value] = []
#         ellipse_outdices_value_dict[value] = []
#     ###
#     k = 2
#     n_points = len(ellipse_outdices)
#     print("the out")
#     n_points_data = data[ellipse_outdices]
#     n_gaussians = centers.shape[0]
#     ellipse_outdices = torch.from_numpy(np.array(ellipse_outdices))
#     log_probs = torch.zeros(n_points, 5, dtype=torch.float64, device=data.device)
#     centers = centers.to(torch.float64)
#     distance_n_points_centers = torch.cdist(data[ellipse_outdices], centers)
#     num_keep = 5
#     nearest_gaussians = torch.topk(distance_n_points_centers, k=num_keep, largest=False).indices
#     # shape: [3000, 11]
#     # 计算每个点在每个高斯分布下的对数概率密度
#     # 3000
#     print("到底是哪里慢")
#     for i in range(n_points):
#         mvn = MultivariateNormal(loc=centers[nearest_gaussians[i]], scale_tril=cov_matrices[nearest_gaussians[i]])
#         log_probs[i, :] = mvn.log_prob(n_points_data[i])
#     print("是这里吗")
#     # 找到每个点对所有高斯分布的对数概率密度最大值的索引
#     top_values, top_indices = torch.topk(log_probs, k, dim=1)
#     top_indices_gather = torch.gather(nearest_gaussians, 1, top_indices)
#     print("不是吧")
#     for idx, out_index in enumerate(ellipse_outdices.tolist()):
#         ellipse_outdices_dict[out_index].extend(top_indices_gather[idx].tolist())
#         ellipse_outdices_value_dict[out_index].extend(top_values[idx].tolist())
#     a = 1
#
#     return ellipse_outdices.tolist(), ellipse_outdices_dict, ellipse_outdices_value_dict
def load_kmeans_result(centers, labels, weight_mu, weight_sigma, weight_mu_grad, min_samples=100):
    outliers_indices = set()

    ############  1. 权重,根据权重选择outliers points ############
    # print(weight_mu)
    indices_larger_02 = torch.where(weight_mu >= 0.2)[0]
    outliers_indices.update(indices_larger_02.tolist())

    indices_less_minus_02 = torch.where(weight_mu <= -0.2)[0]
    outliers_indices.update(indices_less_minus_02.tolist())
    ############  over ############
    ################   2. 处理梯度 ################
    # Replace torch.quantile with torch.topk to avoid large tensor issues
    top_k = int(len(weight_mu_grad) * 0.01)
    top_k = max(top_k, 1)
    top_values, _ = torch.topk(weight_mu_grad, top_k)
    mu_99th_percentile = top_values[-1]
    weight_mu_grad_indices = torch.where(weight_mu_grad >= mu_99th_percentile)[0]
    outliers_indices.update(weight_mu_grad_indices.tolist())
    # print(outliers_indices)
    ############  over ############

    ################ 3. 处理kmeans中少于30个点的cluster ############

    # original_labels = torch.from_numpy(labels).to(device)
    # original_centers = torch.from_numpy(centers).to(device)

    if isinstance(labels, np.ndarray):
        original_labels = torch.from_numpy(labels).to(device)
    elif isinstance(labels, torch.Tensor):
        original_labels = labels

    if isinstance(centers, np.ndarray):
        original_centers = torch.from_numpy(centers).to(device)
    elif isinstance(centers, torch.Tensor):
        original_centers = centers

    unique_labels, counts = torch.unique(original_labels, return_counts=True)
    # 设置最小样本数阈值
    min_samples = 20
    # 过滤样本数少于min_samples的簇
    invalid_indices = counts <= min_samples
    invalid_labels = unique_labels[invalid_indices] # 無效的簇
    for label_outlier in invalid_labels:
        indices_kmeans_cluster_outliers = torch.where(original_labels == label_outlier)[0]
        outliers_indices.update(indices_kmeans_cluster_outliers.tolist())
    ############ end ####################

    ############ 处理映射标签 ##############
    data = torch.stack((weight_mu, weight_sigma), dim=1).to(device)

    outliers_indices = torch.tensor(list(outliers_indices)).to(device)
    mask_inliers = ~torch.isin(torch.arange(data.shape[0]).to(device), outliers_indices).to(device)
    inliers_indices = torch.where(mask_inliers)[0].to(device)  ## 取inliers的下标
    # sum_inliners = sum(mask_inliers)
    inliers_label = original_labels[mask_inliers] # inliers 点的所有的label
    assert inliers_indices.shape[0] + outliers_indices.shape[0] == data.shape[0], "something went wrong"
    unique_labels, counts = torch.unique(inliers_label, return_counts=True)
    data_inliers = data[mask_inliers]
    unique_labels = unique_labels.tolist()
    # original_labels = original_labels.tolist()
    # valid_indices = counts >= 1
    # valid_labels = unique_labels[valid_indices]  # 有效的簇
    # 创建一个映射，将原始标签映射到新的连续标签
    # time_dict = time.time()
    # # 过滤并重新映射标签，对于不在映射中的标签，赋值为-1
    # filtered_labels = np.array([label_map.get(label.item(), -1) for label in original_labels])
    # time_dict_end = time.time()
    # print(f"老方法用时{time_dict_end-time_dict}")
    ##### 新方法 ###
    # time_new = time.time()
    unique_labels_array = np.array(unique_labels)
    # 创建一个大于原标签范围的数组，并初始化为-1
    max_label_value = max(unique_labels_array.max(), original_labels.max())
    label_map_array = np.full(max_label_value + 1, -1)
    # 设置新标签
    label_map_array[unique_labels_array] = np.arange(len(unique_labels_array))

    # 使用标签映射数组快速查找和过滤
    label_map_array = torch.from_numpy(label_map_array).to(device)
    filtered_labels = label_map_array[original_labels]

    filterd_labels_inliers = label_map_array[inliers_label]
    # filterd_labels_inliers = torch.from_numpy(filterd_labels_inliers)

    # 创建新的 centers 数组，只包含 valid_labels 对应的 centers
    new_centers = original_centers[unique_labels]
    # print(f"最大值的new_center{max(new_centers)}")
    print(f"new_centers shape {new_centers.shape}")
    cov_matrices = torch.zeros((new_centers.shape[0], 2, 2)).to(device)
    # 计算每个簇的样本协方差矩阵
    # data = torch.stack((weight_mu, weight_sigma), dim=1)
    # outliers_indices = torch.tensor(list(outliers_indices))
    # mask_inliers = ~torch.isin(torch.arange(data.size(0)), outliers_indices)

    # GPU 上迭代 indices
    data_inliers = data_inliers.to(device)
    ellipse_outdices = []
    for i in range(new_centers.shape[0]):
        cluster_mask = torch.where(filterd_labels_inliers == i)[0].to(device)
        cluster_data = data_inliers[cluster_mask]

        cov_matrix = torch.cov(cluster_data.T)
        cov_matrices[i] = torch.linalg.cholesky(cov_matrix)

        ########## ##########
        cov_matrix_inv = torch.linalg.inv(cov_matrix)

        # 获取当前簇的均值
        mean = new_centers[i]

        # 计算每个点到均值的偏差矩阵
        diff = cluster_data - mean

        # 计算 (x - μ)^T Σ^{-1} (x - μ) 使用矩阵操作
        # diff * cov_matrix_inv * diff^T 的每个对角线元素
        ellipse_value = torch.sum(diff @ cov_matrix_inv * diff, dim=1)
        # transformed_diff = diff @ cov_matrix_inv

        # 计算 (x - μ)^T Σ^{-1} (x - μ)
        # mahalanobis_squared = torch.dot(transformed_diff, diff)

        # 判断 (x - μ)^T Σ^{-1} (x - μ) 是否小于等于 1
        outside_ellipse = ellipse_value > 5.991

        # 记录符合条件的点的索引
        ellipse_outdices.extend(cluster_mask[outside_ellipse].tolist())

    ## 把这些点直接加入outliers
    # ellipse_outdices = torch.from_numpy(np.array(ellipse_outdices)).to(outliers_indices.device)
    # outliers_indices = torch.cat((outliers_indices, ellipse_outdices))
    # print(len(ellipse_outdices))
    #### 重新分配label或者新增label
    ellipse_outdices_dict = {}
    ellipse_outdices_value_dict = {}
    for idx, value in enumerate(ellipse_outdices):
        ellipse_outdices_dict[value] = []
        ellipse_outdices_value_dict[value] = []
    ###
    k = 2
    n_points = len(ellipse_outdices)
    n_points_data = data[ellipse_outdices]
    n_gaussians = new_centers.shape[0]
    print(f"n_gaussians is {n_gaussians}")
    log_probs = torch.zeros(n_points, n_gaussians, dtype=torch.float64)

    # 计算每个点在每个高斯分布下的对数概率密度
    for i in range(n_gaussians):
        mvn = MultivariateNormal(loc=new_centers[i], scale_tril=cov_matrices[i])
        log_probs[:, i] = mvn.log_prob(n_points_data).exp()

    # 找到每个点对所有高斯分布的对数概率密度最大值的索引
    top_values, top_indices = torch.topk(log_probs, k, dim=1)
    for idx, out_index in enumerate(ellipse_outdices):
        ellipse_outdices_dict[out_index].extend(top_indices[idx].tolist())
        ellipse_outdices_value_dict[out_index].extend(top_values[idx].tolist())

    return outliers_indices, inliers_indices, filtered_labels, new_centers, cov_matrices, ellipse_outdices, ellipse_outdices_dict, ellipse_outdices_value_dict

###################### GMM 代码 #################################


################## 定义bnn
class ModelBNN():

    def __init__(self, model_cnn, param, inliers_indices, outliers_indices, labes_inliers, centers, covars, ellipse_outdices, ellipse_outdices_dict, ellipse_outdices_value_dict, record_shape):
        ###
        # param -> (n,2)
        # inliers_indices -> (m)
        # outliers_indices -> (k)
        ###
        super().__init__()
        self.model_cnn = model_cnn.to(device)
        # 首先定义 outliers 的sample.
        self.param_weight_model = copy.deepcopy(param[:,:])
        self.param_weight_model.requires_grad = False
        self.outliers_indices = outliers_indices
        self.inliers_indices = inliers_indices
        self.labes_inliers = labes_inliers
        self.centers = centers
        self.covars = covars
        self.record_shape = record_shape
        self.weight_pure_cnn_mu = copy.deepcopy(param[:,0])
        self.weight_pure_cnn_mu.requires_grad = False
        self.weight_pure_cnn_sigma = copy.deepcopy(param[:,1])
        # ellipse_outdices, ellipse_outdices_dict, ellipse_outdices_value_dict
        self.ellipse_outdices = ellipse_outdices
        self.ellipse_outdices_dict = ellipse_outdices_dict
        self.ellipse_outdices_value_dict = ellipse_outdices_value_dict
        # time_start = time.time()
        # distance = self.euclidean_distances(self.param_weight_model[inliers_indices], centers)
        # time_end = time.time()
        # print("distance calculate done, spend{time_end - time_start}")
        # input()



    def reconstruct_model_cnn(self, sample_inliers, sample_outliers, x):
        # print("reconstruct_model_cnn")

        # Update model parameters with inliers and outliers
        self.param_weight_model[self.inliers_indices] = sample_inliers[self.labes_inliers].detach()
        self.param_weight_model[self.outliers_indices, 0] = sample_outliers.detach()
        #### 处理ellipse_outdices
        weight_pre_places = torch.zeros((len(self.ellipse_outdices), 2), device=device,dtype=torch.double)
        pdf_values = torch.stack(
            [torch.from_numpy(np.array(self.ellipse_outdices_value_dict[idx])).to(device) for idx in self.ellipse_outdices])
        scores = F.softmax(pdf_values, dim=1)

        centers_indices = torch.tensor([self.ellipse_outdices_dict[idx] for idx in ellipse_outdices]).to(device)

        mask = (pdf_values > torch.exp(torch.tensor(12.0)).item()).sum(dim=1) == 2
        # 计算加权位置（双中心和单中心的情况）

        weight_pre_places[mask] = self.centers[centers_indices[mask, 0]] * scores[mask, 0][:, None] + self.centers[centers_indices[mask, 1]] * scores[mask, 1][:, None]

        weight_pre_places[~mask] = self.centers[centers_indices[~mask, 0]] * scores[~mask, 0][:, None]

        # 更新 param_weight_model
        self.param_weight_model[ellipse_outdices] = weight_pre_places.to(torch.float32).detach()
        #### 结束
        cluster_value = sample_inliers[:, 0]
        cluster_value_inliers = cluster_value[self.labes_inliers]

        # Update weights with inlier and outlier cluster values
        self.weight_pure_cnn_mu[self.inliers_indices] = cluster_value_inliers
        self.weight_pure_cnn_mu[self.outliers_indices] = sample_outliers
        self.weight_pure_cnn_mu[self.ellipse_outdices] = weight_pre_places[:,0].to(torch.float32)
        self.weight_pure_cnn_sigma[self.inliers_indices] = sample_inliers[:, 1][self.labes_inliers]

        # Ensure correct ordering of weights in self.weight_pure_cnn_mu
        keys_name = list(self.record_shape.keys())
        weight_pure_cnn_mu_copy = self.weight_pure_cnn_mu.detach().clone()

        idx = 0
        for name, module in self.model_cnn.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                name_record, _ = keys_name[idx].rsplit(".", 1)
                assert name_record == name, "Check order of list_mu and list_sigma"

                # Flatten the module weights and reshape them
                len_weight = len(module.weight.flatten())
                reshape_weight = weight_pure_cnn_mu_copy[:len_weight].view(module.weight.shape)
                module.weight.data = reshape_weight.clone()
                module.weight.requires_grad = False

                # Move to the next set of weights in the copy

                weight_pure_cnn_mu_copy = weight_pure_cnn_mu_copy[len_weight:].contiguous()
                idx += 1
        # Perform a forward pass to verify model integrity
        logist = self.model_cnn(x)
        # eval_training(self.model_cnn,0)
        return logist

    def __call__(self, x, y=None):
        ### set outliers sample site
        outliers_loc = self.param_weight_model[outliers_indices][:,0]
        outliers_sigma = self.param_weight_model[outliers_indices][:,1]
        outliers_sigma = outliers_sigma.clamp(min = 1e-10)
        outliers_sample = pyro.sample("outliers",
                           dist.Normal(loc=outliers_loc, scale=outliers_sigma).to_event(1))
        # 设置inliers点,注意inliers的点全部被gmm cover,所以参数只有centers
        locs_gmm = pyro.sample("locs",
                           dist.MultivariateNormal(loc=self.centers, scale_tril=self.covars).to_event(1))
        # 从二维高斯分布中采样一次sd
        logits_out = self.reconstruct_model_cnn(locs_gmm, outliers_sample, x)

        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Categorical(logits=logits_out), obs=y)
        return logits_out

class GuideBNN():
    def __init__(self, model_cnn, param, inliers_indices, outliers_indices, labes_inliers, centers, covars, record_shape):
        super().__init__()
        self.model_cnn = model_cnn
        # 首先定义 outliers 的sample.
        self.param_weight_guide = copy.deepcopy(param)
        self.outliers_indices = outliers_indices
        self.inliers_indices = inliers_indices
        self.labes_inliers = labes_inliers
        self.centers = nn.Parameter(centers)
        self.covars = covars
        self.record_shape = record_shape
        self.outliers_value = self.param_weight_guide[outliers_indices]


    def __call__(self, x, y=None):

        centers_posterior = pyro.param("centers_guide", self.centers).requires_grad_(True)
        scale_tril_posterior = pyro.param("scale_tril_posterior", self.covars,
                                          constraint=constraints.lower_cholesky)
        outliers_posterior_locs = pyro.param("outliers_posterior_loc", self.outliers_value[:,0])
        outliers_posterior_sigma = pyro.param("outliers_posterior_sigma", self.outliers_value[:,1], constraint=constraints.positive)
        # print("前十个locs")
        # print(outliers_posterior_locs[:10])

        # outliers_exp = torch.log(0.03 + torch.exp(outliers_posterior[:,1]))
        pyro.sample("outliers", dist.Normal(loc=outliers_posterior_locs, scale=outliers_posterior_sigma).to_event(1))
        pyro.sample("locs", dist.MultivariateNormal(loc=centers_posterior, scale_tril=scale_tril_posterior).to_event(1))

def sym_matrix_sqrt(mat):
    # 对称矩阵的平方根
    eigenvalues, eigenvectors = torch.linalg.eigh(mat)
    # 处理可能的数值误差导致的负特征值
    eigenvalues = torch.clamp(eigenvalues, min=1e-10)
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    return (eigenvectors * sqrt_eigenvalues.unsqueeze(0)) @ eigenvectors.T

def wasserstein_2d_merge(centers, covars, grad_centers, filtered_label, weights, threshold=1e-2):
    """
    Merge 2D Gaussian distributions based on Wasserstein distance.

    Parameters:
    - centers (Tensor): [3000, 2], means of the Gaussians.
    - covars (Tensor): [3000, 2, 2], Cholesky decompositions of covariance matrices.
    - grad_centers (Tensor): Gradients of the centers.
    - filtered_label (Tensor): Labels indicating which Gaussian each weight belongs to.
    - weights (Tensor): Weights for each point.
    - threshold (float): Threshold for merging Gaussians based on Wasserstein distance.

    Returns:
    - Updated centers, covars, and filtered_label after merging.
    """

    num_gaussians = centers.shape[0]
    merged_centers = centers.clone()
    merged_covars = covars.clone()

    # Step 1: Compute pairwise distances and find the nearest 10 Gaussians
    distances = torch.cdist(centers, centers)  # shape: [3000, 3000]

    # Get the indices of the 10 closest Gaussians for each Gaussian
    nearest_gaussians = torch.topk(distances, k=3, largest=False).indices  # shape: [3000, 11]
    nearest_gaussians = nearest_gaussians.cpu().numpy()
    # Track which Gaussians have been merged
    merged_indices = set()
    merge_map = {}  # Dictionary to map old indices to new indices
    for i in range(num_gaussians):
        # print(i)
        if i in merged_indices:
            continue

        for j in nearest_gaussians[i]:

            if i == j or j in merged_indices:
                continue  # Skip comparing the same Gaussian to itself or already merged ones

            mu1, mu2 = centers[i], centers[j]
            L1, L2 = covars[i], covars[j]
            # Wasserstein distance calculation
            # mean_diff = torch.norm(mu1 - mu2) ** 2
            Sigma1 = L1 @ L1.T
            Sigma2 = L2 @ L2.T
            # sqrt_covar_prod = L1 @ Sigma2 @ L1
            # # Ensure the resulting matrix is positive definite
            # # epsilon = torch.tensor(1e-6,device=sqrt_covar_prod.device)
            # # sqrt_covar_prod += torch.eye(sqrt_covar_prod.size(0)) * epsilon
            #
            # try:
            #     sqrt_covar_prod = torch.linalg.cholesky(sqrt_covar_prod)
            # except RuntimeError as e:
            #     print(f"Cholesky decomposition of sqrt_covar_prod failed: {e}")
            #     return float('inf')
            #
            # wasserstein_distance = mean_diff + torch.trace(Sigma1 + Sigma2 - 2 * sqrt_covar_prod)
            mean_diff = torch.norm(mu1 - mu2) ** 2
            # 计算协方差矩阵的平方根
            Sigma1_sqrt = sym_matrix_sqrt(Sigma1)
            # 计算中间矩阵
            middle_term = Sigma1_sqrt @ Sigma2 @ Sigma1_sqrt
            # 计算中间矩阵的平方根
            middle_term_sqrt = sym_matrix_sqrt(middle_term)
            # 计算迹
            trace_term = torch.trace(Sigma1 + Sigma2 - 2 * middle_term_sqrt)
            # 返回 Wasserstein 距离的平方
            wasserstein_distance =  mean_diff + trace_term
            # Step 3: Check conditions to merge
            # Compute variance and gradient for each Gaussian
            if len(weights[filtered_label == i]) > 1:
                variance_i = torch.var(weights[filtered_label == i])
            else:
                variance_i = 0
            if len(weights[filtered_label == j]) > 1:
                variance_j = torch.var(weights[filtered_label == j])
            else:
                variance_j = 0
            grad_i = torch.norm(grad_centers[i])
            grad_j = torch.norm(grad_centers[j])
            # print(wasserstein_distance,"确实")
            # If Wasserstein distance is small, variance is small, and gradient is large, merge
            # print("------------")
            # print(wasserstein_distance)
            # print(grad_i)
            # bnns -> strong assumption indenpend
            # 2dgbnns -> weak assumption -> each guassian indepent
            # weights -> gaussians
            # print(variance_i)
            # print("------------")
            if wasserstein_distance < threshold and abs(grad_i * lr) < 0.15 and abs(grad_j* lr) < 0.15 and variance_i < 0.001 and variance_j < 0.001:
                # print("get merging weights")
                # Merge the Gaussians by averaging the centers and covariances
                merged_mu = (mu1 + mu2) / 2
                merged_covar = (Sigma1 + Sigma2) / 2
                merged_L = torch.linalg.cholesky(merged_covar)

                # Update centers and covariances
                merged_centers[i] = merged_mu
                merged_covars[i] = merged_L
                # Mark the second Gaussian as merged
                merged_indices.add(j)

                # Map the old index to the new index
                merge_map[j] = i

                filtered_label[filtered_label == j] = i
    # print("Create a new label mapping")
    # # Step 4: Create a new label mapping
    # new_filtered_label = filtered_label.clone()
    # for old_index, new_index in merge_map.items():
    #     new_filtered_label[filtered_label == old_index] = new_index
    print("Step 5: Adjust remaining labels")
    # Step 5: Adjust remaining labels
    mask = filtered_label != -1
    valid_labels = filtered_label[mask]

    # 对除了 `-1` 外的标签进行重新映射
    start_time = time.time()
    unique_labels, new_filtered_indices = torch.unique(valid_labels, sorted=True, return_inverse=True)
    end_time = time.time()
    print(f"好像很耗时{end_time-start_time}")
    # 创建一个新的标签数组，初始化为 `-1`，确保数据类型与 `filtered_label` 一致
    start_time = time.time()
    new_filtered_label = torch.full_like(filtered_label, -1, dtype=filtered_label.dtype)
    end_time = time.time()
    print(f"好像很耗时2{end_time-start_time}")
    # 将重新映射后的标签放回相应位置
    start_time = time.time()
    new_filtered_label[mask] = new_filtered_indices.to(torch.long)
    filtered_labels = new_filtered_label
    remaining_indices = [i for i in range(num_gaussians) if i not in merged_indices]
    end_time = time.time()
    print(f"好像很耗时3{end_time-start_time}")
    merged_centers = merged_centers[remaining_indices]
    merged_covars = merged_covars[remaining_indices]

    print("返回了啊")
    print(merged_centers)
    print(merged_covars)
    print(filtered_labels)
    print("返回了啊")
    return merged_centers, merged_covars, filtered_labels

################ over end
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="wrn", required=False, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('-weights_path', type=str,
                        default="checkpoint/wrn/cifar100/stochastic/stochastic_wrn.pth",
                        required=False)
    parser.add_argument('-dataset', type=str, default="cifar100", help='cifar10 or cifar100')
    parser.add_argument('-resume', action='store_true', default=True, help='resume training')
    parser.add_argument('-weights_path_origin', type=str,
                        default="checkpoint/wrn/cifar100/origin/wrn.pth",
                        required=False)
    # /home/test/moule/lml/aaai/checkpoint/resnet50/cifar100/kmeans/kmeans_clusters_gpu.json
    parser.add_argument('-kmeans_path', type=str,
                        default="checkpoint/resnet18/cifar100/init_gmm/clusters_gpu.json",
                        required=False)
    parser.add_argument('-output', type=str, default="checkpoint/wrn/cifar100/2DGBNNs", help='output')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
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
    args = parser.parse_args()

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    assert (args.dataset == 'cifar10' or args.dataset == 'cifar100')
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('data', train=True, download=True,
                                                transform=transform_train),
        batch_size=args.b, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('data', train=False, transform=transform_test),
        batch_size=args.b, shuffle=True, **kwargs)

    # create model
    net = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                       args.widen_factor, dropRate=args.droprate)

    model_cnn_original = copy.deepcopy(net)
    net = net.to(device)
    Convertor.orig_to_bayes(net, prior=True)
    ############# 加载权重 #################
    if args.resume:
        pretrained_load = torch.load(args.weights_path, map_location="cpu")
        weights_model = pretrained_load["net"]
        grad = pretrained_load["grad"]
        net.load_state_dict(weights_model)
        print("Weights loaded.")
        weights_path_cnn = args.weights_path_origin
        weight_original = torch.load(weights_path_cnn, map_location="cpu")
        model_cnn_original.load_state_dict(weight_original)
    else:
        pretrained_load = torch.load(args.weights_path, map_location="cpu")
        weights_model = pretrained_load["net"]
        grad = pretrained_load["grad"]

    ############# over #################
    grad_value = []
    keys_grad = grad.keys()
    for item in reversed(keys_grad):
        if "mu" in item and "bias" not in item:
            grad_value.extend(grad[item].detach().cpu().flatten().numpy().tolist())
    grad_value = torch.from_numpy(np.array(grad_value))

    ############# 获取权重,包括mu和sigma,以及每个module的权重的大小 #################
    list_mu = []
    list_sigma = []
    shape_record = {} # 记录每个layer的参数的大小，这个再之后放回参数有作用
    for name, module in net.named_modules():
        if isinstance(module, (BayesConv2d, BayesLinear)):
            list_mu.extend(module.weight_mu.detach().cpu().flatten().numpy().tolist())
            list_sigma.extend(module.weight_sigma.detach().cpu().flatten().numpy().tolist())
            shape_record[name+".weight"] = module.weight_mu.shape

            # if module.bias_mu is not None:
            #     shape_record[name + ".bias"] = module.bias_mu.shape
    list_mu = torch.from_numpy(np.array(list_mu))
    list_sigma = torch.from_numpy(np.array(list_sigma))
    # print(shape_record)
    # print(shape_record["conv1.0"][0])
    ############# over #################

    ############# 根据shape_record来获取参数,这个只是为了测试  #############
    ############# over #############
    # 11210432
    ############# 在gpu中运行kmeans,如果没有json,就重新跑,否则直接读json文件 #############
    path = args.kmeans_path
    # os.makedirs(path, exist_ok=True)
    if not os.path.exists(path):
        pass
    else:
        # 打开并读取 JSON 文件
        with open(path, 'r') as file:
            results = json.load(file)
        original_labels = np.array(results['labels'])
        original_centers = np.array(results['centers'])
    ############# over #############

    ############ 根据重要性以及梯度划分outlier和inliers #############

    start_time = time.time()
    outliers_indices, inliers_indices, filtered_labels, new_centers, cov_matrices, ellipse_outdices, ellipse_outdices_dict, ellipse_outdices_value_dict = load_kmeans_result(original_centers, original_labels, list_mu, list_sigma, grad_value)
    center = new_centers[0].cpu().numpy()
    inliers_indices = inliers_indices.to(device)
    new_centers = new_centers.to(device)
    filtered_labels = filtered_labels.to(device)
    cov_matrices = cov_matrices.to(device)
    # 找到所有属于第一个高斯中心的点的索引

    indices = torch.where(filtered_labels == 0)[0]

    # 使用索引从data_all中提取数据
    list_mu = torch.from_numpy(np.array(list_mu))
    list_sigma = torch.from_numpy(np.array(list_sigma))
    # list_sigma = torch.log(0.03 + torch.exp(list_sigma))
    data_all = torch.stack((list_mu, list_sigma), dim=1).to(torch.float32).to(device)
    selected_data = data_all[indices].cpu().numpy()

    # 可视化
    # plt.figure(figsize=(8, 6))
    # plt.scatter(selected_data[:, 0], selected_data[:, 1], label='Data Points', alpha=0.6)
    # plt.scatter(center[0], center[1], color='red', label='Gaussian Center', s=100)
    # plt.title('Visualization of the First Gaussian Center and Corresponding Data Points')
    # plt.xlabel('mu')
    # plt.ylabel('sigma')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("single.png")
    # plt.show()
    # input()

    end_time = time.time()
    print(f"relabel spent {end_time-start_time}")
    new_centers = new_centers.to(torch.float32).to(device)
    cov_matrices = cov_matrices.to(torch.float32).to(device)

    print(f"outlier percentage: {len(outliers_indices)/len(original_labels) * 100}%")
    print(f"number of centers after filtering: {len(new_centers)}")
    ############ over ############

    ########### 定义gmm ############
    list_mu = torch.from_numpy(np.array(list_mu))
    list_sigma = torch.from_numpy(np.array(list_sigma))
    list_sigma_new = torch.log(0.03 + torch.exp(list_sigma))
    list_sigma_new = list_sigma_new.clamp(min = 1e-10)
    data_all = torch.stack((list_mu, list_sigma_new), dim=1).to(torch.float32).to(device)
    # print("----------------")
    data_inliers = data_all[inliers_indices]  # 得到inliers点
    labels_inliers = filtered_labels[inliers_indices] # 得到每个点的label

    # gmm_gpu = GMMGPU(K=new_centers.shape[0], num_data=inliers_indices.shape[0], data=data_all, labels=labels_inliers,
    #                       centers=new_centers, convs=cov_matrices)
    # print("===Start gmm training===")
    # gmm_gpu.train(1)

    # print("===End gmm training===")
    # print("===Allocate new centers and covars===")
    ############ over ############

    ########### 定义bnn ############
    #  model_cnn, param, inliers_indices, outliers_indices, labes_inliers, centers, covars, record_shape

    model = ModelBNN(model_cnn_original, data_all, inliers_indices, outliers_indices, labels_inliers, new_centers, cov_matrices, ellipse_outdices, ellipse_outdices_dict, ellipse_outdices_value_dict, shape_record)
    guide = GuideBNN(model_cnn_original, data_all, inliers_indices, outliers_indices, labels_inliers, new_centers, cov_matrices, shape_record)

    # model_cnn, model_weight, param, inliers_indices, outliers_indices, labes_inliers, centers, covars, record_shape
    lr = 0.00001
    optimizer = Adam({"lr": 0.00001})
    elbo = Trace_ELBO()
    svi = SVI(model, guide, optimizer, loss=elbo)


    cifar100_test_loader = get_test_dataloader(
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            num_workers=4,
            batch_size=1024,
            shuffle=True,
            dataset="cifar100"
        )
    num_steps = 200
    best_acc = 0.
    loss_record = []
    acc_record = []
    output = args.output
    os.makedirs(output, exist_ok=True)
    ############ 输出参数 ###########

    # os.makedirs(f"pyro/{args.net}", exist_ok=True)
    for step in range(num_steps):
        loss_batch = 0
        for idx, pack_data in enumerate(train_loader):
            # 在每次调用 SVI 之前清理参数存储

            start_time = time.time()

            # pyro.clear_param_store()
            data_iters, labels_iter = pack_data
            data_iter = data_iters.to(device)
            labels_iter = labels_iter.to(device)
            # loss = svi.step(data_iter, labels_iter)
            # loss = svi.step(data_iter, labels_iter)
            with poutine.trace(param_only=True) as param_capture:
                loss = svi.loss_and_grads(model, guide, data_iter, labels_iter)

            params = set(
                site["value"].unconstrained() for site in param_capture.trace.nodes.values()
            )

            loss = loss / len(labels_iter)
            loss_batch += loss
            grad_centers = pyro.param("centers_guide").grad
            print(f"batch {step}/{num_steps}, iteration {idx}/{len(train_loader)}, loss {loss}")
            if idx < len(train_loader) - 1:
                pyro.infer.util.zero_grads(params)
            end_time = time.time()
            if idx == 10 or idx == 11:
                print(f"time consume {end_time - start_time}")
        print(f"the size of gaussian {guide.centers.shape[0]}")

        if step == 0 or step:
            acc, final_acc_top5, nll, ece = model_test_predictive(model, guide, val_loader, num_samples=40)
            best_acc = acc
            ########### save model ###########
            # save weights except for conv2d and linear
            # 先存储除了weight的其他全部权重
            model_cnn_original_weight = model_cnn_original.state_dict()
            name_conv_linear = []
            for name, module in model_cnn_original.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    name_conv_linear.append(name + ".weight")
                    print(name)
            weight_further_storage = model_cnn_original.state_dict()
            weight_further_filter = {}
            for name, value in weight_further_storage.items():
                if name not in name_conv_linear:
                    weight_further_filter[name] = value

            print(weight_further_filter.keys())

            ########### 然后存储guide中的参数,包括如下###########
            # centers: torch.Size([3630, 2])
            # scale_tril_posterior: torch.Size([3630, 2, 2])
            # outliers_posterior: torch.Size([123676, 2])
            params_guide = {}
            for name, value in pyro.get_param_store().items():
                params_guide[name] = value.detach().cpu()
            ########### 然后存储indices ########## 注意用int16存储
            # parameters are as following
            # outliers_indices, inliers_indices, filtered_labels, ellipse_outdices, ellipse_outdices_dict, ellipse_outdices_value_dict
            params_dict_indices = {
                'outliers_indices': outliers_indices,  # 不用存储inliers, 根据outliers计算inliers就行
                'filtered_labels': filtered_labels.to(torch.int16),
                'ellipse_outdices': ellipse_outdices,
                'ellipse_outdices_dict': ellipse_outdices_dict,
                'ellipse_outdices_value_dict': ellipse_outdices_value_dict
            }
            all_dicts = {
                'state_dict_no_weight': weight_further_filter,
                'params_guide': params_guide,
                'params_dict_indices': params_dict_indices
            }
            torch.save(weight_further_filter,
                       os.path.join(output, f'2DGBNNs_state_dict_no_weight_{step}_{acc}_nll_{nll}_ece_{ece}.pth'))
            torch.save(params_guide, os.path.join(output, f'2DGBNNs_params_guide_{step}_{acc}_nll_{nll}_ece_{ece}.pth'))
            torch.save(params_dict_indices,
                       os.path.join(output, f'2DGBNNs_params_dict_indices_{step}_{acc}_nll_{nll}_ece_{ece}.pth'))
            # 保存字典
            output_combination = os.path.join(output, f"2DGBNNs_{step}_{acc}_nll_{nll}_ece_{ece}.pth")
            torch.save(all_dicts, output_combination)

        if step or step == 0:
            grad_centers = pyro.param("centers_guide").grad
            print(guide.centers.detach())
            print(guide.covars.detach())
            print(grad_centers)
            with torch.no_grad():
                merged_centers, merged_covars, filtered_label_wassertein = wasserstein_2d_merge(guide.centers.detach(),
                                                                                                guide.covars.detach(),
                                                                                                grad_centers,
                                                                                                filtered_labels.detach(),
                                                                                                data_all.detach(),
                                                                                                threshold=1.5e-7)
                data_all_new = torch.stack((list_mu, list_sigma), dim=1).to(torch.float32).to(device)
                labels_inliers = filtered_label_wassertein[inliers_indices]
                ellipse_outdices, ellipse_outdices_dict, ellipse_outdices_value_dict = ellipse_outlier(
                    merged_centers, filtered_label_wassertein, list_mu, list_sigma)
            filtered_labels = filtered_label_wassertein
            model = ModelBNN(model_cnn_original, data_all.detach(), inliers_indices.detach(), outliers_indices.detach(),
                             labels_inliers.detach(),
                             merged_centers.detach(), merged_covars.detach(), ellipse_outdices, ellipse_outdices_dict,
                             ellipse_outdices_value_dict, shape_record)
            guide = GuideBNN(model_cnn_original, data_all.detach(), inliers_indices.detach(), outliers_indices.detach(),
                             labels_inliers.detach(),
                             merged_centers.detach(), merged_covars.detach(), shape_record)


            lr = 0.00001
            optimizer = Adam({"lr": 0.00001})
            elbo = Trace_ELBO()
            svi = SVI(model, guide, optimizer, loss=elbo)
            # After redefining model and guide
            pyro.clear_param_store()


