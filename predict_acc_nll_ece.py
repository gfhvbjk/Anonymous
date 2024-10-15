import torch
import torch.nn as nn
import json
import copy
from pyro.infer import Predictive
import argparse
import torch.distributions.constraints as constraints
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
def kmeans_gpu(weight_mu, weight_sigma):
    X_gpu = cudf.DataFrame({
        "weight_mu": weight_mu,
        "weight_sigma_exp": weight_sigma
    })
    print("Shape of X_gpu:", X_gpu.shape)
    # 初始化 KMeans，设置聚类数为500，使用 GPU
    kmeans = KMeans(n_clusters=6000, random_state=0)
    kmeans.fit(X_gpu)


    labels = kmeans.labels_.values_host
    centers = kmeans.cluster_centers_.values_host

    plt.figure(figsize=(50, 40))


    plt.scatter(X_gpu['weight_mu'].values_host, X_gpu['weight_sigma_exp'].values_host, c=labels, cmap='viridis',
                alpha=0.5, s=2)
    plt.title('K-means Clustering on GPU of Weights - All Clusters')
    plt.xlabel('Weight Mu')
    plt.ylabel('Weight Sigma')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    if not os.path.exists('output'):
        os.makedirs('output')
    plt.savefig('output/kmeans_clusters_pyro.png')
    plt.close()

    results = {
        'labels': labels.tolist(),
        'centers': centers.tolist()
    }
    os.makedirs('output', exist_ok=True)
    with open('output/kmeans_results_gpu.json', 'w') as f:
        json.dump(results, f)

    print("Results saved to 'output/kmeans_results_gpu.json'.")
    return centers, labels

# 在load_kmeans_result中确定哪些点需要被gmm覆盖,哪些点不需要被gmm覆盖
# 1. 根据权重值,大于0.2或者小于-0.2的,划分为outliers
# 2. 根据梯度值,大于某个阈值的,划分为outliers
# 3. 根据kmeans聚类的数量, 少于mini-sample点的直接划分为outliers
#######################
# load_kmeans_result for splitting outliers and inliers.
# 1. based on weight's value, greater than 0.2 or less than -0.2 points will be categorized into outliers
# 2. based on weight's grad, higher than threshold points will be categorized into outliers
# 3. based on kmeans, points in less than mini-sample cluster will be categorized into outliers.
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
    mu_99th_percentile = torch.quantile(weight_mu_grad, 0.99)
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

    def __init__(self, model_cnn, outliers_posterior, filtered_label, inliers_indices, outliers_indices, labes_inliers, centers, covars, ellipse_outdices, ellipse_outdices_dict, ellipse_outdices_value_dict, record_shape):
        ###
        # param -> (n,2)
        # inliers_indices -> (m)
        # outliers_indices -> (k)
        ###
        super().__init__()
        self.model_cnn = model_cnn.to(device)
        # 首先定义 outliers 的sample.
        self.outliers_indices = outliers_indices
        self.inliers_indices = inliers_indices
        self.labes_inliers = labes_inliers
        self.centers = centers
        self.covars = covars
        self.record_shape = record_shape
        self.weight_pure_cnn_mu = torch.zeros_like(filtered_label, dtype=torch.float32)

        self.weight_pure_cnn_sigma = torch.zeros_like(filtered_label, dtype=torch.float32)
        # ellipse_outdices, ellipse_outdices_dict, ellipse_outdices_value_dict
        self.ellipse_outdices = ellipse_outdices
        self.ellipse_outdices_dict = ellipse_outdices_dict
        self.ellipse_outdices_value_dict = ellipse_outdices_value_dict
        self.outliers_posterior = outliers_posterior


    def reconstruct_model_cnn(self, sample_inliers, sample_outliers, x):
        # print("reconstruct_model_cnn")

        # Update model parameters with inliers and outliers
        # assert len(self.inliers_indices) + len(self.outliers_indices) == self.param_weight_model.shape[0]

        weight_pre_places = torch.zeros((len(self.ellipse_outdices), 2), device=device,dtype=torch.double)
        pdf_values = torch.stack(
            [torch.from_numpy(np.array(self.ellipse_outdices_value_dict[idx])).to(device) for idx in self.ellipse_outdices])
        scores = F.softmax(pdf_values, dim=1)

        centers_indices = torch.tensor([self.ellipse_outdices_dict[idx] for idx in ellipse_outdices]).to(device)

        mask = (pdf_values > torch.exp(torch.tensor(12.0)).item()).sum(dim=1) == 2
        # 计算加权位置（双中心和单中心的情况）

        weight_pre_places[mask] = self.centers[centers_indices[mask, 0]] * scores[mask, 0][:, None] + self.centers[centers_indices[mask, 1]] * scores[mask, 1][:, None]

        weight_pre_places[~mask] = self.centers[centers_indices[~mask, 0]] * scores[~mask, 0][:, None]

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
        # self.model_cnn.eval()
        logist = self.model_cnn(x)
        # eval_training(self.model_cnn,0)
        return logist
    def __call__(self, x, y=None):
        ### set outliers sample site
        outliers_loc = self.outliers_posterior[:,0]
        outliers_sigma = self.outliers_posterior[:,1]
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
    def __init__(self, model_cnn, outliers_posterior, inliers_indices, outliers_indices, labes_inliers, centers, covars, record_shape):
        super().__init__()
        self.model_cnn = model_cnn
        # 首先定义 outliers 的sample.

        self.outliers_indices = outliers_indices
        self.inliers_indices = inliers_indices
        self.labes_inliers = labes_inliers
        self.centers = centers
        self.covars = covars
        self.record_shape = record_shape
        self.outliers_posterior = outliers_posterior



    def __call__(self, x, y=None):

        centers_posterior = pyro.param("centers", self.centers)
        scale_tril_posterior = pyro.param("scale_tril_posterior", self.covars,
                                          constraint=constraints.lower_cholesky)
        outliers_posterior_locs = pyro.param("outliers_posterior_loc", self.outliers_posterior[:,0])
        outliers_posterior_sigma = pyro.param("outliers_posterior_simga", self.outliers_posterior[:,1], constraint=constraints.positive)
        # print("前十个locs")
        # print(outliers_posterior_locs[:10])

        # outliers_exp = torch.log(0.03 + torch.exp(outliers_posterior[:,1]))
        pyro.sample("outliers", dist.Normal(loc=outliers_posterior_locs, scale=outliers_posterior_sigma).to_event(1))
        pyro.sample("locs", dist.MultivariateNormal(loc=centers_posterior, scale_tril=scale_tril_posterior).to_event(1))


################ over end
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="resnet18", required=False, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-weights_path', type=str,
                        default="checkpoint/resnet18/exp_822/resnet18-21-best-0.7736999988555908.pth",
                        required=False)
    parser.add_argument('-dataset', type=str, default="cifar100", help='cifar10 or cifar100')
    parser.add_argument('-resume', action='store_true', default=True, help='resume training')
    parser.add_argument('-n', type=int, default=50, help='number of samples for predictive distribution')
    args = parser.parse_args()

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=1024,
        shuffle=True,
        dataset=args.dataset
    )
    net = get_network(args)
    model_cnn_original = copy.deepcopy(net)
    net = net.to(device)
    Convertor.orig_to_bayes(net, prior=True)
    ############# 加载权重 #################
    if args.resume:
        weight_path = args.weights_path
        gd_weights = torch.load(weight_path, map_location="cpu")
    ############# state_dict_no_weight #################
    state_dict_no_weight = gd_weights["state_dict_no_weight"]

    model_cnn_original_weight = model_cnn_original.state_dict()
    name_conv_linear = []
    shape_record = {}
    for name, module in model_cnn_original.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            name_weight = name + ".weight"
            state_dict_no_weight[name_weight] = torch.zeros_like(module.weight)
            shape_record[name + ".weight"] = module.weight.shape
    model_cnn_original.load_state_dict(state_dict_no_weight)
    model_cnn_original.to(device)
    # weight_further_storage = model_cnn_original.state_dict()
    # weight_further_filter = {}
    # for name, value in weight_further_storage.items():
    #     if name not in name_conv_linear:
    #         weight_further_filter[name] = value
    ############# params_guide #################
    params_guide = gd_weights["params_guide"] # 包含了
    print(params_guide)
    # centers: torch.Size([3630, 2])
    # scale_tril_posterior: torch.Size([3630, 2, 2])
    # outliers_posterior: torch.Size([123676, 2])
    centers = params_guide["centers"].to(device)
    scale_tril_posterior = params_guide["scale_tril_posterior"].to(device)
    outliers_posterior_loc = params_guide["outliers_posterior_loc"].to(device)
    outliers_posterior_simga = params_guide["outliers_posterior_simga"].to(device)
    outliers_posterior = torch.stack((outliers_posterior_loc, outliers_posterior_simga), dim=1)
    ############# params_dict_indices #################
    params_dict_indices = gd_weights["params_dict_indices"]
    outliers_indices = params_dict_indices["outliers_indices"].to(device) # outliers indices
    filtered_labels = params_dict_indices["filtered_labels"].to(torch.int32).to(device)
    all_indices = torch.arange(filtered_labels.shape[0])
    # 或者使用更高效的方式通过布尔掩码
    mask = torch.ones(all_indices.shape[0], dtype=bool)
    mask[outliers_indices] = False
    inliers_indices = all_indices[mask] # 得到inlier indices

    ellipse_outdices = params_dict_indices["ellipse_outdices"]
    ellipse_outdices_dict = params_dict_indices["ellipse_outdices_dict"]
    ellipse_outdices_value_dict = params_dict_indices["ellipse_outdices_value_dict"]
    # params_dict_indices = {
    #     'outliers_indices': outliers_indices,  # 不用存储inliers, 根据outliers计算inliers就行
    #     'filtered_labels': filtered_labels.to(torch.int16),
    #     'ellipse_outdices': ellipse_outdices,
    #     'ellipse_outdices_dict': ellipse_outdices_dict,
    #     'ellipse_outdices_value_dict': ellipse_outdices_value_dict
    # }
    ############# 获取权重,包括mu和sigma,以及每个module的权重的大小 #################
    # list_mu = []
    # list_sigma = []
    # shape_record = {} # 记录每个layer的参数的大小，这个再之后放回参数有作用
    # for name, module in net.named_modules():
    #     if isinstance(module, (BayesConv2d, BayesLinear)):
    #         list_mu.extend(module.weight_mu.detach().cpu().flatten().numpy().tolist())
    #         list_sigma.extend(module.weight_sigma.detach().cpu().flatten().numpy().tolist())
    #         shape_record[name+".weight"] = module.weight_mu.shape
    # list_mu = torch.from_numpy(np.array(list_mu))
    # list_sigma = torch.from_numpy(np.array(list_sigma))
    ############# over #################

    ########### 定义gmm ############

    # print("----------------")
    # model_cnn, filtered_label, inliers_indices, outliers_indices, labes_inliers, centers, covars, ellipse_outdices, ellipse_outdices_dict, ellipse_outdices_value_dict, record_shape):
    #
    labels_inliers = filtered_labels[inliers_indices]
    model = ModelBNN(model_cnn_original, outliers_posterior, filtered_labels, inliers_indices, outliers_indices, labels_inliers, centers, scale_tril_posterior, ellipse_outdices, ellipse_outdices_dict, ellipse_outdices_value_dict, shape_record)
    guide = GuideBNN(model_cnn_original, outliers_posterior, inliers_indices, outliers_indices, labels_inliers, centers, scale_tril_posterior, shape_record)

    # model_cnn, model_weight, param, inliers_indices, outliers_indices, labes_inliers, centers, covars, record_shape
    optimizer = ClippedAdam({"lr": 0.00001})
    elbo = Trace_ELBO()
    svi = SVI(model, guide, optimizer, loss=elbo)
    for idx, pack_data in enumerate(cifar100_training_loader):
        # 在每次调用 SVI 之前清理参数存储
        pyro.clear_param_store()
        data_iters, labels_iter = pack_data
        data_iter = data_iters.to(device)
        labels_iter = labels_iter.to(device)
        loss = svi.step(data_iter, labels_iter)
        break
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # cifar100_test_loader = get_test_dataloader(
    #         settings.CIFAR100_TRAIN_MEAN,
    #         settings.CIFAR100_TRAIN_STD,
    #         num_workers=4,
    #         batch_size=1024,
    #         shuffle=True,
    #         dataset="cifar100"
    #     )
    # num_steps = 200
    # best_acc = 0.
    # loss_record = []
    # acc_record = []
    os.makedirs("pyro", exist_ok=True)
    ############ 输出参数 ###########

    acc, nll, ece = model_test_predictive(model, guide, cifar100_test_loader, num_samples=50)
