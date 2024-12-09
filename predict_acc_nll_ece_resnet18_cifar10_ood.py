import torch
import torch.nn as nn
import json
import copy
import argparse
import torch.distributions.constraints as constraints
import numpy as np
import torch.nn.functional as F
import os
import pyro
from pyro.infer import SVI, Trace_ELBO, Predictive
import pyro.distributions as dist
from pyro.optim import ClippedAdam
from pyro import poutine
import time
import sys
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# 从您自己的模块中导入必要的函数和设置
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, init_weights
from bnn.Bayes_Modules import Convertor, BayesConv2d, BayesLinear
from conf import settings

# 设置随机种子
torch.manual_seed(settings.SEED)
np.random.seed(settings.SEED)

# 选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def compute_metrics(id_scores, ood_scores):
    # 创建标签：在分布内为1，分布外为0
    labels = [1] * len(id_scores) + [0] * len(ood_scores)
    scores = id_scores + ood_scores

    # 将列表转换为NumPy数组
    labels = np.array(labels)
    scores = np.array(scores)

    # 计算AUROC
    auroc = roc_auc_score(labels, scores)

    # 计算AUPR
    aupr = average_precision_score(labels, scores)

    # 计算FPR95
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.where(tpr >= 0.85)[0]
    if len(idx) == 0:
        fpr95 = fpr[-1]
    else:
        fpr95 = fpr[idx[0]]

    return auroc, aupr, fpr95

class ModelBNN():
    def __init__(self, model_cnn, outliers_posterior, filtered_label, inliers_indices, outliers_indices,
                 labes_inliers, centers, covars, ellipse_outdices, ellipse_outdices_dict,
                 ellipse_outdices_value_dict, record_shape):
        super().__init__()
        self.model_cnn = model_cnn.to(device)
        self.outliers_indices = outliers_indices
        self.inliers_indices = inliers_indices
        self.labes_inliers = labes_inliers
        self.centers = centers
        self.covars = covars
        self.record_shape = record_shape
        self.weight_pure_cnn_mu = torch.zeros_like(filtered_label, dtype=torch.float32)
        self.weight_pure_cnn_sigma = torch.zeros_like(filtered_label, dtype=torch.float32)
        self.ellipse_outdices = ellipse_outdices
        self.ellipse_outdices_dict = ellipse_outdices_dict
        self.ellipse_outdices_value_dict = ellipse_outdices_value_dict
        self.outliers_posterior = outliers_posterior
        self.flag = True
        self.flag_ellipse = True

    def reconstruct_model_cnn(self, sample_inliers, sample_outliers, x):
        if self.flag_ellipse:
            weight_pre_places = torch.zeros((len(self.ellipse_outdices), 2), device=device, dtype=torch.double)
            pdf_values = torch.stack(
                [torch.from_numpy(np.array(self.ellipse_outdices_value_dict[idx])).to(device) for idx in self.ellipse_outdices])
            scores = F.softmax(pdf_values, dim=1)
            centers_indices = torch.tensor([self.ellipse_outdices_dict[idx] for idx in self.ellipse_outdices]).to(device)
            mask = (pdf_values > torch.exp(torch.tensor(12.0))).sum(dim=1) == 2
            weight_pre_places[mask] = self.centers[centers_indices[mask, 0]] * scores[mask, 0][:, None] + \
                                      self.centers[centers_indices[mask, 1]] * scores[mask, 1][:, None]
            weight_pre_places[~mask] = self.centers[centers_indices[~mask, 0]] * scores[~mask, 0][:, None]
            self.weight_pure_cnn_mu[self.ellipse_outdices] = weight_pre_places[:, 0].to(torch.float32)

        cluster_value = sample_inliers[:, 0]
        cluster_value_inliers = cluster_value[self.labes_inliers]

        # 更新权重
        self.weight_pure_cnn_mu[self.inliers_indices] = cluster_value_inliers
        self.weight_pure_cnn_mu[self.outliers_indices] = sample_outliers
        self.weight_pure_cnn_sigma[self.inliers_indices] = sample_inliers[:, 1][self.labes_inliers]

        # 确保权重的顺序正确
        keys_name = list(self.record_shape.keys())
        weight_pure_cnn_mu_copy = self.weight_pure_cnn_mu.detach().clone()

        idx = 0
        with torch.no_grad():
            weight_copy = weight_pure_cnn_mu_copy
            for name, module in self.model_cnn.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    name_record, _ = keys_name[idx].rsplit(".", 1)
                    assert name_record == name, "Check order of list_mu and list_sigma"
                    num_elements = module.weight.numel()
                    module.weight.copy_(weight_copy[:num_elements].view_as(module.weight))
                    module.weight.requires_grad = False
                    weight_copy = weight_copy[num_elements:]
                    idx += 1
        self.model_cnn.eval()
        logits_out = self.model_cnn(x)
        return logits_out

    def __call__(self, x, y=None):
        batch_size = x.shape[0]
        num_outliers = len(self.outliers_indices)

        with pyro.plate("outliers_plate", num_outliers):
            outliers_loc = self.outliers_posterior[:, 0]
            outliers_scale = self.outliers_posterior[:, 1].clamp(min=1e-10)
            outliers_sample = pyro.sample(
                "outliers",
                dist.Normal(loc=outliers_loc, scale=outliers_scale)
            )

        num_clusters = self.centers.shape[0]
        with pyro.plate("clusters_plate", num_clusters):
            locs_gmm = pyro.sample(
                "locs",
                dist.MultivariateNormal(loc=self.centers, scale_tril=self.covars)
            )

        logits_out = self.reconstruct_model_cnn(locs_gmm, outliers_sample, x)
        self.flag_ellipse = False

        with pyro.plate("data_plate", batch_size):
            pyro.sample("obs", dist.Categorical(logits=logits_out), obs=y)

        pyro.deterministic("logits_out", logits_out)
        return logits_out

class GuideBNN():
    def __init__(self, model_cnn, outliers_posterior, inliers_indices, outliers_indices,
                 labes_inliers, centers, covars, record_shape):
        super().__init__()
        self.model_cnn = model_cnn
        self.outliers_indices = outliers_indices
        self.inliers_indices = inliers_indices
        self.labes_inliers = labes_inliers
        self.centers = centers
        self.covars = covars
        self.record_shape = record_shape
        self.outliers_posterior = outliers_posterior

    def __call__(self, x, y=None):
        centers_posterior = pyro.param("centers_guide", self.centers)
        scale_tril_posterior = pyro.param("scale_tril_posterior", self.covars,
                                          constraint=constraints.lower_cholesky)
        outliers_posterior_locs = pyro.param("outliers_posterior_loc", self.outliers_posterior[:, 0])
        outliers_posterior_sigma = pyro.param("outliers_posterior_sigma", self.outliers_posterior[:, 1],
                                              constraint=constraints.positive)

        pyro.sample("outliers", dist.Normal(loc=outliers_posterior_locs, scale=outliers_posterior_sigma).to_event(1))
        pyro.sample("locs", dist.MultivariateNormal(loc=centers_posterior, scale_tril=scale_tril_posterior).to_event(1))

def model_test_predictive_OOD(model, guide, id_loader, ood_loader, num_samples=80):
    model.model_cnn.eval()  # 设置模型为评估模式
    id_scores = []
    ood_scores = []
    predictive = Predictive(model, guide=guide, num_samples=num_samples, return_sites=("obs", "_RETURN"))
    with torch.no_grad():
        # 在分布内数据
        for images, _ in id_loader:
            images = images.to(device)
            sampled_models = predictive(images)
            mean_preds = torch.mean(sampled_models['_RETURN'], dim=0)
            softmaxes = F.softmax(mean_preds, dim=1)
            max_probs, _ = torch.max(softmaxes, dim=1)
            id_scores.extend(max_probs.cpu().numpy())

        # 分布外数据
        for images, _ in ood_loader:
            images = images.to(device)
            sampled_models = predictive(images)
            mean_preds = torch.mean(sampled_models['_RETURN'], dim=0)
            softmaxes = F.softmax(mean_preds, dim=1)
            max_probs, _ = torch.max(softmaxes, dim=1)
            ood_scores.extend(max_probs.cpu().numpy())

    # 计算评估指标
    auroc, aupr, fpr95 = compute_metrics(id_scores, ood_scores)
    print(f"AUROC on OOD dataset: {auroc:.4f}")
    print(f"AUPR on OOD dataset: {aupr:.4f}")
    print(f"FPR95 on OOD dataset: {fpr95:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="resnet18", required=False, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-classes', type=int, default=100, help='number of classes')
    parser.add_argument('-weights_path', type=str,
                        default="checkpoint/resnet18/cifar10/2DGBNNs/2DGBNNs.pth",
                        required=False)
    parser.add_argument('-dataset', type=str, default="cifar10", help='cifar10 or cifar100')
    parser.add_argument('-resume', action='store_true', default=True, help='resume training')
    args = parser.parse_args()

    # CIFAR-10 测试集加载（在分布内数据）
    cifar_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=1024,
        shuffle=False,
        dataset=args.dataset
    )

    # SVHN 测试集加载（分布外数据）
    transform_ood = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728),
                             (0.1980, 0.2010, 0.1970))
    ])

    ood_dataset = datasets.SVHN(
        root='./data',
        split='test',
        transform=transform_ood,
        download=True
    )

    ood_loader = DataLoader(
        ood_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 加载预训练模型
    net = get_network(args)
    model_cnn_original = copy.deepcopy(net)
    net = net.to(device)
    Convertor.orig_to_bayes(net, prior=True)

    if args.resume:
        weight_path = args.weights_path
        gd_weights = torch.load(weight_path, map_location="cpu")

    # 加载模型权重
    state_dict_no_weight = gd_weights["state_dict_no_weight"]
    shape_record = {}
    for name, module in model_cnn_original.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            name_weight = name + ".weight"
            state_dict_no_weight[name_weight] = torch.zeros_like(module.weight)
            shape_record[name + ".weight"] = module.weight.shape
    model_cnn_original.load_state_dict(state_dict_no_weight)
    model_cnn_original.to(device)

    # 加载引导参数
    params_guide = gd_weights["params_guide"]
    centers = params_guide["centers_guide"].to(device)
    scale_tril_posterior = params_guide["scale_tril_posterior"].to(device)
    outliers_posterior_loc = params_guide["outliers_posterior_loc"].to(device)
    outliers_posterior_sigma = params_guide["outliers_posterior_sigma"].to(device)
    outliers_posterior = torch.stack((outliers_posterior_loc, outliers_posterior_sigma), dim=1)

    # 加载索引
    params_dict_indices = gd_weights["params_dict_indices"]
    outliers_indices = params_dict_indices["outliers_indices"].to(device)
    filtered_labels = params_dict_indices["filtered_labels"].to(torch.int32).to(device)
    all_indices = torch.arange(filtered_labels.shape[0])
    mask = torch.ones(all_indices.shape[0], dtype=bool)
    mask[outliers_indices] = False
    inliers_indices = all_indices[mask]

    ellipse_outdices = params_dict_indices["ellipse_outdices"]
    ellipse_outdices_dict = params_dict_indices["ellipse_outdices_dict"]
    ellipse_outdices_value_dict = params_dict_indices["ellipse_outdices_value_dict"]

    labels_inliers = filtered_labels[inliers_indices]

    # 实例化模型和引导
    model = ModelBNN(model_cnn_original, outliers_posterior, filtered_labels, inliers_indices, outliers_indices,
                     labels_inliers, centers, scale_tril_posterior, ellipse_outdices, ellipse_outdices_dict,
                     ellipse_outdices_value_dict, shape_record)
    guide = GuideBNN(model_cnn_original, outliers_posterior, inliers_indices, outliers_indices,
                     labels_inliers, centers, scale_tril_posterior, shape_record)

    optimizer = ClippedAdam({"lr": 0.00001})
    elbo = Trace_ELBO()
    svi = SVI(model, guide, optimizer, loss=elbo)

    os.makedirs("pyro", exist_ok=True)

    # 执行OOD检测
    model_test_predictive_OOD(model, guide, cifar_test_loader, ood_loader, num_samples=100)
