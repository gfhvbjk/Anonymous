import torch
import torch.nn as nn
import json
import copy
from pyro.infer import Predictive
import torchvision.datasets as datasets
import argparse
import torchvision.transforms as transforms
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
from networks.wide_resnet import WideResNet
import torch.optim as optim
from bnn.exp_bayesian_components import Convertor, BayesConv2d, BayesLinear
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
import sys
torch.manual_seed(3407)
np.random.seed(3407)
@torch.no_grad()
def model_test_predictive(model, guide, test_loader, num_samples=80):
    model.model_cnn.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient calculations
        predictive = Predictive(model, guide=guide, num_samples=num_samples, return_sites=("obs", "_RETURN"))
        correct = 0
        total = 0
        all_labels = []
        all_mean_preds = []
        all_outputs = []
        store_for_between_line = []
        i = 0
        for images, labels in test_loader:
            time_start = time.time()
            i += 1
            images, labels = images.to(device), labels.to(device)
            sampled_models = predictive(images)  # Run prediction
            outputs = sampled_models['obs']  # Get output
            mean_preds = torch.mean(sampled_models['_RETURN'], dim=0)  # Take the average of samples as the final output
            max_index = torch.argmax(mean_preds, dim=1)

            mode_predictions = torch.mode(outputs, dim=0)[0]  # Take the mode as the final prediction
            total += labels.size(0)
            correct += (max_index == labels).sum().item()
            # acc_temp = (mode_predictions == labels).sum().item() / labels.size(0)
            acc_temp = correct/total
            print("#" * 20)
            print("Now is running test: ", end="")
            time_end = time.time()
            print(f"Batch {i}/{len(test_loader)}, Acc: {acc_temp:.8f}, spend time: {time_end - time_start}")
            print("#" * 20)
            store_for_between_line.append(sampled_models['_RETURN'])
            all_labels.append(labels)


            all_mean_preds.append(mean_preds)
            all_outputs.append(outputs)

        # Compute overall accuracy
        sys.stdout.write('\n')
        accuracy = 100 * correct / total
        print(f'Accuracy on CIFAR-100 test set: {accuracy}%')
        # print(f'Top5_accuracy on CIFAR-100 test set: {top5_accuracy}%')

        # Concatenate all batches
        all_labels = torch.cat(all_labels)

        all_labels = all_labels.cpu().numpy()

        all_mean_preds = torch.cat(all_mean_preds)
        log_soft = F.log_softmax(all_mean_preds, dim=1)
        all_labels = torch.tensor(all_labels).to(device)
        nll = F.nll_loss(log_soft, all_labels).item()
        print(f"NLL: {nll}")

        # Compute ECE
        ece = compute_ece(all_mean_preds, all_labels)
        print(f"ECE: {ece}")

    model.model_cnn.train()  # Set model back to training mode
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

class ModelBNN():

    def __init__(self, model_cnn, outliers_posterior, filtered_label, inliers_indices, outliers_indices, labes_inliers, centers, covars, ellipse_outdices, ellipse_outdices_dict, ellipse_outdices_value_dict, record_shape):
        ###
        # param -> (n,2)
        # inliers_indices -> (m)
        # outliers_indices -> (k)
        ###
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
        # ellipse_outdices, ellipse_outdices_dict, ellipse_outdices_value_dict
        self.ellipse_outdices = ellipse_outdices
        self.ellipse_outdices_dict = ellipse_outdices_dict
        self.ellipse_outdices_value_dict = ellipse_outdices_value_dict
        self.outliers_posterior = outliers_posterior


    def reconstruct_model_cnn(self, sample_inliers, sample_outliers, x):


        weight_pre_places = torch.zeros((len(self.ellipse_outdices), 2), device=device,dtype=torch.double)
        pdf_values = torch.stack(
            [torch.from_numpy(np.array(self.ellipse_outdices_value_dict[idx])).to(device) for idx in self.ellipse_outdices])
        scores = F.softmax(pdf_values, dim=1)

        centers_indices = torch.tensor([self.ellipse_outdices_dict[idx] for idx in ellipse_outdices]).to(device)

        mask = (pdf_values > torch.exp(torch.tensor(12.0)).item()).sum(dim=1) == 2

        weight_pre_places[mask] = self.centers[centers_indices[mask, 0]] * scores[mask, 0][:, None] + self.centers[centers_indices[mask, 1]] * scores[mask, 1][:, None]

        weight_pre_places[~mask] = self.centers[centers_indices[~mask, 0]] * scores[~mask, 0][:, None]

        cluster_value = sample_inliers[:, 0]
        # print(torch.max(self.labes_inliers))
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
        self.model_cnn.eval()
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
        locs_gmm = pyro.sample("locs",
                           dist.MultivariateNormal(loc=self.centers, scale_tril=self.covars).to_event(1))
        logits_out = self.reconstruct_model_cnn(locs_gmm, outliers_sample, x)

        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Categorical(logits=logits_out), obs=y)
        return logits_out

class GuideBNN():
    def __init__(self, model_cnn, outliers_posterior, inliers_indices, outliers_indices, labes_inliers, centers, covars, record_shape):
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
        outliers_posterior_locs = pyro.param("outliers_posterior_loc", self.outliers_posterior[:,0])
        outliers_posterior_sigma = pyro.param("outliers_posterior_simga", self.outliers_posterior[:,1], constraint=constraints.positive)
        # print(outliers_posterior_locs[:10])

        # outliers_exp = torch.log(0.03 + torch.exp(outliers_posterior[:,1]))
        pyro.sample("outliers", dist.Normal(loc=outliers_posterior_locs, scale=outliers_posterior_sigma).to_event(1))
        pyro.sample("locs", dist.MultivariateNormal(loc=centers_posterior, scale_tril=scale_tril_posterior).to_event(1))


################ over end
if __name__ == '__main__':
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="resnet50", required=False, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=20, help='batch size for dataloader')
    parser.add_argument('-classes', type=int, default=100, help='classes')
    parser.add_argument('-weights_path', type=str,
                        default="checkpoint/wrn20/cifar100/2DGBNNs/2DGBNNs_13_81.17.pth",
                        required=False)
    # torch.Size([1883, 2])
    parser.add_argument('-dataset', type=str, default="cifar100", help='cifar10 or cifar100')
    parser.add_argument('-resume', action='store_true', default=True, help='resume training')
    parser.add_argument('--layers', default=28, type=int,
                        help='total number of layers (default: 28)')
    parser.add_argument('--widen-factor', default=10, type=int,
                        help='widen factor (default: 10)')
    parser.add_argument('--droprate', default=0, type=float,
                        help='dropout probability (default: 0.0)')
    args = parser.parse_args()

    cifar100_test_loader = get_test_dataloader(
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            num_workers=4,
            batch_size=800,
            shuffle=False,
            dataset=args.dataset
        )
    from networks.wide_resnet import WideResNet

    net = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                       args.widen_factor, dropRate=args.droprate)
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
    params_guide = gd_weights["params_guide"]
    centers = params_guide["centers_guide"].to(device)
    # input()
    scale_tril_posterior = params_guide["scale_tril_posterior"].to(device)
    outliers_posterior_loc = params_guide["outliers_posterior_loc"].to(device)
    outliers_posterior_simga = params_guide["outliers_posterior_sigma"].to(device)
    outliers_posterior = torch.stack((outliers_posterior_loc, outliers_posterior_simga), dim=1)
    ############# params_dict_indices #################
    params_dict_indices = gd_weights["params_dict_indices"]
    outliers_indices = params_dict_indices["outliers_indices"].to(device) # outliers indices
    filtered_labels = params_dict_indices["filtered_labels"].to(torch.int32).to(device)
    all_indices = torch.arange(filtered_labels.shape[0])
    mask = torch.ones(all_indices.shape[0], dtype=bool)
    mask[outliers_indices] = False
    inliers_indices = all_indices[mask] # 得到inlier indices

    ellipse_outdices = params_dict_indices["ellipse_outdices"]
    ellipse_outdices_dict = params_dict_indices["ellipse_outdices_dict"]
    ellipse_outdices_value_dict = params_dict_indices["ellipse_outdices_value_dict"]

    print(
        f"Statistics:\n"
        f"  - Outlier Points: {len(outliers_indices)}\n"
        f"  - Ellipse Points: {len(ellipse_outdices)}\n"
        f"  - Gaussian Points: {len(centers)}"
    )
    labels_inliers = filtered_labels[inliers_indices]
    model = ModelBNN(model_cnn_original, outliers_posterior, filtered_labels, inliers_indices, outliers_indices, labels_inliers, centers, scale_tril_posterior, ellipse_outdices, ellipse_outdices_dict, ellipse_outdices_value_dict, shape_record)
    guide = GuideBNN(model_cnn_original, outliers_posterior, inliers_indices, outliers_indices, labels_inliers, centers, scale_tril_posterior, shape_record)

    # model_cnn, model_weight, param, inliers_indices, outliers_indices, labes_inliers, centers, covars, record_shape
    optimizer = ClippedAdam({"lr": 0.00001})
    elbo = Trace_ELBO()
    svi = SVI(model, guide, optimizer, loss=elbo)


    acc, nll, ece = model_test_predictive(model, guide, cifar100_test_loader, num_samples=30)
