import torch
import torch.nn as nn
import json
import copy
from pyro.infer import Predictive
import argparse
import torch.distributions.constraints as constraints
import numpy as np
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
from pyro.optim import Adam, ClippedAdam, PyroLRScheduler,AdamW,SGD
from bnn.Bayes_Modules import Convertor, BayesConv2d, BayesLinear
from utils import get_network
import pyro
from pyro.infer import SVI
import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
torch.manual_seed(0)
np.random.seed(0)

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
        i = 0
        top5_correct = 0
        # Process test data in batches
        for images, labels in test_loader:
            # print(i)
            i += 1
            time_start = time.time()
            images, labels = images.to(device), labels.to(device)
            sampled_models = predictive(images)  # Run prediction
            outputs = sampled_models['obs']  # Get output
            mean_preds = torch.mean(sampled_models['_RETURN'], dim=0)  # Take the average of samples as the final output

            mode_predictions = torch.mode(outputs, dim=0)[0]  # Take the mode as the final prediction

            total += labels.size(0)
            correct += (mode_predictions == labels).sum().item()
            acc_temp = correct / total
            time_end = time.time()
            print("#" * 20)
            print("Now is running test: ", end="")
            print(f"Batch {i}/{len(test_loader)}, Acc: {acc_temp:.8f}, spend time: {time_end - time_start}")
            print("#" * 20)
            _, pred_top5 = mean_preds.topk(5, dim=1, largest=True, sorted=True)

            # Compare the top 5 predictions with the true labels
            correct_top5 = pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5))
            top5_correct += correct_top5.sum().item()

            # Collect all predictions and labels
            all_labels.append(labels)
            all_mean_preds.append(mean_preds)
            all_outputs.append(outputs)
            model.flag_ellipse = True

        # Compute overall accuracy
        top5_accuracy = top5_correct / total
        accuracy = 100 * correct / total
        print(f'Accuracy on CIFAR-100 test set: {accuracy}%')
        print(f'Top5_accuracy on CIFAR-100 test set: {top5_accuracy}%')

        # Concatenate all batches
        all_labels = torch.cat(all_labels)
        all_mean_preds = torch.cat(all_mean_preds)
        log_soft = F.log_softmax(all_mean_preds, dim=1)
        nll = F.nll_loss(log_soft, all_labels).item()
        print(f"NLL: {nll}")

        # Compute ECE
        ece = compute_ece(all_mean_preds, all_labels)
        print(f"ECE: {ece}")

    model.model_cnn.train()  # Set model back to training mode
    return accuracy, top5_accuracy, nll, ece
def eval_training(model_cnn, epoch=0):

    start = time.time()
    model_cnn.eval()
    loss_function = nn.CrossEntropyLoss()

    test_loss = 0.0 # cost function error
    correct = 0.0
    total = 0
    for (images, labels) in val_loader:

        if args.gpu:
            images = images.to(device)
            labels = labels.to(device)

        outputs = model_cnn(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        total += labels.shape[0]

        print(f"acc {correct/total}")

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(val_loader.dataset),
        correct.float() / len(val_loader.dataset),
        finish - start
    ))

#
def compute_ece(preds, labels, n_bins=10):
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
        self.flag = True
        self.flag_ellipse = True

    def reconstruct_model_cnn(self, sample_inliers, sample_outliers,x):
        # print("reconstruct_model_cnn")

        # Update model parameters with inliers and outliers
        # assert len(self.inliers_indices) + len(self.outliers_indices) == self.param_weight_model.shape[0]
        if self.flag_ellipse == True:
            weight_pre_places = torch.zeros((len(self.ellipse_outdices), 2), device=device,dtype=torch.double)
            pdf_values = torch.stack(
                [torch.from_numpy(np.array(self.ellipse_outdices_value_dict[idx])).to(device) for idx in self.ellipse_outdices])
            scores = F.softmax(pdf_values, dim=1)

            centers_indices = torch.tensor([self.ellipse_outdices_dict[idx] for idx in ellipse_outdices]).to(device)

            mask = (pdf_values > torch.exp(torch.tensor(12.0)).item()).sum(dim=1) == 2
            # 计算加权位置（双中心和单中心的情况）

            weight_pre_places[mask] = self.centers[centers_indices[mask, 0]] * scores[mask, 0][:, None] + self.centers[centers_indices[mask, 1]] * scores[mask, 1][:, None]

            weight_pre_places[~mask] = self.centers[centers_indices[~mask, 0]] * scores[~mask, 0][:, None]
            self.weight_pure_cnn_mu[self.ellipse_outdices] = weight_pre_places[:, 0].to(torch.float32)

        cluster_value = sample_inliers[:, 0]
        # print(torch.max(self.labes_inliers))
        cluster_value_inliers = cluster_value[self.labes_inliers]

        # Update weights with inlier and outlier cluster values
        self.weight_pure_cnn_mu[self.inliers_indices] = cluster_value_inliers
        self.weight_pure_cnn_mu[self.outliers_indices] = sample_outliers
        self.weight_pure_cnn_sigma[self.inliers_indices] = sample_inliers[:, 1][self.labes_inliers]
        # Ensure correct ordering of weights in self.weight_pure_cnn_mu
        keys_name = list(self.record_shape.keys())
        weight_pure_cnn_mu_copy = self.weight_pure_cnn_mu.detach().clone()

        idx = 0
        with torch.no_grad():
            weight_copy = weight_pure_cnn_mu_copy  # 使用一个局部变量以提高效率
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
            outliers_loc = self.outliers_posterior[:,0]
            outliers_scale = self.outliers_posterior[:,1].clamp(min=1e-10)
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

        # if self.flag_ellipse == True:
        logits_out = self.reconstruct_model_cnn(locs_gmm, outliers_sample, x)
        # eval_training(self.model_cnn, 0)
        # input()
        self.flag_ellipse = False
        # if self.flag == True:
        #     self.reconstruct_model_cnn(locs_gmm, outliers_sample)
        #     self.flag = False
        #
        # logits_out = self.model_cnn(x)

        with pyro.plate("data_plate", batch_size):
            pyro.sample("obs", dist.Categorical(logits=logits_out), obs=y)

        pyro.deterministic("logits_out", logits_out)
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
        num_outliers = len(self.outliers_indices)
        num_clusters = self.centers.shape[0]

        centers_posterior = pyro.param("centers_guide", self.centers)
        scale_tril_posterior = pyro.param(
            "scale_tril_posterior", self.covars, constraint=constraints.lower_cholesky)
        outliers_posterior_locs = pyro.param(
            "outliers_posterior_loc", self.outliers_posterior[:, 0])
        outliers_posterior_sigma = pyro.param(
            "outliers_posterior_sigma", self.outliers_posterior[:, 1], constraint=constraints.positive)

        with pyro.plate("outliers_plate", num_outliers):
            pyro.sample(
                "outliers",
                dist.Normal(loc=outliers_posterior_locs, scale=outliers_posterior_sigma)
            )

        with pyro.plate("clusters_plate", num_clusters):
            pyro.sample(
                "locs",
                dist.MultivariateNormal(loc=centers_posterior, scale_tril=scale_tril_posterior)
            )


################ over end
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="resnet18imagenet", required=False, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=200, help='batch size for dataloader')
    parser.add_argument('-weights_path', type=str,
                        default="checkpoint/resnet18/imagenet/2DGBNNs/2DGBNNs.pth",
                        required=False)
    parser.add_argument('-dataset', type=str, default="cifar100", help='cifar10 or cifar100')
    parser.add_argument('-resume', action='store_true', default=True, help='resume training')
    parser.add_argument('-dataset_path', type=str, default="data/imagenet", help='path to dataset')
    args = parser.parse_args()

    data_dir = args.dataset_path



    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    val_dataset = datasets.ImageFolder(root=f'{data_dir}/val', transform=val_transform)

    val_loader = DataLoader(val_dataset, batch_size=args.b, shuffle=False, num_workers=4)
    net = get_network(args)
    model_cnn_original = copy.deepcopy(net)
    net = net.to(device)
    Convertor.orig_to_bayes(net, prior=True)
    ############# loading weights#################
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
            shape_record[name_weight] = module.weight.shape

    model_cnn_original.load_state_dict(state_dict_no_weight)
    model_cnn_original.to(device)

    ############# params_guide #################
    params_guide = gd_weights["params_guide"]

    centers = params_guide["centers_guide"].to(device)

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
    inliers_indices = all_indices[mask]

    ellipse_outdices = params_dict_indices["ellipse_outdices"]
    print(
        f"Statistics:\n"
        f"  - Outlier Points: {len(outliers_indices)}\n"
        f"  - Ellipse Points: {len(ellipse_outdices)}\n"
        f"  - Gaussian Points: {len(centers)}"
    )

    ellipse_outdices_dict = params_dict_indices["ellipse_outdices_dict"]
    ellipse_outdices_value_dict = params_dict_indices["ellipse_outdices_value_dict"]

    labels_inliers = filtered_labels[inliers_indices]
    model = ModelBNN(model_cnn_original, outliers_posterior, filtered_labels, inliers_indices, outliers_indices, labels_inliers, centers, scale_tril_posterior, ellipse_outdices, ellipse_outdices_dict, ellipse_outdices_value_dict, shape_record)
    guide = GuideBNN(model_cnn_original, outliers_posterior, inliers_indices, outliers_indices, labels_inliers, centers, scale_tril_posterior, shape_record)

    # model_cnn, model_weight, param, inliers_indices, outliers_indices, labes_inliers, centers, covars, record_shape
    optimizer = ClippedAdam({"lr": 0.00001})
    elbo = Trace_ELBO()
    svi = SVI(model, guide, optimizer, loss=elbo)
    acc, acc50, nll, ece = model_test_predictive(model, guide, val_loader, num_samples=30)
