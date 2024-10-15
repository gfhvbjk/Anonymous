# Stochastic Weight Sharing for Bayesian Neural Networks

---

## Overview


- This repository contains the source code and documentation for the paper "Stochastic Weight Sharing for Bayesian Neural Networks" submitted to the AISTATS conference. 


## Repository Structure

```
.
├── .git/                                # Git repository files
├── bnn/                                 # Bayesian Neural Network-related code
├── checkpoints/                         # Directory for storing checkpoints
├── conf/                                # Configuration files
├── data/                                # Directory for datasets
├── models/                              # Directory for model definitions
├── params/                              # Directory for temp storing parameter files
├── README.md                            # Project overview and setup instructions
├── gmm_train.py                         # Training script for Gaussian Mixture Model
├── stochastic_nn.py                     # Script for stochastic neural network
├── train.py                             # Training script
├── utils.py                             # Utility functions
└── visualize_stochastic_weights_scatter.py  # Visualization script for stochastic weights scatter
```

## Installation

```
To get started, clone the repository and install the required dependencies:

git clone git@github.com:gfhvbjk/Anonymous.git
cd Anonymouse
pip install -r requirements.txt
```

Note: we use **pyro** as a VI tool, therefore, pls install pyro as follows:

```bash
pip3 install pyro-ppl
```

## Usage

### Data Preparation

- CIFAR-10, CIFAR-100, and MINST will be downloaded automatically
- ImageNet1k needs to be put into the `data` folder
    - put the ImageNet1k as "data/train" and "data/val" (recommendation)
    - or you can specify the specific the path of ImageNet by adding `-dataset_path = your path` (not recommendation)


## Evaluation and Checkpoints

We provide the following `Evaluation` scripts in this repository.

#### ImageNet1k by ResNet-18
- ImageNet1k by ResNet-18 script: predict_acc_nll_ece_resnet18_imagenet.py
    - Download our pre-trained model [here](https://drive.google.com/file/d/16KPKHEAaV7o-x_C-orPRCOxJJdsm9YLi/view?usp=drive_link) and put it under **checkpoint/resnet18/imagenet/2DGBNNs/**
    - ```bash
      python predict_acc_nll_ece_resnet18_imagenet.py
      ```
    - Or you can put the pre-trained model anywhere by running
    - ```bash
      python predict_acc_nll_ece_resnet18_imagenet.py -weights_path your_path
      ```
- The outputs include `Outliers`, `Ellipses`, `Gaussians`, `Accuracy`, `NLL`, and `ECE`

|       Metric        |      Value       |
|:-------------------:|:----------------:|
|      Outliers       |      23013       |
|      Ellipses       |      10885       |
|      Gaussians      |       2217       |
|      Accuracy       |   68.11 ± 0.03   |
|         NLL         |  1.250 ± 0.005   |
|         ECE         |  0.019 ± 0.003   |

#### CIFAR-100 by ResNet-18

- CIFAR-100 by ResNet-18 script: predict_acc_nll_ece_resnet18_cifar100.py
    - Download our pre-trained model [here](https://drive.google.com/file/d/1XlQud73NjqE_g4ymi0lglLYvus9xs_oL/view?usp=drive_link) and put it under **checkpoint/resnet18/cifar100/2DGBNNs/**
    - ```bash
      python predict_acc_nll_ece_resnet18_cifar100.py
      ```
    - Or you can put the pre-trained model anywhere by running
    - ```bash
      python predict_acc_nll_ece_resnet18_cifar100.py -weights_path your_path
      ```
- The outputs include `Outliers`, `Ellipses`, `Gaussians`, `Accuracy`, `NLL`, and `ECE`

|       Metric        |     Value     |
|:-------------------:|:-------------:|
|      Outliers       |     14624     |
|      Ellipses       |      260      |
|      Gaussians      |     2387      |
|      Accuracy       |  74.7 ± 0.1   |
|         NLL         | 1.049 ± 0.003 |
|         ECE         | 0.042± 0.003  |


#### CIFAR-10 by ResNet-18

- CIFAR-10 by ResNet-18 script: predict_acc_nll_ece_resnet18_cifar10.py
    - Download our pre-trained model [here](https://drive.google.com/file/d/1G1uvjUKeQ3ir7RIzagxVTjryho3p8JB_/view?usp=drive_link) and put it under **checkpoint/resnet18/cifar10/2DGBNNs/**
    - ```bash
      python predict_acc_nll_ece_resnet18_cifar10.py
      ```
    - Or you can put the pre-trained model anywhere by running
    - ```bash
      python predict_acc_nll_ece_resnet18_cifar10.py -weights_path your_path
      ```
- The outputs include `Outliers`, `Ellipses`, `Gaussians`, `Accuracy`, `NLL`, and `ECE`

|       Metric        |     Value     |
|:-------------------:|:-------------:|
|      Outliers       |    123310     |
|      Ellipses       |      57       |
|      Gaussians      |     1569      |
|      Accuracy       |  91.8 ± 0.1   |
|         NLL         | 0.313 ± 0.003 |
|         ECE         |    0.034 ± 0.003     |




## Train
> During training, three stages in total, including weights init, stochastic network training, and finally the `2DGBNNs` training.


- We provide all stage scripts and instructions for training of WRN-28-10 in CIFAR-100 step by step. Also, `checkpoint(pre-trained weights)` is provided if it exists and can extremely save you time by just putting them into the corresponding folder.
- All pre-trained can be downloaded in this folder. [here](https://drive.google.com/drive/folders/1_j9tCvX91UT4EEBeXOZ3Nm7owKi_km4k?usp=drive_link)
- **But we highly recommend you download them one by one following the steps.**
### Algorithm: Scaling BNNs to Large Models and Datasets

## 2DGBNN

**Input:**  
- NN architecture $f^{\mathbf{w}}$  
- Training data $\mathcal{D} = \{(\mathbf{X}, \mathbf{y})\}$  
- Algorithm thresholds: $\tau_w$, $\tau_d$, $\tau_g$, $\tau_v$  
- BNN prior $p(\mathbf{w})$  

**Output:**  
- Stochastic weight-sharing trained BNN
## Stage 0: Pre-trained model

1. **Train deterministic neural network**

Run the following script:
```bash
python train_deterministic_network.py -net wrn -dataset cifar100
```

You can download our pre-trained model [here](https://drive.google.com/file/d/1G1uvjUKeQ3ir7RIzagxVTjryho3p8JB_/view?usp=drive_link)
And put it under "checkpoint/wrn/cifar100/origin"

---
## Stage 1: Initialise GMM

1. **Initialize GMM Parameters**  
   Initialize $\mu$, $\sigma$ according to $p(\mathbf{w})$.

2. **Pre-training Loop**  
   For each epoch:
   - Sample weights:  
     $$\mathbf{w} = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
   - Update $\mu$, $\sigma$ by training on $\mathcal{D}$.

3. **Identify Outliers**  
   For each weight $w_i$:
   - If $|w_i| > \tau_w$ **or** $|\nabla_{w_i}|$ is in the top $1\%$, then $w_i$ is an outlier.
   - Else, $w_i$ is an inlier.
   - 
4. **Learn GMM on Inliers**  
   Learn GMM on inlier parameters $\Theta_{in}$.

> Pick a deterministic pre-trained model and get its path 

Run the following script:
```bash
python exp_stochastic_nn.py -net wrn -dataset cifar100 -weights_path "checkpoint/wrn/origin/wrn.pth" (pls it replace by your path)
```
This pre-trained model can downloaded in [here](https://drive.google.com/file/d/1QUOlqPydz0pH7tDYnOGenN6E4rm7Jn1D/view?usp=drive_link)

> Now, we need to initialize the GMM, pls pick a stochastic neural network and get its path

Run the following script:

Note that: we need to install cuml and cudf for GMM weights initiation (must installation)
Install them by following:
```bash
pip install cudf-cu11 cuml-cu11
```
Then run the following bash command:
```bash
python init_gmm.py -weights_path "checkpoint/wrn/cifar100/stochastic/wrn_stochastic.pth" (pls replace it by your path)
```
This pre-trained model can download [here](https://drive.google.com/file/d/1QUOlqPydz0pH7tDYnOGenN6E4rm7Jn1D/view?usp=drive_link)

> The initiation gmm configuration will be stored under "checkpoint/wrn/cifar100/init_gmm" 

---
## Stage 2: Refine GMM

5. **Mahalanobis Distance Check**  
   For each inlier weight $w_i$:
   - Perform Mahalanobis distance check.
   - If $w_i$ is outside the 95th percentile:
     - Assign $w_i$ to multiple clusters.
   - Else:
     - Assign $w_i$ to the closest Gaussian.

6. **Alpha-Blending for Ellipse Points**  
   Apply alpha-blending for ellipse points.

7. **Merge Gaussians**  
   Repeat until no more Gaussians can be merged:
   - For each pair $(\mathcal{N}_1, \mathcal{N}_2)$ in GMM:
     - If $W(\mathcal{N}_1, \mathcal{N}_2) < \tau_d$, $\Delta_g < \tau_g$, and $\Delta_v < \tau_v$:
       - Merge $\mathcal{N}_1$ and $\mathcal{N}_2$.

8. **Final Training Loop**  
   For each epoch:
   - For each weight $w_i$:
     - If $w_i$ is an inlier:
       - Sample $w_i \sim \sum_{k=1}^{K} \pi_k \mathcal{N}(\mu_k, \Sigma_k)$.
     - Else:
       - Use $w_i \sim \mathcal{N}(\mu_{w_i}, \sigma^2_{w_i})$.
   - Perform minimizing step for $\hat{\mathcal{L}}(\mathcal{D}, q)$.

- To train the model, run the following command:

```bash
python 2DGBNNs_train.py -weights_stochastic_path "checkpoint/wrn/cifar100/stochastic/stochastic_wrn.pth" (pls replace it by your path) \
-weights_path_origin "checkpoint/wrn/cifar100/origin/wrn.pth" (pls replace it by your path) \
-init_gmm "checkpoint/wrn/cifar100/init_gmm/clusters_gpu.json" (pls replace it by your path) \
-output "checkpoint/wrn/cifar100/2DGBNNs"
```

This pre-trained model can download [here](https://drive.google.com/file/d/1L9n2hV0HV0YiYujbU2vfKjc-Sh-tI6br/view?usp=drive_link)


## License

```
This project is licensed under the MIT License. See the `LICENSE` file for details.
```
