import argparse
import os
import shutil
import time
from utils import get_network
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
# used for logging to TensorBoard


parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar100', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('-net', default='wrn', type=str,
                    help='model selection')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
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
parser.add_argument('--resume', default='True', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('-weights_path', type=str, default="checkpoint/wrn20/cifar100/origin/wrn.pth", help='weight path')
parser.set_defaults(augment=True)

best_prec1 = 0
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
gradients = {}
def save_grad(module_name):
    def hook(grad):
        gradients[module_name] = grad.clone()
        # print(f"{module_name}的梯度已记录，形状: {grad.shape}")
    return hook
def main():
    global args, best_prec1
    args = parser.parse_args()


    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
        	transforms.ToTensor(),
        	transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
        						(4,4,4,4),mode='reflect').squeeze()),
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
    assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('data', train=True, download=True,
                         transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    print(f"args.droprate is {args.droprate}")
    input()
    # create model
    if args.net == "wrn":
        model = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                           args.widen_factor, dropRate=args.droprate)
    else:
        model = get_network(args)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.to(device)


    # optionally resume from a checkpoint
    if args.resume:

        print("=> loading checkpoint '{}'".format(args.weights_path))
        checkpoint = torch.load(args.weights_path)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = 0.0
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


    Convertor.orig_to_bayes(model, prior=True)
    model = model.to(device)
    cudnn.benchmark = True


    # 定义一个 backward hook 函数
    # conv2_x.0.residual_function.0.weight_mu torch.Size([64, 64, 3, 3])
    # conv2_x.0.residual_function.0.weight_sigma torch.Size([64, 64, 3, 3])
    # conv2_x.0.residual_function.0.bias_mu torch.Size([64])
    # conv2_x.0.residual_function.0.bias_sigma torch.Size([64])
    # 注册 backward hook
    print(model)

    for name, module in model.named_modules():
        if isinstance(module, (BayesConv2d, BayesLinear)):
            module.weight_mu.register_hook(save_grad(f"{name}.weight_mu"))
            module.weight_sigma.register_hook(save_grad(f"{name}.weight_sigma"))
            if module.bias is not None:
                module.bias_mu.register_hook(save_grad(f"{name}.bias_mu"))
                module.bias_sigma.register_hook(save_grad(f"{name}.bias_sigma"))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    weight_params = []
    bias_params = []
    conv_weight_params = []
    conv_sigma_params = []
    conv_weight_params_bias = []
    conv_sigma_params_bias = []

    linear_weight_params = []
    linear_sigma_params = []
    linear_weight_params_bias = []
    linear_sigma_params_bias = []

    # 遍历网络中的所有模块
    for name, module in model.named_modules():
        if isinstance(module, BayesConv2d):
            conv_weight_params.append(module.weight_mu)
            conv_sigma_params.append(module.weight_sigma)
            if module.bias_mu is not None:
                conv_weight_params_bias.append(module.bias_mu)
            if module.bias_sigma is not None:
                conv_sigma_params_bias.append(module.bias_sigma)
        elif isinstance(module, BayesLinear):
            linear_weight_params.append(module.weight_mu)
            linear_sigma_params.append(module.weight_sigma)
            if module.bias_mu is not None:
                linear_weight_params_bias.append(module.bias_mu)
            if module.bias_sigma is not None:
                linear_sigma_params_bias.append(module.bias_sigma)

    optimizer = optim.SGD([
        {'params': conv_weight_params, 'lr': 0.00001},
        {'params': conv_weight_params_bias, 'lr': 0.00001},
        {'params': conv_sigma_params, 'lr': 0.001},
        {'params': conv_sigma_params_bias, 'lr': 0.001},
        {'params': linear_weight_params, 'lr': 0.00001},
        {'params': linear_weight_params_bias, 'lr': 0.00001},
        {'params': linear_sigma_params, 'lr': 0.001},
        {'params': linear_sigma_params_bias, 'lr': 0.001}
    ], momentum=args.momentum, nesterov = args.nesterov,
                                weight_decay=args.weight_decay)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum, nesterov = args.nesterov,
    #                             weight_decay=args.weight_decay)

    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'net': model.state_dict(),
            'best_prec1': best_prec1,
            'grad': gradients
        }, is_best)
    print('Best accuracy: ', best_prec1)

def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))

def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    print("now is val")
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        with torch.no_grad():
            output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    print("val end")
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    p = "checkpoint/wrn20/cifar100/stochastic"
    os.makedirs(p,exist_ok=True)

    if is_best:
        filename = f"checkpoint/wrn20/cifar100/stochastic/wrn_{state['epoch']}-{state['best_prec1']}.pth"
        torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()