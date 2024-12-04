import argparse
import time
import torch

import os

import torchattacks

# os.environ["CUDA_VISIBLE_DEVICES"]="7"
from models.models import CNN7, CNNa, CNNb, CNNc, FCModel
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from datasets import load_cifar10, load_mnist, load_staliro

# from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import numpy as np
import yaml
import mlflow
from test import test_robust, test_robust_regression
from utils import ParamScheduler, RecursiveNamespace
import wandb

torch.manual_seed(123123)


def eval_only(
    model,
    trainloader,
    testloader,
    loss_fn,
    cfg,
    device="cuda",
    lirpa_model=None,
    start_epoch=0,
    benchmark_loader=None,
):
    best_model_loss = float('inf')
    test_eps = 0.1
    epochs = 50
    bounding_method = "bern"
    worst_loss = float('-inf')
    worst_loss_test = float('-inf')
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(tqdm(trainloader)):
            x, target = x.to(device), target.to(device)
            out = model(x)
            batch_loss = loss_fn(out, target)
            worst_loss = max(batch_loss.max().item(), worst_loss)
        for batch_idx, (x, target) in enumerate(tqdm(testloader)):
            x, target = x.to(device), target.to(device)
            out = model(x)
            batch_loss = loss_fn(out, target)
            worst_loss_test = max(batch_loss.max().item(), worst_loss_test)
    print(f"Worst loss in train: {worst_loss}")
    print(f"Worst loss in test: {worst_loss_test}")

if __name__ == "__main__":
    torch.cuda.set_device(1)
    # device = "cuda:1"
    device = "cpu"
    min_ = -12.947212219238281
    max_ = 7.999631881713867
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    params = torch.load('/home/koh/work/DeepBern-Nets/experiments/staliro/optuna/optuna_normalized_7/checkpoint_best_model.pth')
    model = FCModel([17,512,512,512,512,1], 4).to(device)
    input_bounds_ = torch.tensor([[0.0, 1.0] for _ in range(17)]).to(device)
    model.load_state_dict(params['model_state_dict'])
    model.input_bounds = input_bounds_

    is_FC_model = True
    batch_size = 512
    trainloader, testloader = load_staliro(
        batch_size=batch_size, flatten=is_FC_model
    )
    _, benchmark_testloader = load_staliro(
        batch_size=batch_size, flatten=is_FC_model
    )

    print("==>>> Trainig set size = {}".format(len(trainloader.dataset)))
    print("==>>> Test set size = {}".format(len(testloader.dataset)))
    print(
        "==>>> Robustness Test set size = {}".format(len(benchmark_testloader.dataset))
    )

    in_shape = torch.tensor(next(iter(trainloader))[0][0].shape)
    num_outs = len(trainloader.dataset.classes)

    print(model)

    start_epoch = 0

    bounding_method = "bern"
    lirpa_model = None
    cfg=None

    criterion = nn.L1Loss(reduction="none")
    eval_only(
        model,
        trainloader,
        testloader,
        criterion,
        cfg,
        device=device,
        lirpa_model=lirpa_model,
        start_epoch=start_epoch,
        benchmark_loader=benchmark_testloader,
    )
    data_point = trainloader.dataset[1][0].to(device)
    model(data_point.unsqueeze(0))