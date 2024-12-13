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

scaling_factor = {
  "min": [
    0.00012228660424362658,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -12.947212219238281
  ],
  "max": [
    7.999964949027126,
    8.48736771478237,
    8.973167500638938,
    9.452481768993627,
    9.925128063251766,
    10.394770541981519,
    10.811635473914432,
    11.293867117582742,
    11.773110825827766,
    12.248974776305648,
    12.721085244405165,
    13.189033979699555,
    13.644411232334877,
    14.073094462889278,
    14.464576232984038,
    14.811990889865044,
    15.15830120608507,
    15.510364709758887,
    15.872620274371885,
    16.34860212117229,
    16.775598523290164,
    17.267232982560255,
    17.75560763595966,
    18.23467056595263,
    18.698969207133995,
    19.14366880987464,
    19.56455244032151,
    19.95802098039758,
    20.321093127801905,
    20.65140539600957,
    20.94721211427172,
    7.999631881713867
  ]
}

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
    torch.cuda.set_device(2)
    device = "cuda:2"
    # device = "cpu"
    min_ = -12.947212219238281
    max_ = 7.999631881713867
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    params = torch.load('/home/koh/work/DeepBern-Nets/experiments/staliro/state_robust/state_robust_02/checkpoint_best_model.pth')
    input_dimension = len(scaling_factor['min'][:-1])
    model = FCModel([input_dimension,1024,1024,1024,1024,1], 8).to(device)
    input_bounds_ = torch.tensor([[0.0, 1.0] for _ in range(input_dimension)]).to(device)
    model.load_state_dict(params['model_state_dict'])
    model.input_bounds = input_bounds_

    is_FC_model = True
    batch_size = 512
    trainloader, testloader = load_staliro(
        batch_size=batch_size, flatten=is_FC_model, type_num=1
    )
    _, benchmark_testloader = load_staliro(
        batch_size=batch_size, flatten=is_FC_model, type_num=1
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