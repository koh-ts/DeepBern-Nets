import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import os
import glob
import json

class StaliroDataset(Dataset):
    def __init__(self, train=True, transform=None):
        # self.data_path_train = "/home/koh/work/DeepBern-Nets/data/staliro/20241105_132530.json"
        self.data_path_train = "/home/koh/work/matiec_rampo/examples/tankcontrol_flowrate/data/done/data_80.json"
        self.data_path_test = "/home/koh/work/matiec_rampo/examples/tankcontrol_flowrate/data/done/data_20.json"
        # self.data_path_test = "/home/koh/work/DeepBern-Nets/data/staliro/20241105_132534.json"
        self.train = train
        self.classes = [0]
        self.transform = transform
        if self.train:
            with open(self.data_path_train, 'r') as f:
                self.data = json.load(f)
        else:
            with open(self.data_path_test, 'r') as f:
                self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_signal = [self.data[idx]['init_cond']] + self.data[idx]['samples']
        if not self.transform == None:
            input_signal = self.transform(torch.tensor(input_signal))
        else:
            input_signal = torch.tensor(input_signal)
        return input_signal, torch.tensor([self.data[idx]['robustness']], dtype=torch.float32)

class MinMaxNormalize1D(object):
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value
        self.range = max_value - min_value

    def __call__(self, tensor):
        if self.range == 0:
            return tensor - self.min
        else:
            return (tensor - self.min) / self.range

def load_mnist(root_dir="./data", batch_size=512, flatten=True, samples_dist=0):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    if flatten:
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.0,), (1.0,)),
                transforms.Lambda(torch.flatten),
            ]
        )
    else:
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
        )
    train_set = datasets.MNIST(
        root=root_dir, train=True, transform=trans, download=True
    )
    test_set = datasets.MNIST(
        root=root_dir, train=False, transform=trans, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=8
    )
    if samples_dist > 0:
        sampler = torch.utils.data.SubsetRandomSampler(
            [i for i in range(0, 10000, samples_dist)]
        )
    else:
        sampler = None
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        sampler=sampler,
    )
    return train_loader, test_loader


def load_cifar10(root_dir="./data", batch_size=64, flatten=True, samples_dist=0):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    if flatten:
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.0,), (1.0,)),
                transforms.Lambda(torch.flatten),
            ]
        )
    else:
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
        )
    train_set = datasets.CIFAR10(
        root=root_dir, train=True, transform=trans, download=True
    )
    test_set = datasets.CIFAR10(
        root=root_dir, train=False, transform=trans, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    if samples_dist > 0:
        sampler = torch.utils.data.SubsetRandomSampler(
            [i for i in range(0, 10000, samples_dist)]
        )
    else:
        sampler = None
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        sampler=sampler,
    )

    return train_loader, test_loader

def load_staliro(root_dir="./data", batch_size=64, flatten=True, samples_dist=0):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    data_path_train = "/home/koh/work/DeepBern-Nets/data/staliro/20241105_132530.json"
    data_path_test = "/home/koh/work/DeepBern-Nets/data/staliro/20241105_132534.json"

    min_value, max_value = compute_min_max([data_path_train, data_path_test])

    trans = transforms.Compose(
        [
            MinMaxNormalize1D(min_value, max_value),
        ]
    )

    train_set = StaliroDataset(train=True, transform=trans)
    test_set = StaliroDataset(train=False, transform=trans)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=4,
    )
    # if samples_dist > 0:
    #     sampler = torch.utils.data.SubsetRandomSampler(
    #         [i for i in range(0, 10000, samples_dist)]
    #     )
    # else:
    #     sampler = None
    # test_loader = torch.utils.data.DataLoader(
    #     dataset=test_set,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     sampler=sampler,
    # )

    return train_loader, test_loader

def compute_min_max(data_paths):
    all_inputs = []
    for data_path in data_paths:
        with open(data_path, 'r') as f:
            data = json.load(f)
            for sample in data:
                input_signal = [sample['init_cond']] + sample['samples']
                all_inputs.extend(input_signal)
    all_inputs = torch.tensor(all_inputs, dtype=torch.float32)
    min_value = torch.min(all_inputs).item()
    max_value = torch.max(all_inputs).item()
    return min_value, max_value