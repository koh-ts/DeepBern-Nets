import torch
from models.models import FCModel
import copy 
import numpy as np
import json
import sys
sys.path.append('/home/koh/work/matiec_rampo/examples/misc')
sys.path.append('/home/koh/work/staliro')
print(sys.path)
from tree import *

# Scaling factor
scaling_factor = {
  "min": [
    0.00012228660424362658,
    1.3601257419226798e-06,
    1.747870493229442e-06,
    2.096539176710266e-07,
    8.04051494518454e-06,
    3.1278121113142987e-06,
    1.075282831219937e-06,
    3.4564845544649003e-06,
    4.2816793421884825e-06,
    0.00060746404559886,
    8.050800770487143e-06,
    5.2486354079617215e-05,
    0.00021376569467324025,
    9.383422990283385e-05,
    0.0007122989004015867,
    0.00010284856790665486,
    0.0002594730554306146,
    -12.947212219238281
  ],
  "max": [
    7.999964949027126,
    0.9999949975924147,
    0.9999977945055926,
    0.9999970493007093,
    0.9999984583796007,
    0.9999983435977366,
    0.9999993161401693,
    0.9999924671771248,
    0.9999953125686163,
    49.99944142860811,
    49.99999445728556,
    49.999760904294114,
    49.99999652643017,
    49.999950259612184,
    49.9999297322792,
    49.99972069263225,
    49.99982517352103,
    7.999631881713867
  ]
}

# Max error after scaling
max_error = 0.4389359652996063

# data_240_scaled_min_max.json
# min = -12.947212219238281
# max = 7.999631881713867
# min max scaling: x' = (x - min) / (max - min)
# reverse scaling x = x' * (max - min) + min

# device = 'cuda:4'
device = 'cpu'
# torch.cuda.set_device(4)

def main():
    tree = HistoryTrie(height=4, num_child=4)
    min_ = scaling_factor['min'][-1]
    max_ = scaling_factor['max'][-1]
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    params = torch.load('/home/koh/work/DeepBern-Nets/experiments/staliro/optuna/optuna_normalized_7/checkpoint_best_model.pth')
    model = FCModel([17,512,512,512,512,1], 4).to(device)
    input_bounds_ = torch.tensor([[0.0, 1.0] for _ in range(17)]).to(device)
    model.load_state_dict(params['model_state_dict'])
    model.input_bounds = input_bounds_

    x = torch.tensor([0.5034891637283067, 0.9306942060723306, 0.9375638754635564, 0.2775548965217323, 0.21546631017574988, 0.5303836078887472, 0.036676176324594086, 0.6726495645096164, 0.9281323163040858, 0.4377231715260603, 0.876642879734375, 0.07307733003533934, 0.8026005806972524, 0.6791905662337453, 0.4266436088407155, 0.4770551447887879, 0.6094008948987218]).to(device)
    input_bounds = torch.tensor([[0.503, 0.504],
                                [0.930, 0.931],
                                [0.937, 0.938],
                                [0.277, 0.278],
                                [0.215, 0.216],
                                [0.530, 0.531],
                                [0.036, 0.037],
                                [0.672, 0.673],
                                [0.928, 0.929],
                                [0.437, 0.438],
                                [0.876, 0.877],
                                [0.073, 0.074],
                                [0.802, 0.803],
                                [0.679, 0.680],
                                [0.426, 0.427],
                                [0.477, 0.478],
                                [0.609, 0.610]]).to(device)
    init_bounds = torch.tensor([[0.0, 1.0]] * 17).to(device)
    y = model(x, with_bounds=True)
    # bounds = model.forward_subinterval(input_bounds)
    bounds = model.forward_subinterval(init_bounds)
    bounds_ = bounds * (max_ - min_) + min_
    print('lb: {}\nub: {}'.format(float(bounds[0][0][0]), float(bounds[0][0][1])))

    with open('/home/koh/work/matiec_rampo/examples/tankcontrol_flowrate/min_max.json', 'r') as f:
        ranges = json.load(f)

    print('aaaa')

    # {"init_cond": 0.5034891637283067,
    # "samples": [0.9306942060723306, 0.9375638754635564, 0.2775548965217323, 0.21546631017574988, 0.5303836078887472, 0.036676176324594086, 0.6726495645096164, 0.9281323163040858, 0.4377231715260603, 0.876642879734375, 0.07307733003533934, 0.8026005806972524, 0.6791905662337453, 0.4266436088407155, 0.4770551447887879, 0.6094008948987218],
    # "robustness": 0.5404148848784927},

    # {"init_cond": 0.8465674954817122,
    # "samples": [0.6029759092253901, 0.825726591599673, 0.6533003920578835, 0.9933126425543068, 0.8653978945342509, 0.5464428297963971, 0.13031947366269853, 0.6163169302482682, 0.26235594059822626, 0.5364096838013326, 0.6720809224209511, 0.15477127175634828, 0.6347689799440032, 0.3117690851135224, 0.7172214858926735, 0.4082692718566702],
    # "robustness": 0.4379270270499142}
    # x = torch.tensor([0.8465674954817122, 0.6029759092253901, 0.825726591599673, 0.6533003920578835, 0.9933126425543068, 0.8653978945342509, 0.5464428297963971, 0.13031947366269853, 0.6163169302482682, 0.26235594059822626, 0.5364096838013326, 0.6720809224209511, 0.15477127175634828, 0.6347689799440032, 0.3117690851135224, 0.7172214858926735, 0.4082692718566702]).to('cuda:0')

def scaling_input(x):
  min_max = torch.tensor([scaling_factor['min'], scaling_factor['max']]).T.to(device)
  x = (x - min_max[:,0][:-1]) / (min_max[:,1][:-1] - min_max[:,0][:-1])
  return x

def rescaling_input(x):
  min_max = torch.tensor([scaling_factor['min'], scaling_factor['max']]).T.to(device)
  x = x * (min_max[:,1][:-1] - min_max[:,0][:-1]) + min_max[:,0][:-1]
  return x

def scaling_output(y):
  min_max = torch.tensor([scaling_factor['min'], scaling_factor['max']]).T.to(device)
  y = (y - min_max[:,0][-1]) / (min_max[:,1][-1] - min_max[:,0][-1])
  return y

def rescaling_output(y):
  min_max = torch.tensor([scaling_factor['min'], scaling_factor['max']]).T.to(device)
  y = y * (min_max[:,1][-1] - min_max[:,0][-1]) + min_max[:,0][-1]
  return y

def scaling_element(x_i, i):
  min_max = torch.tensor([scaling_factor['min'], scaling_factor['max']]).T.to(device)
  x_i = (x_i - min_max[i][0]) / (min_max[i][1] - min_max[i][0])
  return x_i

def rescaling_element(x_i, i):
  min_max = torch.tensor([scaling_factor['min'], scaling_factor['max']]).T.to(device)
  x_i = x_i * (min_max[i][1] - min_max[i][0]) + min_max[i][0]
  return x_i

if __name__ == '__main__':
    main()