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
from graphviz import *

import time

import argparse

# parser = argparse.ArgumentParser(description='Integrated Analysis')
# parser.add_argument('--cp', type=int, required=True, help='Control points')
# args = parser.parse_args()
# cp = args.cp
cp = 10

# entire_bound = [0.06592662632465363, 0.20799323916435242]

# device = 'cuda:4'
device = 'cpu'

def main():
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    params = torch.load('/home/koh/work/DeepBern-Nets/experiments/staliro/vehicle_engine_new/vehicle_engine_01/checkpoint_best_model.pth')
    # input_dimension = len(scaling_factor['min'][:-1])
    input_dimension = params['model_state_dict']['net.0.weight'].shape[1]
    num_neurons = params['model_state_dict']['net.0.weight'].shape[0]
    # Not sure if this is correct
    num_layers = (len(params['model_state_dict']._metadata) - 3) // 2
    print('num_layers: {}'.format(num_layers))
    num_layers = 5
    model = FCModel([input_dimension] + [num_neurons] * num_layers + [1], 8).to(device)
    input_bounds_ = torch.tensor([[0.0, 1.0] for _ in range(input_dimension)]).to(device)
    model.load_state_dict(params['model_state_dict'])
    model.input_bounds = input_bounds_

    init_bounds = torch.tensor([[0.0, 1.0]] * input_dimension).to(device)
    entire_bounds = model.forward_subinterval(init_bounds)
    # entire_bounds_ = rescaling_output(entire_bounds.squeeze(0).squeeze(0))
    # print('Entire bound: {}'.format(entire_bounds_.tolist()))

    path_table = {0: [0.0, 5.0], 1: [5.0, 7.0], 2: [7.0, 10.0], 3: [10.0, 10.1]}
    sim_time = 30
    interval = sim_time / cp
    timing = [i * interval for i in range(cp)]

    # tmp = []
    # tmp += [[0.0, 10.1] for _ in range(31 - len(tmp))]
    # input_bound = scaling_input_bound(torch.tensor(tmp).to(device))

    input_bound_list = []
    points_list = []
    num_samples = 100
    result_ = {}
    for i in range(10):
      input_bound = torch.tensor([[0.0, 1.0]] * input_dimension).to(device)
      p_list = []
      for dim in range(input_dimension):
          # Generate random subinterval within [0.0, 1.0]
          r = torch.rand(2)
          lower = r.min()
          upper = r.max()
          input_bound[dim] = torch.tensor([lower, upper]).to(device)
          points = torch.empty(num_samples).uniform_(lower, upper)
          p_list.append(points)
      res = torch.stack(p_list, dim=0)
      points_list.append(res.T)
      input_bound_list.append(input_bound)


    points_list = torch.stack(points_list, dim=0)
    input_bound_list = torch.stack(input_bound_list, dim=0)
    count_list = []
    y_list = []
    bound_list = []
    for i in range(10):
      count = 0
      input_bound = input_bound_list[i]
      bound = model.forward_subinterval(input_bound)
      bound_raw = bound.squeeze(0).squeeze(0)
      y_ = model(points_list[i].to(device))
      y_list.append(y_.tolist())
      bound_list.append(bound_raw.tolist())

      for y in y_:
        # Check if the generated bound is within the entire bound
        is_lower_ok = bound_raw[0] <= y
        is_upper_ok = bound_raw[1] >= y
        is_within = is_lower_ok and is_upper_ok
        if not is_within:
          count += 1
          print(f'Bound:{bound_raw.tolist()}')
          print(f'y:    {y.item()}')
      count_list.append(count)
      
      result_[i] = {
        'input_bound': input_bound_list[i].tolist(),
        'sample_points': points_list[i].tolist(),
        'y': y_list[i],
        'bound_raw': bound_list[i],
        'out_count': count_list[i],
        'entire_bounds': entire_bounds.squeeze(0).squeeze(0).tolist()
      }
    with open('/home/koh/work/DeepBern-Nets/accuracy_result_engine_100.json', 'w') as f:
      json.dump(result_, f, indent=2)
    print('Done')

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

def scaling_input_bound(b):
  min_max = torch.tensor([scaling_factor['min'], scaling_factor['max']]).T.to(device)
  lb = (b[:,0] - min_max[:,0][:-1]) / (min_max[:,1][:-1] - min_max[:,0][:-1])
  ub = (b[:,1] - min_max[:,0][:-1]) / (min_max[:,1][:-1] - min_max[:,0][:-1])
  return torch.stack([lb, ub], dim=1)

def scaling_element(x_i, i):
  min_max = torch.tensor([scaling_factor['min'], scaling_factor['max']]).T.to(device)
  x_i = (x_i - min_max[i][0]) / (min_max[i][1] - min_max[i][0])
  return x_i

def rescaling_element(x_i, i):
  min_max = torch.tensor([scaling_factor['min'], scaling_factor['max']]).T.to(device)
  x_i = x_i * (min_max[i][1] - min_max[i][0]) + min_max[i][0]
  return x_i

def negative_value_check(b):
  if torch.any(b < 0):
    return True
  return False

def red_yellow_check(b):
  if b[1] < 0.0:
    return True
  return False

if __name__ == '__main__':
    main()