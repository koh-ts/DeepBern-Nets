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
    print('Entire bound non scaling: {}'.format(entire_bounds.squeeze(0).squeeze(0).tolist()))

    # sim_time = 30
    # interval = sim_time / cp
    # timing = [i * interval for i in range(cp)]

    # tmp = []
    # tmp += [[0.0, 10.1] for _ in range(31 - len(tmp))]
    # input_bound = scaling_input_bound(torch.tensor(tmp).to(device))
    input_bound = torch.tensor([[0.0, 1.0]] * input_dimension).to(device)


    for i in range(1000):
      # Generate random subset input bounds
      num_dims_to_randomize = torch.randint(1, input_dimension + 1, (1,)).item()
      dims_to_randomize = torch.randperm(input_dimension)[:num_dims_to_randomize]

      # Start with full range for all dimensions
      input_bound = torch.tensor([[0.0, 1.0]] * input_dimension).to(device)

      # Randomize selected dimensions
      for dim in dims_to_randomize:
        # Generate random subinterval within [0.0, 1.0]
        lower = torch.rand(1).item() * 0.7  # Ensure enough space for upper bound
        upper = lower + torch.rand(1).item() * (1.0 - lower)  # Ensure upper > lower
        input_bound[dim] = torch.tensor([lower, upper]).to(device)

      # print(f"Randomized {num_dims_to_randomize} out of {input_dimension} dimensions")
      # print(f"Random input bounds: {input_bound.tolist()}")


      bound = model.forward_subinterval(input_bound)
      # compare with the entire bound
      # Check if the generated bound is within the entire bound
      bound_raw = bound.squeeze(0).squeeze(0)
      entire_bounds_raw = entire_bounds.squeeze(0).squeeze(0)
      # print(entire_bounds_raw)

      # Check if the generated bound is within the entire bound
      is_lower_ok = bound_raw[0] >= entire_bounds_raw[0]
      is_upper_ok = bound_raw[1] <= entire_bounds_raw[1]
      is_within = is_lower_ok and is_upper_ok

      print(f"Generated bound before rescaling: {bound_raw.tolist()}")
      print(f"Entire bound before rescaling:    {entire_bounds_raw.tolist()}")
      print(f"Is within entire bound:           {is_within}")

      if not is_within:
        if not is_lower_ok:
          print(f"Lower bound {bound_raw[0].item()} is less than entire lower bound {entire_bounds_raw[0].item()}")
        if not is_upper_ok:
          print(f"Upper bound {bound_raw[1].item()} is greater than entire upper bound {entire_bounds_raw[1].item()}")

    # bound_ = rescaling_output(bound.squeeze(0).squeeze(0))

    # random_input = torch.rand(input_dimension).to(device).squeeze()
    # y = model(random_input)


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