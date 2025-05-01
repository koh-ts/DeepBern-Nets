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

from staliro.models import Trace
from staliro.specifications import rtamt

# parser = argparse.ArgumentParser(description='Integrated Analysis')
# parser.add_argument('--cp', type=int, required=True, help='Control points')
# args = parser.parse_args()
# cp = args.cp
cp = 10

# entire_bound = [0.06592662632465363, 0.20799323916435242]

# device = 'cuda:4'
device = 'cpu'

# scaling_factor = {
#   "speed_min": -2.8882162915020855,
#   "speed_max": 166.5363382521565,
#   "rpm_min": 600.0,
#   "rpm_max": 6000.0,
#   "robustness_min": -65.37969970703125,
#   "robustness_max": 1079.2451171875,
#   "train_len": 22400,
#   "test_len": 5600
# }

with open('/home/koh/work/matiec_rampo/examples/vehicle_engine/data_025/done/data_vehicle_engine_min_max_025_large01.json', 'r') as f:
    scaling_factor = json.load(f)

safety_speed = 100
safety_rpm = 4300
phi = "!(F[0,30] (Speed >= " + str(safety_speed) +") and F[0,30] (RPM >= " + str(safety_rpm) + "))"# specification = TLTK(phi, {"TankHeight": 0, "InValve": 1, "OutValve": 2})
print('phi: {}'.format(phi))
specification = rtamt.parse_dense(phi, {"Speed": 0, "RPM": 1})

def main():
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    params = torch.load('/home/koh/work/DeepBern-Nets/experiments/staliro/vehicle_engine_025/vehicle_engine_03/checkpoint_50.pth')
    # input_dimension = len(scaling_factor['min'][:-1])
    input_dimension = params['model_state_dict']['net.0.weight'].shape[1]
    num_neurons = params['model_state_dict']['net.0.weight'].shape[0]
    # Not sure if this is correct
    num_layers = (len(params['model_state_dict']._metadata) - 3) // 2
    print('num_layers: {}'.format(num_layers))
    # num_layers = 5
    degree = params['model_state_dict']['net.1._basis_indices'].shape[1]-1
    model = FCModel([input_dimension] + [num_neurons] * num_layers + [1], degree).to(device)
    input_bounds_ = torch.tensor([[0.0, 1.0] for _ in range(input_dimension)]).to(device)
    model.load_state_dict(params['model_state_dict'])
    model.input_bounds = input_bounds_

    init_bounds = torch.tensor([[0.0, 1.0]] * input_dimension).to(device)
    entire_bounds = model.forward_subinterval(init_bounds)
    # entire_bounds_ = rescaling_output(entire_bounds.squeeze(0).squeeze(0))
    # print('Entire bound: {}'.format(entire_bounds_.tolist()))

    sim_time = 30
    interval = sim_time / cp
    timing = [i * interval for i in range(cp)]

    # tmp = []
    # tmp += [[0.0, 10.1] for _ in range(31 - len(tmp))]
    # input_bound = scaling_input_bound(torch.tensor(tmp).to(device))

    input_bound_list = []
    points_list = []
    num_samples = 100
    num_try = 100
    result_ = []
    for i in range(num_try):
      input_bound = torch.tensor([[0.0, 1.0]] * input_dimension).to(device)
      p_list = []
      # Pick random dimension indices from input_dimension to limit the search space
      # dim_indices = np.random.choice(input_dimension, size=5, replace=False)
      for dim in range(input_dimension):
          # Generate random subinterval within [0.0, 1.0]
          # Here, implement gradually sliding ranges since dimensions are related in time
          # if dim == 0:
          #   # For the first dimension, generate a completely random subinterval
          #   width = 0.2 + 0.4 * torch.rand(1).item()  # Random width between 0.2 and 0.6
          #   center = torch.rand(1).item() * (1.0 - width)  # Random center position
          #   lower = center
          #   upper = center + width
          # else:
          #   # For subsequent dimensions, gradually slide the range from previous dimension
          #   prev_lower = input_bound[dim-1][0].item()
          #   prev_upper = input_bound[dim-1][1].item()
          #   prev_width = prev_upper - prev_lower
            
          #   # Small random shift with bounds to ensure smooth transition
          #   max_shift = 0.05  # Maximum shift per dimension
          #   shift = (torch.rand(1).item() * 2 - 1) * max_shift  # Random shift between -max_shift and max_shift
            
          #   # Small random change in width
          #   width_change = (torch.rand(1).item() * 2 - 1) * 0.02  # Random width change
          #   new_width = max(0.1, min(0.7, prev_width + width_change))
            
          #   # Apply shift and ensure we stay within [0.0, 1.0]
          #   lower = max(0.0, min(1.0 - new_width, prev_lower + shift))
          #   upper = lower + new_width

          r = torch.rand(2)
          lower = r.min()
          upper = r.max()

          input_bound[dim] = torch.tensor([lower, upper]).to(device)
          points = np.array(torch.empty(num_samples).uniform_(lower, upper).tolist()).T
          p_list.append(points) 
          # p_list.append(points.tolist())          
      # res = np.array(p_list)
      # points_list.append(res.T)
      # input_bound_list.append(input_bound)


    # points_list = np.array(points_list)
    # count_list = []
    # y_list = []
    # bound_list = []
    # for i in range(num_try):
      p_list = np.array(p_list).T
      count = 0
      # input_bound = input_bound_list[i]
      bound = model.forward_subinterval(input_bound)
      bound_raw = bound.squeeze(0).squeeze(0)
      tmp_y = []
      # for p in points_list[i]:
      for p in p_list:
        p_speed, p_rpm = rescaling_input_engine_each(p)
        l = len(p) // 2
        pp = [[p_speed[i],p_rpm[i]] for i in range(l)]
        interval = sim_time / (l-1)
        times = [i * interval for i in range(l)]
        p_trace = Trace(states=pp, times=times)
        spec_result = specification.evaluate(p_trace)
        y_ = scaling_output_engine(spec_result.value)
        tmp_y.append(y_.item())
        y = y_.item()
      # y_ = model(points_list[i].to(device))
      # y_list.append(tmp_y)
      # bound_list.append(bound_raw.tolist())

      # for y in tmp_y:
        # Check if the generated bound is within the entire bound
        is_lower_ok = bound_raw[0] <= y
        is_upper_ok = bound_raw[1] >= y
        is_within = is_lower_ok and is_upper_ok
        if not is_within:
          count += 1
          # print(f'Bound:{bound_raw.tolist()}')
          # print(f'y:    {y}')
      # count_list.append(count)
      print('Count: {}'.format(count))
      result_.append({
        'input_bound': input_bound.tolist(),
        'sample_points': points.tolist(),
        'y': tmp_y,
        'bound_raw': bound_raw.tolist(),
        'out_count': count,
        'entire_bounds': entire_bounds.squeeze(0).squeeze(0).tolist()
      })
    with open('/home/koh/work/DeepBern-Nets/accuracy_result_engine_100_new_01.json', 'w') as f:
      json.dump(result_, f, indent=2)
    print('Done')
    # for i in range(10):
    #   input_bound = torch.tensor([[0.0, 1.0]] * input_dimension).to(device)
    #   p_list = []
    #   for dim in range(input_dimension):
    #       # Generate random subinterval within [0.0, 1.0]
    #       r = torch.rand(2)
    #       lower = r.min()
    #       upper = r.max()
    #       input_bound[dim] = torch.tensor([lower, upper]).to(device)
    #       points = torch.empty(num_samples).uniform_(lower, upper)
    #       p_list.append(points)
    #   res = torch.stack(p_list, dim=0)
    #   points_list.append(res.T)
    #   input_bound_list.append(input_bound)


    # points_list = torch.stack(points_list, dim=0)
    # input_bound_list = torch.stack(input_bound_list, dim=0)
    # count_list = []
    # y_list = []
    # bound_list = []
    # for i in range(10):
    #   count = 0
    #   input_bound = input_bound_list[i]
    #   bound = model.forward_subinterval(input_bound)
    #   bound_raw = bound.squeeze(0).squeeze(0)
    #   y_ = model(points_list[i].to(device))
    #   y_list.append(y_.tolist())
    #   bound_list.append(bound_raw.tolist())

    #   for y in y_:
    #     # Check if the generated bound is within the entire bound
    #     is_lower_ok = bound_raw[0] <= y
    #     is_upper_ok = bound_raw[1] >= y
    #     is_within = is_lower_ok and is_upper_ok
    #     if not is_within:
    #       count += 1
    #       print(f'Bound:{bound_raw.tolist()}')
    #       print(f'y:    {y.item()}')
    #   count_list.append(count)
      
    #   result_[i] = {
    #     'input_bound': input_bound_list[i].tolist(),
    #     'sample_points': points_list[i].tolist(),
    #     'y': y_list[i],
    #     'bound_raw': bound_list[i],
    #     'out_count': count_list[i],
    #     'entire_bounds': entire_bounds.squeeze(0).squeeze(0).tolist()
    #   }
    # with open('/home/koh/work/DeepBern-Nets/accuracy_result_engine_100.json', 'w') as f:
    #   json.dump(result_, f, indent=2)
    # print('Done')

def scaling_input(x):
  min_max = torch.tensor([scaling_factor['min'], scaling_factor['max']]).T.to(device)
  x = (x - min_max[:,0][:-1]) / (min_max[:,1][:-1] - min_max[:,0][:-1])
  return x

def rescaling_input(x):
  min_max = torch.tensor([scaling_factor['min'], scaling_factor['max']]).T.to(device)
  x = x * (min_max[:,1][:-1] - min_max[:,0][:-1]) + min_max[:,0][:-1]
  return x

def rescaling_input_engine(x):
  half = len(x) // 2
  x1 = x[:half]
  x2 = x[half:]
  x1 = x1 * (scaling_factor['speed_max'] - scaling_factor['speed_min']) + scaling_factor['speed_min']
  x2 = x2 * (scaling_factor['rpm_max'] - scaling_factor['rpm_min']) + scaling_factor['rpm_min']
  return x1, x2

def rescaling_input_engine_each(x):
  x1 = x[:len(x)//2]
  x2 = x[len(x)//2:]
  speed_range = np.array(scaling_factor['speed_max']) - np.array(scaling_factor['speed_min'])
  rpm_range = np.array(scaling_factor['rpm_max']) - np.array(scaling_factor['rpm_min'])
  speed_range[speed_range == 0] = 1
  rpm_range[rpm_range == 0] = 1
  x1 = x1 * speed_range + np.array(scaling_factor['speed_min'])
  x2 = x2 * rpm_range + np.array(scaling_factor['rpm_min'])
  return x1, x2


def scaling_output(y):
  min_max = torch.tensor([scaling_factor['min'], scaling_factor['max']]).T.to(device)
  y = (y - min_max[:,0][-1]) / (min_max[:,1][-1] - min_max[:,0][-1])
  return y

def scaling_output_engine(y):
  y = (y - scaling_factor['robustness_min']) / (scaling_factor['robustness_max'] - scaling_factor['robustness_min'])
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