import torch
from models.models import FCModel
import copy 
import numpy as np
import json
import sys
sys.path.append('/home/koh/work/matiec_rampo/examples/misc')
sys.path.append('/home/koh/work/staliro')
sys.path.append('/home/koh/work/RampoNN')
print(sys.path)
from tree import *
from graphviz import *

import time

from staliro.models import Trace
from staliro.specifications import rtamt

from TNN import STL2NN_TankHeight_BernMin

import argparse

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

# parser = argparse.ArgumentParser(description='Integrated Analysis')
# parser.add_argument('--cp', type=int, required=True, help='Control points')
# args = parser.parse_args()
# cp = args.cp
cp = 10

entire_bound = [-0.0716,  2.9049]

# device = 'cuda:4'
device = 'cpu'
# torch.cuda.set_device(4)

phi = "(always[0,30] (TankHeight <= 8))"
specification = rtamt.parse_dense(phi, {"TankHeight": 0, "InValve": 1, "OutValve": 2})
# specification = rtamt.parse_dense(phi, {"TankHeight": 0})

def main():
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    params = torch.load('/home/koh/work/DeepBern-Nets/experiments/staliro/state_robust/state_robust_02/checkpoint_best_model.pth')
    # params = torch.load('/home/koh/work/DeepBern-Nets/experiments/staliro/state_robust/state_robust_01/checkpoint_best_model.pth')
    # params = torch.load('/home/koh/work/DeepBern-Nets/experiments/staliro/state_robust/state_robust_00/checkpoint_best_model.pth')
    input_dimension = len(scaling_factor['min'][:-1])
    model = FCModel([input_dimension,1024,1024,1024,1024,1], 8).to(device)
    # model = FCModel([input_dimension,1024,1024,1024,1024,1], 4).to(device)
    # model = FCModel([input_dimension,512,512,512,512,1], 4).to(device)
    model.eval()
    input_bounds_ = torch.tensor([[0.0, 1.0] for _ in range(input_dimension)]).to(device)
    model.load_state_dict(params['model_state_dict'])
    model.input_bounds = input_bounds_

    lb = -0.01
    ub = 1.01
    model_bernmin_node = STL2NN_TankHeight_BernMin(horizon=31, degree=3, lb=lb, ub=ub)

    init_bounds = torch.tensor([[0.0, 1.0]] * input_dimension).to(device)
    entire_bounds = model.forward_subinterval(init_bounds)
    entire_bounds_ = rescaling_output(entire_bounds.squeeze(0).squeeze(0))
    print('Entire bound: {}'.format(entire_bounds_.tolist()))

    path_table = {0: [0.0, 5.0], 1: [5.0, 7.0], 2: [7.0, 10.0], 3: [10.0, 10.1]}
    sim_time = 30
    # generate 0 to 30 sec timing
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]
    interval = sim_time / cp
    timing = [i * interval for i in range(cp)]

    # tmp = []
    # tmp += [[0.0, 10.1] for _ in range(31 - len(tmp))]
    # input_bound = scaling_input_bound(torch.tensor(tmp).to(device))
    num_try = 100
    input_bound_list = []
    points_list = []
    num_samples = 100
    result_ = []
    for i in range(num_try):
      input_bound = torch.tensor([[0.0, 1.0]] * input_dimension).to(device)
      p_list = []
      # Pick random dimension indices from input_dimension to limit the search space
      # dim_indices = np.random.choice(input_dimension, size=5, replace=False)
      
      # for dim in range(input_dimension):
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

          # r = torch.rand(2)
          # lower = r.min()
          # upper = r.max()
          # input_bound[dim] = torch.tensor([lower, upper]).to(device)
          # points = np.array(torch.empty(num_samples).uniform_(lower, upper).tolist()).T
          # p_list.append(points)

      # Generate initial bounds for the first dimension
      r = torch.rand(2)
      lower = r.min().item()
      upper = r.max().item()
      input_bound[0] = torch.tensor([lower, upper]).to(device)
      points = torch.empty(num_samples).uniform_(lower, upper)
      p_list.append(points)
      
      # For subsequent dimensions, ensure bounds are close to previous dimension
      for dim in range(1, input_dimension):
          # Get previous dimension's bounds
          prev_lower, prev_upper = input_bound[dim-1]
          
          # Generate new bounds with some overlap/proximity to previous bounds
          # Allow maximum shift of 0.2 in either direction
          max_shift = 0.1
          new_lower = max(0.0, min(1.0, prev_lower + (torch.rand(1).item() - 0.5) * max_shift))
          new_upper = max(new_lower + 0.05, min(1.0, prev_upper + (torch.rand(1).item() - 0.5) * max_shift))
          
          input_bound[dim] = torch.tensor([new_lower, new_upper]).to(device)
          points = torch.empty(num_samples).uniform_(new_lower, new_upper)
          p_list.append(points)          
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

      bound_bern = model_bernmin_node.ibp_forward_bern(input_bound.unsqueeze(0))

      tmp_y = []
      # for p in points_list[i]:
      for p in p_list:
        p_ = rescaling_input(p).tolist()
        pp = [[ppp,0.0,0.0] for ppp in p_]
        p_trace = Trace(states=pp, times=times)
        spec_result = specification.evaluate(p_trace)
        y_ = scaling_output(spec_result.value)
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
      if count < 70:
        result_.append({
          'input_bound': input_bound.tolist(),
          'sample_points': points.tolist(),
          'y': tmp_y,
          'bound_raw': bound_raw.tolist(),
          'out_count': count,
          'entire_bounds': entire_bounds.squeeze(0).squeeze(0).tolist()
        })
    with open('/home/koh/work/DeepBern-Nets/accuracy_result_tank_100_new___01.json', 'w') as f:
      json.dump(result_, f, indent=2)
    print('Done')

def scaling_input(x):
  min_max = torch.tensor([scaling_factor['min'], scaling_factor['max']]).T.to(device)
  x = (x - min_max[:,0][:-1]) / (min_max[:,1][:-1] - min_max[:,0][:-1])
  return x

def rescaling_input(x):
  # min_max = torch.tensor([scaling_factor['min'], scaling_factor['max']]).T.to(device)
  min_max = np.array([scaling_factor['min'], scaling_factor['max']]).T
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