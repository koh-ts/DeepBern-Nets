import torch
from models.models import FCModel
import copy 
import numpy as np
import json
import sys
sys.path.append('/home/koh/work/matiec_rampo/examples/misc')
sys.path.append('/home/koh/work/staliro')
sys.path.insert(0, '/home/koh/miniconda3/envs/deepbern/lib/python3.9/site-packages')
sys.path.append('/home/koh/work')
print(sys.path)
from tree import *
from graphviz import *

from staliro.core.interval import Interval
from staliro.core.model import BasicResult, Model, ModelInputs, ModelResult, Trace
from staliro.core.result import best_eval, best_run, worst_run, worst_eval, Evaluation
from staliro.core.signal import Signal
from staliro.optimizers import DualAnnealing
from staliro.options import Options, SignalOptions
from staliro.specifications import TLTK, RTAMTDense
# from staliro.staliro import simulate_model, staliro, staliro_
from staliro.staliro import simulate_model, staliro

sys.path.append('/home/koh/work/matiec_rampo/examples/tankcontrol_flowrate')
from psy_taliro_tankcontrol_flowrate import TankControlFlowRate, path_extraction

try:
    import matlab
    import matlab.engine
except ImportError:
    _has_matlab = False
else:
    _has_matlab = True

import time
import logging

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

# min max scaling: x' = (x - min) / (max - min)
# reverse scaling x = x' * (max - min) + min

# device = 'cuda:4'
device = 'cpu'

sim_model = TankControlFlowRate()
ranges = {
  "TankHeight": [[0.0, 5.0], [5.0, 7.0], [7.0, 10.0], [10.0, 100.0]],
  "InValve": [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
    "OutValve": [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
}
phi = "(always[0,30] (TankHeight <= 8))"
# specification = TLTK(phi, {"TankHeight": 0, "InValve": 1, "OutValve": 2})
specification = RTAMTDense(phi, {"TankHeight": 0, "InValve": 1, "OutValve": 2})
# specification = rtamt.parse_dense(phi, {"TankHeight": 0, "InValve": 1, "OutValve": 2})
optimizer = DualAnnealing()
signals = [
    SignalOptions(control_points=[(0, 1)] * 10),
    SignalOptions(control_points=[(0, 1)] * 10),
    SignalOptions(control_points=[(30, 100)] * 10),
    SignalOptions(control_points=[(30, 100)] * 10), 
]
options = Options(runs=1, iterations=100, interval=(0, 30), signals=signals)

def main():
    logging.basicConfig(level=logging.DEBUG)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    params = torch.load('/home/koh/work/DeepBern-Nets/experiments/staliro/state_robust/state_robust_02/checkpoint_best_model.pth')
    input_dimension = len(scaling_factor['min'][:-1])
    model = FCModel([input_dimension,1024,1024,1024,1024,1], 8).to(device)
    model.eval()
    input_bounds_ = torch.tensor([[0.0, 1.0] for _ in range(input_dimension)]).to(device)
    model.load_state_dict(params['model_state_dict'])
    model.input_bounds = input_bounds_
    init_bounds = torch.tensor([[0.0, 1.0]] * input_dimension).to(device)
    entire_bounds = model.forward_subinterval(init_bounds)
    entire_bounds_ = rescaling_output(entire_bounds.squeeze(0).squeeze(0))
    print('Entire bound: {}'.format(entire_bounds_.tolist()))

    t = HistoryTrie(height=10, num_child=4)
    # G = Digraph(format='png')
    # G.attr('node', shape='circle')
    # G.node('root', label='root')
    tree = t.root
    path_table = {0: [0.0, 5.0], 1: [5.0, 7.0], 2: [7.0, 10.0], 3: [10.0, 10.1]}
    q = []
    safe_range = []
    unsafe_range = []
    red_range = []
    yellow_range = []
    falsified_list = []
    not_falsified_list = []
    # q.append(tree)
    # for c in tree.children:
    #   q.append(c)

    res = staliro(sim_model, specification, optimizer, options)
    res.runs[0].history.sort(key=lambda x: x.cost)
    best_sample = worst_eval(worst_run(res)).sample
    best_result = simulate_model(sim_model, options, best_sample)
    path = path_extraction(best_result)

    start = time.time()
    node = t.root
    # node = node.children[path[0]]
    bound_ = [-0.1, 0.1]
    while t.root.visited == False:
      # Here , we do the NN reachability analysis
      # The result can be the following patterns
      # 1. Yellow: lb is negative and ub is positive
      #    1.a: The node is a intermediate node -> Go down by one depth to the leaf node
      #    1.b: The node is a leaf node -> Falsification on the leaf node
      # 2. Red: lb is negative and ub is negative -> Falsification on the parent node excluding the visited children nodes
      # 3. White: lb is positive and ub is positive -> Falsification on the parent node excluding the visited children nodes
      h = node.height
      for j in range(h):

        # As long as the NN reachability result is yellow, we need to go down to the leaf node
        if bound_[0] < 0.0 and bound_[1] > 0.0:
          tmp = []
          if node.height == 0:
            # This is when the node is a leaf and still yellow result
            break
          else:
            if not node.children[path[j]].visited:
              node = node.children[path[j]]
          for i, p in enumerate(node.path):
            tmp_val = path_table[p]
            # For cp == 10
            tmp += [tmp_val] * 3
          tmp += [[0.0, 10.1] for _ in range(31 - len(tmp))]

          # NN reachability analysis
          input_bound = scaling_input_bound(torch.tensor(tmp).to(device))
          bound = model.forward_subinterval(input_bound)
          bound_ = rescaling_output(bound.squeeze(0).squeeze(0))

        # If the NN reachability result is red or white, we need to go up to the parent node and
        # do the falsification with the actual model excluding the child node that has previously been visited
        else:
          if bounds_[0] > 0.0:
            # This means that the entire output bound is positive (white)
            safe_range.append(node.path)
          else:
            # This means that the entire output bound is negative (red)
            red_range.append(node.path)
          node.visited = True
          node = node.parent

      # Here, we do the falsification with the actual model
      # There are several cases from the previous NN reachability step
      # 1b: leaf yellow node -> Falsification with only the path constraints
      #    Falsified: Store the path as a potential vulnerable node, and go up to the parent node
      #    Not falsified: Store the path as a safe node, and go up to the parent node
      # 2: red node -> Falsification with the path constraints down until one above the red node, and the pruning constraints
      # 3: white node -> Falsification with the path constraints down until one above the white node, and the pruning constraints
      falsified = False
      while not falsified:
        res, best_result = falsification_with_actual_model(node)
        falsified = res.runs[0].history[0].cost < 0.0
        if falsified:
          if node.height == 0:
            # If the node is a leaf node and it's falsified, we store this node as a potential vulnerable node
            falsified_list.append([node.path, res, best_result])
            node.visited = True
            node = node.parent
            falsified = False
          else:
            # If the node is not a leaf node but it's falsified, then we extract the path and go towards the leaf node by one depth
            # This path should not include the visited children nodes because we exclude them in the falsification attemp by adding the extra constraints
            path = path_extraction(best_result)
            l = len(node.path)
            node = node.children[path[l]]
            break
        else:
          # If the result is not falsified, then the entire subtree can regarded as safe, meaning there will be no vulnerable node
          node.visited = True
          if node == t.root:
            break
          node = node.parent
      
      if node == t.root:
        continue

      tmp = []
      for i, p in enumerate(node.path):
        tmp_val = path_table[p]
        # For cp == 10
        tmp += [tmp_val] * 3
      tmp += [[0.0, 10.1] for _ in range(31 - len(tmp))]
      # NN reachability analysis
      input_bound = scaling_input_bound(torch.tensor(tmp).to(device))
      bound = model.forward_subinterval(input_bound)       
      bound_ = rescaling_output(bound.squeeze(0).squeeze(0))
      

          
          

      # If it's a leaf node and the color is yellow, that means we need to do the falsification with the actual model on the leaf node
      # if bound_[0] < 0.0 and bound_[1] > 0.0:
      #   res, best_result = falsification_with_actual_model(node)
      #   falsified = res.runs[0].history[0].cost < 0.0
      #   if falsified:
      #     # Since the result is falsified, we store this node as a potential vulnerabile node
      #     node.visited = True
      #     falsified_list.append(node.path)
      #   else:
      #     # If the result is not falsified, we store this node as a safe node
      #     node.visited = True
      #     not_falsified_list.append(node.path)
      # else:
      #   # Regardless of whether it's red or while, we just mark the node as visited and move up to the parent node
      #   node.visited = True
      #   node = node.parent
      #   res, best_result = falsification_with_actual_model(node)
      #   falsified = res.runs[0].history[0].cost < 0.0
      #   if falsified:
      #     # Extract the path and go all the way to the leaf node
      #     extracted_path = path_extraction(best_result)
      #     current_depth = len(path)
      #     for i in range(current_depth, 10):
      #       node = node.children[extracted_path[i]]
      #   node = node.parent




      # If it's safe result, meaning that the entire output bound is positive

      # if negative_value_check(bound_):
      #   node.visited = True
      #   for c in node.children:
      #     q.append(c)
      # if node.height == 0:
      #   if negative_value_check(bound_):

      #     if red_yellow_check(bound_):
      #        red_range.append(path)
      #     else:
      #        yellow_range.append(path)
      #     print('Path: {}'.format(path))
      #     print('Output bound: {}'.format(bound_.tolist()))
      #     node.visited = True
      #   else:
      #     safe_range.append(path)
    
      #   with open('/home/koh/work/DeepBern-Nets/safe_ranges.json', 'w') as f:
      #       json.dump(safe_range, f)
      #   with open('/home/koh/work/DeepBern-Nets/red_ranges.json', 'w') as f:
      #       json.dump(red_range, f)
      #   with open('/home/koh/work/DeepBern-Nets/yellow_ranges.json', 'w') as f:
      #       json.dump(yellow_range, f)
    
    # end = time.time()
    # print('Time: {}'.format(end - start))
    # print('# of safe range: {}'.format(len(safe_range)))
    # print('# of red range: {}'.format(len(red_range)))
    # print('# of yellow range: {}'.format(len(yellow_range)))
    # for c in t.root.children:
    #     G.node(str(c.path), label=str(c.path))
    #     G.edge('root', str(c.path))
    #     for cc in c.children:
    #         G.node(str(cc.path), label=str(cc.path))
    #         G.edge(str(c.path), str(cc.path))
    #         for ccc in cc.children:
    #             G.node(str(ccc.path), label=str(ccc.path))
    #             G.edge(str(cc.path), str(ccc.path))
    #             for cccc in ccc.children:
    #                 G.node(str(cccc.path), label=str(cccc.path))
    #                 G.edge(str(ccc.path), str(cccc.path))
    # for r in red_range:
    #     G.node(str(r), label=str(r), style='filled', fillcolor='red')
    # for y in yellow_range:
    #     G.node(str(y), label=str(y), style='filled', fillcolor='yellow')
    # G.render('tree', outfile='/home/koh/work/DeepBern-Nets/tree_reachability_red_yellow.png')
    print('Done')

def falsification_with_actual_model(node):
  path = node.path
  num_cp = node.height + len(node.path)
  sim_time = options.interval.upper - options.interval.lower
  interval = sim_time / num_cp
  cp_array = [i * interval for i in range(num_cp)]
  Phi = "(G[0,30] (TankHeight <= 8))"
  phi = ''
  epsilon = 0.1
  for i, p in enumerate(path):
    phi += '(F[{}, {}] (TankHeight >= {} /\ TankHeight <= {}))' \
            .format(max(0, cp_array[i] - epsilon), min(cp_array[i] + epsilon, sim_time), \
            ranges['TankHeight'][p][0], ranges['TankHeight'][p][1])
    if i != len(path) - 1:
        phi += ' /\ '
  if phi != '':
    phi = Phi + ' \/ ! (' + phi + ')'
  else:
    phi = Phi

  # Here, we need to consider the pruning children by checking the node's children list
  # If the node's children list is empty, then nothing to do
  # If the node's children list is not empty, then we need to generate the extra constraints
  pruning_children = []
  for c in node.children:
      if c.visited:
        pruning_children.append(c.idx)
  if len(node.children) > 0:
    # Here, we need to generate the extra constraints
    j = len(node.path)
    extra_phi = ''
    # c is one of [0, 1, 2, 3]
    for k, c in enumerate(pruning_children):
      extra_phi += '(F[{}, {}] (TankHeight >= {} /\ TankHeight <= {}))' \
            .format(max(0, cp_array[j] - epsilon), min(cp_array[j] + epsilon, sim_time), \
            ranges['TankHeight'][c][0], ranges['TankHeight'][c][1])
      if k != len(pruning_children) - 1:
        extra_phi += ' \/ '
    phi += ' \/ (' + extra_phi + ')'

  print('Searching Node: {}, Phi: {}'.format(path, phi))

  # spec = TLTK(phi, {'TankHeight': 0, 'InValve': 1, 'OutValve': 2})
  spec = RTAMTDense(phi, {'TankHeight': 0, 'InValve': 1, 'OutValve': 2})
  res = staliro(sim_model, spec, optimizer, options)
  res.runs[0].history.sort(key=lambda x: x.cost)
  best_sample = worst_eval(worst_run(res)).sample
  best_result = simulate_model(sim_model, options, best_sample)
  if res.runs[0].history[0].cost < 0:
      # red_falsified.append([d, res.runs[0].history[0].cost])
      print('Falsified')
      print('Cost: {}'.format(res.runs[0].history[0].cost))
  else:
      # red_not_falsified.append([d, res.runs[0].history[0].cost])
      print('Not falsified')
      print('Cost: {}'.format(res.runs[0].history[0].cost))
  return res, best_result

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

def red_white_check(b):
  if b[0] < 0.0:
    return True
  return False

if __name__ == '__main__':
    main()