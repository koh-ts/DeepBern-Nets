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

# Scaling factor
# scaling_factor = {
#   "min": [
#     0.00012228660424362658,
#     1.3601257419226798e-06,
#     1.747870493229442e-06,
#     2.096539176710266e-07,
#     8.04051494518454e-06,
#     3.1278121113142987e-06,
#     1.075282831219937e-06,
#     3.4564845544649003e-06,
#     4.2816793421884825e-06,
#     0.00060746404559886,
#     8.050800770487143e-06,
#     5.2486354079617215e-05,
#     0.00021376569467324025,
#     9.383422990283385e-05,
#     0.0007122989004015867,
#     0.00010284856790665486,
#     0.0002594730554306146,
#     -12.947212219238281
#   ],
#   "max": [
#     7.999964949027126,
#     0.9999949975924147,
#     0.9999977945055926,
#     0.9999970493007093,
#     0.9999984583796007,
#     0.9999983435977366,
#     0.9999993161401693,
#     0.9999924671771248,
#     0.9999953125686163,
#     49.99944142860811,
#     49.99999445728556,
#     49.999760904294114,
#     49.99999652643017,
#     49.999950259612184,
#     49.9999297322792,
#     49.99972069263225,
#     49.99982517352103,
#     7.999631881713867
#   ]
# }

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

# data_240_state_robust.json
# Max error before scaling: 0.25424838066101074
# After scaling
# max_error = 


# data_240_scaled_min_max.json
# min = -12.947212219238281
# max = 7.999631881713867
# Max error after scaling
# max_error = 0.4389359652996063

# min max scaling: x' = (x - min) / (max - min)
# reverse scaling x = x' * (max - min) + min

# device = 'cuda:4'
device = 'cpu'
# torch.cuda.set_device(4)

def main():
    # torch.manual_seed(123)
    # torch.cuda.manual_seed(123)
    # torch.backends.cudnn.enabled=False
    # torch.backends.cudnn.deterministic=True
    # params = torch.load('/home/koh/work/DeepBern-Nets/experiments/staliro/state_robust/state_robust_02/checkpoint_best_model.pth')
    # input_dimension = len(scaling_factor['min'][:-1])
    # model = FCModel([input_dimension,1024,1024,1024,1024,1], 8).to(device)
    # input_bounds_ = torch.tensor([[0.0, 1.0] for _ in range(input_dimension)]).to(device)
    # model.load_state_dict(params['model_state_dict'])
    # model.input_bounds = input_bounds_

    # with open('/home/koh/work/matiec_rampo/examples/tankcontrol_flowrate/data/done/data_40_state_robust.json', 'r') as f:
    #     data = json.load(f)
    # init_bounds = torch.tensor([[0.0, 1.0]] * input_dimension).to(device)
    # entire_bounds = model.forward_subinterval(init_bounds)
    # entire_bounds_ = rescaling_output(entire_bounds.squeeze(0).squeeze(0))
    # print('Entire bound: {}'.format(entire_bounds_.tolist()))

    t = HistoryTrie(height=4, num_child=4)
    G = Digraph(format='png')
    G.attr('node', shape='circle')
    G.node('root', label='root')
    tree = t.root
    # path_table = {0: [0.0, 5.0], 1: [5.0, 7.0], 2: [7.0, 10.0], 3: [10.0, 10.1]}
    # q = []
    # safe_range = []
    # unsafe_range = []
    # red_range = []
    # yellow_range = []
    # for c in tree.children:
    #   q.append(c)

    # start = time.time()
    
    # while q:
    #   node = q.pop(0)
    #   path = node.path
    #   tmp = []
    #   for i, p in enumerate(path):
    #     if i == 2:
    #       tmp += [path_table[p] for _ in range(7)]
    #     else:
    #       tmp += [path_table[p] for _ in range(8)]
    #   tmp += [[0.0, 10.1] for _ in range(31 - len(tmp))]
    #   input_bound = scaling_input_bound(torch.tensor(tmp).to(device))
    #   bound = model.forward_subinterval(input_bound)
    #   bound_ = rescaling_output(bound.squeeze(0).squeeze(0))
    #   if negative_value_check(bound_):
    #     node.visited = True
    #     for c in node.children:
    #       q.append(c)
    #   if node.height == 0:
    #     if negative_value_check(bound_):

    #       if red_yellow_check(bound_):
    #          red_range.append(path)
    #       else:
    #          yellow_range.append(path)
    #       print('Path: {}'.format(path))
    #       print('Output bound: {}'.format(bound_.tolist()))
    #       node.visited = True
    #     else:
    #       safe_range.append(path)
    
        # with open('/home/koh/work/DeepBern-Nets/safe_ranges.json', 'w') as f:
        #     json.dump(safe_range, f)
        # with open('/home/koh/work/DeepBern-Nets/red_ranges.json', 'w') as f:
        #     json.dump(red_range, f)
        # with open('/home/koh/work/DeepBern-Nets/yellow_ranges.json', 'w') as f:
        #     json.dump(yellow_range, f)

    # end = time.time()

    # Open the json files
    rf_ = '/home/koh/work/DeepBern-Nets/red_falsified_2.json'
    rn_ = '/home/koh/work/DeepBern-Nets/red_not_falsified_2.json'
    yf_ = '/home/koh/work/DeepBern-Nets/yellow_falsified_2.json'
    yn_ = '/home/koh/work/DeepBern-Nets/yellow_not_falsified_2.json'
    rf = open_json_file(rf_)
    rn = open_json_file(rn_)
    yf = open_json_file(yf_)
    yn = open_json_file(yn_)

    # print('Time: {}'.format(end - start))
    # print('# of safe range: {}'.format(len(safe_range)))
    # print('# of red range: {}'.format(len(red_range)))
    # print('# of yellow range: {}'.format(len(yellow_range)))
    for c in t.root.children:
        G.node(str(c.path), label=str(c.path))
        G.edge('root', str(c.path))
        for cc in c.children:
            G.node(str(cc.path), label=str(cc.path))
            G.edge(str(c.path), str(cc.path))
            for ccc in cc.children:
                G.node(str(ccc.path), label=str(ccc.path))
                G.edge(str(cc.path), str(ccc.path))
                for cccc in ccc.children:
                    G.node(str(cccc.path), label=str(cccc.path))
                    G.edge(str(ccc.path), str(cccc.path))
    for rr in rf:
        r = rr[0]
        G.node(str(r), label=str(r), style='filled', fillcolor='red')
    for rr in rn:
        r = rr[0]
        G.node(str(r), label=str(r), style='filled', fillcolor='pink')
    for yy in yf:
        y = yy[0]
        G.node(str(y), label=str(y), style='filled', fillcolor='orange')
    for yy in yn:
        y = yy[0]
        G.node(str(y), label=str(y), style='filled', fillcolor='yellow')
    G.render('tree', outfile='/home/koh/work/DeepBern-Nets/falsification_with_actual_model_2.png')
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

def open_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

if __name__ == '__main__':
    main()