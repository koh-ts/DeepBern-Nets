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
    # tree = HistoryTrie(height=4, num_child=4)
    min_ = scaling_factor['min'][-1]
    max_ = scaling_factor['max'][-1]
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

    with open('/home/koh/work/matiec_rampo/examples/tankcontrol_flowrate/data/done/data_40_state_robust.json', 'r') as f:
        data = json.load(f)
    init_bounds = torch.tensor([[0.0, 1.0]] * input_dimension).to(device)
    whole_bounds = model.forward_subinterval(init_bounds)
    whole_bounds_ = rescaling_output(whole_bounds.squeeze(0).squeeze(0))
    print('Entire bound: {}'.format(whole_bounds_.tolist()))
    ok_cnt = 0
    ng_cnt = 0
    # for i, d in enumerate(data):
    #     if rescaling_output(d['robustness']) < 0:
    #         idx = i
    #         x_ = data[idx]['states']
    #         x_ = torch.tensor(x_).to(device)
    #         y_ = model(x_, with_bounds=True)
    #         y = rescaling_output(y_)
    #         label = data[idx]['robustness']
    #         # 0.32, Correct: 9459, Incorrect: 1 with the test data
    #         # input_bounds = torch.tensor([[xx - 0.32, xx + 0.32] for xx in x_]).to(device)
    #         # Worst loss in train: 0.25424838066101074
    #         # Worst loss in test: 0.22922194004058838
    #         # 0.25424838066101074, Correct: 9380, Incorrect: 80 with the test data
    #         worst_loss = 0.25424838066101074
    #         # input_bounds = torch.tensor([[xx - worst_loss, xx + worst_loss] for xx in x_]).to(device)
    #         input_bounds = [[0,5] for _ in range(8)] + [[0, 12] for _ in range(31 - 8)]
    #         bounds = model.forward_subinterval(input_bounds)
    #         bounds_ = rescaling_output(bounds.squeeze(0).squeeze(0))
            
    #         if rescaling_output(label) >= bounds_[0] and rescaling_output(label) <= bounds_[1]:
    #             # print('Correct')
    #             ok_cnt+=1
    #         else:
    #             ng_cnt+=1
    #             print('Output bound: {}'.format(bounds_.tolist()))
    #             print('Inference: {}'.format(rescaling_output(y).tolist()[0]))
    #             print('Label: {}'.format(rescaling_output(label)))
    #             print('Incorrect')
            # print('=======================\n')

    t = HistoryTrie(height=4, num_child=4)
    tree = t.root
    path_table = {0: [0.0, 5.0], 1: [5.0, 7.0], 2: [7.0, 10.0], 3: [10.0, 10.1]}
    q = []
    safe_range = []
    unsafe_range = []
    for c in tree.children:
      q.append(c)
    while q:
      node = q.pop(0)
      path = node.path
      tmp = []
      for i, p in enumerate(path):
        if i == 2:
          tmp += [path_table[p] for _ in range(7)]
        else:
          tmp += [path_table[p] for _ in range(8)]
      tmp += [[0.0, 10.1] for _ in range(31 - len(tmp))]
      input_bound = scaling_input_bound(torch.tensor(tmp).to(device))
      bound = model.forward_subinterval(input_bound)
      bound_ = rescaling_output(bound.squeeze(0).squeeze(0))
      if negative_value_check(bound_):
        node.visited = True
        for c in node.children:
          q.append(c)
      if node.height == 0:
        if negative_value_check(bound_):
          print('Path: {}'.format(path))
          print('Output bound: {}'.format(bound_.tolist()))
          unsafe_range.append(path)
          node.visited = True
        else:
          safe_range.append(path)
    
    
      

    # input_bounds = torch.tensor([[0.0,5.0] for _ in range(8)] + [[0.0, 12.0] for _ in range(31 - 8)]).to(device)
    # t0p0 = rescaling_output(model.forward_subinterval(scaling_input_bound(input_bounds)))
    # print(negative_value_check(t0p0))

    # input_bounds = torch.tensor([[5.0,7.0] for _ in range(8)] + [[0.0, 12.0] for _ in range(31 - 8)]).to(device)
    # t0p1 = rescaling_output(model.forward_subinterval(scaling_input_bound(input_bounds)))
    # print(negative_value_check(t0p1))

    # input_bounds = torch.tensor([[7.0,10.0] for _ in range(8)] + [[0.0, 12.0] for _ in range(31 - 8)]).to(device)
    # t0p2 = rescaling_output(model.forward_subinterval(scaling_input_bound(input_bounds)))
    # print(negative_value_check(t0p2))

    # input_bounds = torch.tensor([[10.0,10.1] for _ in range(8)] + [[10.0, 10.1] for _ in range(31 - 8)]).to(device)
    # t0p3 = rescaling_output(model.forward_subinterval(scaling_input_bound(input_bounds)))
    # print(negative_value_check(t0p3))

    with open('/home/koh/work/matiec_rampo/examples/tankcontrol_flowrate/min_max.json', 'r') as f:
        ranges = json.load(f)

    # print('Correct: {}\nIncorrect: {}'.format(ok_cnt, ng_cnt))
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

if __name__ == '__main__':
    main()