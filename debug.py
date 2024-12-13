import torch
from models.models import FCModel
import copy 
import numpy as np
import json

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

# scaling_factor = {
#   "min": [
#     1.3601257419226798e-06,
#     1.3601257419226798e-06,
#     1.3601257419226798e-06,
#     1.3601257419226798e-06,
#     1.3601257419226798e-06,
#     1.3601257419226798e-06,
#     1.3601257419226798e-06,
#     1.3601257419226798e-06,
#     0.002937264261383299,
#     1.747870493229442e-06,
#     1.747870493229442e-06,
#     1.747870493229442e-06,
#     1.747870493229442e-06,
#     1.747870493229442e-06,
#     1.747870493229442e-06,
#     2.096539176710266e-07,
#     2.096539176710266e-07,
#     2.096539176710266e-07,
#     2.096539176710266e-07,
#     2.096539176710266e-07,
#     2.096539176710266e-07,
#     2.096539176710266e-07,
#     2.096539176710266e-07,
#     0.002348103932300505,
#     8.04051494518454e-06,
#     8.04051494518454e-06,
#     8.04051494518454e-06,
#     8.04051494518454e-06,
#     8.04051494518454e-06,
#     8.04051494518454e-06,
#     8.04051494518454e-06,
#     -12.947212219238281
#   ],
#   "max": [
#     0.9999949975924147,
#     0.9999949975924147,
#     0.9999949975924147,
#     0.9999949975924147,
#     0.9999949975924147,
#     0.9999949975924147,
#     0.9999949975924147,
#     0.9999949975924147,
#     0.9977847945926406,
#     0.9999977945055926,
#     0.9999977945055926,
#     0.9999977945055926,
#     0.9999977945055926,
#     0.9999977945055926,
#     0.9999977945055926,
#     0.9999970493007093,
#     0.9999970493007093,
#     0.9999970493007093,
#     0.9999970493007093,
#     0.9999970493007093,
#     0.9999970493007093,
#     0.9999970493007093,
#     0.9999970493007093,
#     0.9994515591419667,
#     0.9999984583796007,
#     0.9999984583796007,
#     0.9999984583796007,
#     0.9999984583796007,
#     0.9999984583796007,
#     0.9999984583796007,
#     0.9999984583796007,
#     7.999631881713867
#   ]
# }

def debug():
  with open('/home/koh/work/matiec_rampo/examples/tankcontrol_flowrate/data/done/data_40_state_robust.json', 'r') as f:
    data = json.load(f)
  # for i, d in enumerate(data):
  #   if 0.0 in d['samples']:
  #     for ii, dd in enumerate(d['samples']):
  #       if dd == 0.0:
  #         print(i, ii)
  return data

def main():
    # min_ = -12.947212219238281
    # max_ = 7.999631881713867
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    params = torch.load('/home/koh/work/DeepBern-Nets/experiments/staliro/state_robust/state_robust_03/checkpoint_best_model.pth')
    input_dimension = len(scaling_factor['min'][:-1])
    model = FCModel([input_dimension,512,512,512,512,1], 4).to('cuda:0')
    input_bounds_ = torch.tensor([[0.0, 1.0] for _ in range(input_dimension)]).to('cuda:0')
    # model = FCModel([17,2048,2048,2048,2048,1], 4, input_bounds=input_bounds_).to('cuda:0')
    model.load_state_dict(params['model_state_dict'])
    model.input_bounds = input_bounds_
    data = debug()
    idx = -1
    for i, d in enumerate(data):
      if rescaling_robustness(d['robustness']) < 0:
        idx = i
    x_ = data[idx]['samples']
    x_ = torch.tensor(x_).to('cuda:0')
    # make a input bounds from x by adding/subtracting 0.1 for each lb and ub
    input_bounds = torch.tensor([[xx - 0.01, xx + 0.01] for xx in x_]).to('cuda:0')


    # input_bounds = torch.tensor([[xx - 0.1, xx + 0.1] for _ in range(input_dimension)]).to('cuda:0')
    x = x_.unsqueeze(0)
    y = model(x, with_bounds=True)
    print(y)
    label = data[idx]['robustness']
    # x = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]).to('cuda:0')
    # x = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]).to('cuda:0')
    # y = model(x, with_bounds=True)
    # input_bounds = torch.tensor([[0.23, 0.24] for _ in range(17)]).to('cuda:0')
    # bounds = model.forward_subinterval(input_bounds)
    # print('lb: {}\nub: {}'.format(float(bounds[0][0][0]), float(bounds[0][0][1])))
    # input_bounds = torch.tensor([[0.5, 0.6] for _ in range(17)]).to('cuda:0')
    # bounds = model.forward_subinterval(input_bounds)
    # print('lb: {}\nub: {}'.format(float(bounds[0][0][0]), float(bounds[0][0][1])))
    # with open('/home/koh/work/matiec_rampo/examples/tankcontrol_flowrate/data/done/data_40_s.json', 'r') as f:
    #     data = json.load(f)
    
    # input_bounds = torch.tensor([[0.77395, 0.87396],
    #                             [0.2616, 0.3617],
    #                             [0.2984, 0.3985],
    #                             [0.8142, 0.9143],
    #                             [0.0919, 0.1920],
    #                             [0.6001, 0.7002],
    #                             [0.7285, 0.8286],
    #                             [0.1879, 0.2880],
    #                             [0.0551, 0.1552],
    #                             [0.137485, 0.237486],
    #                             [0.328717, 0.428718],
    #                             [0.281133, 0.381134],
    #                             [0.75031, 0.85032], 
    #                             [0.216315, 0.316316], 
    #                             [0.334649, 0.434650], 
    #                             [0.211392, 0.311393], 
    #                             [0.316592, 0.416593]]).to('cuda:0')
    # bounds = model.forward_subinterval(input_bounds)
    # print('lb: {}\nub: {}'.format(float(bounds[0][0][0]), float(bounds[0][0][1])))
    # yy = model.forward_with_bounds(x)

    # x = torch.tensor([0.5034891637283067, 0.9306942060723306, 0.9375638754635564, 0.2775548965217323, 0.21546631017574988, 0.5303836078887472, 0.036676176324594086, 0.6726495645096164, 0.9281323163040858, 0.4377231715260603, 0.876642879734375, 0.07307733003533934, 0.8026005806972524, 0.6791905662337453, 0.4266436088407155, 0.4770551447887879, 0.6094008948987218]).to('cuda:0')
    # Create a narrow interval based on the input x
    # input_bounds = torch.tensor([[0.503, 0.504],
    #                             [0.930, 0.931],
    #                             [0.937, 0.938],
    #                             [0.277, 0.278],
    #                             [0.215, 0.216],
    #                             [0.530, 0.531],
    #                             [0.036, 0.037],
    #                             [0.672, 0.673],
    #                             [0.928, 0.929],
    #                             [0.437, 0.438],
    #                             [0.876, 0.877],
    #                             [0.073, 0.074],
    #                             [0.802, 0.803],
    #                             [0.679, 0.680],
    #                             [0.426, 0.427],
    #                             [0.477, 0.478],
    #                             [0.609, 0.610]]).to('cuda:0')
    # y = model(x, with_bounds=True)
    bounds = model.forward_subinterval(input_bounds)
    # bounds_ = bounds * (max_ - min_) + min_
    # print('lb: {}\nub: {}'.format(float(bounds[0][0][0]), float(bounds[0][0][1])))

    print('aaaa')

    # {"init_cond": 0.5034891637283067,
    # "samples": [0.9306942060723306, 0.9375638754635564, 0.2775548965217323, 0.21546631017574988, 0.5303836078887472, 0.036676176324594086, 0.6726495645096164, 0.9281323163040858, 0.4377231715260603, 0.876642879734375, 0.07307733003533934, 0.8026005806972524, 0.6791905662337453, 0.4266436088407155, 0.4770551447887879, 0.6094008948987218],
    # "robustness": 0.5404148848784927},

    # {"init_cond": 0.8465674954817122,
    # "samples": [0.6029759092253901, 0.825726591599673, 0.6533003920578835, 0.9933126425543068, 0.8653978945342509, 0.5464428297963971, 0.13031947366269853, 0.6163169302482682, 0.26235594059822626, 0.5364096838013326, 0.6720809224209511, 0.15477127175634828, 0.6347689799440032, 0.3117690851135224, 0.7172214858926735, 0.4082692718566702],
    # "robustness": 0.4379270270499142}
    # x = torch.tensor([0.8465674954817122, 0.6029759092253901, 0.825726591599673, 0.6533003920578835, 0.9933126425543068, 0.8653978945342509, 0.5464428297963971, 0.13031947366269853, 0.6163169302482682, 0.26235594059822626, 0.5364096838013326, 0.6720809224209511, 0.15477127175634828, 0.6347689799440032, 0.3117690851135224, 0.7172214858926735, 0.4082692718566702]).to('cuda:0')

# data_240_scaled_min_max.json
# min = -12.947212219238281
# max = 7.999631881713867
# min max scaling: x' = (x - min) / (max - min)
# reverse scaling x = x' * (max - min) + min

def rescaling_robustness(r):
  min_ = scaling_factor['min'][-1]
  max_ = scaling_factor['max'][-1]
  return r * (max_ - min_) + min_

if __name__ == '__main__':
  main()
  # debug()
