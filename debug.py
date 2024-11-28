import torch
from models.models import FCModel
import copy 
import numpy as np
import json

if __name__ == '__main__':
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    params = torch.load('/home/koh/work/DeepBern-Nets/experiments/staliro/optuna/optuna_normalized_2/checkpoint_best_model.pth')
    model = FCModel([17,2048,2048,2048,2048,1], 4).to('cuda')
    input_bounds_ = torch.tensor([[0.0, 1.0] for _ in range(17)]).to('cuda:0')
    # model = FCModel([17,2048,2048,2048,2048,1], 4, input_bounds=input_bounds_).to('cuda:0')
    model.load_state_dict(params['model_state_dict'])
    model.input_bounds = input_bounds_
    x = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]).to('cuda:0')
    # x = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]).to('cuda:0')
    y = model(x, with_bounds=True)
    input_bounds = torch.tensor([[0.23, 0.24] for _ in range(17)]).to('cuda:0')
    bounds = model.forward_subinterval(input_bounds)
    print('lb: {}\nub: {}'.format(float(bounds[0][0][0]), float(bounds[0][0][1])))
    input_bounds = torch.tensor([[0.5, 0.6] for _ in range(17)]).to('cuda:0')
    bounds = model.forward_subinterval(input_bounds)
    print('lb: {}\nub: {}'.format(float(bounds[0][0][0]), float(bounds[0][0][1])))
    with open('/home/koh/work/matiec_rampo/examples/tankcontrol_flowrate/data/done/data_40_s.json', 'r') as f:
        data = json.load(f)
    
    input_bounds = torch.tensor([[0.77395, 0.87396],
                                [0.2616, 0.3617],
                                [0.2984, 0.3985],
                                [0.8142, 0.9143],
                                [0.0919, 0.1920],
                                [0.6001, 0.7002],
                                [0.7285, 0.8286],
                                [0.1879, 0.2880],
                                [0.0551, 0.1552],
                                [0.137485, 0.237486],
                                [0.328717, 0.428718],
                                [0.281133, 0.381134],
                                [0.75031, 0.85032], 
                                [0.216315, 0.316316], 
                                [0.334649, 0.434650], 
                                [0.211392, 0.311393], 
                                [0.316592, 0.416593]]).to('cuda:0')
    bounds = model.forward_subinterval(input_bounds)
    print('lb: {}\nub: {}'.format(float(bounds[0][0][0]), float(bounds[0][0][1])))
    yy = model.forward_with_bounds(x)

    print('aaaa')