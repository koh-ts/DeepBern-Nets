import torch
from models.models import FCModel
import copy 
import numpy as np
import json
import sys
sys.path.append('/home/koh/work/matiec_rampo/examples/misc')
sys.path.append('/home/koh/work/angr-staliro/models')
# sys.path.append('/home/koh/work/angr-staliro/models')
# sys.path.insert(0, '/home/koh/work/psy-taliro/src')
# sys.path.insert(0, '/home/koh/miniconda3/envs/deepbern/lib/python3.9/site-packages')
# sys.path.append('/home/koh/work')
print(sys.path)
from tree import *
from graphviz import *
import argparse

from staliro import Sample, SignalInput, TestOptions, staliro
from staliro.models import Model, Result
from staliro.optimizers import DualAnnealing
from staliro.specifications import rtamt

# from staliro import models, optimizers, specifications
# from staliro.options import TestOptions
# from staliro.staliro import staliro

# from staliro.core.result import worst_run, worst_eval
# from staliro.options import Options, SignalOptions
# from staliro.specifications import TLTK, RTAMTDense
# from staliro.staliro import simulate_model, staliro, staliro_
# from staliro.staliro import simulate_model, staliro

# sys.path.append('/home/koh/work/matiec_rampo/examples/tankcontrol_flowrate')

try:
    import matlab
    import matlab.engine
except ImportError:
    _has_matlab = False
else:
    _has_matlab = True

import time
import logging

# old engine spec data
# scaling_factor = {
#   "speed_min": 0.0,
#   "speed_max": 135.54785661249386,
#   "rpm_min": 600.0,
#   "rpm_max": 4775.429858741292,
#   "robustness_min": -15.5478515625,
#   "robustness_max": 2332.626220703125,
# }

# new enigne spec data for small
scaling_factor = {
  "speed_min": 0.0,
  "speed_max": 134.413849817066,
  "rpm_min": 600.0,
  "rpm_max": 4775.234658160327,
  "robustness_min": -34.413848876953125,
  "robustness_max": 2120.63232421875,
  "train_len": 23200,
  "test_len": 5800
}


# new engine spec data for medium
# scaling_factor = {
#   "speed_min": -2.971394183178259,
#   "speed_max": 166.5363382521565,
#   "rpm_min": 600.0,
#   "rpm_max": 6000.0,
#   "robustness_min": -65.91999816894531,
#   "robustness_max": 1079.2451171875,
#   "train_len": 64800,
#   "test_len": 16200
# }

class VehicleEngine(Model[list[float], None]):
    MODEL_NAME = "sldemo_autotrans_mod03"

    def __init__(self) -> None:
        if not _has_matlab:
            raise RuntimeError(
                "Simulink support requires the MATLAB Engine for Python to be installed"
            )

        # engine = matlab.engine.start_matlab()
        engine = matlab.engine.connect_matlab(matlab.engine.find_matlab()[0])
        # engine.addpath("examples")

        is_loaded = engine.bdIsLoaded(self.MODEL_NAME)
        if not is_loaded:
            engine.open_system(self.MODEL_NAME, nargout=0)
        # engine.set_param(self.MODEL_NAME + '/open_loop', 'Value', '1', nargout=0)

        model_opts = engine.simget(self.MODEL_NAME)

        self.sampling_step = 0.2
        self.engine = engine
        self.model_opts = engine.simset(model_opts, "SaveFormat", "Array")

    def simulate(self, sample: Sample) -> Result[list[float], None]:
        tstart, tend = sample.signals.tspan
        duration = tend - tstart
        sim_t = matlab.double([0, tend])
        n_times = duration // self.sampling_step
        signal_times = np.linspace(tstart, tend, num=int(n_times))
        signal_values = np.array(
            [[signal.at_time(t) for t in signal_times] for signal in sample.signals]
        )
        model_input = matlab.double(np.row_stack((signal_times, signal_values)).T.tolist())

        timestamps, _, data = self.engine.sim(
            self.MODEL_NAME, sim_t, self.model_opts, model_input, nargout=3
        )

        times: list[float] = np.array(timestamps).flatten().tolist()
        states: list[list[float]] = list(data)

        return Result(times=times, states=states, extra=None)

def path_extraction(best_result):
    # result.runs[0].history.sort(key=lambda x: x.cost)
    # best_sample = worst_eval(worst_run(result)).sample
    # best_result = simulate_model(model, options, best_sample)
    path = []

    # This works only for the current setting: cp = 10, sim_time = 30
    # l = len(best_result.trace.states)
    # cp = 10
    # interval = l // cp
    # extract_point_list = [int(i * interval) for i in range(cp)]
    # extract_point_list = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    # This works only for the current setting: cp = 4, sim_time = 30
    # extract_point_list = [0, 8, 15, 23]


    # Check if the part below works
    sim_time = best_result.trace.times[-1]
    interval = sim_time / cp
    extract_point_list = [i * interval for i in range(cp)]

    # Find index of nearest time point in trace for each extract point
    indices = []
    for p in extract_point_list:
      # Find the index of the smallest value in times that is >= p
      idx = np.searchsorted(best_result.trace.times, p)
      # Make sure we don't go out of bounds
      if idx >= len(best_result.trace.times):
        idx = len(best_result.trace.times) - 1
      indices.append(idx)

    for p in indices:
        if best_result.trace.states[p][0] >= ranges['speed'][0][0] and best_result.trace.states[p][0] <= ranges['speed'][0][1] and \
            best_result.trace.states[p][1] >= ranges['rpm'][0][0] and best_result.trace.states[p][1] <= ranges['rpm'][0][1]:
            path.append(0)
        elif best_result.trace.states[p][0] >= ranges['speed'][1][0] and best_result.trace.states[p][0] <= ranges['speed'][1][1] and \
            best_result.trace.states[p][1] >= ranges['rpm'][1][0] and best_result.trace.states[p][1] <= ranges['rpm'][1][1]:
            path.append(1)
        elif best_result.trace.states[p][0] >= ranges['speed'][2][0] and best_result.trace.states[p][0] <= ranges['speed'][2][1] and \
            best_result.trace.states[p][1] >= ranges['rpm'][2][0] and best_result.trace.states[p][1] <= ranges['rpm'][2][1]:
            path.append(2)
        elif best_result.trace.states[p][0] >= ranges['speed'][3][0] and best_result.trace.states[p][0] <= ranges['speed'][3][1] and \
            best_result.trace.states[p][1] >= ranges['rpm'][3][0] and best_result.trace.states[p][1] <= ranges['rpm'][3][1]:
            path.append(3)
        else:
            path.append(-1)
    return path

# min max scaling: x' = (x - min) / (max - min)
# reverse scaling x = x' * (max - min) + min

# device = 'cuda:4'
device = 'cpu'

sim_model = VehicleEngine()
ranges = {
	"throttle": [[87.7, 100], [0, 81.599], [0, 63.099], [0, 87.298]],
	"rpm": [[0, 3300], [0, 3300], [3300, 4500], [3300, 4500]],
	"speed": [[0, 80], [80, 120], [0, 80], [80, 120]]
}
safety_speed = 100
safety_rpm = 4300
phi = "!(F[0,30] (Speed >= " + str(safety_speed) +") and F[0,30] (RPM >= " + str(safety_rpm) + "))"# specification = TLTK(phi, {"TankHeight": 0, "InValve": 1, "OutValve": 2})
print('phi: {}'.format(phi))
specification = rtamt.parse_dense(phi, {"Speed": 0, "RPM": 1})
min_cost = 0.01
optimizer = DualAnnealing(min_cost=min_cost)
signals = {
    "throttle": SignalInput(control_points=[(0, 100)] * 10),
}
options = TestOptions(runs=1, iterations=100, tspan=(0, 30), signals=signals, seed=123)


parser = argparse.ArgumentParser(description='Integrated Analysis')
parser.add_argument('--cp', type=int, required=True, help='Control points')
args = parser.parse_args()
cp = args.cp
# cp = 10

def main():
    logging.basicConfig(level=logging.DEBUG)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    # params = torch.load('/home/koh/work/DeepBern-Nets/experiments/staliro/vehicle_engine/vehicle_engine_06/checkpoint_best_model.pth')
    params = torch.load('/home/koh/work/DeepBern-Nets/experiments/staliro/vehicle_engine_new/vehicle_engine_01/checkpoint_best_model.pth')
    input_dimension = params['model_state_dict']['net.0.weight'].shape[1]
    num_neurons = params['model_state_dict']['net.0.weight'].shape[0]
    num_layers = (len(params['model_state_dict']._metadata) - 3) // 2
    degree = params['model_state_dict']['net.1._basis_indices'].shape[1]-1
    model = FCModel([input_dimension] + [num_neurons] * num_layers + [1], degree).to(device)
    # model = FCModel([input_dimension,515,515,515,515,515,1], 8).to(device)
    model.eval()
    input_bounds_ = torch.tensor([[0.0, 1.0] for _ in range(input_dimension)]).to(device)
    model.load_state_dict(params['model_state_dict'])
    model.input_bounds = input_bounds_
    init_bounds = torch.tensor([[0.0, 1.0]] * input_dimension).to(device)
    entire_bounds = model.forward_subinterval(init_bounds)
    entire_bounds_ = rescaling_output_engine(entire_bounds.squeeze(0).squeeze(0))
    print('Entire bound: {}'.format(entire_bounds_.tolist()))

    t = HistoryTrie(height=cp, num_child=4)
    tree = t.root
    # path table is from angr-staliro/misc/min_max_control04.json
    path_table = {0: [[scaling_factor['rpm_min'], 3300], [scaling_factor['speed_min'], 80]], 1: [[scaling_factor['rpm_min'], 3300], [80, scaling_factor['speed_max']]], 2: [[3300, scaling_factor['rpm_max']], [scaling_factor['speed_min'], 80]], 3: [[3300, scaling_factor['rpm_max']], [80, scaling_factor['speed_max']]]}
    q = []
    safe_range = []
    unsafe_range = []
    red_range = []
    yellow_range = []
    falsified_list = []
    not_falsified_list = []
    start = time.time()

    res = staliro(sim_model, specification, optimizer, options)
    res[0].evaluations.sort(key=lambda x: x.cost)
    best_sample = res[0].evaluations[0].sample.values
    best_result = res[0].evaluations[0].extra
    path = path_extraction(best_result)

    node = t.root
    bound_ = [-0.1, 0.1]
    v_count = 0
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
          yellow_range.append((node.path, v_count))
          v_count += 1
          if node.height == 0:
            # This is when the node is a leaf and still yellow result
            break
          else:
            if not node.children[path[j]].visited:
              node = node.children[path[j]]
            # What if the child node is visited?
          # one_dimension: num of states for one state veriable (e.g., Speed, RPM)
          one_dimension = input_dimension // 2

          # each_interval: num of states for one control point
          # If one_dimension = 751, meaning 0.04 sec for sampling time for 30 sec of simulation time,
          # and cp = 10, then each_interval = 75, meaning all this 75 states fall in the same branch
          # in the control prgram for one control point duration
          each_interval = one_dimension // cp
          tmp = []
          tmp_speed = []
          tmp_rpm = []
          for i, p in enumerate(node.path):
            tmp_val = path_table[p]
            tmp_speed += [tmp_val[1]] * each_interval
            tmp_rpm += [tmp_val[0]] * each_interval

          tmp_speed += [[0, 135.6] for _ in range(one_dimension - len(tmp_speed))]
          tmp_rpm += [[600, 4775.5] for _ in range(one_dimension - len(tmp_rpm))]
          tmp = tmp_speed + tmp_rpm

          # NN reachability analysis
          input_bound = scaling_input_bound_engine(torch.tensor(tmp).to(device))
          bound = model.forward_subinterval(input_bound)
          bound_ = rescaling_output_engine(bound.squeeze(0).squeeze(0))

        # If the NN reachability result is red or white, we need to go up to the parent node and
        # do the falsification with the actual model excluding the child node that has previously been visited
        else:
          if bound_[0] > 0.0:
            # This means that the entire output bound is positive (white)
            safe_range.append((node.path, v_count))
            v_count += 1
          else:
            # This means that the entire output bound is negative (red)
            red_range.append((node.path, v_count))
            v_count += 1
          node.visited = True
          node = node.parent
          break

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
        falsified = res[0].evaluations[0].cost < 0.0
        if falsified:
          if node.height == 0:
            # If the node is a leaf node and it's falsified, we store this node as a potential vulnerable node
            falsified_list.append([node.path, res, best_result, v_count])
            v_count += 1
            node.visited = True
            node = node.parent
            falsified = False
          else:
            # If the node is not a leaf node but it's falsified, then we extract the path and go towards the leaf node by one depth
            # This path should not include the visited children nodes because we exclude them in the falsification attemp by adding the extra constraints
            path = path_extraction(best_result)
            l = len(node.path)
            # if node.children[path[l]].visited:
              # for c in node.children:
              #   if not c.visited:
              #     node = c
            if node.children[path[l]].visited:
              # If the child node is visited, we need to go up to the parent node
              node.visited = True
              node = node.parent
            else:
              node = node.children[path[l]]
            break
        else:
          # If the result is not falsified, then the entire subtree can regarded as safe, meaning there will be no vulnerable node
          node.visited = True
          if node == t.root:
            break
          elif node.height == 0:
            not_falsified_list.append([node.path, res, best_result, v_count])
            v_count += 1
          node = node.parent
      
      if node == t.root:
        continue

      tmp = []
      tmp_speed = []
      tmp_rpm = []
      for i, p in enumerate(node.path):
        tmp_val = path_table[p]
        tmp_speed += [tmp_val[1]] * each_interval
        tmp_rpm += [tmp_val[0]] * each_interval

      tmp_speed += [[0, 135.6] for _ in range(one_dimension - len(tmp_speed))]
      tmp_rpm += [[600, 4775.5] for _ in range(one_dimension - len(tmp_rpm))]
      tmp = tmp_speed + tmp_rpm

      # NN reachability analysis
      input_bound = scaling_input_bound_engine(torch.tensor(tmp).to(device))
      bound = model.forward_subinterval(input_bound)
      bound_ = rescaling_output_engine(bound.squeeze(0).squeeze(0))
    end = time.time()
    # print('Time: {}'.format(end - start))
    # print('safe range: {}'.format(safe_range))
    # print('red range: {}'.format(red_range))
    # print('falsified list: {}'.format(falsified_list))
    # print('not falsified list: {}'.format(not_falsified_list))
    print('Time: {}'.format(end - start))
    stamp = str(time.strftime("%Y%m%d_%H%M%S")) + '_' + str(cp)

    execution_info = {
      'execution_time': end - start,
      'safe_range_count': len(safe_range),
      'red_range_count': len(red_range),
      'yellow_range_count': len(yellow_range),
      'falsified_list_count': len(falsified_list),
      'not_falsified_list_count': len(not_falsified_list),
      'v_count': v_count
    }
    with open('/home/koh/work/DeepBern-Nets/result/integrated_analysis_engine/' + stamp + '_numbers.json', 'w') as f:
      json.dump(execution_info, f)
    save_falsification_result(falsified_list, '/home/koh/work/DeepBern-Nets/result/integrated_analysis_engine/' + stamp + '_falsified.json')
    save_falsification_result(not_falsified_list, '/home/koh/work/DeepBern-Nets/result/integrated_analysis_engine/' + stamp + '_not_falsified.json')
    save_reachability_result(safe_range, '/home/koh/work/DeepBern-Nets/result/integrated_analysis_engine/' + stamp + '_safe_ranges.json')
    save_reachability_result(red_range, '/home/koh/work/DeepBern-Nets/result/integrated_analysis_engine/' + stamp + '_red_ranges.json')
    save_reachability_result(yellow_range, '/home/koh/work/DeepBern-Nets/result/integrated_analysis_engine/' + stamp + '_yellow_ranges.json')
    print('Done')

def save_falsification_result(raw_data, filename):
  path_list = []
  cost_list = []
  sample_list = []
  states_list = []
  times_list = []
  v_count_list = []

  for r in raw_data:
    result = r[1][0]
    best_res = result.evaluations[0]
    best_sample = best_res.sample.values
    best_result = best_res.extra.trace
    states = [list(i) for i in best_result.states]
    times = [float(i) for i in best_result.times]

    path_list.append(r[0])
    cost_list.append(best_res.cost)
    sample_list.append(best_sample)
    states_list.append(states)
    times_list.append(times)
    v_count_list.append(r[3])

  processed_data = {
    'path': path_list,
    'cost': cost_list,
    'sample': sample_list,
    'states': states_list,
    'times': times_list,
    'v_count': v_count_list
  }

  with open(filename, 'w') as f:
    json.dump(processed_data, f)

def save_reachability_result(raw_data, filename):
  path_list = []
  v_count_list = []
  for r in raw_data:
    path_list.append(r[0])
    v_count_list.append(r[1])
  processed_data = {
    'path': path_list,
    'v_count': v_count_list
  }
  with open(filename, 'w') as f:
    json.dump(processed_data, f)

def falsification_with_actual_model(node):
  path = node.path
  num_cp = node.height + len(node.path)
  tstart, tend = options.tspan
  sim_time = tend - tstart
  interval = sim_time / num_cp
  cp_array = [i * interval for i in range(num_cp)]
  Phi = "!(F[0,30] (Speed >= " + str(safety_speed) +") and F[0,30] (RPM >= " + str(safety_rpm) + "))"
  phi = ''
  epsilon = 0.0
  for i, p in enumerate(path):
    phi += '(G[{}, {}] (Speed >= {} and Speed <= {} and RPM >= {} and RPM <= {}))' \
        .format(max(0, cp_array[i] - epsilon), min(cp_array[i] + epsilon, sim_time), \
        ranges['speed'][p][0], ranges['speed'][p][1], \
        ranges['rpm'][p][0], ranges['rpm'][p][1])
    if i != len(path) - 1:
        phi += ' and '
  if phi != '':
    phi = Phi + ' or ! (' + phi + ')'
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
      extra_phi += '(G[{}, {}] (Speed >= {} and Speed <= {} and RPM >= {} and RPM <= {}))' \
        .format(max(0, cp_array[j] - epsilon), min(cp_array[j] + epsilon, sim_time), \
        ranges['speed'][c][0], ranges['speed'][c][1], \
        ranges['rpm'][c][0], ranges['rpm'][c][1])
      if k != len(pruning_children) - 1:
        extra_phi += ' or '
    if extra_phi != '':
      phi += ' or (' + extra_phi + ')'

  print('Searching Node: {}, Phi: {}'.format(path, phi))

  spec = rtamt.parse_dense(phi, {'Speed': 0, 'RPM': 1})
  res = staliro(sim_model, spec, optimizer, options)
  res[0].evaluations.sort(key=lambda x: x.cost)
  best_sample = res[0].evaluations[0].sample.values
  best_result = res[0].evaluations[0].extra
  if res[0].evaluations[0].cost < min_cost:
      # red_falsified.append([d, res[0].evaluations[0].cost])
      print('Falsified')
      print('Cost: {}'.format(res[0].evaluations[0].cost))
  else:
      # red_not_falsified.append([d, res[0].evaluations[0].cost])
      print('Not falsified')
      print('Cost: {}'.format(res[0].evaluations[0].cost))
  return res, best_result

def scaling_input_bound_engine(b):
  # First 751 elements in b are for Speed, and the rest 751 elements are for RPM
  # We have speed_min, speed_max, rpm_min, rpm_max
  # Min max scaling: x' = (x - min) / (max - min)
  lb_speed = (b[:751, 0] - scaling_factor['speed_min']) / (scaling_factor['speed_max'] - scaling_factor['speed_min'])
  ub_speed = (b[:751, 1] - scaling_factor['speed_min']) / (scaling_factor['speed_max'] - scaling_factor['speed_min'])
  lb_rpm = (b[751:, 0] - scaling_factor['rpm_min']) / (scaling_factor['rpm_max'] - scaling_factor['rpm_min'])
  ub_rpm = (b[751:, 1] - scaling_factor['rpm_min']) / (scaling_factor['rpm_max'] - scaling_factor['rpm_min'])
  lb = torch.cat([lb_speed, lb_rpm])
  ub = torch.cat([ub_speed, ub_rpm])
  return torch.stack([lb, ub], dim=1)

def rescaling_output_engine(y):
  min_max = torch.tensor([scaling_factor['robustness_min'], scaling_factor['robustness_max']]).T.to(device)
  y = y * (min_max[1] - min_max[0]) + min_max[0]
  return y

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