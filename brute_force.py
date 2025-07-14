import torch
from models.models import FCModel
import copy 
import numpy as np
import json
import sys
sys.path.append('/home/koh/work/matiec_rampo/examples/misc')

print(sys.path)
from tree import *

from staliro import Sample, SignalInput, TestOptions, staliro
from staliro.models import Model, Result
from staliro.optimizers import DualAnnealing
from staliro.specifications import rtamt

import argparse

try:
    import matlab
    import matlab.engine
except ImportError:
    _has_matlab = False
else:
    _has_matlab = True

import time
import logging
import random


class TankControlFlowRate(Model[list[float], None]):
    MODEL_NAME = "tankcontrol_flowrate"

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
        engine.set_param(self.MODEL_NAME + '/open_loop', 'Value', '1', nargout=0)

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

    # # This works only for the current setting: cp = 10, sim_time = 30
    # extract_point_list = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    # # This works only for the current setting: cp = 4, sim_time = 30
    # # extract_point_list = [0, 8, 15, 23]
    sim_time = best_result.trace.times[-1]
    interval = sim_time / cp
    timing = [i * interval for i in range(cp)]

    # Find index of nearest time point in trace for each extract point
    extract_point_list = []
    for p in timing:
      # Find the index of the smallest value in times that is >= p
      idx = np.searchsorted(best_result.trace.times, p)
      # Make sure we don't go out of bounds
      if idx >= len(best_result.trace.times):
        idx = len(best_result.trace.times) - 1
      extract_point_list.append(idx)

    for p in extract_point_list:
        if best_result.trace.states[p][0] >= ranges['TankHeight'][0][0] and best_result.trace.states[p][0] <= ranges['TankHeight'][0][1]:
            path.append(0)
        elif best_result.trace.states[p][0] >= ranges['TankHeight'][1][0] and best_result.trace.states[p][0] <= ranges['TankHeight'][1][1]:
            path.append(1)
        elif best_result.trace.states[p][0] >= ranges['TankHeight'][2][0] and best_result.trace.states[p][0] <= ranges['TankHeight'][2][1]:
            path.append(2)
        elif best_result.trace.states[p][0] >= ranges['TankHeight'][3][0] and best_result.trace.states[p][0] <= ranges['TankHeight'][3][1]:
            path.append(3)
        else:
            path.append(-1)
    return path

# min max scaling: x' = (x - min) / (max - min)
# reverse scaling x = x' * (max - min) + min

# device = 'cuda:4'
device = 'cpu'

parser = argparse.ArgumentParser(description='Integrated Analysis')
parser.add_argument('--cp', type=int, required=True, help='Control points')
args = parser.parse_args()
cp = args.cp


sim_model = TankControlFlowRate()
ranges = {
  "TankHeight": [[0.0, 5.0], [5.0, 7.0], [7.0, 10.0], [10.0, 100.0]],
  "InValve": [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
    "OutValve": [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
}
phi = "(always[0,30] (TankHeight <= 8))"
specification = rtamt.parse_dense(phi, {"TankHeight": 0, "InValve": 1, "OutValve": 2})
optimizer = DualAnnealing(min_cost=0.0)
signals = {
    "InValve": SignalInput(control_points=[(0, 1)] * cp),
    "OutValve": SignalInput(control_points=[(0, 1)] * cp),
    "InValveRate": SignalInput(control_points=[(30, 100)] * cp),
    "OutValveRate": SignalInput(control_points=[(30, 100)] * cp), 
}
options = TestOptions(runs=1, iterations=100, tspan=(0, 30), signals=signals)

def main():
    logging.basicConfig(level=logging.DEBUG)

    t = HistoryTrie(height=cp, num_child=4)
    tree = t.root
    start = time.perf_counter()
    # randomly pick a int from {0, 1, 2, 3}
    path = [random.randint(0, 3) for _ in range(cp)]
    node = t.root
    for p in path:
      node = node.children[p]

    res, best_result = falsification_with_actual_model(node)
    end = time.perf_counter()

    print('Time: {}'.format(end - start))


    stamp = str(time.strftime("%Y%m%d_%H%M%S")) + '_brute_force' + str(cp)
    execution_info = {
      'execution_time': end - start
    }
    with open('/home/koh/work/DeepBern-Nets/result/brute_force/' + stamp + '_numbers.json', 'w') as f:
      json.dump(execution_info, f)
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
  Phi = "(G[0,30] (TankHeight <= 8))"
  phi = ''
  epsilon = 0.0
  for i, p in enumerate(path):
    phi += '(G[{:.2f}, {:.2f}] (TankHeight >= {} and TankHeight <= {}))' \
            .format(max(0, cp_array[i] - epsilon), min(cp_array[i] + epsilon, sim_time), \
            ranges['TankHeight'][p][0], ranges['TankHeight'][p][1])
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
      extra_phi += '(G[{:.2f}, {:.2f}] (TankHeight >= {} and TankHeight <= {}))' \
            .format(max(0, cp_array[j] - epsilon), min(cp_array[j] + epsilon, sim_time), \
            ranges['TankHeight'][c][0], ranges['TankHeight'][c][1])
      if k != len(pruning_children) - 1:
        extra_phi += ' or '
    if extra_phi != '':
      phi += ' or (' + extra_phi + ')'

  print('Searching Node: {}, Phi: {}'.format(path, phi))

  # spec = TLTK(phi, {'TankHeight': 0, 'InValve': 1, 'OutValve': 2})
  spec = rtamt.parse_dense(phi, {'TankHeight': 0, 'InValve': 1, 'OutValve': 2})
  res = staliro(sim_model, spec, optimizer, options)
  res[0].evaluations.sort(key=lambda x: x.cost)
  best_sample = res[0].evaluations[0].sample.values
  best_result = res[0].evaluations[0].extra
  if res[0].evaluations[0].cost < 0:
      # red_falsified.append([d, res[0].evaluations[0].cost])
      print('Falsified')
      print('Cost: {}'.format(res[0].evaluations[0].cost))
  else:
      # red_not_falsified.append([d, res[0].evaluations[0].cost])
      print('Not falsified')
      print('Cost: {}'.format(res[0].evaluations[0].cost))
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