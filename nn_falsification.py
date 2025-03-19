import torch
from models.models import FCModel
import numpy as np
import sys
sys.path.append('/home/koh/work/matiec_rampo/examples/misc')
sys.path.append('/home/koh/work/staliro')
import json
from tree import *
import logging
import time
from graphviz import *

from staliro.core.interval import Interval
from staliro.core.model import BasicResult, Model, ModelInputs, ModelResult, Trace
from staliro.core.result import best_eval, best_run
from staliro.core.signal import Signal
from staliro.optimizers import DualAnnealing
from staliro.options import Options, SignalOptions
from staliro.specifications import TLTK
from staliro.staliro import simulate_model, staliro
from staliro.models import Blackbox


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
    
class NNBlackbox:
    def __init__(self):
        self.device = 'cuda:4'
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)
        torch.backends.cudnn.enabled=False
        torch.backends.cudnn.deterministic=True
        params = torch.load('/home/koh/work/DeepBern-Nets/experiments/staliro/state_robust/state_robust_02/checkpoint_best_model.pth')
        input_dimension = len(scaling_factor['min'][:-1])
        self.model = FCModel([input_dimension,1024,1024,1024,1024,1], 8).to(self.device)
        self.input_bounds_ = torch.tensor([[0.0, 1.0] for _ in range(input_dimension)]).to(self.device)
        self.model.load_state_dict(params['model_state_dict'])
        self.model.input_bounds = self.input_bounds_
        self.model.eval()

    def __call__(self, static_parameters, signal_times, signal_values) -> ModelResult:
        """
        Args:
            static_parameters: Not used for this NN
            signal_times: Time points for simulation
            signal_values: Input values at each time point (32 inputs)
        """
        with torch.no_grad():
            # Process the 32-dimensional input
            inputs_tensor = torch.FloatTensor(signal_values).to(self.device)
            output = self.model(inputs_tensor.squeeze(0))
            output_value = output.cpu().numpy().flatten()

        # Create trace with single time point (since it's not time-dependent)
        trace = Trace(
            times=[0.0],  # Single time point
            states=[output_value]  # Single output
        )
        
        return BasicResult(trace)

# Instantiate the network and load the saved parameters.
nn = NNBlackbox()
model = Blackbox(nn, sampling_interval=1.0)

# Equation for the re-scaling
# x_scaled = (x - x_min) / (x_max - x_min)
# x_rescaled = x * (x_max - x_min) + x_min  
rescaled_zero = (0.0 - scaling_factor['min'][-1]) / (scaling_factor['max'][-1] - scaling_factor['min'][-1])
orginal_zero = rescaled_zero * (scaling_factor['max'][-1] - scaling_factor['min'][-1]) + scaling_factor['min'][-1]


phi = "always[0,30] (Output >= " + str(rescaled_zero) + ")"

specification = TLTK(phi, {"Output": 0})

optimizer = DualAnnealing()

path_table = {0: [0.0, 5.0], 1: [5.0, 7.0], 2: [7.0, 10.0], 3: [10.0, 10.1]}


# Adjust the scaling for the state depending on the simulation time
# In our case, the simulation time is 30 seconds, and the state points are 1 second apart (31 points)
# The control signals are applied 4 times during the simulation, so we need to scale the path table accordingly
# This way, the states can be classified into 4 different paths based on the control signal applied
timing_table = {0: [0, 7], 1: [8, 15], 2: [16, 22], 3: [23, 30]}

rescaled_path_table = []
for t in range(31):  # 31 time points
    rescaled_path_table.append({})
    for p, (l, u) in path_table.items():
        min_scale = scaling_factor['min'][t]
        max_scale = scaling_factor['max'][t]
        rescaled_path_table[t][p] = ((l - min_scale) / (max_scale - min_scale), (u - min_scale) / (max_scale - min_scale))



if __name__ == "__main__":
    tree = HistoryTrie(height=4, num_child=4)
    G = Digraph(format='png')
    G.attr('node', shape='circle')
    G.node('root', label='root')
    # logging.basicConfig(level=logging.DEBUG)
    falsified_count = 0
    for c in tree.root.children:
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
                    path = cccc.path
                    print("Current node: ", path)

                    signal = []
                    # for t in range(31):
                    #     if t in range(timing_table[0][0], timing_table[0][1]+1):
                    #         signal.append(SignalOptions(control_points=[rescaled_path_table[t][path[0]]], signal_times=[0.0]))
                    #     elif t in range(timing_table[1][0], timing_table[1][1]+1):
                    #         signal.append(SignalOptions(control_points=[rescaled_path_table[t][path[1]]], signal_times=[0.0]))
                    #     elif t in range(timing_table[2][0], timing_table[2][1]+1):
                    #         signal.append(SignalOptions(control_points=[rescaled_path_table[t][path[2]]], signal_times=[0.0]))
                    #     else:
                    #         signal.append(SignalOptions(control_points=[rescaled_path_table[t][path[3]]], signal_times=[0.0]))
                    # signals = signal

                    for t in range(31):
                        if t in range(timing_table[0][0], timing_table[0][1]+1):
                            signal.append(rescaled_path_table[t][path[0]])
                        elif t in range(timing_table[1][0], timing_table[1][1]+1):
                            signal.append(rescaled_path_table[t][path[1]])
                        elif t in range(timing_table[2][0], timing_table[2][1]+1):
                            signal.append(rescaled_path_table[t][path[2]])
                        else:
                            signal.append(rescaled_path_table[t][path[3]])
                    signals = [SignalOptions(control_points=signal)]

                    options = Options(runs=1, iterations=100, interval=(0, 31), signals=signals)
                    result = staliro(model, specification, optimizer, options)
                    result.runs[0].history.sort(key=lambda x: x.cost)
                    best_res = result.runs[0].history[0]
                    if result.runs[0].history[0].cost < 0:
                        print("Falsified")
                        print(best_res.cost)
                        falsified_count += 1
                        signals_ = best_res.sample.values
                        for i, d in enumerate(signals_):
                            dd = scaling_factor['min'][i] + d * (scaling_factor['max'][i] - scaling_factor['min'][i])
                            print("Signal ", i, ": ", dd)
                        G.node(str(path), label=str(path), style='filled', fillcolor='red')
                        print("A")
                    else:
                        print("Not falsified")
                        print(best_res.cost)
    
    # G.node('[0, 2, 3]', label='[0, 2, 3]', style='filled', fillcolor='green')
    G.render('tree', outfile='/home/koh/work/DeepBern-Nets/tree.png')
    print("done")
    print("Falsified count: ", falsified_count)