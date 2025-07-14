
from staliro import Sample, SignalInput, TestOptions, staliro
from staliro.models import Model, Result
from staliro.optimizers import DualAnnealing
from staliro.specifications import rtamt

import numpy as np
import json

try:
    import matlab
    import matlab.engine
except ImportError:
    _has_matlab = False
else:
    _has_matlab = True

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

sim_model = TankControlFlowRate()
cp = 10
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
    res = staliro(
        model=sim_model,
        options=options,
        specification=specification,
        optimizer=optimizer,
    )
    print('Done.')

if __name__ == '__main__':
    main()
