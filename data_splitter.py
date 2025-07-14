import json
import glob
import os
import numpy as np
import re

from datasets import StaliroDataset

# data_list = ['20241107_103833.json','20241107_105313.json','20241107_110806.json','20241107_112257.json','20241107_113754.json']
# data_path = '/home/koh/work/matiec_rampo/examples/tankcontrol_flowrate/data_variable_cp'
data_path = '/home/koh/work/matiec_rampo/examples/vehicle_engine/data_025'
data_list = glob.glob(os.path.join(data_path, '*.json'))

# d_train = []
# d_test = []

# feature_vec_train = []
# feature_vec_test = []

# scaling_factor = {
#   "min": [
#     0.00012228660424362658,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     0.0,
#     -12.947212219238281
#   ],
#   "max": [
#     7.999964949027126,
#     8.48736771478237,
#     8.973167500638938,
#     9.452481768993627,
#     9.925128063251766,
#     10.394770541981519,
#     10.811635473914432,
#     11.293867117582742,
#     11.773110825827766,
#     12.248974776305648,
#     12.721085244405165,
#     13.189033979699555,
#     13.644411232334877,
#     14.073094462889278,
#     14.464576232984038,
#     14.811990889865044,
#     15.15830120608507,
#     15.510364709758887,
#     15.872620274371885,
#     16.34860212117229,
#     16.775598523290164,
#     17.267232982560255,
#     17.75560763595966,
#     18.23467056595263,
#     18.698969207133995,
#     19.14366880987464,
#     19.56455244032151,
#     19.95802098039758,
#     20.321093127801905,
#     20.65140539600957,
#     20.94721211427172,
#     7.999631881713867
#   ]
# }

def diff_cp_min_max_scaling():
    
    for i, file in enumerate(data_list):
        feature_vec_test = []
        d_test = []
        cp = re.search(r"_(\d+)\.json$", file).group(1)

        with open(os.path.join(data_path, file), 'r') as f:
            data = json.load(f)
            for d in data:
                states = np.array(d['states'])[:,0]
                d_ = np.concatenate((states, np.array([d['robustness']])))
                feature_vec_test.append(d_)
        
        feature_vec_test = np.array(feature_vec_test)

        min_train = np.array(scaling_factor['min'])
        max_train = np.array(scaling_factor['max'])
        # min_train = feature_vec_train.min(axis=0)
        # max_train = feature_vec_train.max(axis=0)

        range_values = max_train - min_train
        range_values[range_values == 0] = 1

        scaled_feature_test = (feature_vec_test - min_train) / range_values

        scaled_test = []

        for d in scaled_feature_test:
            scaled_states = d[:-1].tolist()
            scaled_robustness = d[-1]
            scaled_item = {
                'states': scaled_states,
                'robustness': scaled_robustness
            }
            scaled_test.append(scaled_item)

        print('a')

        with open(os.path.join(data_path + '/done', str(cp) + '.json'), 'w') as f:
            json.dump(scaled_test, f)

    # with open(os.path.join(data_path + '/done', 'data_240_state_robust_min_max___.json'), 'w') as f:
    #     json.dump({'min': min_train.tolist(), 'max': max_train.tolist()}, f, indent=2)

    print('ddddd')

def state_next_state():
    for i, file in enumerate(data_list):
        if i < 200:
            with open(os.path.join(data_path, file), 'r') as f:
                data = json.load(f)
                for d in data:
                    d_ = np.concatenate((np.array([d['samples']]), np.array(d['states']), np.array([d['robustness']])))
                    feature_vec_train.append(d_)
        elif i < 240:
            with open(os.path.join(data_path, file), 'r') as f:
                data = json.load(f)
                for d in data:
                    d_ = np.concatenate((np.array([d['samples']]), np.array(d['states']), np.array([d['robustness']])))
                    feature_vec_test.append(d_)
        else:
            break

def state_robust():
    feature_vec_train = []
    feature_vec_test = []
    d_train = []
    d_test = []
    for i, file in enumerate(data_list):
        if i < 200:
        # if i < 10:
            with open(os.path.join(data_path, file), 'r') as f:
                data = json.load(f)
                for d in data:
                    states = np.array(d['states'])[:,0]
                    d_ = np.concatenate((states, np.array([d['robustness']])))
                    feature_vec_train.append(d_)
        elif i < 240:
        # elif i < 12:
            with open(os.path.join(data_path, file), 'r') as f:
                data = json.load(f)
                for d in data:
                    states = np.array(d['states'])[:,0]
                    d_ = np.concatenate((states, np.array([d['robustness']])))
                    feature_vec_test.append(d_)
        else:
            break
    
    feature_vec_train = np.array(feature_vec_train)
    feature_vec_test = np.array(feature_vec_test)
    min_train = feature_vec_train.min(axis=0)
    max_train = feature_vec_train.max(axis=0)

    range_values = max_train - min_train
    range_values[range_values == 0] = 1

    scaled_feature_train = (feature_vec_train - min_train) / range_values
    scaled_feature_test = (feature_vec_test - min_train) / range_values

    scaled_train = []
    scaled_test = []

    for d in scaled_feature_train:
        scaled_states = d[:-1].tolist()
        scaled_robustness = d[-1]
        scaled_item = {
            'states': scaled_states,
            'robustness': scaled_robustness
        }
        scaled_train.append(scaled_item)

    for d in scaled_feature_test:
        scaled_states = d[:-1].tolist()
        scaled_robustness = d[-1]
        scaled_item = {
            'states': scaled_states,
            'robustness': scaled_robustness
        }
        scaled_test.append(scaled_item)

    print('a')

    with open(os.path.join(data_path + '/done', 'data_200_state_robust.json'), 'w') as f:
        json.dump(scaled_train, f)

    with open(os.path.join(data_path + '/done', 'data_40_state_robust.json'), 'w') as f:
        json.dump(scaled_test, f)

    with open(os.path.join(data_path + '/done', 'data_240_state_robust_min_max.json'), 'w') as f:
        json.dump({'min': min_train.tolist(), 'max': max_train.tolist()}, f, indent=2)

    print('ddddd')

def signal_robust():
    for i, file in enumerate(data_list):
        if i < 200:
            with open(os.path.join(data_path, file), 'r') as f:
                data = json.load(f)
                for d in data:
                    d_ = np.concatenate((np.array([d['init_cond']]), np.array(d['samples']), np.array([d['robustness']])))
                    feature_vec_train.append(d_)
        elif i < 240:
            with open(os.path.join(data_path, file), 'r') as f:
                data = json.load(f)
                for d in data:
                    d_ = np.concatenate((np.array([d['init_cond']]), np.array(d['samples']), np.array([d['robustness']])))
                    feature_vec_test.append(d_)
        else:
            break

    feature_vec_train = np.array(feature_vec_train)
    feature_vec_test = np.array(feature_vec_test)
    min_train = feature_vec_train.min(axis=0)
    max_train = feature_vec_train.max(axis=0)

    range_values = max_train - min_train
    range_values[range_values == 0] = 1

    scaled_feature_train = (feature_vec_train - min_train) / range_values
    scaled_feature_test = (feature_vec_test - min_train) / range_values

    scaled_train = []
    scaled_test = []

    for d in scaled_feature_train:
        scaled_init_cond = d[0]
        scaled_samples = d[1:-1].tolist()
        scaled_robustness = d[-1]
        scaled_item = {
            'init_cond': scaled_init_cond,
            'samples': scaled_samples,
            'robustness': scaled_robustness
        }
        scaled_train.append(scaled_item)

    for d in scaled_feature_test:
        scaled_init_cond = d[0]
        scaled_samples = d[1:-1].tolist()
        scaled_robustness = d[-1]
        scaled_item = {
            'init_cond': scaled_init_cond,
            'samples': scaled_samples,
            'robustness': scaled_robustness
        }
        scaled_test.append(scaled_item)

    # Get the mean of the samples for each dimension
    # samples = np.array([d['samples'] for d in d_train])
    # mean = samples.mean(axis=0)
    # std = samples.std(axis=0)
    # Get the mean of the init_cond
    # init_cond = np.array([d['init_cond'] for d in d_train])
    # init_cond_mean = init_cond.mean()
    # init_cond_std = init_cond.std()
    # Get the mean of the robustness
    # robustness = np.array([d['robustness'] for d in d_train])
    # robustness_mean = robustness.mean()
    # robustness_std = robustness.std()

    # for d in d_train:
    #     d['robustness'] = (d['robustness'] - robustness_mean) / robustness_std
    # for d in d_test:
    #     d['robustness'] = (d['robustness'] - robustness_mean) / robustness_std

    print('a')

    with open(os.path.join(data_path + '/done', 'data_200_scaled.json'), 'w') as f:
        json.dump(scaled_train, f)

    with open(os.path.join(data_path + '/done', 'data_40_scaled.json'), 'w') as f:
        json.dump(scaled_test, f)

    with open(os.path.join(data_path + '/done', 'data_240_scaled_min_max.json'), 'w') as f:
        json.dump({'min': min_train.tolist(), 'max': max_train.tolist()}, f, indent=2)


    # with open(os.path.join(data_path + '/done', 'data_200_s.json'), 'w') as f:
    #     json.dump(d_train, f)

    # with open(os.path.join(data_path + '/done', 'data_40_s.json'), 'w') as f:
    #     json.dump(d_test, f)

    # with open(os.path.join(data_path + '/done', 'data_240_s_mean_std.json'), 'w') as f:
    #     json.dump({'robustness_mean': robustness_mean, 'robustness_std': robustness_std}, f, indent=2)

    print('ddddd')

def vehicle_engine():
    feature_vec_train = []
    feature_vec_test = []
    d_train = []
    d_test = []
    speed_train_ = []
    speed_test_ = []
    rpm_train_ = []
    rpm_test_ = []
    robustness_train_ = []
    robustness_test_ = []
    # speed_max, speed_min, rpm_max, rpm_min, robustness_max, robustness_min = 0, np.inf, 0, np.inf, 0, np.inf
    # files = ['20250328_183504_10.json']
    # files = ['20250322_221719_10.json',
    #         '20250323_141926_10.json',
    #         '20250323_144659_10.json',
    #         '20250323_151436_10.json',
    #         '20250323_154214_10.json',
    #         '20250323_160953_10.json',
    #         '20250323_163734_10.json',
    #         '20250323_170515_10.json',
    #         '20250323_173315_10.json',
    #         '20250323_180106_10.json',
    #         '20250323_182856_10.json']

    # files = ['20250328_183504_10.json',
    #         '20250328_184048_10.json',
    #         '20250328_184916_10.json',
    #         '20250328_185458_10.json',
    #         '20250328_190040_10.json',
    #         '20250328_190625_10.json',
    #         '20250328_191207_10.json',
    #         '20250328_191753_10.json',
    #         '20250328_192341_10.json',
    #         '20250328_192922_10.json',
    #         '20250328_193508_10.json',
    #         '20250328_194053_10.json',
    #         '20250328_194638_10.json',
    #         '20250328_195221_10.json',
    #         '20250328_195803_10.json',
    #         '20250328_200348_10.json',
    #         '20250328_200933_10.json',
    #         '20250328_201515_10.json',
    #         '20250328_202102_10.json',
    #         '20250328_202654_10.json',
    #         '20250328_203237_10.json',
    #         '20250328_203825_10.json',
    #         '20250328_204415_10.json',
    #         '20250328_204959_10.json',
    #         '20250328_205544_10.json',
    #         '20250328_210139_10.json',
    #         '20250328_210723_10.json',
    #         '20250328_211311_10.json'] 025 small

    # Get all json files in the directory /home/koh/work/matiec_rampo/examples/vehicle_engine/data_025
    files = [f for f in os.listdir(data_path) if f.endswith('.json')]

    total_train_len = 0
    total_test_len = 0
    for file in files:   
        with open(os.path.join(data_path, file), 'r') as f:
            data = json.load(f)

        train_len = int(len(data) * 0.8)
        test_len = len(data) - train_len
        total_train_len += train_len
        total_test_len += test_len

        for i, d in enumerate(data):
            # if i < train_len:
            #     speed = np.array(d['states'])[:, 0]
            #     speed_max = max(speed_max, speed.max())
            #     speed_min = min(speed_min, speed.min())
            #     speed_train_.append(speed)
            #     rpm = np.array(d['states'])[:, 1]
            #     rpm_max = max(rpm_max, rpm.max())
            #     rpm_min = min(rpm_min, rpm.min())
            #     rpm_train_.append(rpm)
            #     robustness = np.array(d['robustness'])
            #     robustness_min = min(robustness_min, robustness.min())
            #     robustness_max = max(robustness_max, robustness.max())
            #     robustness_train_.append(robustness)
            # else:
            #     speed_test_.append(np.array(d['states'])[:, 0])
            #     rpm_test_.append(np.array(d['states'])[:, 1])
            #     robustness_test_.append(np.array(d['robustness']))
            if i < train_len:
                speed = np.array(d['states'])[:, 0]
                speed_train_.append(speed)
                rpm = np.array(d['states'])[:, 1]
                rpm_train_.append(rpm)
                robustness = np.array(d['robustness'])
                robustness_train_.append(robustness)
            else:
                speed_test_.append(np.array(d['states'])[:, 0])
                rpm_test_.append(np.array(d['states'])[:, 1])
                robustness_test_.append(np.array(d['robustness']))
    # added for dimension wise scaling
    speed_train_ = np.array(speed_train_)
    speed_test_ = np.array(speed_test_)
    rpm_train_ = np.array(rpm_train_)
    rpm_test_ = np.array(rpm_test_)
    robustness_train_ = np.array(robustness_train_)
    robustness_test_ = np.array(robustness_test_)
    speed_max = speed_train_.max(axis=0)
    speed_min = speed_train_.min(axis=0)
    rpm_max = rpm_train_.max(axis=0)
    rpm_min = rpm_train_.min(axis=0)
    robustness_max = robustness_train_.max()
    robustness_min = robustness_train_.min()

    speed_range_values = speed_max - speed_min
    speed_range_values[speed_range_values == 0] = 1
    rpm_range_values = rpm_max - rpm_min
    rpm_range_values[rpm_range_values == 0] = 1
    robustness_range_values = robustness_max - robustness_min

    speed_train_ = (np.array(speed_train_) - speed_min) / speed_range_values
    rpm_train_ = (np.array(rpm_train_) - rpm_min) / rpm_range_values
    robustness_train_ = (np.array(robustness_train_) - robustness_min) / robustness_range_values
    speed_test_ = (np.array(speed_test_) - speed_min) / speed_range_values
    rpm_test_ = (np.array(rpm_test_) - rpm_min) / rpm_range_values
    robustness_test_ = (np.array(robustness_test_) - robustness_min) / robustness_range_values 
    # till here

    # speed_train_ = (np.array(speed_train_) - speed_min) / (speed_max - speed_min)
    # rpm_train_ = (np.array(rpm_train_) - rpm_min) / (rpm_max - rpm_min)
    # robustness_train_ = (np.array(robustness_train_) - robustness_min) / (robustness_max - robustness_min)
    scaled_feature_train = np.concatenate((speed_train_, rpm_train_, robustness_train_.reshape(-1, 1)), axis=1)
    scaled_feature_test = np.concatenate((speed_test_, rpm_test_, robustness_test_.reshape(-1, 1)), axis=1)

    # speed_test_ = (np.array(speed_test_) - speed_min) / (speed_max - speed_min)
    # rpm_test_ = (np.array(rpm_test_) - rpm_min) / (rpm_max - rpm_min)
    # robustness_test_ = (np.array(robustness_test_) - robustness_min) / (robustness_max - robustness_min)
    # scaled_feature_test = np.concatenate((speed_test_, rpm_test_, robustness_test_.reshape(-1, 1)), axis=1)


    # feature_vec_train = np.array(feature_vec_train)
    # feature_vec_test = np.array(feature_vec_test)
    # min_train = feature_vec_train.min(axis=0)
    # max_train = feature_vec_train.max(axis=0)

    # range_values = max_train - min_train
    # range_values[range_values == 0] = 1

    # scaled_feature_train = (feature_vec_train - min_train) / range_values
    # scaled_feature_test = (feature_vec_test - min_train) / range_values

    scaled_train = []
    scaled_test = []

    for d in scaled_feature_train:
        scaled_states = d[:-1].tolist()
        scaled_robustness = float(d[-1])
        scaled_item = {
            'states': scaled_states,
            'robustness': scaled_robustness
        }
        scaled_train.append(scaled_item)

    for d in scaled_feature_test:
        scaled_states = d[:-1].tolist()
        scaled_robustness = float(d[-1])
        scaled_item = {
            'states': scaled_states,
            'robustness': scaled_robustness
        }
        scaled_test.append(scaled_item)

    print('a')

    with open(os.path.join(data_path + '/done', 'data_train_vehicle_engine_025_large01.json'), 'w') as f:
        json.dump(scaled_train, f)

    with open(os.path.join(data_path + '/done', 'data_test_vehicle_engine_025_large01.json'), 'w') as f:
        json.dump(scaled_test, f)

    with open(os.path.join(data_path + '/done', 'data_vehicle_engine_min_max_025_large01.json'), 'w') as f:
        json.dump({'speed_min': speed_min.tolist(),
                    'speed_max': speed_max.tolist(),
                    'rpm_min': rpm_min.tolist(),
                    'rpm_max': rpm_max.tolist(),
                    'robustness_min': robustness_min,
                    'robustness_max': robustness_max,
                    'train_len': total_train_len,
                    'test_len': total_test_len}, f, indent=2)

    print('ddddd')

if __name__ == "__main__":
    # signal_robust()
    # state-next_state()
    # state_robust()
    # diff_cp_min_max_scaling()
    vehicle_engine()