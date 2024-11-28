import json
import glob
import os
import numpy as np

from datasets import StaliroDataset

# data_list = ['20241107_103833.json','20241107_105313.json','20241107_110806.json','20241107_112257.json','20241107_113754.json']
data_path = '/home/koh/work/matiec_rampo/examples/tankcontrol_flowrate/data'

data_list = glob.glob(os.path.join(data_path, '*.json'))


d_train = []
d_test = []
# for i, data in enumerate(data_list):
#     if i < 200:
#         with open(os.path.join(data_path, data), 'r') as f:
#             data = json.load(f)
#             for d in data:
#                 d['init_cond'] /= 10
#                 d_ = np.array(d['samples'])
#                 d1 = (d_[0:4] - d_[0:4].min()) / (d_[0:4].max() - d_[0:4].min())
#                 d2 = (d_[4:8] - d_[4:8].min()) / (d_[4:8].max() - d_[4:8].min())
#                 d3 = (d_[8:12] - d_[8:12].min()) / (d_[8:12].max() - d_[8:12].min())
#                 d4 = (d_[12:16] - d_[12:16].min()) / (d_[12:16].max() - d_[12:16].min())
#                 d['samples'] = list(np.concatenate((d1, d2, d3, d4)))
#             d_train += data
#     elif i < 240:
#         with open(os.path.join(data_path, data), 'r') as f:
#             data = json.load(f)
#             for d in data:
#                 d['init_cond'] /= 10
#                 d_ = np.array(d['samples'])
#                 d1 = (d_[0:4] - d_[0:4].min()) / (d_[0:4].max() - d_[0:4].min())
#                 d2 = (d_[4:8] - d_[4:8].min()) / (d_[4:8].max() - d_[4:8].min())
#                 d3 = (d_[8:12] - d_[8:12].min()) / (d_[8:12].max() - d_[8:12].min())
#                 d4 = (d_[12:16] - d_[12:16].min()) / (d_[12:16].max() - d_[12:16].min())
#                 d['samples'] = list(np.concatenate((d1, d2, d3, d4)))
#             d_test += data
#     else:
#         break


feature_vec_train = []
feature_vec_test = []
for i, file in enumerate(data_list):
    if i < 200:
        with open(os.path.join(data_path, file), 'r') as f:
            data = json.load(f)
            for d in data:
                d_ = np.concatenate((np.array([d['init_cond']]), np.array(d['samples'], np.array([d['robustness']]))))
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
    scaled_samples = d[1:].tolist()
    scaled_robustness = d[-1]
    scaled_item = {
        'init_cond': scaled_init_cond,
        'samples': scaled_samples,
        'robustness': scaled_robustness
    }
    scaled_train.append(scaled_item)

for d in scaled_feature_test:
    scaled_init_cond = d[0]
    scaled_samples = d[1:].tolist()
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