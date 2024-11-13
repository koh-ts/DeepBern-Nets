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
for i, data in enumerate(data_list):
    if i < 8:
        with open(os.path.join(data_path, data), 'r') as f:
            data = json.load(f)
            for d in data:
                d['init_cond'] /= 10
                d_ = np.array(d['samples'])
                d1 = (d_[0:4] - d_[0:4].min()) / (d_[0:4].max() - d_[0:4].min())
                d2 = (d_[4:8] - d_[4:8].min()) / (d_[4:8].max() - d_[4:8].min())
                d3 = (d_[8:12] - d_[8:12].min()) / (d_[8:12].max() - d_[8:12].min())
                d4 = (d_[12:16] - d_[12:16].min()) / (d_[12:16].max() - d_[12:16].min())
                d['samples'] = list(np.concatenate((d1, d2, d3, d4)))
            d_train += data
    elif i < 10:
        with open(os.path.join(data_path, data), 'r') as f:
            data = json.load(f)
            for d in data:
                d['init_cond'] /= 10
                d_ = np.array(d['samples'])
                d1 = (d_[0:4] - d_[0:4].min()) / (d_[0:4].max() - d_[0:4].min())
                d2 = (d_[4:8] - d_[4:8].min()) / (d_[4:8].max() - d_[4:8].min())
                d3 = (d_[8:12] - d_[8:12].min()) / (d_[8:12].max() - d_[8:12].min())
                d4 = (d_[12:16] - d_[12:16].min()) / (d_[12:16].max() - d_[12:16].min())
                d['samples'] = list(np.concatenate((d1, d2, d3, d4)))
            d_test += data
    else:
        break

with open(os.path.join(data_path + '/done', 'data_8_normalized_each.json'), 'w') as f:
    json.dump(d_train, f)

with open(os.path.join(data_path + '/done', 'data_2_normalized_each.json'), 'w') as f:
    json.dump(d_test, f)

print('ddddd')
# /home/koh/work/DeepBern-Nets/data/staliro/20241105_132530.json
# /home/koh/work/matiec_rampo/examples/tankcontrol_flowrate/data