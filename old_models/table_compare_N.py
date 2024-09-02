import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample

N = 16

# Load and preprocess the data
data = pd.read_csv("io_latency.csv", header=None)
data.columns = ['sector', 'size', 'op', 'tag0', 'tag1', 'tag2', 'tag3', 'tag4', 'queue', 'latency']

cols = ['size', 'op', 'tag0', 'tag1', 'tag2', 'tag3', 'tag4', 'queue']

cols_to_merge = [data[cols]]

biggers = []

for i in range(1, N):
    cols_to_merge.append(data['size'].shift(i))
    cols_to_merge.append(data['op'].shift(i))
    cols_to_merge.append(data['tag0'].shift(i))
    cols_to_merge.append(data['tag1'].shift(i))
    cols_to_merge.append(data['tag2'].shift(i))
    cols_to_merge.append(data['tag3'].shift(i))
    cols_to_merge.append(data['tag4'].shift(i))
    cols_to_merge.append(data['queue'].shift(i))
    cols.append(f'size_prev_{i}')
    cols.append(f'op_prev_{i}')
    cols.append(f'tag0_prev_{i}')
    cols.append(f'tag1_prev_{i}')
    cols.append(f'tag2_prev_{i}')
    cols.append(f'tag3_prev_{i}')
    cols.append(f'tag4_prev_{i}')
    cols.append(f'queue_prev_{i}')
    
    biggers.append(data['latency'] > data['latency'].shift(i))
    
data = pd.concat(cols_to_merge, axis=1)
data.columns = cols

biggers = pd.concat(biggers, axis=1)

print(np.shape(data))

def upsample(data, count):
    to_merge = []
    targets = data['bigger'].unique()
    mid = (max(targets) + min(targets)) // 2
    for target in targets:
        cur_count = int(count*(0.7*abs(target-mid) + 0.3))
        cur = resample(data[data['bigger'] == target], replace=True, n_samples=cur_count, random_state=42)
        cur['bigger_sq'] = (target - (N-1)/2) * abs(target - (N-1)/2)
        to_merge.append(cur)
    return pd.concat(to_merge)

data['bigger'] = biggers.sum(axis=1)

data = data.dropna()

data = upsample(data, 20000)

targets = data['bigger_sq']
features = data[cols]

features.to_csv('features.csv', index=False)
targets.to_csv('targets.csv', index=False)