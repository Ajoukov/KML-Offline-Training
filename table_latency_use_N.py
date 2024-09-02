import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import resample
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

N = 16
M=32

# Load and preprocess the data
data = pd.read_csv("io_latency.csv", header=None)

# data = data[~data.apply(lambda row: row.astype(str).str.contains('999999')).any(axis=1)]

cols_features = []

for i in range(0, N):
    cols_features.append(f'size_prev_{i}')
    cols_features.append(f'tag0_prev_{i}')
    cols_features.append(f'tag1_prev_{i}')
    cols_features.append(f'tag2_prev_{i}')
    cols_features.append(f'tag3_prev_{i}')
    cols_features.append(f'tag4_prev_{i}')

# cols.append('latency')

data.columns = cols_features + ['latency']

# cols_completions = [f'lp{i}' for i in range(16)]

# data.columns = ['sector', 'size', 'op', 'tag0', 'tag1', 'tag2', 'tag3', 'tag4', 'queue', 'latency'] + cols_completions
# cols = []
# cols_to_merge = []

# latencies = data['latency']
# completions = np.mean(data[cols_completions], axis=1)

# for i in range(0, N):
#     cols_to_merge.append(data['size'].shift(i))
#     # cols_to_merge.append(data['op'].shift(i))
#     cols_to_merge.append(data['tag0'].shift(i))
#     cols_to_merge.append(data['tag1'].shift(i))
#     cols_to_merge.append(data['tag2'].shift(i))
#     cols_to_merge.append(data['tag3'].shift(i))
#     cols_to_merge.append(data['tag4'].shift(i))
#     # cols_to_merge.append(data['queue'].shift(i))
#     cols.append(f'size_prev_{i}')
#     # cols.append(f'op_prev_{i}')
#     cols.append(f'tag0_prev_{i}')
#     cols.append(f'tag1_prev_{i}')
#     cols.append(f'tag2_prev_{i}')
#     cols.append(f'tag3_prev_{i}')
#     cols.append(f'tag4_prev_{i}')
#     # cols.append(f'queue_prev_{i}')
    
# data = pd.concat(cols_to_merge, axis=1)
# data.columns = cols

def upsample(data, field, count):
    to_merge = []
    targets = data[field].unique()
    mid = (max(targets) + min(targets)) // 2
    for target in targets:
        cur_count = int(count*(0.7*abs(target-mid) + 0.3))
        cur = resample(data[data[field] == target], replace=True, n_samples=cur_count, random_state=42)
        to_merge.append(cur)
    return pd.concat(to_merge)

# data['latency'] = np.log(latencies)
data['latency'] = np.log(data['latency'])
# data['completion_avg'] = np.log(completions)
# cols.append('completion_avg')

data['mean'] = data['latency'].rolling(window=32).mean().shift(-16)
data['std'] = data['latency'].rolling(window=32).std().shift(-16)

data['z-score'] = (data['latency'] - data['mean']) / data['std']

data = data[abs(data['z-score']) < 2.5]
data = data.dropna()
# data = data.head(10000)

print(np.shape(data))

data['z-score'] = ((data['z-score'].abs() * 5) ** 1.5).astype(int) * data['z-score'].apply(lambda x: -1 if x < 0 else 1)

print(data.nunique())

# data['z-score'] = (data['z-score'] ** 3).astype(int)
data = upsample(data, 'z-score', 200)

print(np.shape(data))

# targets = (data['latency'] - np.mean(data['latency'])) * (data['latency'] - np.mean(data['latency'])) * (data['latency'] - np.mean(data['latency']))

targets = data['z-score']
# targets = data['latency']

# metadata = data[['mean', 'std']]
features = data[cols_features]

# features.to_csv('features_latency_use_N.csv', index=False)
# metadata.to_csv('metadata_latency_use_N.csv', index=False)
# targets.to_csv('targets_latency_use_N.csv', index=False)

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
y_train = torch.tensor(np.array(y_train.values), dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(np.array(y_test.values), dtype=torch.float32).view(-1, 1)

with open('X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open('X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open('y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open('y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)