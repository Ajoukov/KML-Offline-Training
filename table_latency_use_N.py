from __global_paths import *

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import resample
import numpy as np
from sklearn.model_selection import train_test_split
import pickle


# Load and preprocess the data
data = pd.read_csv(INP_DIR + "inputs.csv", header=None)

# data = data[~data.apply(lambda row: row.astype(str).str.contains('999999')).any(axis=1)]

cols_features = []

for i in range(0, N):
    cols_features.append(f'size_prev_{i}')
    cols_features.append(f'op_prev_{i}')
    cols_features.append(f'lba_diff_prev_{i}')
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

def upsample(data, field, count, slope):
    to_merge = []
    targets = sorted(data[field].unique())
    mid = len(targets) // 2
    for loc, target in enumerate(targets):
        if loc <= mid:
            cur_count = count - int(count * (slope * (loc / mid)))
        else:
            cur_count = count - int(count * (slope * ((len(targets) - 1 - loc) / mid)))
        cur_data = data[data[field] == target]
        if cur_count > len(cur_data):
            cur_count = len(cur_data)
        cur = cur_data.sample(n=cur_count, replace=False, random_state=42)
        to_merge.append(cur)
    return pd.concat(to_merge)

# data['latency'] = np.log(latencies)
data['latency'] = np.log(data['latency'])
# data['completion_avg'] = np.log(completions)
# cols.append('completion_avg')

data['mean'] = data['latency'].rolling(window=32).mean().shift(-16)
data['std'] = data['latency'].rolling(window=32).std().shift(-16)

data['z-score'] = (data['latency'] - data['mean']) / data['std']

data = data[abs(data['z-score']) < 1.2]
# data = data[abs(data['latency']) < 2000]
data = data.dropna()
# data = data.head(10000)

print(np.shape(data))

data['z-score'] = (abs(data['z-score'].abs() * FAC) ** EXP).astype(int) * data['z-score'].apply(lambda x: -1 if x < 0 else 1)

nunique = data['z-score'].nunique()

print(f"Number of buckets: {nunique}")

count = int(TARGET_ROWS / nunique * 2) # rough estimate

data = upsample(data, 'z-score', count, -0.2)

print(np.shape(data))

# targets = (data['latency'] - np.mean(data['latency'])) * (data['latency'] - np.mean(data['latency'])) * (data['latency'] - np.mean(data['latency']))

targets = data['z-score']
# targets = data['latency']

# metadata = data[['mean', 'std']]
features = data[cols_features]

# features.to_csv(OUT_DIR + 'features_latency_use_N.csv', index=False)
# metadata.to_csv(OUT_DIR + 'metadata_latency_use_N.csv', index=False)
# targets.to_csv(OUT_DIR + 'targets_latency_use_N.csv', index=False)

# pd.set_option('display.max_columns', None)  # Show all columns
# pd.set_option('display.width', 1000)        # Set display width to a larger value
print(data[cols_features[:8]].describe())

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
y_train = torch.tensor(np.array(y_train.values), dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(np.array(y_test.values), dtype=torch.float32).view(-1, 1)

with open(TMP_DIR + 'X_train_latency.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open(TMP_DIR + 'X_test_latency.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open(TMP_DIR + 'y_train_latency.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open(TMP_DIR + 'y_test_latency.pkl', 'wb') as f:
    pickle.dump(y_test, f)