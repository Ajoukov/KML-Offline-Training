from __global_paths import *

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import resample
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from scipy.signal import find_peaks
import seaborn as sns


# Load and preprocess the data
data = pd.read_csv(INP_DIR + "inputs.csv", header=None)

# data = data[~data.apply(lambda row: row.astype(str).str.contains('999999')).any(axis=1)]

cols_features = []
data = data.iloc[:, 9 * (DATA_N - N):]

for i in range(DATA_N - N, DATA_N):
    cols_features.append(f'size_prev_{i}')
    cols_features.append(f'op_prev_{i}')
    cols_features.append(f'lba_diff_prev_{i}')
    cols_features.append(f'q_size_{i}')
    cols_features.append(f'tag0_prev_{i}')
    cols_features.append(f'tag1_prev_{i}')
    cols_features.append(f'tag2_prev_{i}')
    cols_features.append(f'tag3_prev_{i}')
    cols_features.append(f'tag4_prev_{i}')

data.columns = cols_features + ['latency']

print(data)

log_lat = np.log2(data['latency']) * FAC
plt.figure(figsize=(12, 6))
sns.histplot(log_lat[log_lat < 50], kde=True)
plt.title('Latency log Distribution')
plt.xlabel('Latency')
plt.ylabel('Frequency')
plt.savefig('output_data/latency_dist.png')

import pandas as pd
import numpy as np

def upsample(data, field, count):
    upsampled_data = []
    unique_values = np.unique(data[field])

    for value in unique_values:
        bin_data = data[data[field] == value]
        if len(bin_data) < count:
            sampled_data = bin_data.sample(n=count, replace=True, random_state=42)
            numeric_cols = sampled_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                noise = np.random.normal(0, 0.01 * sampled_data[col].std(), size=len(sampled_data))
                sampled_data[col] += noise
        else:
            sampled_data = bin_data.sample(n=count, replace=False, random_state=42)
        upsampled_data.append(sampled_data)

    return pd.concat(upsampled_data)


if (TAKE_LOG):
    data['latency'] = np.log2(data['latency']) * FAC
    data['latency'] = data['latency'].astype(int)
    data = data[data['latency'] <= LATENCY_MAX]
    data = data[data['latency'] >= LATENCY_MIN]

if (GROUP):
    def assign_to_nearest_peak(val, peak_values):
        return np.abs(peak_values - val).argmin()

    data['latency'] = np.log2(data['latency']) * FAC
    data = data[data['latency'] <= LATENCY_MAX]
    data = data[data['latency'] >= LATENCY_MIN]
    data['latency'] *= 5 / FAC
    data['latency'] = data['latency'].astype(int)
    _, counts = np.unique(sorted(data['latency']), return_counts=True)
    print(len(counts))
    print(counts)
    peaks, _ = find_peaks(counts, prominence=100)
    print(len(peaks))
    print(peaks)
    peak_values = np.sort(np.unique(data['latency']))[peaks]
    print(f"Peak values: {peak_values}")
    data['latency'] = data['latency'].apply(lambda x: assign_to_nearest_peak(x, peak_values))
    print(np.unique(data['latency'], return_counts=True))

if (USE_ZSCORE):
    if (GLOBAL_ZSCORE):
        data['mean'] = data['latency'].mean()
        data['std'] = data['latency'].std()
    else:
        data['mean'] = data['latency'].rolling(window=SHIFT * 2).mean().shift(-SHIFT)
        data['std'] = data['latency'].rolling(window=SHIFT * 2).std().shift(-SHIFT)

    data['z-score'] = (data['latency'] - data['mean']) / data['std']

    data = data[abs(data['z-score']) < ZSCORE_MAX]
    data = data.dropna()
    print(np.shape(data))

    data['z-score'] = abs(data['z-score'].abs() ** EXP) * data['z-score'].apply(lambda x: -1 if x < 0 else 1)
else:
    data['z-score'] = data['latency']
    data = data.dropna()

count = int(TARGET_ROWS / 10) # rough estimate

data = upsample(data, 'z-score', count)
print(np.unique(data['latency'], return_counts=True))

targets = data['z-score']
features = data[cols_features]

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
if USE_BUCKETS:
    y_train = torch.tensor(np.array(y_train.values), dtype=torch.long).view(-1)
    y_test = torch.tensor(np.array(y_test.values), dtype=torch.long).view(-1)
else:
    y_train = torch.tensor(np.array(y_train.values), dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(np.array(y_test.values), dtype=torch.float32).view(-1, 1)
    
print(np.shape(y_test))

plt.figure(figsize=(12, 6))
sns.histplot(y_test, bins=100, kde=True)
plt.title('Raw z-score Distribution')
plt.xlabel('z-score')
plt.ylabel('Frequency')
plt.savefig('output_data/z-score_distr.png')

with open(TMP_DIR + 'X_train_latency.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open(TMP_DIR + 'X_test_latency.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open(TMP_DIR + 'y_train_latency.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open(TMP_DIR + 'y_test_latency.pkl', 'wb') as f:
    pickle.dump(y_test, f)