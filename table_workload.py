from __global_paths import *

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import resample
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

data = pd.DataFrame()

cols_features = []

for i in range(0, N):
    cols_features.append(f'size_prev_{i}')
    cols_features.append(f'op_prev_{i}')
    cols_features.append(f'lba_diff_prev_{i}')
    cols_features.append(f'q_size{i}')
    cols_features.append(f'tag0_prev_{i}')
    cols_features.append(f'tag1_prev_{i}')
    cols_features.append(f'tag2_prev_{i}')
    cols_features.append(f'tag3_prev_{i}')
    cols_features.append(f'tag4_prev_{i}')

for wl in workloads:
    df = pd.read_csv(INP_DIR + wl + ".csv", header=None)
    for other_wl in workloads:
        df[other_wl] = 0
    df[wl] = 1
    data = pd.concat([data, df], ignore_index=True)


data.columns = cols_features + ['latency'] + workloads

# data.drop('latency', axis=1)
data = data.dropna()


data['category'] = data[workloads].idxmax(axis=1)

# Separate the data by category
category_counts = data['category'].value_counts()
min_category_size = category_counts.min()  # Smallest category size

resampled_data = []

for category in data['category'].unique():
    category_data = data[data['category'] == category]
    resampled_category = resample(category_data, replace=True, n_samples=min_category_size * 20, random_state=123)
    resampled_data.append(resampled_category)

data = pd.concat(resampled_data)
data = data.drop(columns=['category'])

for wl in workloads:
    print(f"{wl}: {data[wl].sum()}")

features = data[cols_features + ['latency']]
targets = data[workloads]

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
y_test = torch.tensor(np.array(y_test), dtype=torch.float32)

with open(TMP_DIR + 'X_train_wl.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open(TMP_DIR + 'X_test_wl.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open(TMP_DIR + 'y_train_wl.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open(TMP_DIR + 'y_test_wl.pkl', 'wb') as f:
    pickle.dump(y_test, f)