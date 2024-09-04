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
    cols_features.append(f'tag0_prev_{i}')
    cols_features.append(f'tag1_prev_{i}')
    cols_features.append(f'tag2_prev_{i}')
    cols_features.append(f'tag3_prev_{i}')
    cols_features.append(f'tag4_prev_{i}')

for wl in workloads:
    df = pd.read_csv(INP_DIR + wl + "_small.csv", header=None)
    for other_wl in workloads:
        df[other_wl] = 0
    df[wl] = 1
    data = pd.concat([data, df], ignore_index=True)

print(np.shape(data))

data.columns = cols_features + ['latency'] + workloads
data[workloads] = 0

data.drop('latency', axis=1)
data = data.dropna()

features = data[cols_features]
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