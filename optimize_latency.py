import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Global paths and settings
INP_DIR = "input_data/"
OUT_DIR = "output_data/"
TMP_DIR = "tmp_data/"
WTS_DIR = OUT_DIR + "weights/"
TST_DIR = OUT_DIR + "tests/"

N = 16
WL_M = 128
M = 64
TARGET_ROWS = 100000
workloads = ["mixgraph", "updaterandom", "readrandom", "readrandomwriterandom", "readwhilewriting", "readreverse", "readseq", "fillseq", "fill100k"]
L = len(workloads)
EPOCHS = 1024 * 16
SHIFT = 64

import itertools

# Define possible values for each hyperparameter
take_log_options = [True, False]
use_zscore_options = [True, False]
fac_options = [50, 100, 200]  # You can adjust this range
exp_options = [1.2, 1.5, 2.0]  # You can adjust this range
zscore_max_options = [1.5, 2, 3]  # You can adjust this range
latency_max_options = [2000, 2500, 3000]  # You can adjust this range
latency_min_options = [50, 100, 150]  # You can adjust this range

# Use itertools.product to create all combinations of the hyperparameters
hyperparameter_combinations = list(itertools.product(
    take_log_options,
    use_zscore_options,
    fac_options,
    exp_options,
    zscore_max_options,
    latency_max_options,
    latency_min_options
))

# Convert the tuples into dictionaries with appropriate keys
hyperparameter_combinations_dicts = [
    {
        "TAKE_LOG": combination[0],
        "USE_ZSCORE": combination[1],
        "FAC": combination[2],
        "EXP": combination[3],
        "ZSCORE_MAX": combination[4],
        "LATENCY_MAX": combination[5],
        "LATENCY_MIN": combination[6]
    }
    for combination in hyperparameter_combinations
]


class Workload(nn.Module):
    def __init__(self):
        super(Workload, self).__init__()
        self.fc1 = nn.Linear(int(N*9 + 1), int(WL_M))
        self.fc2 = nn.Linear(int(WL_M), int(WL_M/2))
        self.fc3 = nn.Linear(int(WL_M/2), int(WL_M/4))
        self.fc4 = nn.Linear(int(WL_M/4), int(WL_M/8))
        self.fc5 = nn.Linear(int(WL_M/8), int(WL_M/8))
        self.fc6 = nn.Linear(int(WL_M/8), L)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(int(N*9), int(M))
        self.fc2 = nn.Linear(int(M), int(M/2))
        self.fc3 = nn.Linear(int(M/2), int(M/4))
        self.fc4 = nn.Linear(int(M/4), int(M/8))
        self.fc5 = nn.Linear(int(M/8), int(M/8))
        self.fc6 = nn.Linear(int(M/8), 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

def load_and_preprocess_data(config):
    # Load and preprocess the data
    data = pd.read_csv(INP_DIR + "inputs.csv", header=None)

    cols_features = []
    for i in range(0, N):
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

    # Filter data based on latency
    data = data[data['latency'] < config["LATENCY_MAX"]]
    data = data[data['latency'] > config["LATENCY_MIN"]]

    if config["TAKE_LOG"]:
        data['latency'] = np.log(data['latency'])

    if config["USE_ZSCORE"]:
        data['mean'] = data['latency'].rolling(window=SHIFT * 2).mean().shift(-SHIFT)
        data['std'] = data['latency'].rolling(window=SHIFT * 2).std().shift(-SHIFT)
        data['z-score'] = (data['latency'] - data['mean']) / data['std']
        data = data[abs(data['z-score']) < config["ZSCORE_MAX"]]
        data = data.dropna()
        data['z-score'] = (abs(data['z-score'].abs() * config["FAC"]) ** config["EXP"]).astype(int) * data['z-score'].apply(lambda x: -1 if x < 0 else 1)
    else:
        data['z-score'] = data['latency']
        data = data.dropna()

    # Upsample data
    nunique = data['z-score'].nunique()
    count = int(TARGET_ROWS / nunique * 2)
    data = upsample(data, 'z-score', count, -0.2)

    return data[cols_features], data['z-score']

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
        cur = cur_data.sample(n=cur_count, replace=True, random_state=42)
        to_merge.append(cur)
    return pd.concat(to_merge)

def train_and_evaluate(config, X_train, X_test, y_train, y_test):
    # Initialize and train the model
    model = NeuralNetwork()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, EPOCHS):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Save the model
    torch.save(model.state_dict(), TMP_DIR + f"latency_{config}.pth")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
        y_test_np = y_test.numpy()
        mse = np.mean((predictions - y_test_np) ** 2)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_np, predictions)
        r, _ = pearsonr(y_test_np.flatten(), predictions.flatten())

    print(f'Config: {config}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R: {r:.4f}')

# Main script to run hyperparameter optimization
for config in hyperparameter_combinations_dicts:
    print(f"Running with config: {config}")
    features, targets = load_and_preprocess_data(config)
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train.values), dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(np.array(y_test.values), dtype=torch.float32).view(-1, 1)

    train_and_evaluate(config, X_train, X_test, y_train, y_test)
