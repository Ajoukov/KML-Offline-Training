import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

seed=43
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

INP_DIR = "input_data/"
OUT_DIR = "output_data/"
TMP_DIR = "tmp_data/"
WTS_DIR = OUT_DIR + "weights/"
TST_DIR = OUT_DIR + "tests/"

DATA_N = 16
N = 4
WL_M = 128
# M = 128
M = 32
TARGET_ROWS = 200000
workloads = ["mixgraph", "updaterandom", "readrandom", "readrandomwriterandom", "readwhilewriting", "readreverse", "readseq", "fillseq", "fill100k"]
L = len(workloads)

EPOCHS = 1024 * 128
LR = 0.0015

TAKE_LOG=True
USE_ZSCORE=False
GLOBAL_ZSCORE=False
GROUP=False

USE_BUCKETS=False
NUM_LATENCIES = 32 if USE_BUCKETS else 1

SHIFT = 16
EXP = 1.5
# EXP = 1
FAC=2
ZSCORE_MAX = 3

# mongo
# LATENCY_MIN = 100
# LATENCY_MAX = 4000

# oltp
# LATENCY_MIN = 17 #Mongo:100
# LATENCY_MAX = 27 #Mongo:4000

# varmail
# LATENCY_MIN = 13
# LATENCY_MAX = 23

# mixgraph
LATENCY_MIN = 13
LATENCY_MAX = 33

HIDDEN_LAYERS=4

class Workload(nn.Module):
    def __init__(self):
        super(Workload, self).__init__()
        self.weights = [nn.Linear(int(M/(2 ** i)), int(M/(2 ** i)/2)) for i in range(0, HIDDEN_LAYERS)]
        self.start = nn.Linear(int(N*9), int(M))
        self.end = nn.Linear(int(M / 2 ** HIDDEN_LAYERS), L)

    def forward(self, x):
        x = torch.relu(self.start(x))
        for i in range(0, HIDDEN_LAYERS):
            x = torch.relu(self.weights[i](x))
        x = self.end(x)
        return x

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.weights = nn.ModuleList([nn.Linear(int(M/(2 ** i)), int(M/(2 ** i)/2)) for i in range(0, HIDDEN_LAYERS)])
        self.start = nn.Linear(int(N*9), int(M))
        self.end = nn.Linear(int(M / 2 ** HIDDEN_LAYERS), NUM_LATENCIES)
        print(self.start)
        print(self.weights)
        print(self.end)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = torch.relu(self.start(x))
        for i in range(0, HIDDEN_LAYERS):
            # x = torch.relu(self.weights[i](x))
            x = self.leaky_relu(self.weights[i](x))
        x = self.end(x)
        return x

def save_predictions(model, X_test, y_test):
    y_test = y_test.cpu().numpy()
    model.eval()
    device = next(model.parameters()).device
    X_test = X_test.to(device)
    with torch.no_grad():
        predictions = model(X_test)
    predictions = predictions.cpu().numpy()
    plt.figure(figsize=(15, 10))
    B = int(max(y_test) - min(y_test)) + 1
    min_val = y_test.min()
    max_val = y_test.max()
    bins = np.linspace(min_val, max_val + 1, B + 1)
    predictions_list = []
    labels = []
    for i in range(B):
        # mask = (y_test >= bins[i]) & (y_test < bins[i])
        mask = y_test == bins[i]
        predictions_list.append(predictions[mask].flatten())
        labels.append(f'{int(bins[i])}')
            
    box = plt.boxplot(predictions_list, showfliers=False, labels=labels, patch_artist=True, whis=[5, 95])
    colors = ['#FF9999', '#FFCC99', '#FFFF99', '#99FF99', '#99CCFF', '#CC99FF']
    cmap = LinearSegmentedColormap.from_list('custom_gradient', colors, N=B)
    colors = [cmap(i / (B)) for i in range(B)]

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.xlabel('True Value')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True Values (with 5th and 95th Percentiles)')
    plt.grid(True)
    plt.savefig(OUT_DIR + 'box_plots_predictions_with_percentiles.png')
    plt.show()