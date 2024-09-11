import torch
import torch.nn as nn
import torch.optim as optim
import pickle

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

TAKE_LOG=True
USE_ZSCORE=False

SHIFT = 64
FAC = 50
EXP = 1.5
ZSCORE_MAX = 3

LATENCY_MAX = 2500
LATENCY_MIN = 100

print(WL_M/16, L)

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