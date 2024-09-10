import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from torch.optim.lr_scheduler import StepLR

from __global_paths import *

class Workload(nn.Module):
    def __init__(self):
        super(Workload, self).__init__()
        self.fc1 = nn.Linear(int(N*8 + 1), int(WL_M))
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

N = 16
WL_M = 128
M = 64
workloads = ["mixgraph", "updaterandom", "readrandom", "readrandomwriterandom", "readwhilewriting", "readreverse", "readseq", "fillseq", "fill100k"]
L = len(workloads)
EPOCHS = 1024 * 24

FAC = 10
EXP = 1
TARGET_ROWS = 100000

with open(TMP_DIR + 'X_train_wl.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open(TMP_DIR + 'X_test_wl.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open(TMP_DIR + 'y_train_wl.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open(TMP_DIR + 'y_test_wl.pkl', 'rb') as f:
    y_test = pickle.load(f)

def train_model(model, X_train, y_train, optimizer, criterion, epochs=EPOCHS):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        # print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    return loss.item()

model = Workload()
criterion = nn.CrossEntropyLoss()

# Save the original data shape
original_shape = X_train.shape

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Store losses for each removed feature
feature_importances = {}

# Iterate through each feature
for feature_idx in range(9):
    # Create a copy of the training data
    exclude_indices = list(range(feature_idx, N * 9 + feature_idx, 9))
    X_train_copy = np.delete(X_train, exclude_indices, axis=1)
    
    # Convert the data back to tensor
    X_train_tensor = torch.tensor(X_train_copy, dtype=torch.float32)

    # Initialize a new model for each feature removal
    model = Workload()

    # Reinitialize optimizer for the new model
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model with the modified dataset
    print("training")
    final_loss = train_model(model, X_train_tensor, y_train, optimizer, criterion, epochs=1024 * 12)

    # Store the final loss associated with removing this feature
    feature_importances[feature_idx] = final_loss

    print(f"Feature {feature_idx} removed, final loss: {final_loss}")

# Print the feature importances
sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1])



print("Feature importance (based on final loss):")
for feature, loss in sorted_importances:
    print(f"Feature {feature}: Loss = {loss:.4f}")

# Save the final model
torch.save(model.state_dict(), TMP_DIR + "workload.pth")
