import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from __global_paths import *

with open(TMP_DIR + 'X_train_latency.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open(TMP_DIR + 'X_test_latency.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open(TMP_DIR + 'y_train_latency.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open(TMP_DIR + 'y_test_latency.pkl', 'rb') as f:
    y_test = pickle.load(f)

# Load data from CSV files
# features = pd.read_csv('features_latency_use_N.csv')
# targets = pd.read_csv('targets_latency_use_N.csv')


model = NeuralNetwork()

from torch.optim.lr_scheduler import StepLR

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=100, gamma=0.90)

# Training the model
for epoch in range(1, EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    # scheduler.step()

    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')
    # current_lr = scheduler.get_last_lr()[0]
    # print(f'Epoch {epoch}/{EPOCHS}, Current Learning Rate: {current_lr}')

torch.save(model.state_dict(), TMP_DIR + "latency.pth")
