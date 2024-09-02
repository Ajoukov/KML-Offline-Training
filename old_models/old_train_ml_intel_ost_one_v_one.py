import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.utils import resample

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        return x

# Function to upsample data
def upsample(data, target_col, count):
    upsampled_data = pd.DataFrame()
    for target in data[target_col].unique():
        upsampled_data = pd.concat([upsampled_data, 
                                    resample(data[data[target_col] == target],
                                             replace=True,  # sample with replacement
                                             n_samples=count,
                                             random_state=42)])
    return upsampled_data

# Load and preprocess the data
data = pd.read_csv("io_latency.csv", header=None)
data.columns = ['sector', 'size', 'op', 'tag', 'latency']

# Create latency buckets
data['latency_bucket'] = pd.qcut(data['latency'], 10, labels=False)

# Upsample data to make each latency bucket represented equally
upsampled_data = upsample(data, 'latency_bucket', data['latency_bucket'].value_counts().max())

# Generate pairwise comparisons
def generate_pairwise_comparisons(data, limit=100000):
    comparisons = []
    indices = np.random.choice(data.index, size=(limit, 2), replace=True)
    for i, j in indices:
        row_i = data.iloc[i]
        row_j = data.iloc[j]
        comparison = {
            'size_1': row_i['size'],
            'op_1': row_i['op'],
            'tag_1': row_i['tag'],
            'size_2': row_j['size'],
            'op_2': row_j['op'],
            'tag_2': row_j['tag'],
            'target': int(row_i['latency'] <= row_j['latency'])
        }
        comparisons.append(comparison)
    return pd.DataFrame(comparisons)

pairwise_data = generate_pairwise_comparisons(upsampled_data)

features = pairwise_data[['size_1', 'op_1', 'tag_1', 'size_2', 'op_2', 'tag_2']]
targets = pairwise_data['target']

features.to_csv('features.csv', index=False)
targets.to_csv('targets.csv', index=False)

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
y_train = torch.tensor(np.array(y_train.values), dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(np.array(y_test.values), dtype=torch.float32).view(-1, 1)

model = NeuralNetwork()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % (epochs // 10) == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), "compare_neural_network.pth")
print("Model saved to compare_neural_network.pth")
