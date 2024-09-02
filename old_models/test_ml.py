import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(21, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

# Load features and targets
features = pd.read_csv('features.csv')
targets = pd.read_csv('targets.csv')

# Normalize the features
# scaler = StandardScaler()
# features = scaler.fit_transform(features)

# Load the trained model
model = NeuralNetwork()
model.load_state_dict(torch.load("improved_neural_network.pth"))
model.eval()

data = pd.read_csv("io_latency.csv", header=None)
data.columns = ['sector', 'size', 'op', 'tag', 'latency']

data = data.dropna()

# Function to use the ML model to determine where to place a new row
def predict_bigger_than(model, row, sample_rows):
    input_features = []
    input_features.append(row[['size', 'op', 'tag']].values)
    for sample in sample_rows:
        input_features.append(sample[['size', 'op', 'tag']].values)
    
    input_features = [item for sublist in input_features for item in sublist]
    
    input_tensor = torch.tensor(input_features, dtype=torch.float32)
#     print(np.shape(input_tensor))
    with torch.no_grad():
        output = model(input_tensor)
        # print(output)
    return output.mean().item() # Scale back to the range 0-6

ordered_data = []

def get_sample_rows(start, size):
    step = max(len(size) // 6, 1)
    sample_rows = [ordered_data[start + j] for j in range(0, size, step)]
    sample_rows = sample_rows[0:6]
    while(len(sample_rows) < 6):
        sample_rows.append(ordered_data[-1])
    return sample_rows

for index, new_row in data.iterrows():
    if not ordered_data:
        ordered_data.append(new_row)
    else:
        sample_rows = get_sample_rows(0, len(ordered_data))
        bigger_than = predict_bigger_than(model, new_row, sample_rows)
        insert_position = int(len(ordered_data) * (bigger_than / 6))
        insert_position = max(0, min(insert_position, len(ordered_data)))
        ordered_data.insert(insert_position, new_row)
    if (index % 100 == 0):
        print(index)

# Extract the orderings and true latencies
ordered_latencies = [row['latency'] for row in ordered_data]
true_latencies = data['latency']

# Plot the ordering of all the orderings vs true latencies
plt.figure(figsize=(10, 6))
plt.plot(ordered_latencies, label='Model Ordering')
plt.plot(sorted(true_latencies), label='True Latencies')
plt.xlabel('Index')
plt.ylabel('Latency')
plt.legend()
plt.title('Model Ordering vs True Latencies')

# Save the plot to a file
plt.savefig('model_ordering_vs_true_latencies.png')
print("Plot saved to 'model_ordering_vs_true_latencies.png'")