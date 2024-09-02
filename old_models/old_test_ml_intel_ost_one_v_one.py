import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Load features and targets
features = pd.read_csv('features.csv')
targets = pd.read_csv('targets.csv')

# Convert to PyTorch tensors
X = torch.tensor(np.array(features), dtype=torch.float32)
y = torch.tensor(np.array(targets.values), dtype=torch.float32).view(-1, 1)

# Load the trained model
model = NeuralNetwork()
model.load_state_dict(torch.load("compare_neural_network.pth"))
model.eval()

# Make predictions
with torch.no_grad():
    outputs = model(X)
    predictions = (outputs > 0.5).float()

# Generate statistics
accuracy = accuracy_score(y, predictions)
conf_matrix = confusion_matrix(y, predictions)
class_report = classification_report(y, predictions, target_names=['<=', '>'])

# Print statistics
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Save statistics to file
with open('model_evaluation.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix))
    f.write("\nClassification Report:\n")
    f.write(class_report)

data = pd.DataFrame(features, columns=['size', 'op', 'tag'])
data['latency'] = pd.read_csv('io_latency.csv', header=None)[4]

def compare(row1, row2):
    input_features = np.array([
        row1['size'], row1['op'], row1['tag'],
        row2['size'], row2['op'], row2['tag']
    ]).reshape(1, -1)
    # input_features = scaler.transform(input_features)
    input_tensor = torch.tensor(input_features, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    return output.item() > 0.5

# Perform 100,000 insertions
ordered_data = []

for i in range(35000):
    new_row = data.sample().iloc[0]
    if not ordered_data:
        ordered_data.append(new_row)
    else:
        low, high = 0, len(ordered_data)
        while low < high:
            mid = (low + high) // 2
            if compare(new_row, ordered_data[mid]):
                low = mid + 1
            else:
                high = mid
        ordered_data.insert(low, new_row)
    if (i % 100 == 0):
        print(i)

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