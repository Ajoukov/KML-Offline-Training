import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Define the neural network model
class LightNeuralNetwork(nn.Module):
    def __init__(self):
        super(LightNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 256)  # Adjusted input layer to have 29 neurons as per the diagram
        self.fc2 = nn.Linear(256, 2)   # Output layer with 2 neurons for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
data = pd.read_csv("io_latency.csv", header=None)
data.columns = ['sector', 'size', 'op', 'latency_us', 'pending_ios']

# data = data.head(10000)

# Helper function to split a number into its individual digits and cap at a max value
def split_digits(numbers, length=4):
    max_value = 10**length - 1
    capped_numbers = np.minimum(numbers, max_value).astype(int)
    return numbers
    str_nums = pd.Series(capped_numbers.astype(str)).str.zfill(length)
    # return np.array([list(map(int, num)) for num in str_nums])

# Extract and preprocess queue lengths and latencies
pending_ios_digits = split_digits(data['pending_ios'].values, length=3)
latency_us_digits = split_digits(data['latency_us'].values, length=4)

N = 5

# Prepare feature matrix
features = []
for i in range(N, len(data)):
    current_features = pending_ios_digits[i-N:i].flatten()  # PQL_ij for i in range(N)
    lat_features = latency_us_digits[i-N:i].flatten()[:16]  # L_ij for i in range(N)
    features.append(np.concatenate([current_features, lat_features]))

features = np.array(features)

# Print the entire first line of features
print("First line of features (expanded):")
print(features[0:2])

# Determine the 95th percentile latency value
latency_threshold = data['latency_us'].quantile(0.95)
print(f"Threshold: {latency_threshold}")

# Create binary target
targets = (data['latency_us'].shift(-N) > latency_threshold).astype(int)[N:].values

print(targets[0:2])

print(targets)

# Standardize the data
# scaler = StandardScaler()
# features = scaler.fit_transform(features)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# Compute class weights
class_counts = np.bincount(y_train.cpu().numpy())
class_weights = 1.0 / class_counts
weights = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(device)

# Create model instance and move it to the device
model = LightNeuralNetwork().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % (epochs // 100) == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    _, predicted_classes = torch.max(predictions, 1)
    accuracy = (predicted_classes == y_test).sum().item() / y_test.size(0)
    print(f'Accuracy: {accuracy:.4f}')

# Save the model
torch.save(model.state_dict(), "light_neural_network.pth")
print("Model saved to light_neural_network.pth")
