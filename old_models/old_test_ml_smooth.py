import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

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

features = pd.read_csv('features.csv')
targets = pd.read_csv('targets.csv')

# Normalize the features
# scaler = StandardScaler()
# features = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
y_train = torch.tensor(np.array(y_train.values), dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(np.array(y_test.values), dtype=torch.float32).view(-1, 1)

model = NeuralNetwork()
model.load_state_dict(torch.load("improved_neural_network.pth"))

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predictions = predictions.numpy()
    predictions = np.clip(predictions, a_min=-0.5, a_max=6.5)
    predictions = np.clip(predictions, a_min=0, a_max=6)
    # predictions = np.where((predictions < 1.2) | (predictions > 6), 3, predictions)
    y_test_np = y_test.numpy()
    mse = np.mean((predictions - y_test_np) ** 2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_np, predictions)
    r2 = r2_score(y_test_np, predictions)
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R²: {r2:.4f}')

# Plot true vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_np, predictions, alpha=0.01)
plt.plot([min(y_test_np), max(y_test_np)], [min(y_test_np), max(y_test_np)], color='red')  # Line y=x
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.savefig('true_vs_predicted.png')
plt.show()
