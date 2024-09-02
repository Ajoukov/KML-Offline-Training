import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample

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

# Load and preprocess the data
data = pd.read_csv("io_latency.csv", header=None)
data.columns = ['sector', 'size', 'op', 'tag', 'latency']

data['size_prev4'] = data['size'].shift(4)
data['op_prev4'] = data['op'].shift(4)
data['tag_prev4'] = data['tag'].shift(4)
data['size_prev3'] = data['size'].shift(3)
data['op_prev3'] = data['op'].shift(3)
data['tag_prev3'] = data['tag'].shift(3)
data['size_prev2'] = data['size'].shift(2)
data['op_prev2'] = data['op'].shift(2)
data['tag_prev2'] = data['tag'].shift(2)
data['size_prev1'] = data['size'].shift(1)
data['op_prev1'] = data['op'].shift(1)
data['tag_prev1'] = data['tag'].shift(1)
data['size_next1'] = data['size'].shift(-1)
data['op_next1'] = data['op'].shift(-1)
data['tag_next1'] = data['tag'].shift(-1)
data['size_next2'] = data['size'].shift(-2)
data['op_next2'] = data['op'].shift(-2)
data['tag_next2'] = data['tag'].shift(-2)

data['bigger1'] = data['latency'] > data['latency'].shift(-2)
data['bigger2'] = data['latency'] > data['latency'].shift(-1)
data['bigger3'] = data['latency'] > data['latency'].shift(1)
data['bigger4'] = data['latency'] > data['latency'].shift(2)
data['bigger5'] = data['latency'] > data['latency'].shift(3)
data['bigger6'] = data['latency'] > data['latency'].shift(4)


def upsample(data, target_col, count):
    upsampled_data = pd.DataFrame()
    for target in data[target_col].unique():
        upsampled_data = pd.concat([upsampled_data, 
                                    resample(data[data[target_col] == target],
                                             replace=True,  # sample with replacement
                                             n_samples=count,
                                             random_state=42)])
    return upsampled_data

biggers = ['bigger1', 'bigger2', 'bigger3', 'bigger4', 'bigger5', 'bigger6']
# data[biggers] = data[biggers].apply(pd.to_numeric, errors='coerce')
data['bigger'] = data[biggers].sum(axis=1)

cols = ['size',
'op',
'tag',
'size_prev4',
'op_prev4',
'tag_prev4',
'size_prev3',
'op_prev3',
'tag_prev3',
'size_prev2',
'op_prev2',
'tag_prev2',
'size_prev1',
'op_prev1',
'tag_prev1',
'size_next1',
'op_next1',
'tag_next1',
'size_next2',
'op_next2',
'tag_next2']

data = data.dropna()

data = upsample(data, 'bigger', 10000)

features = data[cols]
targets = data['bigger']

features.to_csv('features.csv', index=False)
targets.to_csv('targets.csv', index=False)

# Normalize the features
# scaler = StandardScaler()
# features = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
y_train = torch.tensor(np.array(y_train.values), dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(np.array(y_test.values), dtype=torch.float32).view(-1, 1)

model = NeuralNetwork()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Lowered learning rate

# Training the model
epochs = 10000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % (epochs // 100) == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), "improved_neural_network.pth")
print("Model saved to improved_neural_network.pth")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predictions = predictions.numpy()
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
plt.scatter(y_test_np, predictions, alpha=0.5)
plt.plot([min(y_test_np), max(y_test_np)], [min(y_test_np), max(y_test_np)], color='red')  # Line y=x
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.savefig('true_vs_predicted_improved.png')
plt.show()
