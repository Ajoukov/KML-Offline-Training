import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from __global_paths import *

print(f"Using N={N}, Layers=({int(N*6)}, {int(M)}, {int(M/2)}, {int(M/4)}, {1})")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(int(N*6), int(M))
        self.fc2 = nn.Linear(int(M), int(M/2))
        self.fc3 = nn.Linear(int(M/2), int(M/4))
        self.fc4 = nn.Linear(int(M/4), 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


print("loading files")

with open(TMP_DIR + 'X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open(TMP_DIR + 'X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open(TMP_DIR + 'y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open(TMP_DIR + 'y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
    
print("loaded files")

# Load data from CSV files
# features = pd.read_csv('features_latency_use_N.csv')
# targets = pd.read_csv('targets_latency_use_N.csv')

model = NeuralNetwork()

from torch.optim.lr_scheduler import StepLR

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = StepLR(optimizer, step_size=100, gamma=0.90)

# Training the model
epochs = 1024 * 5
for epoch in range(1, epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    # scheduler.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    # current_lr = scheduler.get_last_lr()[0]
    # print(f'Epoch {epoch}/{epochs}, Current Learning Rate: {current_lr}')

# Save the model
torch.save(model.state_dict(), TMP_DIR + "improved_neural_network_latency_use_N.pth")
print(f"Model saved to {TMP_DIR}improved_neural_network_latency_use_N.pth")
