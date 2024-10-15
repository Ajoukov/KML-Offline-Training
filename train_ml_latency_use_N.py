import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from __global_paths import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

with open(TMP_DIR + 'X_train_latency.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open(TMP_DIR + 'X_test_latency.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open(TMP_DIR + 'y_train_latency.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open(TMP_DIR + 'y_test_latency.pkl', 'rb') as f:
    y_test = pickle.load(f)
    

X_train = X_train.clone().detach().float().to(device)
X_test = X_test.clone().detach().float().to(device)
y_train = y_train.clone().detach().float().to(device)
y_test = y_test.clone().detach().float().to(device)

model = NeuralNetwork().to(device)
print(next(model.parameters()).device)  # Check the device after the model is moved

if USE_BUCKETS:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=LR)

save_predictions(model, X_test, y_test)

min_loss = 1000000
for epoch in range(1, EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    loss.backward()
    optimizer.step()

    if (epoch % 20 == 0):
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')
    
    if (epoch % int(EPOCHS / 50) == 0):
        save_predictions(model, X_test, y_test)
        model.train()
    
    if loss.item() < min_loss * 0.98 and epoch > EPOCHS / 3:
        torch.save(model.state_dict(), f"{TMP_DIR}latency_{int(loss.item() * 10)}.pth")
        print(f"Saving to {TMP_DIR}latency_{int(loss.item() * 10)}.pth")
        min_loss = loss.item()

torch.save(model.state_dict(), TMP_DIR + "latency.pth")