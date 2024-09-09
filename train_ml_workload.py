import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.optim.lr_scheduler import StepLR

from __global_paths import *

with open(TMP_DIR + 'X_train_wl.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open(TMP_DIR + 'X_test_wl.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open(TMP_DIR + 'y_train_wl.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open(TMP_DIR + 'y_test_wl.pkl', 'rb') as f:
    y_test = pickle.load(f)

model = Workload()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
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

# Save the model

# with open(TMP_DIR + 'workload.pkl', 'wb') as f:
#     pickle.dump(model, f)
torch.save(model.state_dict(), TMP_DIR + "workload.pth")

model2 = Workload()
model2.load_state_dict(torch.load(TMP_DIR + "workload.pth"))

model2.eval()
with torch.no_grad():
    predictions = model(X_test)
    predictions = predictions.numpy()