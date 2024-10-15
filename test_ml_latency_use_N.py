import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

from __global_paths import *

with open(TMP_DIR + 'X_test_latency.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open(TMP_DIR + 'y_test_latency.pkl', 'rb') as f:
    y_test = pickle.load(f)

model = NeuralNetwork()
model.load_state_dict(torch.load(TMP_DIR + "latency.pth"))

model.eval()
with torch.no_grad():
    predictions = model(X_test)

predictions = predictions.numpy()
# predictions = np.clip(predictions, LATENCY_MIN, LATENCY_MAX)
np.savetxt(OUT_DIR + "predictions_simulate.csv", predictions, delimiter=" ", fmt="%015.6f")
y_test_np = y_test.numpy()
np.savetxt(OUT_DIR + "latencies_simulate.csv", y_test_np, delimiter=" ", fmt="%015.6f")

mse = np.mean((predictions - y_test_np) ** 2)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_np, predictions)
r, _ = pearsonr(y_test_np.flatten(), predictions.flatten())
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'R: {r:.4f}')

model = NeuralNetwork()
model.load_state_dict(torch.load(TMP_DIR + "latency.pth"))
save_predictions(model, X_test, y_test)

import matplotlib.pyplot as plt

def layers_to_csv(layer, num):
    w_np = layer.cpu().state_dict()['weight'].numpy()
    b_np = layer.cpu().state_dict()['bias'].numpy()
    df = pd.DataFrame(w_np) #convert to a dataframe
    df.to_csv(index=False, header=False, sep=" ", path_or_buf=WTS_DIR + f"linear{num}_w.csv", float_format="%015.6f") #save to file
    df = pd.DataFrame(b_np) #convert to a dataframe
    df.to_csv(index=False, header=False, sep=" ", path_or_buf=WTS_DIR + f"linear{num}_b.csv", float_format="%015.6f")

layers_to_csv(model.start, 0)
print(0)
for i in range(0, HIDDEN_LAYERS):
    print(i+1)
    layers_to_csv(model.weights[i], i+1)
print(HIDDEN_LAYERS + 1)
layers_to_csv(model.end, HIDDEN_LAYERS + 1)

def cap_negative(value):
    if value < -9999999.999999:
        return -9999999.999999
    return value

X_test_save = pd.DataFrame(X_test.numpy().astype(float)).map(cap_negative)
predictions_save = pd.DataFrame(predictions.astype(float)).map(cap_negative)
y_test_np_save = pd.DataFrame(y_test_np.astype(float)).map(cap_negative)

X_test_save.to_csv(TST_DIR + "norm_input.csv", index=False, header=False, sep=" ", float_format="%015.6f")
predictions_save.to_csv(TST_DIR + "predictions.csv", index=False, header=False, sep=" ", float_format="%015.6f")
y_test_np_save.to_csv(TST_DIR + "actual.csv", index=False, header=False, sep=" ", float_format="%015.6f")

print(model(torch.tensor(np.array([    
00000032.000000,00000000.000000,00000256.000000,00000000.000000,00000001.000000,00000000.000000,00000000.000000,00000001.000000,00000001.000000,00000032.000000,00000000.000000,00000256.000000,00000000.000000,00000001.000000,00000000.000000,00000000.000000,00000001.000000,00000001.000000,00000032.000000,00000000.000000,00000256.000000,00000000.000000,00000001.000000,00000000.000000,00000000.000000,00000001.000000,00000001.000000,00000032.000000,00000000.000000,00000256.000000,00000000.000000,00000001.000000,00000000.000000,00000000.000000,00000001.000000,00000001.000000,00000032.000000,00000000.000000,00000256.000000,00000000.000000,00000001.000000,00000000.000000,00000000.000000,00000001.000000,00000001.000000,00000032.000000,00000000.000000,00000256.000000,00000000.000000,00000001.000000,00000000.000000,00000000.000000,00000001.000000,00000001.000000,00000032.000000,00000000.000000,00000256.000000,00000000.000000,00000001.000000,00000000.000000,00000000.000000,00000001.000000,00000001.000000,00000032.000000,00000000.000000,00000256.000000,00000000.000000,00000001.000000,00000000.000000,00000000.000000,00000001.000000,00000001.000000,00000032.000000,00000000.000000,00000256.000000,00000000.000000,00000001.000000,00000000.000000,00000000.000000,00000001.000000,00000001.000000,00000032.000000,00000000.000000,00000256.000000,00000000.000000,00000001.000000,00000000.000000,00000000.000000,00000001.000000,00000001.000000,00000032.000000,00000000.000000,00000256.000000,00000000.000000,00000001.000000,00000000.000000,00000000.000000,00000001.000000,00000001.000000,00000032.000000,00000000.000000,00000256.000000,00000000.000000,00000001.000000,00000000.000000,00000000.000000,00000001.000000,00000001.000000,00000032.000000,00000000.000000,00000256.000000,00000000.000000,00000001.000000,00000000.000000,00000000.000000,00000001.000000,00000001.000000,00000032.000000,00000000.000000,00000256.000000,00000000.000000,00000001.000000,00000000.000000,00000000.000000,00000001.000000,00000001.000000,00000032.000000,00000000.000000,00000256.000000,00000000.000000,00000001.000000,00000000.000000,00000000.000000,00000001.000000,00000001.000000,00000032.000000,00000000.000000,00000256.000000,00000000.000000,00000001.000000,00000000.000000,00000000.000000,00000001.000000,00000001.000000
    ]), dtype=torch.float32)))
