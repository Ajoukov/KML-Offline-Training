import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

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

# Box and Whisker Plots
# plt.figure(figsize=(10, 6))
# data.boxplot()
# plt.title("Box and Whisker Plot of Features and Targets")
# plt.xticks(rotation=90)
# plt.savefig('box_and_whisker_plot.png')
# plt.close()

# Histograms
# data.hist(bins=30, figsize=(15, 10))
# plt.suptitle("Histograms of Features and Targets")
# plt.savefig('histograms.png')
# plt.close()

# Z-Score to identify outliers
# z_scores = stats.zscore(data)
# abs_z_scores = abs(z_scores)
# outliers_z = (abs_z_scores > 3).all(axis=1)
# print("Outliers based on Z-Score:\n", data[outliers_z])

# IQR Method to identify outliers
# Q1 = data.quantile(0.25)
# Q3 = data.quantile(0.75)
# IQR = Q3 - Q1

# outliers_iqr = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)
# print("Outliers based on IQR:\n", data[outliers_iqr])

# X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
# X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
# M_train = torch.tensor(np.array(M_train), dtype=torch.float32)
# M_test = torch.tensor(np.array(M_test), dtype=torch.float32)
# y_train = torch.tensor(np.array(y_train.values), dtype=torch.float32).view(-1, 1)
# y_test = torch.tensor(np.array(y_test.values), dtype=torch.float32).view(-1, 1)

model = NeuralNetwork()
model.load_state_dict(torch.load(TMP_DIR + "improved_neural_network_latency_use_N.pth"))

# mean = M_test['mean'].values.reshape(-1, 1)
# std = M_test['std'].values.reshape(-1, 1)

# mask = ((y_test.numpy() * std) - mean) > -10

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predictions = predictions.numpy()
    
    # predictions_all = model(torch.tensor(np.array(features), dtype=torch.float32)).numpy()
    # np.savetxt("predictions_simulate.csv", predictions_all, delimiter=" ")
    
    # Convert predictions and y_test_np back from z-score
    # predictions = predictions * std + mean
    # predictions = np.clip(predictions, a_min=min(y_test.numpy())**2 * -1, a_max=max(y_test.numpy())**2)
    # predictions = np.clip(predictions, a_min=-10, a_max=10)
    np.savetxt(OUT_DIR + "predictions_simulate.csv", predictions, delimiter=" ", fmt="%015.6f")
    # predictions = 2.718 ** predictions
    # predictions = predictions[mask]
    
    y_test_np = y_test.numpy()
    # y_test_np = y_test_np * std + mean
    # y_test_np = 2.718 ** y_test_np
    # y_test_np = y_test_np[mask]
    
    # latencies_all = torch.tensor(np.array(targets.values), dtype=torch.float32).view(-1, 1).numpy()
    # np.savetxt("latencies_simulate.csv", predictions_all, delimiter=" ")
    np.savetxt(OUT_DIR + "latencies_simulate.csv", y_test_np, delimiter=" ", fmt="%015.6f")
    
    mse = np.mean((predictions - y_test_np) ** 2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_np, predictions)
    r, _ = pearsonr(y_test_np.flatten(), predictions.flatten())
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R: {r:.4f}')


# Create side-by-side box plots for each true value with 10th and 90th percentiles
# true_values = sorted(targets['bigger_sq'].unique())
plt.figure(figsize=(15, 10))
# Define the number of buckets and create bucket labels

B = 100
bucket_labels = range(B)

# Create buckets for the true targets
y_test_buckets = pd.cut(y_test_np.flatten(), bins=B, labels=bucket_labels)

# Modify the true values and labels section for bucketed targets
true_values = sorted(bucket_labels)
predictions_list = []
labels = []
for bucket in true_values:
    mask = (y_test_buckets == bucket)
    if np.sum(mask) > 0:
        predictions_list.append(predictions[mask].flatten())
        labels.append(f'Bucket {bucket}')

# Customizing the box plot to show the 10th and 90th percentiles as whiskers
box = plt.boxplot(predictions_list, showfliers=False, labels=labels, patch_artist=True, whis=[5, 95])
colors = ['#FF9999', '#FFCC99', '#FFFF99', '#99FF99', '#99CCFF', '#CC99FF']
cmap = LinearSegmentedColormap.from_list('custom_gradient', colors, N=B)
colors = [cmap(i / (B)) for i in range(B)]

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.xlabel('True Value')
plt.ylabel('Predicted Values')
plt.title('Predicted vs True Values (with 5th and 95th Percentiles)')
# plt.ylim(0, N-1)
plt.grid(True)
plt.savefig(OUT_DIR + 'box_plots_predictions_with_percentiles.png')
plt.show()

import matplotlib.pyplot as plt

# Assuming predictions and y_test_np have been converted back to true latency values as shown earlier

# Scatter plot of true vs predicted latencies
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test_np, predictions, alpha=0.6, edgecolors='w', linewidth=0.5)
# plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--', lw=2)  # Line of perfect prediction

# # Adding labels and title
# plt.xlabel('True Latencies')
# plt.ylabel('Predicted Latencies')
# plt.title('True vs Predicted Latencies')
# plt.grid(True)

# plt.savefig(OUT_DIR + 'true_vs_predicted_latencies.png')
# plt.show()

# Save summary statistics and outlier information to CSV
# summary_stats.to_csv('summary_statistics.csv')
# data[outliers_z].to_csv('outliers_z_score.csv')
# data[outliers_iqr].to_csv('outliers_iqr.csv')

def layers_to_csv(layer, num):
    w_np = layer.cpu().state_dict()['weight'].numpy()
    b_np = layer.cpu().state_dict()['bias'].numpy()
    df = pd.DataFrame(w_np) #convert to a dataframe
    df.to_csv(index=False, header=False, sep=" ", path_or_buf=WTS_DIR + f"linear{num}_w.csv", float_format="%015.6f") #save to file
    df = pd.DataFrame(b_np) #convert to a dataframe
    df.to_csv(index=False, header=False, sep=" ", path_or_buf=WTS_DIR + f"linear{num}_b.csv", float_format="%015.6f")

layers_to_csv(model.fc1, 0)
layers_to_csv(model.fc2, 1)
layers_to_csv(model.fc3, 2)
layers_to_csv(model.fc4, 3)

print(N * 6)

pd.DataFrame(X_test.numpy().astype(float)).to_csv(TST_DIR + "norm_input.csv", index=False, header=False, sep=" ", float_format="%015.6f")
pd.DataFrame(predictions.astype(float)).to_csv(TST_DIR + "predictions.csv", index=False, header=False, sep=" ", float_format="%015.6f")
pd.DataFrame(y_test_np.astype(float)).to_csv(TST_DIR + "actual.csv", index=False, header=False, sep=" ", float_format="%015.6f")