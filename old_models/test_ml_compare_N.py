import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats

N = 16

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(N * 8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        # x = torch.relu(self.fc5(x))
        x = self.fc5(x)
        return x

# Load data from CSV files
features = pd.read_csv('features.csv')
targets = pd.read_csv('targets.csv')

# Combine features and targets into one DataFrame for analysis
data = pd.concat([features, targets], axis=1)

# Summary Statistics
# summary_stats = data.describe()
# print("Summary Statistics:\n", summary_stats)

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
    # predictions = np.clip(predictions, a_min=0, a_max=N-1)
    y_test_np = y_test.numpy()
    mse = np.mean((predictions - y_test_np) ** 2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_np, predictions)
    r2 = r2_score(y_test_np, predictions)
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R²: {r2:.4f}')

# Create side-by-side box plots for each true value with 10th and 90th percentiles
true_values = sorted(targets['bigger_sq'].unique())
plt.figure(figsize=(15, 10))

predictions_list = []
labels = []
for true_value in true_values:
    mask = (y_test_np == true_value).flatten()
    if np.sum(mask) > 0:
        predictions_list.append(predictions[mask].flatten())
        labels.append(f'True {true_value}')

# Customizing the box plot to show the 10th and 90th percentiles as whiskers
box = plt.boxplot(predictions_list, showfliers=False, labels=labels, patch_artist=True, whis=[5, 95])
colors = ['#FF9999', '#FFCC99', '#FFFF99', '#99FF99', '#99CCFF', '#CC99FF']
cmap = LinearSegmentedColormap.from_list('custom_gradient', colors, N=N-1)
colors = [cmap(i / (N - 1)) for i in range(N)]

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.xlabel('True Value')
plt.ylabel('Predicted Values')
plt.title('Predicted vs True Values (with 5th and 95th Percentiles)')
# plt.ylim(0, N-1)
plt.grid(True)
plt.savefig('box_plots_predictions_with_percentiles.png')
plt.show()

# Save summary statistics and outlier information to CSV
# summary_stats.to_csv('summary_statistics.csv')
# data[outliers_z].to_csv('outliers_z_score.csv')
# data[outliers_iqr].to_csv('outliers_iqr.csv')
