import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define the neural network model
class LightNeuralNetwork(nn.Module):
    def __init__(self):
        super(LightNeuralNetwork, self).__init__()
        # self.fc1 = nn.Linear(31, 256)  # Adjusted input layer to have 29 neurons as per the diagram
        # self.fc2 = nn.Linear(256, 2)   # Output layer with 2 neurons for binary classification
        self.fc1 = nn.Linear(10, 256)  # Adjusted input layer to have 29 neurons as per the diagram
        self.fc2 = nn.Linear(256, 2)   # Output layer with 2 neurons for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model
model = LightNeuralNetwork().to(device)
model.load_state_dict(torch.load("light_neural_network.pth"))

def layers_to_csv(layer, num):
    w_np = layer.cpu().state_dict()['weight'].numpy()
    b_np = layer.cpu().state_dict()['bias'].numpy()
    df = pd.DataFrame(w_np) #convert to a dataframe
    df.to_csv(index=False, header=False, sep=" ", path_or_buf=f"torch_model_new/linear{num}_w.csv", float_format="%015.6f") #save to file
    df = pd.DataFrame(b_np) #convert to a dataframe
    df.to_csv(index=False, header=False, sep=" ", path_or_buf=f"torch_model_new/linear{num}_b.csv", float_format="%015.6f")

layers_to_csv(model.fc1, 0)
layers_to_csv(model.fc2, 1)


model.eval()
print("Model loaded from light_neural_network.pth")

# Load data
data = pd.read_csv("io_latency.csv", header=None)
data.columns = ['sector', 'size', 'op', 'latency_us', 'pending_ios']

data = data.head(1000)

# Calculate sector differences
# data['sector_diff'] = data['sector'].diff().fillna(0)

# Helper function to split a number into its individual digits and cap at a max value
def split_digits(numbers, length=4):
    max_value = 10**length - 1
    capped_numbers = np.minimum(numbers, max_value).astype(int)
    return numbers
    str_nums = pd.Series(capped_numbers.astype(str)).str.zfill(length)
    return np.array([list(map(int, num)) for num in str_nums])

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
print(features[0])

# Determine the 95th percentile latency value
# latency_threshold = data['latency_us'].quantile(0.95)
latency_threshold = 9970
print(f"Threshold: {latency_threshold}")

# Create binary target
targets = (data['latency_us'].shift(-N) > latency_threshold).astype(int)[N:].values

# Standardize the data
# scaler = StandardScaler()
# features = scaler.fit_transform(features)

# Convert to torch tensors
X = torch.tensor(features, dtype=torch.float32).to(device)
y = torch.tensor(targets, dtype=torch.long).to(device)

# Get predictions
with torch.no_grad():
    outputs = model(X)
    _, predicted_classes = torch.max(outputs, 1)

# Convert predictions to binary (0/1) values for easy interpretation
predicted_classes = predicted_classes.cpu().numpy()
true_classes = y.cpu().numpy()

# Save predictions to a CSV file
pd.DataFrame(features.astype(float)).to_csv("norm_input.csv", index=False, header=False, sep=" ", float_format="%015.6f")
pd.DataFrame(predicted_classes.astype(float)).to_csv("predictions.csv", index=False, header=False, sep=" ", float_format="%015.6f")
pd.DataFrame(true_classes.astype(float)).to_csv("actual.csv", index=False, header=False, sep=" ", float_format="%015.6f")

print("Predictions saved to predictions.csv")
print("Actual values saved to actual.csv")

# Plot confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
cmd = ConfusionMatrixDisplay(conf_matrix, display_labels=["Below 95th", "Above 95th"])
cmd.plot()
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png')
plt.show()

# Plotting true vs predicted values for debugging purposes
plt.figure(figsize=(10, 6))
plt.scatter(range(len(true_classes)), true_classes, alpha=0.5, label='True')
plt.scatter(range(len(predicted_classes)), predicted_classes, alpha=0.5, label='Predicted')
plt.xlabel('Samples')
plt.ylabel('Class')
plt.title('True vs Predicted Classes')
plt.legend()
plt.grid(True)
plt.savefig('true_vs_predicted_classes.png')
plt.show()
