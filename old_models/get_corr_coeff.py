import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the data from the CSV files
# predictions = pd.read_csv('predictions.csv', header=None, names=['predicted'])
# actual = pd.read_csv('actual.csv', header=None, names=['actual'])

# data = predictions
# data['actual'] = actual


read_data = pd.read_csv('latency_predictions.csv')
read_data.columns = ['actual', 'predicted', "kml"]

data = pd.DataFrame()

# Convert columns to float
data['actual'] = read_data['kml'].astype(float)
data['predicted'] = read_data['predicted'].astype(float)

print(data)

# Compute the correlation coefficient
correlation = np.corrcoef(data['actual'], data['predicted'])[0, 1]
print(f'Correlation between actual and predicted values: {correlation:.6f}')

# Compute RMSE
rmse = np.sqrt(mean_squared_error(data['actual'], data['predicted']))
print(f'Root Mean Squared Error (RMSE): {rmse:.6f}')

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['actual'], data['predicted'], alpha=0.5)
plt.xlabel('Actual Latency')
plt.ylabel('Predicted Latency')
plt.title('Actual vs Predicted Latency')
plt.plot([data['actual'].min(), data['actual'].max()], [data['actual'].min(), data['actual'].max()], color='red')  # Line of perfect correlation
plt.grid(True)
plt.savefig('actual_vs_predicted_latency.png')
plt.show()

# Plot residuals
residuals = data['actual'] - data['predicted']
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, alpha=0.75)
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.grid(True)
plt.savefig('residuals_histogram.png')
plt.show()
