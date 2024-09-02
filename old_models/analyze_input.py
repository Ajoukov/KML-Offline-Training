import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Load data from CSV files
features = pd.read_csv('features.csv')
targets = pd.read_csv('targets.csv')

# Combine features and targets into one DataFrame for analysis
data = pd.concat([features, targets], axis=1)

# Summary Statistics
summary_stats = data.describe()
print("Summary Statistics:\n", summary_stats)

# Box Plots
plt.figure(figsize=(10, 6))
data.boxplot()
plt.title("Box Plot of Features and Targets")
plt.xticks(rotation=90)
plt.show()

# Histograms
data.hist(bins=30, figsize=(15, 10))
plt.suptitle("Histograms of Features and Targets")
plt.show()

# Z-Score to identify outliers
z_scores = stats.zscore(data)
abs_z_scores = abs(z_scores)
outliers_z = (abs_z_scores > 3).all(axis=1)
print("Outliers based on Z-Score:\n", data[outliers_z])

# IQR Method to identify outliers
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

outliers_iqr = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)
print("Outliers based on IQR:\n", data[outliers_iqr])

# Scatter Plots for visualization (using first two features for simplicity)
if len(features.columns) > 1:
    plt.figure(figsize=(10, 6))
    plt.scatter(data[features.columns[0]], data[features.columns[1]], c='blue', label='Data Points')
    plt.xlabel(features.columns[0])
    plt.ylabel(features.columns[1])
    plt.title(f"Scatter Plot of {features.columns[0]} vs {features.columns[1]}")
    plt.legend()
    plt.show()
else:
    print("Not enough features for scatter plot visualization.")

# Save summary statistics and outlier information to CSV
summary_stats.to_csv('summary_statistics.csv')
data[outliers_z].to_csv('outliers_z_score.csv')
data[outliers_iqr].to_csv('outliers_iqr.csv')
