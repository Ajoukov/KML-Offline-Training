import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define latency buckets with finer granularity (powers of 2^0.5)
def latency_to_bucket(latency):
    return min(int(np.log2(latency) * 2), 30)  # 30 is the max bucket index for latencies up to 32768 us with finer granularity

# Load data
data = pd.read_csv("io_latency.csv")
data.columns = ['sector', 'size', 'op', 'latency_us', 'pending_ios']

# Print basic statistics
print("Basic Statistics of Latency Values:")
print(data['latency_us'].describe())

# Convert latencies to buckets
data['latency_bucket'] = data['latency_us'].apply(latency_to_bucket)

# Print unique buckets and their counts
print("Latency Buckets and Counts:")
print(data['latency_bucket'].value_counts().sort_index())

# Plot raw latency distribution
plt.figure(figsize=(12, 6))
sns.histplot(data['latency_us'], bins=100, kde=True)  # Increased number of bins for more granularity
plt.xlim(0, 15000)
plt.title('Raw Latency Distribution')
plt.xlabel('Latency (us)')
plt.ylabel('Frequency')
plt.savefig('raw_latency_distribution.png')

# Ensure all buckets from 0 to 30 are included in the x-axis
all_buckets = pd.Series(range(31))

# Plot bucketed latency distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=all_buckets, y=data['latency_bucket'].value_counts().reindex(all_buckets, fill_value=0))
plt.title('Bucketed Latency Distribution')
plt.xlabel('Latency Bucket (log2 scale, with finer granularity)')
plt.ylabel('Frequency')
plt.savefig('bucketed_latency_distribution.png')
