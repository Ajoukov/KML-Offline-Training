from __global_paths import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# # Define latency buckets with finer granularity (powers of 2^0.5)
# def latency_to_bucket(latency):
#     return min(int(np.log2(latency) * 2), 30)  # 30 is the max bucket index for latencies up to 32768 us with finer granularity

# # data = pd.read_csv(INP_DIR + "input.csv")
# data = pd.read_csv("input_data/inputs.csv")
# # data.columns = ['sector', 'size', 'op', 'latency', 'pending_ios']

# cols_features = []

# for i in range(0, N):
#     cols_features.append(f'size_prev_{i}')
#     cols_features.append(f'op_prev_{i}')
#     cols_features.append(f'lba_diff_prev_{i}')
#     cols_features.append(f'tag0_prev_{i}')
#     cols_features.append(f'tag1_prev_{i}')
#     cols_features.append(f'tag2_prev_{i}')
#     cols_features.append(f'tag3_prev_{i}')
#     cols_features.append(f'tag4_prev_{i}')

# # cols.append('latency')

# data.columns = cols_features + ['latency']
# print(data['latency'].describe())

# # Convert latencies to buckets
# data['latency_bucket'] = data['latency'].apply(latency_to_bucket)

# # Print unique buckets and their counts
# print("Latency Buckets and Counts:")
# print(data['latency_bucket'].value_counts().sort_index())

# # Plot raw latency distribution
# plt.figure(figsize=(12, 6))
# sns.histplot(data['latency'], bins=100, kde=True)  # Increased number of bins for more granularity
# # plt.xlim(0, 15000)
# plt.title('Raw Latency Distribution')
# plt.xlabel('Latency (us)')
# plt.ylabel('Frequency')
# plt.savefig('output_data/raw_latency_distribution.png')

# # Ensure all buckets from 0 to 30 are included in the x-axis
# all_buckets = pd.Series(range(31))

# # Plot bucketed latency distribution
# plt.figure(figsize=(12, 6))
# sns.barplot(x=all_buckets, y=data['latency_bucket'].value_counts().reindex(all_buckets, fill_value=0))
# plt.title('Bucketed Latency Distribution')
# plt.xlabel('Latency Bucket (log2 scale, with finer granularity)')
# plt.ylabel('Frequency')
# plt.savefig('output_data/bucketed_latency_distribution.png')

with open(TMP_DIR + 'X_train_latency.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open(TMP_DIR + 'X_test_latency.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open(TMP_DIR + 'y_train_latency.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open(TMP_DIR + 'y_test_latency.pkl', 'rb') as f:
    y_test = pickle.load(f)

plt.figure(figsize=(12, 6))
sns.histplot(y_train, bins=100, kde=True)
plt.title('Raw z-score Distribution')
plt.xlabel('z-score')
plt.ylabel('Frequency')
plt.savefig('output_data/z-score_distr.png')