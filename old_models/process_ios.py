import pandas as pd

# Load the CSV file
file_path = 'io_latency.csv'  # Replace with your file path
data = pd.read_csv(file_path, header=None)

# Function to adjust the counter using a suffix array
def adjust_counter_with_suffix(data):
    n = len(data)
    suffix_min = [0] * n
    suffix_min[-1] = data.iloc[-1, 4]
    
    # Fill the suffix array with minimum values from the end to the beginning
    for i in range(n-2, -1, -1):
        suffix_min[i] = min(data.iloc[i, 4], suffix_min[i + 1])
    
    # Adjust the values in the fifth column
    adjusted_values = [data.iloc[i, 4] - suffix_min[i] for i in range(n)]
    data[4] = adjusted_values
    return data

adjusted_data = adjust_counter_with_suffix(data)

# Save the adjusted data to a new CSV file
adjusted_file_path = 'adjusted_latency_ios.csv'  # Replace with your desired save path
adjusted_data.to_csv(adjusted_file_path, index=False, header=False)

print(f"Adjusted data saved to {adjusted_file_path}")
