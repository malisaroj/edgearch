from pandas import json_normalize
import ast
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import os
from sklearn.preprocessing import LabelEncoder
import ast
import numpy as np

# Load dataset 
df = pd.read_csv("datasets/borg_traces_data.csv")

columns_to_normalize = ['resource_request', 'average_usage', 'maximum_usage', 'random_sample_usage']

# Iterate over each column and apply normalization
for column in columns_to_normalize:
    df_normalized = json_normalize(df[column].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else None))
    
    # Concatenate the normalized DataFrame with the original DataFrame
    df = pd.concat([df, df_normalized.add_prefix(f"{column}_")], axis=1)

    # Drop the original column
    df.drop(column, axis=1, inplace=True)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

column_to_encode = ['user', 'priority']

# Fit and transform the column with label encoding
# Iterate over each column to encode
for column in column_to_encode:
    # Fit and transform the column with label encoding
    df[column] = label_encoder.fit_transform(df[column])

columns_to_drop = ['instance_events_type', 'scheduling_class',
                   'constraint', 'collections_events_type', 'user', 'collection_name', 
                   'collection_logical_name', 'start_after_collection_ids', 'vertical_scaling', 
                   'scheduler',
                   'cluster', 'event', 'failed', 'random_sample_usage_memory', 'collection_id',
                   'alloc_collection_id', 'collection_type', 
                   'instance_index', 'machine_id']


# Drop the specified columns
df.drop(columns=columns_to_drop, inplace=True)
df = df.drop(df.columns[df.columns.str.contains('Unnamed', case=False, regex=True)][0], axis=1)

# Adding more feature engineering as needed based on  data
# Cross-feature interactions
df['interaction_feature'] = df['maximum_usage_cpus'] * df['random_sample_usage_cpus']

# Creating lag features for memory_demand
df['memory_demand_lag_1'] = df['resource_request_memory'].shift(1)

# Creating rolling window statistics for memory_demand
df['memory_demand_rolling_mean'] = df['resource_request_memory'].rolling(window=3).mean()
df['memory_demand_rolling_std'] = df['resource_request_memory'].rolling(window=3).std()

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['maximum_usage_cpus', 'random_sample_usage_cpus']])
poly_feature_names = [f"poly_{name}" for name in poly.get_feature_names_out(['maximum_usage_cpus', 'random_sample_usage_cpus'])]
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
df = pd.concat([df, poly_df], axis=1)

# Convert timestamp columns to datetime objects
df['start_time'] = pd.to_datetime(df['start_time'], unit='ms')
df['end_time'] = pd.to_datetime(df['end_time'], unit='ms')

# Extract relevant features from timestamps
df['start_hour'] = df['start_time'].dt.hour
df['start_dayofweek'] = df['start_time'].dt.dayofweek

# Calculate the duration between start_time and end_time
df['duration_seconds'] = (df['end_time'] - df['start_time']).dt.total_seconds()


# Check for empty values
empty_values = df.isnull().sum()
print("Empty Values:\n", empty_values)

# Check for zero values
zero_values = (df == 0).sum()
print("\nZero Values:\n", zero_values)

# Fill NaN values with the mean of each column (excluding datetime columns)
df.fillna(df.mean(numeric_only=True), inplace=True)

# check again for empty or zero values
updated_empty_values = df.isnull().sum()
updated_zero_values = (df == 0).sum()

print("\nUpdated Empty Values:\n", updated_empty_values)
print("\nUpdated Zero Values:\n", updated_zero_values)

correlation_matrix = df.corr()
print(correlation_matrix)

# Specify the file name
file_name = "preprocessed_data.csv"

# Check if the file exists
if os.path.exists(file_name):
    # If it exists, replace it
    os.remove(file_name)
# Save the final DataFrame to a CSV file
df.to_csv(file_name, index=False)
# Print the first few rows of the DataFrame
print(df.head())
