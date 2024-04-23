import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import os
import numpy as np
from keras.utils import pad_sequences

# Load the saved model
model_save_path = Path(".cache") / "trained_model.h5" 
model = tf.keras.models.load_model(model_save_path)

# Read the entire dataset
df = pd.read_csv("preprocessed_data.csv")

df['cpu_usage_distribution'] = df['cpu_usage_distribution'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
df['tail_cpu_usage_distribution'] = df['tail_cpu_usage_distribution'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

# Padding sequences
max_seq_length = df['cpu_usage_distribution'].apply(len).max()  # Find the maximum length of sequences
df['cpu_usage_distribution_padded'] = df['cpu_usage_distribution'].apply(lambda x: pad_sequences([x], maxlen=max_seq_length, padding='post', dtype='float32')[0])
tail_max_seq_length = df['tail_cpu_usage_distribution'].apply(len).max()  # Find the maximum length of sequences
df['tail_cpu_usage_distribution_padded'] = df['tail_cpu_usage_distribution'].apply(lambda x: pad_sequences([x], maxlen=tail_max_seq_length, padding='post', dtype='float32')[0])

# Extract the relevant features for prediction
# Feature Scaling

scaler = StandardScaler()
df['cpu_usage_distribution_scaled'] = df['cpu_usage_distribution_padded'].apply(lambda x: scaler.fit_transform(x.reshape(-1, 1)))
df['tail_cpu_usage_distribution_scaled'] = df['tail_cpu_usage_distribution_padded'].apply(lambda x: scaler.fit_transform(x.reshape(-1, 1)))

scaled_features = scaler.fit_transform(df[[ 'resource_request_cpus', 'resource_request_memory',  'poly_maximum_usage_cpus random_sample_usage_cpus', 
                                            'maximum_usage_cpus',  'poly_random_sample_usage_cpus', 'poly_random_sample_usage_cpus^2', 'memory_demand_rolling_mean',
                                            'maximum_usage_memory',  'interaction_feature', 'poly_maximum_usage_cpus^2', 'memory_demand_lag_1',
                                            'random_sample_usage_cpus', 'assigned_memory',  'poly_maximum_usage_cpus', 'memory_demand_rolling_std', 
                                            'start_hour', 'start_dayofweek', 'duration_seconds', 'sample_rate', 'cycles_per_instruction', 
                                            'memory_accesses_per_instruction', 'page_cache_memory', 'priority',
                                        ]])

# Labels
labels = df[['average_usage_cpus', 'average_usage_memory']]

# Convert numpy arrays to pandas DataFrames
scaled_features_df = pd.DataFrame(scaled_features, columns=[
    'resource_request_cpus', 'resource_request_memory', 'poly_maximum_usage_cpus random_sample_usage_cpus',
    'maximum_usage_cpus', 'poly_random_sample_usage_cpus', 'poly_random_sample_usage_cpus^2', 'memory_demand_rolling_mean',
    'maximum_usage_memory', 'interaction_feature', 'poly_maximum_usage_cpus^2', 'memory_demand_lag_1',
    'random_sample_usage_cpus', 'assigned_memory', 'poly_maximum_usage_cpus', 'memory_demand_rolling_std',
    'start_hour', 'start_dayofweek', 'duration_seconds', 'sample_rate', 'cycles_per_instruction',
    'memory_accesses_per_instruction', 'page_cache_memory', 'priority'])

# Confirm dimensions and data types before concatenating scaled features
print("Scaled Features Dimensions:")
print("  df['cpu_usage_distribution_scaled']: ", df['cpu_usage_distribution_scaled'].shape)
print("  df['tail_cpu_usage_distribution_scaled']: ", df['tail_cpu_usage_distribution_scaled'].shape)
print("  scaled_features_df: ", scaled_features_df.shape)

print("\nData Types:")
print("  df['cpu_usage_distribution_scaled'].dtype: ", df['cpu_usage_distribution_scaled'].dtype)
print("  df['tail_cpu_usage_distribution_scaled'].dtype: ", df['tail_cpu_usage_distribution_scaled'].dtype)
print("  scaled_features_df.dtypes: ", scaled_features_df.dtypes)

# Reshape the data to 2D array
cpu_usage_reshaped = np.vstack(df['cpu_usage_distribution_scaled']).reshape(-1, max_seq_length)

# Create DataFrame
cpu_usage_df = pd.DataFrame(cpu_usage_reshaped, columns=[f'cpu_usage_{i}' for i in range(max_seq_length)])

# Reshape the data to 2D array
tail_cpu_usage_reshaped = np.vstack(df['tail_cpu_usage_distribution_scaled']).reshape(-1, tail_max_seq_length)

# Create DataFrame
tail_cpu_usage_df = pd.DataFrame(tail_cpu_usage_reshaped, columns=[f'tail_cpu_usage_{i}' for i in range(tail_max_seq_length)])

# Concatenate all DataFrames
features_for_prediction = pd.concat([cpu_usage_df, tail_cpu_usage_df, scaled_features_df], axis=1)

features_for_prediction_reshaped = tf.reshape(features_for_prediction, 
                                              (features_for_prediction.shape[0], 1, features_for_prediction.shape[1]))

# Choose the number of predictions
num_predictions = 200

# Make predictions for the first 'num_predictions' rows
predictions = model.predict(features_for_prediction_reshaped[:num_predictions])

# Clip negative predictions to 0
predictions = np.maximum(predictions, 0)

#'predictions' is a NumPy array
print("Predictions:\n", predictions)

# Plot the predictions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(df['average_usage_cpus'][:num_predictions], predictions[:, 0], alpha=0.5)
plt.title('Predicted vs Actual CPU Usage')
plt.xlabel('Actual CPU Usage')
plt.ylabel('Predicted CPU Usage')

plt.subplot(1, 2, 2)
plt.scatter(df['average_usage_memory'][:num_predictions], predictions[:, 1], alpha=0.5)
plt.title('Predicted vs Actual Memory Usage')
plt.xlabel('Actual Memory Usage')
plt.ylabel('Predicted Memory Usage')

plt.tight_layout()
plt.show()
