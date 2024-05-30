import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pandas import json_normalize
import ast
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures

# Load the saved model
model_save_path = Path(".cache") / "trained_model" 
model = tf.keras.models.load_model(model_save_path)

# Load your dataset (assuming the dataset is in a pandas DataFrame)
df = pd.read_csv("datasets/borg_traces_data.csv")

columns_to_normalize = ['resource_request', 'average_usage', 'maximum_usage', 'random_sample_usage']

# Iterate over each column and apply normalization
for column in columns_to_normalize:
    # Assuming the column contains dictionary-like strings
    df_normalized = json_normalize(df[column].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else None))
    
    # Concatenate the normalized DataFrame with the original DataFrame
    df = pd.concat([df, df_normalized.add_prefix(f"{column}_")], axis=1)

    # Drop the original column
    df.drop(column, axis=1, inplace=True)

# Adding more feature engineering as needed based on  data
# Cross-feature interactions
df['interaction_feature'] = df['maximum_usage_cpus'] * df['random_sample_usage_cpus']

# Creating lag features for memory_demand
df['memory_demand_lag_1'] = df['resource_request_memory'].shift(1)

# Creating rolling window statistics for memory_demand
df['memory_demand_rolling_mean'] = df['resource_request_memory'].rolling(window=3).mean()
df['memory_demand_rolling_std'] = df['resource_request_memory'].rolling(window=3).std()

# Reshape the input features if needed# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['maximum_usage_cpus', 'random_sample_usage_cpus']])
poly_feature_names = [f"poly_{name}" for name in poly.get_feature_names_out(['maximum_usage_cpus', 'random_sample_usage_cpus'])]
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
df = pd.concat([df, poly_df], axis=1)

# Extract the relevant features for prediction
# Feature Scaling
scaler = StandardScaler()
features_for_prediction = scaler.fit_transform(df[[ 'resource_request_cpus', 'resource_request_memory',  'poly_maximum_usage_cpus random_sample_usage_cpus', 
                                            'maximum_usage_cpus',  'poly_random_sample_usage_cpus', 'poly_random_sample_usage_cpus^2', 'memory_demand_rolling_mean',
                                            'maximum_usage_memory',  'interaction_feature', 'poly_maximum_usage_cpus^2', 'memory_demand_lag_1',
                                            'random_sample_usage_cpus', 'assigned_memory',  'poly_maximum_usage_cpus', 'memory_demand_rolling_std',
]])

features_for_prediction_reshaped = tf.reshape(features_for_prediction, 
                                              (features_for_prediction.shape[0], 1, features_for_prediction.shape[1]))

# Choose the number of predictions
num_predictions = 200

# Make predictions for the first 'num_predictions' rows
predictions = model.predict(features_for_prediction_reshaped[:num_predictions])

# Clip negative predictions to 0
predictions = np.maximum(predictions, 0)

# 'predictions' is a NumPy array
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
