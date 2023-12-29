import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import json_normalize
import ast
from sklearn.preprocessing import PolynomialFeatures

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


columns_to_drop = ['time', 'instance_events_type', 'scheduling_class', 'priority',
                   'constraint', 'collections_events_type', 'user', 'collection_name', 
                   'collection_logical_name', 'start_after_collection_ids', 'vertical_scaling', 
                   'scheduler', 'cpu_usage_distribution', 'tail_cpu_usage_distribution', 
                   'cluster', 'event', 'failed', 'random_sample_usage_memory', 'collection_id',
                   'alloc_collection_id', 'collection_type','start_time', 'end_time', 'sample_rate'
                   , 'cycles_per_instruction', 'memory_accesses_per_instruction', 'page_cache_memory',
                   'instance_index', 'machine_id',]


# Drop the specified columns
df.drop(columns=columns_to_drop, inplace=True)
df = df.drop(df.columns[df.columns.str.contains('Unnamed', case=False, regex=True)][0], axis=1)

# Adding more feature engineering as needed based on  data
# Cross-feature interactions
df['interaction_feature'] = df['maximum_usage_cpus'] * df['random_sample_usage_cpus']

# Creating lag features for memory_demand
df['memory_demand_lag_1'] = df['memory_demand'].shift(1)

# Creating rolling window statistics for memory_demand
df['memory_demand_rolling_mean'] = df['memory_demand'].rolling(window=3).mean()
df['memory_demand_rolling_std'] = df['memory_demand'].rolling(window=3).std()

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['maximum_usage_cpus', 'random_sample_usage_cpus']])
poly_feature_names = [f"poly_{name}" for name in poly.get_feature_names_out(['maximum_usage_cpus', 'random_sample_usage_cpus'])]
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
df = pd.concat([df, poly_df], axis=1)

# Check for empty values
empty_values = df.isnull().sum()
print("Empty Values:\n", empty_values)

# Check for zero values
zero_values = (df == 0).sum()
print("\nZero Values:\n", zero_values)

df.fillna(df.mean(), inplace=True)

# Now, check again for empty or zero values
updated_empty_values = df.isnull().sum()
updated_zero_values = (df == 0).sum()

print("\nUpdated Empty Values:\n", updated_empty_values)
print("\nUpdated Zero Values:\n", updated_zero_values)

correlation_matrix = df.corr()
print(correlation_matrix)

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[[  'resource_request_cpus', 'resource_request_memory',  'poly_maximum_usage_cpus random_sample_usage_cpus',
                                            'maximum_usage_cpus',  'poly_random_sample_usage_cpus', 'poly_random_sample_usage_cpus^2', 'memory_demand_lag_1',
                                            'maximum_usage_memory',  'interaction_feature', 'poly_maximum_usage_cpus^2', 'memory_demand_rolling_mean',
                                            'random_sample_usage_cpus', 'assigned_memory',  'poly_maximum_usage_cpus', 'memory_demand_rolling_std',
]])

# Labels
labels = df[['average_usage_cpus', 'average_usage_memory']]
print(df.head)

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Display the correlation matrix
print(correlation_matrix)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

# Convert NumPy arrays back to TensorFlow tensors
X_train = tf.constant(X_train, dtype=tf.float32)
X_test = tf.constant(X_test, dtype=tf.float32)
y_train = tf.constant(y_train, dtype=tf.float32)
y_test = tf.constant(y_test, dtype=tf.float32)

# Custom Dataset class
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

# Reshape the input features to add a third dimension for time steps
X_train_reshaped = tf.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = tf.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Define the model using TensorFlow layers

#model = tf.keras.Sequential([
#        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True), input_shape=(1, X_train.shape[1])),
#        tf.keras.layers.Dense(units=2)
#])

#model = tf.keras.Sequential([
#    tf.keras.layers.LSTM(units=64, return_sequences=True, input_shape=(1, X_train.shape[1])),
#    tf.keras.layers.LSTM(units=32, activation='relu'),
#    tf.keras.layers.Dense(units=2)
#])  


model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True), input_shape=(1, X_train.shape[1])),
    tf.keras.layers.GRU(units=32, activation='relu'),
    tf.keras.layers.Dense(units=2)
])


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print the model summary
model.summary()

# Train the model
#model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32)  # Adjust the number of epochs based on your dataset and problem

# Evaluate the model on the test data
loss = model.evaluate(X_test_reshaped, y_test)
print("Test Loss:", loss)

model.save("hybrid_model.h5")

# Train the model and collect the training history
history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test))

# Plot both training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the plot
plt.savefig('training_loss_plot.png')
