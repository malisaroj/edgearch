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
scaled_features_concatenated = pd.concat([cpu_usage_df, tail_cpu_usage_df, scaled_features_df], axis=1)

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Display the correlation matrix
print(correlation_matrix)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features_concatenated, labels, test_size=0.2, random_state=42)

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
'''


# Model with only GRU layer
model = tf.keras.Sequential([
    tf.keras.layers.GRU(units=128, activation='relu', input_shape=(1, X_train.shape[1])),
    tf.keras.layers.Dense(units=2, activation='linear')  
])

# Model with only Bidirectional LSTM layer
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=512, return_sequences=False), input_shape=(1, X_train.shape[1])),
    tf.keras.layers.Dense(units=2, activation='linear')  
])

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=512, return_sequences=True, input_shape=(1, X_train.shape[1])),
    tf.keras.layers.LSTM(units=128, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='linear')     
])  


'''
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=512, return_sequences=True), input_shape=(1, X_train.shape[1])),
    tf.keras.layers.GRU(units=128, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='linear')  
])

# Define the TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')
# Compile the model with the TensorBoard callback
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse'), 'accuracy'])

# Print the model summary
model.summary()

# Train the model
#model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32)  # Adjust the number of epochs based on your dataset and problem

# Evaluate the model on the test data
loss,  mae, rmse, accuracy = model.evaluate(X_test_reshaped, y_test)
print("Test Loss:", loss)
print("Test MAE:", mae)
print("Test RMSE:", rmse)
print("Test Accuracy:", accuracy)

# Save the trained model after the training is completed
model_save_path = Path(".cache") / "trained_model.h5"

# Check if the model directory already exists, and replace it if necessary
if model_save_path.exists():
    print("A trained model directory already exists. Replacing it.")
    try:
        for file in os.listdir(model_save_path):
            file_path = os.path.join(model_save_path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Error removing existing file/directory: {e}")
    except Exception as e:
        print(f"Error accessing the model directory: {e}")
else:
    print("No existing model directory found.")

# Save the new model in the SavedModel format
try:
    model.save(model_save_path)
except Exception as e:
    print(f"Error saving the model: {e}")

# Train the model and collect the training history
history = model.fit(X_train_reshaped, y_train, epochs=25, batch_size=32,
                    validation_data=(X_test_reshaped, y_test), callbacks=[tensorboard_callback])

# Plot both training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
