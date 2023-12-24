import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (assuming the dataset is in a pandas DataFrame)
df = pd.read_csv("datasets/synthetic_dataset.csv")

# Use LabelEncoder to encode the 'device_name' column
label_encoder = LabelEncoder()
df['device_name_encoded'] = label_encoder.fit_transform(df['device_name'])

# Feature Engineering

## Spatial Features
df['distance_to_edge'] = np.sqrt((df['iot_x_locations'] - df['edge_x_locations'])**2 + (df['iot_y_locations'] - df['edge_y_locations'])**2)
df['direction_to_edge'] = np.arctan2(df['iot_y_locations'] - df['edge_y_locations'], df['iot_x_locations'] - df['edge_x_locations'])

## Resource Utilization
df['resource_task_ratio'] = df['computation_resources'] / df['task_sizes']


## Interaction Features
df['latency_distance_interaction'] = df['latency'] * df['distance_to_edge']
df['distance_network_interaction'] = df['distances'] * df['network_conditions']

#Cost Related Features
df['cost_resource_interaction'] = df['cost'] * df['computation_resources']
df['cost_per_computing_demand'] = df['cost'] / df['computing_demands']


## Statistical Features
mean_energy_by_device = df.groupby('device_name_encoded')['energy_consumption'].mean()
df['mean_energy_device'] = df['device_name_encoded'].map(mean_energy_by_device)

## Task Complexity
df['task_complexity'] = df['task_sizes'] * df['network_conditions']
df['task_energy_ratio'] = df['task_sizes'] / df['energy_consumption']


## Cost-Related Features
df['energy_efficiency'] = 1 / (df['cost'] * df['energy_consumption'])

# Scatter plot example
sns.scatterplot(x='cost_per_computing_demand', y='computing_demands', data=df)
plt.show()

correlation_matrix = df.corr()
print(correlation_matrix)

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['device_name_encoded', 'device_id', 'computation_resources', 'task_sizes',
                                          'network_conditions', 'iot_x_locations', 'iot_y_locations', 'latency', 'distances', 'energy_consumption',
                                          'edge_x_locations', 'edge_y_locations', 'cost', 'resource_task_ratio', 'cost_per_computing_demand']])

# Labels
labels = df[['computing_demands']]

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
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True), input_shape=(1, X_train.shape[1])),
    tf.keras.layers.GRU(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print the model summary
model.summary()

# Train the model
#model.fit(X_train_reshaped, y_train, epochs=200, batch_size=32)  # Adjust the number of epochs based on your dataset and problem

# Evaluate the model on the test data
loss = model.evaluate(X_test_reshaped, y_test)
print("Test Loss:", loss)