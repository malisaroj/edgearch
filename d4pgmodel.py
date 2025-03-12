import tensorflow_federated as tff
import tensorflow as tf
import numpy as np
from datetime import datetime
from collections import deque
import tensorflow_probability as tfp
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import csv
import matplotlib.pyplot as plt

# Load EMNIST dataset
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(only_digits=False, cache_dir=None)

G = 100
NUM_CLIENTS = 10
BATCH_SIZE = 64
L = 20  # Local Training Iterations
LR = 0.05  # Learning Rate
MU = 0.01  # Regularization Strength
EPSILON = 0.1  # Exploration rate for DQN
TARGET_UPDATE_INTERVAL = 5  # Update target Q-network every 5 rounds
MAX_BUFFER_SIZE = 10000  # Maximum buffer size
ALPHA = 0.6  # Priority exponent (higher alpha gives higher priority to experiences with higher TD error)
BETA = 0.4  # Importance sampling exponent
GAMMA = 0.99  # Discount factor for future rewards
NUM_CLASSES = 62
IMG_SIZE = (28, 28)
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 1)
# Define different noise levels to compare
noise_level = 0.1

# Update model input shape and preprocessing function
def preprocess(element):
    return tf.reshape(element['pixels'], [28, 28, 1]), tf.one_hot(element['label'], depth=NUM_CLASSES)

# Use client IDs directly without shuffling
client_ids = emnist_train.client_ids[:NUM_CLIENTS]

federated_train_data = [
    emnist_train.create_tf_dataset_for_client(client_id).map(preprocess).shuffle(buffer_size=10000).batch(BATCH_SIZE)
    for client_id in client_ids
]

# Preprocess the dataset and batch it
dataset = emnist_test.create_tf_dataset_from_all_clients().map(preprocess).batch(BATCH_SIZE)

#train_data = emnist_train.create_tf_dataset_from_all_clients().map(preprocess).batch(BATCH_SIZE)

#len_train_data = train_data.reduce(0, lambda x, _: x + 1).numpy()

# Determine the size of the dataset efficiently
dataset_size = dataset.reduce(0, lambda x, _: x + 1).numpy()

# Compute the size of validation and test sets
validation_size = dataset_size // 2

# Split the dataset into validation and test sets
validation_data = dataset.take(validation_size)
test_data = dataset.skip(validation_size)

'''
# Update the global model architecture
def create_global_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])

# Define the global model architecture
def create_global_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])

    return model

    '''
    # Define the global model architecture
def create_global_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])

    return model

# Define the D4PG agent
class D4PGAgent:
    def __init__(self, observation_space, action_space, buffer_size=MAX_BUFFER_SIZE, alpha=ALPHA, beta=BETA):
        self.observation_space = observation_space
        self.action_space = action_space
        self.q_network = self.build_q_network()
        self.buffer_size = buffer_size
        self.experience_replay_buffer = PrioritizedReplayBuffer(buffer_size=buffer_size, alpha=alpha, beta=beta)

    def build_q_network(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.observation_space,)),
            tf.keras.layers.Dense(64, activation='relu'),
            # Output layer with distributional parameters
            tf.keras.layers.Dense(self.action_space * 2)  # Double the output for mean and standard deviation
        ])

    def get_observation(self, global_model, client_resources):
        global_state = global_model.get_weights()

        # Flatten the global state
        global_state_flattened = np.concatenate([np.array(w).flatten() for w in global_state])

        # Flatten the UE resource information
        client_resources_flattened = np.concatenate([np.array(list(resources.values())) for resources in client_resources.values()])

        # Concatenate global state and UE resource information
        observation = np.concatenate([global_state_flattened, client_resources_flattened])

        # Reshape to match the expected input shape of the model
        return observation.reshape(1, -1)


    def update_action_space(self, new_action_space_size):
        self.action_space = new_action_space_size

    '''

    def select_action(self, state):
        # Sample actions from distribution for each state in the batch
        distribution_params = self.q_network.predict(state)
        means = distribution_params[:, :self.action_space]
        stds = tf.math.softplus(distribution_params[:, self.action_space:])

        actions = []
        for mean, std in zip(means, stds):
            action_distribution = tfp.distributions.Normal(loc=mean, scale=std)
            action = action_distribution.sample()
            actions.append(action)

        return np.array(actions)  # Convert to NumPy array for compatibility

    def select_action(self, state):
        # Sample action from distribution
        distribution_params = self.q_network.predict(state.reshape(1, -1))
        means, stds = distribution_params[:, :self.action_space], tf.math.softplus(distribution_params[:, self.action_space:])
        action_distributions = tfp.distributions.Normal(loc=means, scale=stds)
        actions = action_distributions.sample()
        return actions.numpy().squeeze()

    '''
    def select_action(self, state):
        q_values = self.q_network.predict(state.reshape(1, -1))
        if np.random.rand() < EPSILON:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(q_values)


    def record_experience(self, state, action, reward, next_state, td_error=None):
        if next_state is not None:
            self.experience_replay_buffer.record_experience(state, action, reward, next_state, td_error)

    def update_q_network(self):
        filtered_indices, filtered_experiences = self.experience_replay_buffer.get_all_experiences()

        if len(filtered_experiences) > BATCH_SIZE:
            minibatch, indices, weights = self.experience_replay_buffer.sample_batch(BATCH_SIZE)
            sampled_experiences = [filtered_experiences[i] for i in minibatch]

            states, actions, rewards, next_states = zip(*sampled_experiences)
            states = np.vstack(states)
            next_states = np.vstack(next_states)

            target_distributions = self.compute_target_distributions(next_states)

            # Compute TD error and update priorities
            td_errors = self.compute_td_errors(states, actions, rewards, target_distributions)
            self.experience_replay_buffer.update_priorities(indices, td_errors)

            # Train the Q-network
            self.q_network.fit(states, target_distributions, sample_weight=weights, epochs=1, verbose=0)

    def compute_target_distributions(self, next_states):
        target_distribution_params = self.q_network.predict(next_states)
        means, stds = target_distribution_params[:, :self.action_space], tf.math.softplus(target_distribution_params[:, self.action_space:])
        target_distributions = tfp.distributions.Normal(loc=means, scale=stds)
        return target_distributions

    def compute_td_errors(self, states, actions, rewards, target_distributions):
        current_distribution_params = self.q_network.predict(states)
        means, stds = current_distribution_params[:, :self.action_space], tf.math.softplus(current_distribution_params[:, self.action_space:])
        current_distributions = tfp.distributions.Normal(loc=means, scale=stds)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        action_probs = current_distributions.prob(actions)
        target_probs = target_distributions.prob(actions)
        td_errors = tf.reduce_mean(tf.square(target_probs / action_probs) * (rewards + GAMMA * target_distributions.mean() - current_distributions.mean()), axis=-1)
        return td_errors

# Define the Prioritized Replay Buffer class
class PrioritizedReplayBuffer:
    def __init__(self, buffer_size=MAX_BUFFER_SIZE, alpha=ALPHA, beta=BETA):
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=self.buffer_size)
        self.priorities = deque(maxlen=self.buffer_size)

    def record_experience(self, state, action, reward, next_state, td_error=None):
        if next_state is not None:
            priority = (np.abs(td_error) + 1e-5) ** self.alpha
            self.buffer.append((state, action, reward, next_state))
            self.priorities.append(priority)

    def sample_batch(self, batch_size):
        priorities = np.array(self.priorities)
        probabilities = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities)
        samples = [self.buffer[i] for i in indices]

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            priority = (np.abs(td_error) + 1e-5) ** self.alpha
            self.priorities[i] = priority

    def get_all_experiences(self):
        return list(self.buffer), list(self.priorities)

# Update the target network with Polyak averaging
def update_target_network(target_model, model, tau=0.001):
    target_weights = target_model.get_weights()
    model_weights = model.get_weights()
    new_weights = [(1 - tau) * t + tau * m for t, m in zip(target_weights, model_weights)]
    target_model.set_weights(new_weights)

def dynamic_sampling(global_model, client_resources):
    state = d4pg_agent.get_observation(global_model, client_resources)

    # Select clients initially
    selected_clients = [d4pg_agent.select_action(state) for _ in range(NUM_CLIENTS)]

    # Remove duplicates by converting to a set and then back to a list
    unique_clients = list(set(selected_clients))

    # Ensure all client IDs are within the valid range
    unique_clients = [client_id % NUM_CLIENTS for client_id in unique_clients]

    # If we don't have enough unique clients, select more until we do
    while len(unique_clients) < max_selected_clients:
        client_index = d4pg_agent.select_action(state)
        if client_index not in unique_clients:
            unique_clients.append(client_index % NUM_CLIENTS)

    # Trim the list to the desired number of clients if necessary
    unique_clients = unique_clients[:max_selected_clients]

    print(unique_clients)
    return unique_clients

def client_update(global_model, global_weights, local_data):
    local_model = tf.keras.models.clone_model(global_model)
    local_model.set_weights(global_weights)
    optimizer = tf.keras.optimizers.SGD(learning_rate=LR)

    for epoch in range(L):  # Iterate over epochs

        for x, y in local_data:
            features = tf.cast(x, dtype=tf.float32)
            labels = tf.cast(y, dtype=tf.float32)
            with tf.GradientTape() as tape:
                #features = tf.reshape(features, [-1] + list(INPUT_SHAPE))  # Reshape to original shape
                logits = local_model(features, training=True)
                loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                loss = loss_fn(labels, logits)

            # Compute and apply gradients
            gradients = tape.gradient(loss, local_model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
            # Update the local model using SGD optimizer
            optimizer.apply_gradients(zip(gradients, local_model.trainable_variables))

            # Calculate accuracy
            accuracy_fn = tf.keras.metrics.CategoricalAccuracy()
            accuracy_fn.update_state(labels, logits)
            accuracy = accuracy_fn.result().numpy()

    return local_model.get_weights()

# Define the aggregate models function
def aggregate_models_fn(global_weights, local_models):
    total_weight = 0
    weighted_sum = [tf.zeros_like(w) for w in global_weights]
    for local_model, dataset_size in local_models:
        weight = dataset_size
        total_weight += weight
        weighted_sum = [ws + w * weight for ws, w in zip(weighted_sum, local_model)]

    averaged_weights = [ws / total_weight for ws in weighted_sum]
    return averaged_weights

# Specify the range for random resource values
RESOURCE_RANGE = {'processing_power': (50, 150),
                  'memory': (8, 32),
                  'disk_space': (200, 800),
                  'bandwidth': (50, 150),
                  'power_availability': (0.5, 1.0)}

# Generate random resource values for each client
client_resources = {client_id: {
                        'processing_power': np.random.uniform(*RESOURCE_RANGE['processing_power']),
                        'memory': np.random.uniform(*RESOURCE_RANGE['memory']),
                        'disk_space': np.random.uniform(*RESOURCE_RANGE['disk_space']),
                        'bandwidth': np.random.uniform(*RESOURCE_RANGE['bandwidth']),
                        'power_availability': np.random.uniform(*RESOURCE_RANGE['power_availability']),
                    }
                    for client_id in range(NUM_CLIENTS)}

'''
def assign_weights_to_resources(client_resources):
    # Define weights for each resource dimension
    weights = {
        'processing_power': 3.0,
        'memory': 1.5,
        'disk_space': 2.0,
        'bandwidth': 2.5,
        'power_availability': 1.0,
    }

    # Assign weights directly to resources
    weighted_resources = {
        client_id: {
            resource: client[resource] * weights[resource]
            for resource in client.keys()
        }
        for client_id, client in client_resources.items()
    }

    return weighted_resources

'''

def assign_weights_to_resources(client_resources):
    # Define weights for each resource dimension
    weights = {
        'processing_power': 3.0,
        'memory': 1.5,
        'disk_space': 2.0,
        'bandwidth': 2.5,
        'power_availability': 1.0,
    }

    # Apply weights to each resource dimension
    weighted_resources = {}
    for client_id, resources in client_resources.items():
        weighted_client_resources = {}
        for resource, value in resources.items():
            weighted_client_resources[resource] = value * weights[resource]
        weighted_resources[client_id] = weighted_client_resources

    # Normalize each resource dimension separately
    normalized_resources = {}
    for resource in weights.keys():
        # Extract weighted values for the current resource dimension
        values = [client[resource] for client in weighted_resources.values()]
        # Find min and max values
        min_value = min(values)
        max_value = max(values)
        # Normalize values to be between 0 and 1
        normalized_values = [(value - min_value) / (max_value - min_value) for value in values]
        # Assign normalized values to the resource dimension for each client
        for i, client_id in enumerate(weighted_resources.keys()):
            if client_id not in normalized_resources:
                normalized_resources[client_id] = {}
            normalized_resources[client_id][resource] = normalized_values[i]

    return normalized_resources


def compute_fairness_feedback(selected_clients, federated_train_data, client_resources):
    fairness_feedback = []

    weighted_resources = assign_weights_to_resources(client_resources)

    for client_id in selected_clients:
        client_data = federated_train_data[client_id]
        class_distribution = compute_class_distribution(client_data)
        # Combine class distribution and normalized resources for fairness feedback
        feedback = {'class_distribution': class_distribution, 'resources': weighted_resources[client_id]}
        fairness_feedback.append(feedback)

    return fairness_feedback

def compute_class_distribution(client_data):
    class_counts = [0] * NUM_CLASSES
    total_samples = 0

    for features, labels in client_data:
        total_samples += len(labels)
        class_counts += tf.reduce_sum(labels, axis=0).numpy()

    class_distribution = class_counts / total_samples
    return class_distribution

def compute_reward(fairness_feedback, accuracy, fairness_weight=0.4, resource_weight=0.4, accuracy_weight=0.2):

    # Extract class distribution and resources from fairness feedback
    class_distribution = fairness_feedback['class_distribution']
    resources = fairness_feedback['resources']

    # Compute fairness component of the reward
    fairness_reward = np.mean(class_distribution)

    # Compute resource component of the reward
    resource_reward = np.mean(list(resources.values()))  # Average of all resource values

    # Combine fairness and resource rewards using specified weights
    total_reward = (fairness_weight * fairness_reward) + (resource_weight * resource_reward) + (accuracy_weight * accuracy)


    return total_reward

# Create a summary writer for TensorBoard
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

# Function to log metrics to TensorBoard
def log_metrics(round_num, loss, accuracy):
    with summary_writer.as_default():
        tf.summary.scalar('Loss', loss, step=round_num)
        tf.summary.scalar('Accuracy', accuracy, step=round_num)

# Evaluate the global model on the test dataset
def evaluate_model(global_model, test_data):
    loss, accuracy = global_model.evaluate(test_data)
    return loss, accuracy

# Function to add noise to weights
def add_noise(weights, noise_factor):
    return [w + np.random.normal(0, noise_factor, w.shape) for w in weights]

# Federated learning algorithm
global_model = create_global_model()

#history = model.fit(train_data, steps_per_epoch=len_train_data // 32, epochs=20, validation_data=validation_data)


#model.save('saved_model')

#global_model = tf.keras.models.load_model('saved_model')

global_weights = global_model.get_weights()
obs_space_clients_info = 5  # resource features for each Clients (processing power, memory, bandwidth, etc.)
# Length of global model weights
obs_space_global_model = sum(np.prod(w.shape) for w in global_weights)

# Length of concatenated client resource information
obs_space_clients_info = NUM_CLIENTS * obs_space_clients_info  # obs_space_clients_info is the length of concatenated client resource information for each client

# Total observation space
obs_space = obs_space_global_model + obs_space_clients_info
# Set an initial estimated maximum number of selected clients
max_selected_clients = 5
act_space = NUM_CLIENTS
d4pg_agent = D4PGAgent(obs_space, act_space)

# Compile the global model
global_model.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

rewards = []  # To store rewards for each round

for round_num in range(G):
    print("Round:", round_num)
    # Ensure that the sampled_clients list contains valid client IDs
    sampled_clients = dynamic_sampling(global_model, client_resources)
    sampled_clients = [client_id % len(federated_train_data) for client_id in sampled_clients]
    federated_models = []

    for client_id in sampled_clients:
            # Get the client-specific datasets
            client_data = federated_train_data[client_id]
            client_data = client_data.take(1)  # for faster experimentation during development.
            client_model = client_update(global_model, global_model.get_weights(), client_data)
            dataset_size = tf.data.experimental.cardinality(client_data).numpy()
            federated_models.append((client_model, dataset_size))

    global_weights = aggregate_models_fn(global_weights, federated_models)
    global_weights = add_noise(global_weights, noise_level)  # Apply noise to weights
    global_model.set_weights(global_weights)

    # Compute fairness feedback based on class distribution and client resources
    fairness_feedback = compute_fairness_feedback(sampled_clients, federated_train_data, client_resources)

    # Evaluate global model accuracy
    val_loss, val_accuracy = evaluate_model(global_model, validation_data)

    # Log metrics
    log_metrics(round_num, val_loss, val_accuracy)

    round_reward = 0  # Accumulator for round reward

    for i, client_id in enumerate(sampled_clients):

        # Compute reward based on fairness feedback and accuracy
        reward = compute_reward(fairness_feedback[i], val_accuracy)

        round_reward += reward  # Add reward to round_reward accumulator
    d4pg_agent.update_q_network()

    rewards.append(round_reward)  # Append round_reward to rewards list

# Write rewards to CSV file
csv_filename = 'rewards.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['Round', 'Reward']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for round_num, reward in enumerate(rewards):
        writer.writerow({'Round': round_num, 'Reward': reward})

# Plot rewards
rounds = list(range(G))
plt.plot(rounds, rewards)
plt.xlabel('Round')
plt.ylabel('Reward')
plt.title('Rewards over Rounds')
plt.grid(True)
plt.show()

# Evaluate the global model on test data
test_loss, test_accuracy = evaluate_model(global_model, test_data)

# Make predictions on the test dataset and grab true labels
predictions_with_labels = []

for batch in test_data:
    images, labels = batch
    predictions = global_model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(labels, axis=1)
    predictions_with_labels.extend(list(zip(predicted_labels, true_labels)))

# Now predictions_with_labels contains tuples of (predicted_label, true_label)
# Compute precision, recall, and F1-score
y_pred_labels, y_true_labels = zip(*predictions_with_labels)
precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

# Print the test loss and accuracy
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Close the summary writer
summary_writer.close()
# End of the federated learning algorithm
