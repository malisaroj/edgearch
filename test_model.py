#Importing Edgesimpy components
from edge_sim_py import *
import networkx as nx
import numpy as np
import tensorflow as tf
import json 
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from utils import set_pandas_display_options, highlight_rows, load_datasets, display_user_mobility_traces, plot_edge_servers_power_consumption
import os
from keras.utils import pad_sequences
from tensorflow.keras.initializers import GlorotUniform
from keras.saving import register_keras_serializable

# Set Pandas display options
set_pandas_display_options()

def worst_fit_algorithm(parameters):
    for service in Service.all():
        # We don't want to migrate services are are already being migrated
        if not service.being_provisioned:
            # We need to sort edge servers based on amount of free resources they have. To do so, we are going to use Python's
            # "sorted" method (you can learn more about "sorted()" in this link: https://docs.python.org/3/howto/sorting.html). As
            # the capacity of edge servers is modeled in three layers (CPU, memory, and disk), we calculate the geometric mean
            # between these to get the average resource utilization of edge servers. Finally, we set the sorted method "reverse"
            # attribute as "True" as we want to sort edge servers by their free resources in descending order
            edge_servers = sorted(
                EdgeServer.all(),
                key=lambda s: ((s.cpu - s.cpu_demand) * (s.memory - s.memory_demand) )** 0.5,
                reverse=True,
            )

            for edge_server in edge_servers:
                # Checking if the edge server has resources to host the service
                if edge_server.has_capacity_to_host(service=service):
                    # We just need to migrate the service if it's not already in the least occupied edge server
                    if service.server != edge_server:
                        print(f"[STEP {parameters['current_step']}] Migrating {service} From {service.server} to {edge_server}")
                        service.provision(target_server=edge_server)
                        # After start migrating the service we can move on to the next service
                        break

# Implementing the best-fit algorithm
def best_fit_algorithm(parameters):
    for service in Service.all():
        if not service.being_provisioned:
            edge_servers = sorted(
                EdgeServer.all(),
                key=lambda s: ((s.cpu - s.cpu_demand) * (s.memory - s.memory_demand)) ** 0.5,
                reverse=False,
            )

            for edge_server in edge_servers:
                if edge_server.has_capacity_to_host(service=service):
                    if service.server != edge_server:
                        print(f"[BEST-FIT - STEP {parameters['current_step']}] Migrating {service} from {service.server} to {edge_server}")
                        # Record the provision time when a service is provisioned
                        service.provision(target_server=edge_server)
                        service.provision_time = parameters['current_step']
                        break

#Implementing the first fit algorithm 
def first_fit_algorithm(parameters):
    for service in Service.all():
        if not service.being_provisioned:
            for edge_server in EdgeServer.all():
                if edge_server.has_capacity_to_host(service=service):
                    if service.server != edge_server:
                        print(f"[FIRST-FIT - STEP {parameters['current_step']}] Migrating {service} from {service.server} to {edge_server}")
                        service.provision(target_server=edge_server)
                        service.provision_time = parameters['current_step']
                        break  # Break out of the loop once a suitable server is found
                    else:
                        # Service is already hosted on this server, no need to migrate
                        break


# implementing the hybrid  migration algorithm
def hybrid_offloading_algorithm(parameters):

    # Let's iterate over the list of services using the 'all()' helper method
    print("\n\n")

    # Read data from the JSON file
    with open('datasets/test.json', 'r') as file:
        edge_servers_data = json.load(file)

    # List to store features for prediction
    features_for_prediction = []

    # Extracting cpu_demand and memory_demand for each EdgeServer
    for edge_server in edge_servers_data["EdgeServer"]:
        cpu_demand = edge_server["attributes"]["cpu_demand"]
        memory_demand = edge_server["attributes"]["memory_demand"]
        maximum_usage_cpus = edge_server["attributes"]["maximum_usage_cpus"]
        maximum_usage_memory = edge_server["attributes"]["maximum_usage_memory"]
        random_sample_usage_cpus = edge_server["attributes"]["random_sample_usage_cpus"]
        assigned_memory = edge_server["attributes"]["assigned_memory"]
        time = edge_server["attributes"]["time"]
        priority = edge_server["attributes"]["priority"]
        start_time = edge_server["attributes"]["start_time"]
        end_time = edge_server["attributes"]["end_time"]
        page_cache_memory = edge_server["attributes"]["page_cache_memory"]
        cycles_per_instruction = edge_server["attributes"]["cycles_per_instruction"]
        memory_accesses_per_instruction = edge_server["attributes"]["memory_accesses_per_instruction"]
        sample_rate = edge_server["attributes"]["sample_rate"]
        cpu_usage_distribution = edge_server["attributes"]["cpu_usage_distribution"]
        tail_cpu_usage_distribution = edge_server["attributes"]["tail_cpu_usage_distribution"]
        average_usage_cpus = edge_server["attributes"]["average_usage_cpus"]
        average_usage_memory = edge_server["attributes"]["average_usage_memory"]
    
        # Creating a dictionary for each EdgeServer's features
        features = {
                    "cpu_demand": cpu_demand,
                    "memory_demand": memory_demand,
                    "maximum_usage_cpus": maximum_usage_cpus,
                    "maximum_usage_memory": maximum_usage_memory,
                    "random_sample_usage_cpus": random_sample_usage_cpus,
                    "assigned_memory": assigned_memory,
                    "time": time,
                    "priority": priority,
                    "start_time": start_time,
                    "end_time": end_time,
                    "assigned_memory": assigned_memory,
                    "page_cache_memory": page_cache_memory,
                    "cycles_per_instruction": cycles_per_instruction,
                    "memory_accesses_per_instruction": memory_accesses_per_instruction,
                    "sample_rate": sample_rate,
                    "cpu_usage_distribution": cpu_usage_distribution,
                    "tail_cpu_usage_distribution": tail_cpu_usage_distribution,
                    "average_usage_cpus": average_usage_cpus,
                    "average_usage_memory": average_usage_memory,
                    "maximum_usage_cpus": maximum_usage_cpus,
                    "maximum_usage_memory": maximum_usage_memory,
                    "random_sample_usage_cpus": random_sample_usage_cpus,  

                }

        # Appending the features dictionary to the list
        features_for_prediction.append(features)

    # Convert the list of dictionaries to a Pandas DataFrame
    df_for_prediction = pd.DataFrame(features_for_prediction)

    # Adding more feature engineering as needed based on  data
    # Cross-feature interactions
    df_for_prediction['interaction_feature'] = df_for_prediction['maximum_usage_cpus'] * df_for_prediction['random_sample_usage_cpus']

    # Creating lag features for memory_demand
    df_for_prediction['memory_demand_lag_1'] = df_for_prediction['memory_demand'].shift(1)

    # Creating rolling window statistics for memory_demand
    df_for_prediction['memory_demand_rolling_mean'] = df_for_prediction['memory_demand'].rolling(window=3).mean()
    df_for_prediction['memory_demand_rolling_std'] = df_for_prediction['memory_demand'].rolling(window=3).std()

    # Polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df_for_prediction[['maximum_usage_cpus', 'random_sample_usage_cpus']])
    poly_feature_names = [f"poly_{name}" for name in poly.get_feature_names_out(['maximum_usage_cpus', 'random_sample_usage_cpus'])]
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    df_for_prediction = pd.concat([df_for_prediction, poly_df], axis=1)

    # Convert timestamp columns to datetime objects
    df_for_prediction['start_time'] = pd.to_datetime(df_for_prediction['start_time'], unit='ms')
    df_for_prediction['end_time'] = pd.to_datetime(df_for_prediction['end_time'], unit='ms')

    # Extract relevant features from timestamps
    df_for_prediction['start_hour'] = df_for_prediction['start_time'].dt.hour
    df_for_prediction['start_dayofweek'] = df_for_prediction['start_time'].dt.dayofweek

    # Calculate the duration between start_time and end_time
    df_for_prediction['duration_seconds'] = (df_for_prediction['end_time'] - df_for_prediction['start_time']).dt.total_seconds()

    df_for_prediction['cpu_usage_distribution'] = df_for_prediction['cpu_usage_distribution'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    df_for_prediction['tail_cpu_usage_distribution'] = df_for_prediction['tail_cpu_usage_distribution'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

    # Padding sequences
    max_seq_length = df_for_prediction['cpu_usage_distribution'].apply(len).max()  # Find the maximum length of sequences
    df_for_prediction['cpu_usage_distribution_padded'] = df_for_prediction['cpu_usage_distribution'].apply(lambda x: pad_sequences([x], maxlen=max_seq_length, padding='post', dtype='float32')[0])
    tail_max_seq_length = df_for_prediction['tail_cpu_usage_distribution'].apply(len).max()  # Find the maximum length of sequences
    df_for_prediction['tail_cpu_usage_distribution_padded'] = df_for_prediction['tail_cpu_usage_distribution'].apply(lambda x: pad_sequences([x], maxlen=tail_max_seq_length, padding='post', dtype='float32')[0])

    # Feature Scaling
    scaler = StandardScaler()
    df_for_prediction['cpu_usage_distribution_scaled'] = df_for_prediction['cpu_usage_distribution_padded'].apply(lambda x: scaler.fit_transform(x.reshape(-1, 1)))
    df_for_prediction['tail_cpu_usage_distribution_scaled'] = df_for_prediction['tail_cpu_usage_distribution_padded'].apply(lambda x: scaler.fit_transform(x.reshape(-1, 1)))

    scaled_features = scaler.fit_transform(df_for_prediction[[ 'average_usage_cpus', 'average_usage_memory',  'poly_maximum_usage_cpus random_sample_usage_cpus', 
                                                'maximum_usage_cpus',  'poly_random_sample_usage_cpus', 'poly_random_sample_usage_cpus^2', 'memory_demand_rolling_mean',
                                                'maximum_usage_memory',  'interaction_feature', 'poly_maximum_usage_cpus^2', 'memory_demand_lag_1',
                                                'random_sample_usage_cpus', 'assigned_memory',  'poly_maximum_usage_cpus', 'memory_demand_rolling_std', 
                                                'start_hour', 'start_dayofweek', 'duration_seconds', 'sample_rate', 'cycles_per_instruction', 
                                                'memory_accesses_per_instruction', 'page_cache_memory', 'priority',
                                            ]])

    # Convert numpy arrays to pandas DataFrames
    scaled_features_df = pd.DataFrame(scaled_features, columns=[
        'average_usage_cpus', 'average_usage_memory', 'poly_maximum_usage_cpus random_sample_usage_cpus',
        'maximum_usage_cpus', 'poly_random_sample_usage_cpus', 'poly_random_sample_usage_cpus^2', 'memory_demand_rolling_mean',
        'maximum_usage_memory', 'interaction_feature', 'poly_maximum_usage_cpus^2', 'memory_demand_lag_1',
        'random_sample_usage_cpus', 'assigned_memory', 'poly_maximum_usage_cpus', 'memory_demand_rolling_std',
        'start_hour', 'start_dayofweek', 'duration_seconds', 'sample_rate', 'cycles_per_instruction',
        'memory_accesses_per_instruction', 'page_cache_memory', 'priority'])

    # Reshape the data to 2D array
    cpu_usage_reshaped = np.vstack(df_for_prediction['cpu_usage_distribution_scaled']).reshape(-1, max_seq_length)
    tail_cpu_usage_reshaped = np.vstack(df_for_prediction['tail_cpu_usage_distribution_scaled']).reshape(-1, tail_max_seq_length)

    # Create DataFrame
    cpu_usage_df = pd.DataFrame(cpu_usage_reshaped, columns=[f'cpu_usage_{i}' for i in range(max_seq_length)])
    tail_cpu_usage_df = pd.DataFrame(tail_cpu_usage_reshaped, columns=[f'tail_cpu_usage_{i}' for i in range(tail_max_seq_length)])

    # Concatenate all DataFrames
    scaled_features_concatenated = pd.concat([cpu_usage_df, tail_cpu_usage_df, scaled_features_df], axis=1)


    # Convert NumPy arrays back to TensorFlow tensors
    scaled_features_concatenated = tf.constant(scaled_features_concatenated, dtype=tf.float32)

    #Reshape the input features to add a third dimension for time steps
    X_pred_reshaped = tf.reshape(scaled_features_concatenated, (scaled_features_concatenated.shape[0], 1, scaled_features_concatenated.shape[1]))

    #load trained BiLSTM model for task prediction
    @register_keras_serializable()
    # Custom Attention Layer
    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self):
            super(AttentionLayer, self).__init__()

        def build(self, input_shape):
            self.W_a = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer=GlorotUniform(), trainable=True)
            self.U_a = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer=GlorotUniform(), trainable=True)
            self.v_a = self.add_weight(shape=(input_shape[-1], 1), initializer=GlorotUniform(), trainable=True)

        def call(self, hidden_states):
            # Score computation
            score_first_part = tf.tensordot(hidden_states, self.W_a, axes=1)
            h_t = tf.tensordot(hidden_states, self.U_a, axes=1)
            score = tf.nn.tanh(score_first_part + h_t)
            
            # Attention weights
            attention_weights = tf.nn.softmax(tf.tensordot(score, self.v_a, axes=1), axis=1)
            
            # Context vector computation
            context_vector = tf.reduce_sum(attention_weights * hidden_states, axis=1)
            return context_vector
        
    # Load the model
    model = tf.keras.models.load_model('C:/Users/rog/Desktop/fededgearch/.cache/trained_model.keras' , custom_objects={'AttentionLayer': AttentionLayer})

    # Show the model architecture
    model.summary()

    #Make predictions on the new data
    predictions_pred = model.predict(X_pred_reshaped)

    #Reshape the predictions to match the original shape of labels
    predictions_pred = np.reshape(predictions_pred, (predictions_pred.shape[0], 2))

    #Convert the Numpy arrays back to a Pandas DataFrame for better analysis or comparision
    predictions_pred_df = pd.DataFrame(predictions_pred, columns=['predicted_cpus', 'predicted_memory'])
    # Preprocess the predictions to fill NaN values with 0
    predictions_pred_df.fillna(0, inplace=True)

    # Combine predictions with edge server information
    edge_servers_info = pd.DataFrame(
        {
            "EdgeServer": [server for server in EdgeServer.all()],
            "predicted_cpus": predictions_pred_df['predicted_cpus'],
            "predicted_memory": predictions_pred_df['predicted_memory'],
            "current_cpu": [server.cpu - server.cpu_demand for server in EdgeServer.all()],
            "current_memory": [server.memory - server.memory_demand for server in EdgeServer.all()]
        }
    )

    # Calculate decision factor (consider both predicted demands and current resources)
    edge_servers_info['available_cpu'] = edge_servers_info['current_cpu']
    edge_servers_info['available_memory'] = edge_servers_info['current_memory']
    edge_servers_info['decision_factor'] = edge_servers_info['predicted_cpus'] * edge_servers_info['available_cpu'] + \
                                           edge_servers_info['predicted_memory'] * edge_servers_info['available_memory']


    print(edge_servers_info['decision_factor'] )

    # Sort edge servers based on the decision factor
    edge_servers_info = edge_servers_info.sort_values(by='decision_factor', ascending=True)

    local_processing_threshold=0
    
    # Iterate over services
    for service in Service.all():
        if not service.being_provisioned:
            local_processing = False #Flag to determine if local processing is chosen

            #Check if the decsion factor is below the threshold for local processing
            if edge_servers_info['decision_factor'].iloc[0] < local_processing_threshold:
                local_processing = True
            
            if local_processing:
                #Perform local processing (no migration)
                print(f"[STEP {parameters['current_step']}] Processing {service} locally on {service.server}")
            else:

                for edge_server_info in edge_servers_info.itertuples():
                    edge_server = edge_server_info.EdgeServer
                    if edge_server.has_capacity_to_host(service=service):
                        if service.server != edge_server:
                            print(f"[STEP {parameters['current_step']}] Migrating {service} from {service.server} to {edge_server}")
                            service.provision(target_server=edge_server)

                            break

def stopping_criterion(model: object):
    # Defining a variable that will help us to count the number of services successfully provisioned within the infrastructure
    provisioned_services = 0
    
    # Iterating over the list of services to count the number of services provisioned within the infrastructure
    for service in Service.all():

        # Initially, services are not hosted by any server (i.e., their "server" attribute is None).
        # Once that value changes, we know that it has been successfully provisioned inside an edge server.
        if service.server != None:
            provisioned_services += 1
    
    # As EdgeSimPy will halt the simulation whenever this function returns True, its output will be a boolean expression
    # that checks if the number of provisioned services equals to the number of services spawned in our simulation
    return provisioned_services == Service.count()

# Creating Simulator objects for each algorithm
simulator = Simulator(
    tick_duration=1,
    tick_unit="seconds",
    stopping_criterion=stopping_criterion,
    resource_management_algorithm=hybrid_offloading_algorithm,
)

# Loading a sample dataset from GitHub for each simulator
simulator.initialize(input_file="datasets/test.json")

# Executing the simulations
simulator.run_model()

# Gathering the list of msgpack files in the current directory
logs_directory = f"{os.getcwd()}/logs"
datasets = load_datasets(logs_directory)

# Display user mobility traces
display_user_mobility_traces(datasets)

# Plot edge servers' power consumption
plot_edge_servers_power_consumption(datasets)
