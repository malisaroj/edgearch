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

# implementing the hybrid  migration algorithm
def hybrid_offloading_algorithm(parameters):

    # Let's iterate over the list of services using the 'all()' helper method
    print("\n\n")

    # Read data from the JSON file
    with open('datasets/sample_dataset2.json', 'r') as file:
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

        # Creating a dictionary for each EdgeServer's features
        features = {
                    "cpu_demand": cpu_demand,
                    "memory_demand": memory_demand,
                    "maximum_usage_cpus": maximum_usage_cpus,
                    "maximum_usage_memory": maximum_usage_memory,
                    "random_sample_usage_cpus": random_sample_usage_cpus,
                    "assigned_memory": assigned_memory
                }

        # Appending the features dictionary to the list
        features_for_prediction.append(features)

    # Convert the list of dictionaries to a Pandas DataFrame
    df_for_prediction = pd.DataFrame(features_for_prediction)

    # Adding more feature engineering as needed based on  data
    # Cross-feature interactions
    df_for_prediction['interaction_feature'] = df_for_prediction['maximum_usage_cpus'] * df_for_prediction['random_sample_usage_cpus']

    # Creating lag features for memory_demand
    #df_for_prediction['memory_demand_lag_1'] = df_for_prediction['memory_demand'].shift(1)

    # Creating rolling window statistics for memory_demand
    #df_for_prediction['memory_demand_rolling_mean'] = df_for_prediction['memory_demand'].rolling(window=3).mean()
    #df_for_prediction['memory_demand_rolling_std'] = df_for_prediction['memory_demand'].rolling(window=3).std()

    # Polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df_for_prediction[['maximum_usage_cpus', 'random_sample_usage_cpus']])
    poly_feature_names = [f"poly_{name}" for name in poly.get_feature_names_out(['maximum_usage_cpus', 'random_sample_usage_cpus'])]
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    df_for_prediction = pd.concat([df_for_prediction, poly_df], axis=1)

    #Scale the features 
    scaler = StandardScaler()
    scaled_features_pred = scaler.fit_transform(df_for_prediction)

    #Reshape the input features to add a third dimension for time steps
    X_pred_reshaped = tf.reshape(scaled_features_pred, (scaled_features_pred.shape[0], 1, scaled_features_pred.shape[1]))

    #load trained BiLSTM model for task prediction
    model = tf.keras.models.load_model('hybrid_model.h5')

    # Show the model architecture
    model.summary()

    #Make predictions on the new data
    predictions_pred = model.predict(X_pred_reshaped)

    #Reshape the predictions to match the original shape of labels
    predictions_pred = np.reshape(predictions_pred, (predictions_pred.shape[0], 2))

    #Convert the Numpy arrays back to a Pandas DataFrame for better analysis or comparision
    predictions_pred_df = pd.DataFrame(predictions_pred, columns=['predicted_cpus', 'predicted_memory'])
    
    # Combine predictions with edge server information
    edge_servers_info = pd.DataFrame(
        {
            "EdgeServer": [server for server in EdgeServer.all()],
            "predicted_cpus": predictions_pred_df['predicted_cpus'],
            "predicted_memory": predictions_pred_df['predicted_memory'],
        }
    )

    # Determine the decision factor (e.g., use the sum of predicted CPUs and memory as the decision factor)
    edge_servers_info['decision_factor'] = edge_servers_info['predicted_cpus'] + edge_servers_info['predicted_memory']
    print(edge_servers_info['decision_factor'] )

    # Sort edge servers based on the decision factor
    edge_servers_info = edge_servers_info.sort_values(by='decision_factor', ascending=False)

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
    resource_management_algorithm=best_fit_algorithm,
)

# Loading a sample dataset from GitHub for each simulator
simulator.initialize(input_file="datasets/sample_dataset2.json")

# Executing the simulations
simulator.run_model()

# Gathering the list of msgpack files in the current directory
logs_directory = f"{os.getcwd()}/logs"
datasets = load_datasets(logs_directory)

# Display user mobility traces
display_user_mobility_traces(datasets)

# Plot edge servers' power consumption
plot_edge_servers_power_consumption(datasets)
