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

# Global variable to store migration counts
migration_counts = {'worst_fit': 0, 'best_fit': 0, 'hybrid': 0}
energy_consumption = {'worst_fit': 0, 'best_fit': 0, 'hybrid': 0}

# Implementing the worst-fit algorithm
def worst_fit_algorithm(parameters):
    global migration_counts
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
                        # Calculate the energy consumption during migration
                        migration_data_size = calculate_migration_data_size(service, edge_server)
                        migration_energy = calculate_migration_energy(edge_server, migration_data_size)
                        energy_consumption['worst_fit'] += migration_energy

                        #Calculate energy consumption during task execution
                        compute_energy = calculate_compute_energy(edge_server, service)
                        energy_consumption['worst_fit'] += compute_energy
                        print(f"[WORST-FIT - STEP {parameters['current_step']}] Migrating {service} from {service.server} to {edge_server}")
                        service.provision(target_server=edge_server)
                        migration_counts['worst_fit'] += 1
                        break

# Define a function to calculate migration data size 
def calculate_migration_data_size(service, target_server):
    return service.attributes.data_size

# Define a function to calculate migration energy consumption
def calculate_migration_energy(edge_server, migration_data_size):
    # Extract power model parameters from the edge server
    max_power_consumption = edge_server.attributes.power_model_parameters["max_power_consumption"]
    static_power_percentage = edge_server.attributes.power_model_parameters["static_power_percentage"]

    # Calculate energy consumption during migration (linear power model)
    migration_energy = max_power_consumption * migration_data_size + static_power_percentage * max_power_consumption

    return migration_energy

# Define a function to calculate computational energy consumption during task execution
def calculate_compute_energy(edge_server, service):
    # Extract the maximum power consumption from the edge server attributes
    max_power_consumption = edge_server.attributes.power_model_parameters["max_power_consumption"]

    # Extract the static power percentage
    static_power_percentage = edge_server.attributes.power_model_parameters["static_power_percentage"]

    # Calculate the static power consumption (power consumption when CPU is idle) in watts
    baseline_power_consumption = max_power_consumption * static_power_percentage 

    # Power consumption under the specific workload (in watts)
    workload_power_consumption = edge_server.cpu_demand * edge_server.power_model_parameters["alpha"]

    # Compute power per compute
    power_per_compute = workload_power_consumption - baseline_power_consumption

    # Compute the energy consumption during task execution
    compute_workload = service.attributes.cpu_demand 
    compute_energy = power_per_compute * compute_workload

    return compute_energy

# Implementing the best-fit algorithm
def best_fit_algorithm(parameters):
    global migration_counts
    for service in Service.all():
        if not service.being_provisioned:
            edge_servers = sorted(
                EdgeServer.all(),
                key=lambda s: ((s.cpu - s.cpu_demand) * (s.memory - s.memory_demand)) ** 0.5,
                reverse=True,
            )
            for edge_server in edge_servers:
                if edge_server.has_capacity_to_host(service=service):
                    if service.server != edge_server:
                        print(f"[BEST-FIT - STEP {parameters['current_step']}] Migrating {service} from {service.server} to {edge_server}")
                        # Record the provision time when a service is provisioned
                        service.provision(target_server=edge_server)
                        service.provision_time = parameters['current_step']
                        migration_counts['best_fit'] += 1
                        break

# implementing the hybrid  migration algorithm
def hybrid_offloading_algorithm(parameters):
    global migration_counts

    # Let's iterate over the list of services using the 'all()' helper method
    print("\n\n")

    # Read data from the JSON file
    with open('datasets/sample_dataset1.json', 'r') as file:
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

    # Sort edge servers based on the decision factor
    edge_servers_info = edge_servers_info.sort_values(by='decision_factor', ascending=False)

    local_processing_threshold=0.5
    
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
                            migration_counts['hybrid'] += 1

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
simulator_hybrid = Simulator(
    tick_duration=1,
    tick_unit="seconds",
    stopping_criterion=stopping_criterion,
    resource_management_algorithm=hybrid_offloading_algorithm,
)

simulator_worst_fit = Simulator(
    tick_duration=1,
    tick_unit="seconds",
    stopping_criterion=stopping_criterion,
    resource_management_algorithm=worst_fit_algorithm,
)

simulator_best_fit = Simulator(
    tick_duration=1,
    tick_unit="seconds",
    stopping_criterion=stopping_criterion,
    resource_management_algorithm=best_fit_algorithm,
)

# Loading a sample dataset from GitHub for each simulator
simulator_hybrid.initialize(input_file="datasets/sample_dataset1.json")
simulator_worst_fit.initialize(input_file="datasets/sample_dataset1.json")
simulator_best_fit.initialize(input_file="datasets/sample_dataset1.json")

# Executing the simulations
simulator_hybrid.run_model()
simulator_worst_fit.run_model()
simulator_best_fit.run_model()

a = 1.0  # Baseline execution time
b = 0.5  # Coefficient for CPU demand

# Create a function to calculate execution time for a service on an edge server
def calculate_execution_time(edge_server, service):
    cpu_demand = service.attributes.cpu_demand
    max_cpu_capacity = edge_server.attributes.max_cpu_capacity

    #Normalize CPU and memory demands
    normalized_cpu_demand = cpu_demand / max_cpu_capacity
    execution_time = a + b * normalized_cpu_demand
    return execution_time

# Calculate process completion time for each service
process_completion_times = []

for service in Service.all():
    if service.server is not None:
        completion_time = service.provision_time + calculate_execution_time(service.server)
        process_completion_times.append(completion_time)

# Visualize process completion times
plt.hist(process_completion_times, bins=20, edgecolor='black')
plt.title('Process Completion Times')
plt.xlabel('Time')
plt.ylabel('Number of Services')
plt.show()

# Visualizing migration counts
labels = migration_counts.keys()
values = migration_counts.values()

plt.bar(labels, values)
plt.title('Service Migration Counts')
plt.xlabel('Migration Algorithm')
plt.ylabel('Number of Migrations')
plt.show()

