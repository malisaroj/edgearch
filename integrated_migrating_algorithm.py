#Importing Edgesimpy components
from edge_sim_py import *
import numpy as np
import tensorflow as tf
import json 
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from pathlib import Path
import matplotlib.pyplot as plt
import msgpack
import os
from utils import set_pandas_display_options, highlight_rows, load_datasets, display_user_mobility_traces, plot_edge_servers_power_consumption

# Set Pandas display options
set_pandas_display_options()

# Placeholder values
initial_number_of_resources = 10
U_th = 80  # Upper workload threshold triggering scaling out
L_th = 20  # Lower workload threshold triggering scaling in
Delta_N_out = 2  # Number of resources to add during scaling out
Delta_N_in = 1  # Number of resources to remove during scaling in
T_th = 0  # Threshold for local processing in the task offloading algorithm
min_number_of_resources = 5

def measure_and_update_workload(parameters):
    current_cpu_workload = 0
    current_memory_workload = 0

    # Iterate over all services and update the workload based on their resource demands
    for service in Service.all():
        if service.server is not None:
            current_cpu_workload += service.cpu_demand 
            current_memory_workload += service.memory_demand

    # Combine CPU and memory workloads using a weighted sum (you can adjust the weights)
    combined_workload = 0.7 * current_cpu_workload + 0.3 * current_memory_workload

    parameters['current_workload'] = combined_workload

    return parameters['current_workload']

def PredictResourceRequirements():

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

    df_for_prediction.fillna(df_for_prediction.mean(), inplace=True)

    # Scale the features 
    scaler = StandardScaler()
    scaled_features_pred = scaler.fit_transform(df_for_prediction)

    # Reshape the input features to add a third dimension for time steps
    X_pred_reshaped = tf.reshape(scaled_features_pred, (scaled_features_pred.shape[0], 1, scaled_features_pred.shape[1]))

    # Load the saved model
    model_save_path = Path(".cache") / "trained_model" 
    model = tf.keras.models.load_model(model_save_path)

    # Make predictions on the new data
    predictions_pred = model.predict(X_pred_reshaped)

    # Reshape the predictions to match the original shape of labels
    predictions_pred = np.reshape(predictions_pred, (predictions_pred.shape[0], 2))

    # Convert the Numpy arrays back to a Pandas DataFrame for better analysis or comparision
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

    sorted_edge_servers_info = edge_servers_info.sort_values(by='decision_factor', ascending=False)


    # Determine the decision factor (Use the geometric mean to account for both CPU and memory utilization in a multiplicative manner.)
    #edge_servers_info['decision_factor'] = np.sqrt(edge_servers_info['predicted_cpus'] * edge_servers_info['predicted_memory'])

    return sorted_edge_servers_info


# Integrated Offloading and Dynamic Scaling Algorithm
def integrated_offloading_and_scaling_algorithm(parameters):
    global migration_counts, ProvisionedCount

    # Initialization
    N = initial_number_of_resources  # actual initial number of resources
    ProvisionedCount = 0

    # Continuous Monitoring (function to measure and update workload W(t))
    W_t = measure_and_update_workload(parameters)

    # Workload Analysis (analyze workload patterns if needed)

    # Scaling Decision
    scaling_decision = ScalingDecision(W_t, U_th, L_th)

    # Elastic Resource Provisioning
    N = ElasticResourceProvisioning(N, Delta_N_out, Delta_N_in, scaling_decision)

    # Task Offloading Decision
    decision_factors = PredictResourceRequirements()
    sorted_edge_servers_info = decision_factors

    for service in Service.all():
        print(f"Service: {service}, Server: {service.server}")
        offloading_action = None  # Initialize offloading_action variable

        if not service.being_provisioned:
            #P_decision = decision_factors[service.server]  # Retrieve decision factor for the current service
            P_decision = decision_factors

            offloading_action = OffloadingDecision(P_decision, T_th)

            if offloading_action == "Process Locally":
                # Process the service locally
                print(f"[STEP {parameters['current_step']}] Processing {service} locally on {service.server}")
            elif offloading_action == "Migrate":
                for edge_server_info in sorted_edge_servers_info.itertuples():
                    edge_server = edge_server_info.EdgeServer
                    if edge_server.has_capacity_to_host(service=service):
                        if service.server != edge_server:
                            # Provision service to edge server
                            print(f"[STEP {parameters['current_step']}] Migrating {service} from {service.server} to {edge_server}")
                            service.provision(target_server=edge_server)

                            break


        # Handle the case where offloading_action is still None
        if offloading_action is None:
            print(f"[WARNING] offloading_action is not assigned. Skipping decision for {service}.")


# Function to make scaling decisions based on workload metrics and thresholds
def ScalingDecision(W_t, U_th, L_th):
    if W_t > U_th:
        return "Scale Out"
    elif W_t < L_th:
        return "Scale In"
    else:
        return "No Change"

# Function to adjust the number of active resources based on scaling decisons
def ElasticResourceProvisioning(N, Delta_N_out, Delta_N_in, scaling_decision):
    if scaling_decision == "Scale Out":
        N += Delta_N_out
    elif scaling_decision == "Scale In":
        N -= Delta_N_in
        if N < min_number_of_resources:
            N = min_number_of_resources  # Ensure minimum number of resources
    # No change if scaling_decision is "No Change"
    return N

# Function to decide whether to process a service locally or migrate based on the task offloading algorithm
def OffloadingDecision(P_decision, T_th):
    if P_decision['decision_factor'].iloc[0] < T_th:

        return "Process Locally"
    else:
        return "Migrate"

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
    resource_management_algorithm=integrated_offloading_and_scaling_algorithm,
)

# Loading a sample dataset from datasets for each simulator
simulator_hybrid.initialize(input_file="datasets/sample_dataset2.json")

# Executing the simulations
simulator_hybrid.run_model()

# Gathering the list of msgpack files in the current directory
logs_directory = f"{os.getcwd()}/logs"
datasets = load_datasets(logs_directory)

# Display user mobility traces
display_user_mobility_traces(datasets)

# Plot edge servers' power consumption
plot_edge_servers_power_consumption(datasets)