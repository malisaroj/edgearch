#Importing Edgesimpy components
from edge_sim_py import *
import numpy as np
import tensorflow as tf
import json 
from sklearn.preprocessing import StandardScaler
import pandas as pd


#Load the JSON data
with open('datasets/sample_dataset1.json', 'r') as file:
    data = json.load(file)

# Extract the relevant attributes and relationships
flat_data_pred = pd.json_normalize(data['Service'])

# Drop unnecessary columns
columns_to_drop_pred = ["attributes.label", "relationships.application.class", "attributes._available", "relationships.server.class", "attributes.image_digest"]
flat_data_pred.drop(columns=columns_to_drop_pred, inplace=True)

#Fill missing values with mean
flat_data_pred.fillna(flat_data_pred.mean(), inplace=True)

#Scale the features 
scaler = StandardScaler()
scaled_features_pred = scaler.fit_transform(flat_data_pred)

#Reshape the input features to add a third dimension for time steps
X_pred_reshaped = tf.reshape(scaled_features_pred, (scaled_features_pred.shape[0], 1, scaled_features_pred.shape[1]))
task_offloading_decision = 1

def predict_resource_demand(X_pred_reshaped):
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

    #print the predictions DataFrame
    print(predictions_pred_df)

predict_resource_demand(X_pred_reshaped)

#implementing the migration algorithm
def my_algorithm(parameters):
    #iterating over the list of services 
    for service in Service.all():
        #Check whether the services is being already migrated or not,  Thing is we don
        #want to migrate the already migrated services
        if service.server is None and not service.being_provisioned:

            #Predict task offloading decision using BiLSTM
            predict_resource_demand(X_pred_reshaped)            

            if task_offloading_decision == 1:
                #sorting the edge server based on amount of free resources 
                edge_servers = sorted(
                                    EdgeServer.all(),
                                    key=lambda s: ((s.cpu - s.cpu_demand) * (s.memory - s.memory_demand) * (s.disk - s.disk_demand)) ** (1 / 3),
                                    reverse=True,  
                                ) 

                #Apply hybrid Worst-Fit and Best-Fit heuristic for offloading
                for edge_server in edge_servers:
                    #Checking if the edge server has resources to host the service
                        if edge_server.has_capacity_to_host(service=service):
                        #Migrate service if it's not already in the least occupied edge server
                            if service.server != edge_server:
                                print("Migrating the task to the target server using First-Fit heuristic")

                                print(f"[STEP {parameters['current_step']}] Migrating {service} From {service.server} to {edge_server}")

                                service.provision(target_server=edge_server)
                                #After start migrating the service we can move on to the next service
                                break

            else:
                #Perform local processing if task_offloading_decision is 0
                print("Processing the task locally")

