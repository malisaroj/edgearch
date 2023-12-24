#importing edgesimpy componenets and its built-in libraries networkx, msgpck
from edge_sim_py import *
import networkx as nx
import msgpack
import matplotlib as plt
import pandas as pd
import os

#Visualizing the dataset
def visualizing_dataset():
    #Customizing network visualization
    positions = {}
    labels = {}
    sizes = []
    colors = []

    for node in Topology.first().nodes():
        positions[node] = node.coordinates
        labels[node] = f"ID: {node.id}\n{node.coordinates}"

        if len(node.base_station.users) > 0:
            sizes.append(3500)
        else:
            sizes.append(1000)

        if len(node.base_station.edge_servers) > 0:
            colors.append("red")
        else:
            colors.append("black")

    #Drawing the network topology
    nx.draw(
        Topology.first(),
        pos=positions,
        node_color=colors,
        node_size=sizes,
        labels=labels,
        font_size=7,
        font_weight="bold",
        font_color="whitesmoke"
    )    

def parsing_simulation_logs():
    pd.options.display.max_columns = 1000
    pd.options.display.max_rows = 5000
    pd.options.display.max_colwidth = 199
    pd.options.display.width = 1000

    def highlight_rows(dataframe):
        df = dataframe.copy()
        mask = df['Time Step'] % 2 == 0
        df.loc[mask, :] = 'background-color: #222222; color: white'
        df.loc[~mask, :] = 'background-color: #333333; color: white'

        return df
    
    #Gathering the list of msgpack files in the current directory
    logs_directory = f"{os.getcwd()}/logs"
    dataset_files = [file for file in os.listdir(logs_directory) if ".msgpack" in file]

    #Reading msgpack files found
    datasets = {}
    for file in dataset_files:
        with open(f"logs/{file}", "rb") as data_file:
            datasets[file.replace(".msgpack", "")] = pd.DataFrame(msgpack.unpackb(data_file.read(), strict_map_key=False))

    datasets["EdgeServer"].style.apply(highlight_rows, axis=None)
    datasets["User"].style.apply(highlight_rows, axis=None)

    print("=== User Mobility Traces ===")
    users_coordinates = dict(datasets["User"].groupby('Object')['Coordinates'].apply(list))
    for user, mobility_logs in users_coordinates.items():
        print(f"{user}. Mobility Logs: {mobility_logs}")

    print("\n\n")

    print("=== Edge Servers' Power Consumption ===")
    edge_servers_power_consumption = dict(datasets["EdgeServer"].groupby('Object')['Power Consumption'])
    for edge_server, power_consumption in edge_servers_power_consumption.items():
        print(f"{edge_server}. Power Consumption Per Step: {power_consumption}")

    #Defining the data fram columns that will be exhibited
    properties = ['Coordinates', 'CPU Demand', 'RAM Demand', 'Disk Demand', 'Services']
    columns = ['Time Step', 'Instance ID'] + properties

    dataframe = datasets["EdgeServer"].filter(items=columns)
    dataframe


