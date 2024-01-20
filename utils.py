# utils.py
import os
import pandas as pd
import msgpack
import matplotlib.pyplot as plt
import networkx as nx
from edge_sim_py import *

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

def set_pandas_display_options():
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

def load_datasets(logs_directory):
    dataset_files = [file for file in os.listdir(logs_directory) if ".msgpack" in file]

    datasets = {}
    for file in dataset_files:
        with open(os.path.join(logs_directory, file), "rb") as data_file:
            datasets[file.replace(".msgpack", "")] = pd.DataFrame(msgpack.unpackb(data_file.read(), strict_map_key=False))

    return datasets

def display_user_mobility_traces(datasets):
    print("=== USER MOBILITY TRACES ===")
    users_coordinates = dict(datasets["User"].groupby('Object')['Coordinates'].apply(list))
    for user, mobility_logs in users_coordinates.items():
        print(f"{user}. Mobility Logs: {mobility_logs}")

    print("\n\n")

def plot_edge_servers_power_consumption(datasets):
    plt.figure(figsize=(10, 6))
    print("=== EDGE SERVERS' POWER CONSUMPTION ===")
    edge_servers_power_consumption = dict(datasets["EdgeServer"].groupby('Object')['Power Consumption'].apply(list))
    for edge_server, power_consumption in edge_servers_power_consumption.items():
        print(f"{edge_server}. Power Consumption per Step: {power_consumption}")

        plt.plot(power_consumption, label=edge_server)

    plt.title("Edge Servers' Power Consumption")
    plt.xlabel("Step")
    plt.ylabel("Power Consumption")
    plt.legend()
    plt.grid(True)
    plt.show()
