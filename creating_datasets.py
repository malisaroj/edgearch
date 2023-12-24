#Importing EdgeSimpy components
from edge_sim_py import *
import networkx as nx
import msgpack
import os
import random
import copy
import matplotlib.pyplot as plt


#creating dataset
#creating the hexagonal cell of given map size
map_coordinates = hexagonal_grid(x_size=100, y_size=100)
#creating the base station and network switch for the giveen map coordinates

for coordinates in map_coordinates:
    #creating the basestation
    base_station = BaseStation()
    base_station.wireless_delay = 0
    base_station.coordinates = coordinates

    #creating network switch
    network_switch = sample_switch()
    base_station._connect_to_network_switch(network_switch=network_switch)

partially_connected_hexagonal_mesh(
    network_nodes=NetworkSwitch.all(),
    link_specifications=[
        {"number_of_objects": 29601, "delay": 1, "bandwidth": 10},
    ],

)


# create 10 edge servers with different computational capacities
edge_servers = []
for _ in range(10):
    edge_server = EdgeServer()
    # define individual computational capacity for each edge server
    edge_server.cpu = random.randint(5, 15)
    edge_server.memory = random.randint(2048, 8192)
    edge_server.disk = random.randint(51200, 204800)
    

    # Power-related attributes
    edge_server.power_model_parameters = {
        "max_power_consumption": 110,
        "static_power_percentage": 0.1,
    }

    # Specifying the edge server's power model
    edge_server.power_model = LinearServerPowerModel

    # Connecting the edge server to a random base station with no attached edge server
    base_stations_without_servers = [base_station for base_station in BaseStation.all() if len(base_station.edge_servers) == 0]
    base_station = random.choice(base_stations_without_servers)
    base_station._connect_to_edge_server(edge_server=edge_server)

    edge_servers.append(edge_server)


# create users in a loop
num_users = 100
for _ in range(num_users):
    user = User()
    user.mobility_model = pathway
    user._set_initial_position(coordinates=random.choice(map_coordinates))

#exporting the dataset
dataset = ComponentManager.export_scenario(save_to_file='True', file_name='custom_dataset')




