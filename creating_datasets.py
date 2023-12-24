#Importing EdgeSimpy components
from edge_sim_py import *
import networkx as nx
import msgpack
import os
import random
import copy

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

#custom power model 
class CustomPowerModel:
    def get_pwer_consumption(device):
        power_consumption = device.cpu_demand * device.power_model_parameters['alpha']
        return power_consumption

# create 10 edge servers with different computational capacities
edge_servers = []
for _ in range(10):
    edge_server = EdgeServer()
    # define individual computational capacity for each edge server
    edge_server.cpu = random.randint(5, 15)
    edge_server.memory = random.randint(2048, 8192)
    edge_server.disk = random.randint(51200, 204800)

    # power-related attributes
    edge_server.power_model_parameters = {
        "alpha": 2,
    }
    # specify the power consumption model
    edge_server.power_model =  CustomPowerModel

    # connect the edge server to a random base station
    base_stations_without_servers = [base_station for base_station in BaseStation.all() if len(base_station.edge_servers) == 0]
    if base_stations_without_servers:
        base_station = random.choice(base_stations_without_servers)
        base_station._connect_to_edge_server(edge_server=edge_server)

    edge_servers.append(edge_server)


#Creating the user with  a custom mobility model , which will move the user
#to  a random position at each beta time steps 
def user_mobility_model(user: object):
    #Gathering teh user's mobility model parameters. If no parameter was specified, set ""
    if hasattr(user, 'mobility_model_parameters') and "beta" in user.mobility_model_parameters:
        parameters = user.mobility_model_parameters
    else:
        parameters = {"beta": 1}
    # moving the user to random coordinates. 
    random_base_station = user.base_station
    while random_base_station == user.base_station:
        random_base_station = random.choice(BaseStation.all())

    #Setting the user's coordinates trace to the random base station position and
    #instructing edge simpy that the user will stay in that position for "beta time steps"

    new_coordinates = [random_base_station.coordinates for _ in range(parameters["beta"])]
    user.coordinates_trace.extend(new_coordinates)

# create users in a loop
num_users = 100
for _ in range(num_users):
    user = User()
    user.mobility_model = user_mobility_model
    user.mobility_model_parameters = {"beta": 3}
    user._set_initial_position(coordinates=random.choice(map_coordinates), 
                               number_of_replicates=user.mobility_model_parameters["beta"])
    
class EdgeServer(ComponentManager):

    ...

    def _to_dict(self) -> dict:
        access_patterns = {}
        for app_id, access_pattern in self.access_patterns.items():
            access_patterns[app_id] = {"class": access_pattern.__class__.__name__, "id": access_pattern.id}

        dictionary = {
            "attributes": {
                "id": self.id,
                "coordinates": self.coordinates,
                "delays": copy.deepcopy(self.delays),
            },
            "relationships": {
                "mobility_model": self.mobility_model.__name__,
                "base_station": {"class": type(self.base_station).__name__, "id": self.base_station.id},
                "applications": [{"class": type(app).__name__, "id": app.id} for app in self.applications],
            },
        }
        return dictionary

#exporting the dataset
dataset = ComponentManager.export_scenario(save_to_file='True', file_name='custom_dataset')




