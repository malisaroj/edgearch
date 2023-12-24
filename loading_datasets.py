# Importing EdgeSimPy components
from edge_sim_py import *
from placement_algorithm import cool_resource_management_policy
from manipulating_components import my_algorithm, stopping_criterion
from creating_datasets import CustomPowerModel, user_mobility_model
from utils import parsing_simulation_logs, visualizing_dataset

# Instantiating the simulator
simulator = Simulator(
    dump_interval=5,
    tick_unit="minutes",
    tick_duration=1,
    stopping_criterion=stopping_criterion,
    resource_management_algorithm=cool_resource_management_policy,
    user_defined_functions=[CustomPowerModel, user_mobility_model]
)

# Loading the dataset file from the external JSON filec
simulator.initialize(input_file="datasets\custom_dataset.json")

# Displaying some of the objects loaded from the dataset
for user in User.all():
    print(f"{user}. Coordinates: {user.coordinates}")

for edge_server in EdgeServer.all():
    print(f"{edge_server}. CPU Capacity: {edge_server.cpu}  cores")

#Checking the placement output
for service in Service.all():
    print(f"{service}. Host: {service.server}")

#Visualizing the datasets
visualizing_dataset()

#Analyzing the simulation results
parsing_simulation_logs()