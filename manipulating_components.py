#mporting edge simpy compnenets
from edge_sim_py import *
#Importing Python libraries
import os
import random
import msgpack
import pandas as pd

def assign_priority(task):
    #Assign priority based on the inverse of the deadline
    #Higher priority for tasks with earlier deadlines
    priority = 1 / task['deadline']
    return priority

def check_premption_feasibility(current__task, new_task):
    #Check if the priority of the new task is higher than the current task
    if new_task['priority'] > current__task['priority']:
        return True
    

def prempt_task(task):
    #Finding the lowest priority task currently executing
    lowest_priority_task = min(tasks, key=lambda x: x['priority'])

    #Check if the new task has a higher priority than the lowest priority task
    if new_priority > lowest_priority_task['priority']:


def dynamic_adjustment_after_execution(priorities, local_utilization, remote_utilization):
    for task in priorities:
        priorities[task] = adjust_priority_after_execution(priorities[task], local_utilization, remote_utilization)

def offloading_decision(tasks, local_resources):
    offload_tasks = select_tasks_to_offload(tasks, local_resources)
    for task in offload_tasks:
        if not local_resources.can_accomodate(task):
            feasible = check_preemption_feasibilty(task)
            if feasible:
                preempt_low_priority_task(task)

def deadline_compliance_check(tasks, completions):
    for task in tasks:
        check_deadline_complinance(task, completions[task])





def my_algorithm(parameters: dict):
    current_time_step = parameters["current_step"]
    for service in Service.all():
        service['priority'] = assign_priority(service)
    #sort tasks based on priority(higher priority first)
    sorted_service = sorted(service, key=lambda x:x['priority'], reverse=True)
    for service in sorted_service:
        if service.server == None and not service.being_provisioned:
            edge_servers = sorted(EdgeServer.all(),
                                  key = lambda a : ((a.cpu - a.cpu.demand)*
                                                    (a.disk -a.disk.demand)*
                                                    (a.memory-a.memory.demand) ** 1/3),
                                  reverse = True)
            for  edge_server in edge_servers():
                edge_server.cpu_demand = current_time_step

                if edge_server.has_capacity_to_host(service=service):
                    if service.server != edge_server:
                        print("migrating the tasks to the target server")
                        service.provision(target_server=edge_server)
                        break


def stopping_criterion():
    provisioned_services = 0
    for service in Service.all():
        if service.server != None:
            provisioned_services += 1

    return provisioned_services == Service.Count()

#Checking the placement output
for service in Service.all():
    print(f"{service}. Host: {service.server}")