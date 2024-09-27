import yaml
import simpy
import random
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional
import copy
import ast
import csv
import re
from utils import *



def run(manuf_line):
    """
    Run the manufacturing line and get the simulation results.
    
    :param manuf_line: Manufacturing line object.
    :return: Parts produced per machine, waiting times, cycle time, breakdowns.
    """
    manuf_line.run()
    return manuf_line.get_results(save=False, track=False)

def save_global_settings(manuf_line, configuration, references_config, line_data, buffer_sizes=[]):
    """
    Save global settings and configure the manufacturing line based on the input configuration.
    
    :param manuf_line: Manufacturing line object.
    :param configuration: Dictionary with various settings for the manufacturing line.
    :param references_config: Reference configurations.
    :param line_data: Configuration data for machines.
    :param buffer_sizes: Buffer sizes for machines (optional).
    """
    # Enable or disable breakdowns and random seed
    manuf_line.breakdowns_switch = configuration.get("enable_breakdowns", False)
    manuf_line.randomseed = configuration.get("enable_random_seed", False)

    # Configure the manufacturing line parameters
    manuf_line.stock_capacity = float(configuration["stock_capacity"])
    manuf_line.stock_initial = float(configuration["initial_stock"])
    manuf_line.reset_shift_dec = bool(configuration["reset_shift"])
    manuf_line.breakdown_law = str(configuration["breakdown_dist_distribution"])
    manuf_line.safety_stock = float(configuration["safety_stock"])
    manuf_line.refill_size = float(configuration["refill_size"])
    manuf_line.n_robots = float(configuration["n_robots"])
    manuf_line.n_repairmen = int(configuration["n_repairmen"])
    
    # Set repairmen as preemptive resources and stock containers
    manuf_line.repairmen = simpy.PreemptiveResource(manuf_line.env, capacity=manuf_line.n_repairmen)
    manuf_line.supermarket_in = simpy.Container(manuf_line.env, capacity=manuf_line.stock_capacity, init=manuf_line.stock_initial)
    manuf_line.shop_stock_out = simpy.Container(manuf_line.env, capacity=float(manuf_line.config["shopstock"]["capacity"]), init=float(manuf_line.config["shopstock"]["initial"]))

    # Set simulation time and takt time
    manuf_line.sim_time = float(configuration["sim_time"])
    manuf_line.takt_time = float(configuration["takt_time"])

    manuf_line.references_config = references_config
    manuf_line.machine_config_data = line_data

    # Update buffer sizes if provided
    if buffer_sizes:
        for i in range(len(manuf_line.machine_config_data)):
            manuf_line.machine_config_data[i][3] = buffer_sizes[i]

    # Create machines in the manufacturing line based on the configuration
    manuf_line.create_machines(manuf_line.machine_config_data)

def buffer_optim_costfunction(buffer_sizes, configuration, references_config, line_data):
    """
    Cost function to optimize buffer sizes in the manufacturing line.
    
    :param buffer_sizes: List of buffer capacities for each machine.
    :param configuration: Manufacturing line configuration.
    :param references_config: Reference configurations.
    :param line_data: Configuration data for machines.
    :return: Simulation results (parts produced, waiting times, cycle time, breakdowns).
    """
    tasks = []
    env = simpy.Environment()
    manuf_line = ManufLine(env, tasks, config_file='config.yaml')

    save_global_settings(manuf_line, configuration, references_config, line_data, buffer_sizes)
    return run(manuf_line)

def function_to_optimize(buffer_capacities, configuration, references_config, line_data, waiting_ref, invent_cost=10, unit_revenue=100):
    """
    Objective function to optimize buffer capacities for a manufacturing line.
    
    :param buffer_capacities: List of buffer capacities  (candidates that we will optimize)
    :param configuration: Manufacturing line configuration.
    :param references_config: Reference configurations.
    :param line_data: Configuration data for machines.
    :param waiting_ref: Reference waiting times for comparison.
    :param invent_cost: Inventory cost per unit of buffer (default=10).
    :param unit_revenue: Revenue per unit of product (default=100).
    :return: Total cost and waiting time differences.
    """
    sim_results, waiting_times, cycle_time, breakdowns = buffer_optim_costfunction(buffer_capacities, configuration, references_config, line_data)
    
    result_values = []
    
    # Calculate waiting time differences (blockage and starvation)
    for i in range(len(waiting_times)):
        if i == len(sim_results) - 1:  # Last element only considers blockage
            result_values.append(waiting_times[i][1] - waiting_ref[i][1])
        else:
            starvation_diff = waiting_times[i+1][0] - waiting_ref[i+1][0]
            blockage_diff = waiting_times[i][1] - waiting_ref[i][1]
            result_values.append((blockage_diff + starvation_diff) / 2)

    # If no buffer candidate was given, start with [1, 1, 1, ...]
    if not buffer_capacities:
        buffer_capacities = [1] * len(sim_results)
        
    # Calculate inventory costs
    inventory_cost = [invent_cost * i for i in buffer_capacities]

    # Calculate total cost (negative revenue + inventory cost)
    total_cost = [-unit_revenue * a_i / cycle_time - b_i for a_i, b_i in zip(result_values, inventory_cost)]
    
    print(f"Buffer Capacities = {buffer_capacities} - Waiting times = {result_values}")
    print(f"Total Cost: {total_cost}")

    return total_cost, result_values
