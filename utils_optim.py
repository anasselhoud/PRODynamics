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



def run(manuf_line, save=False, track=False):
    manuf_line.run()
    waiting_times,cycle_time, breakdowns  = manuf_line.get_results(save=save, track=track)

    return waiting_times, cycle_time, breakdowns
   

def save_global_settings(manuf_line, configuration, references_config, line_data, buffer_sizes=[]):

    if configuration["enable_breakdowns"]:
        manuf_line.breakdowns_switch = True
    else:
        manuf_line.breakdowns_switch = False

    if configuration["enable_random_seed"]:
        manuf_line.randomseed = True
    else:
        manuf_line.randomseed = False

    available_strategies = ["Balanced Strategy", "Greedy Strategy"]

    manuf_line.stock_capacity = float(configuration["stock_capacity"])
    manuf_line.stock_initial = float(configuration["initial_stock"])
    manuf_line.reset_shift_dec = bool(configuration["reset_shift"])
    manuf_line.breakdown_law = str(configuration["breakdown_dist_distribution"])
    
    manuf_line.safety_stock = float(configuration["safety_stock"])
    manuf_line.refill_size = float(configuration["refill_size"])
    manuf_line.n_robots = float(configuration["n_robots"])
    manuf_line.n_repairmen = int(configuration["n_repairmen"])
    manuf_line.robot_strategy = int(available_strategies.index(configuration["strategy"]))
    manuf_line.repairmen = simpy.PreemptiveResource(manuf_line.env, capacity=int(configuration["n_repairmen"]))
    manuf_line.supermarket_in = simpy.Container(manuf_line.env, capacity=manuf_line.stock_capacity, init=manuf_line.stock_initial)
    manuf_line.shop_stock_out = simpy.Container(manuf_line.env, capacity=float(manuf_line.config["shopstock"]["capacity"]), init=float(manuf_line.config["shopstock"]["initial"]))
    
    manuf_line.sim_time = eval(str(configuration["sim_time"]))
    print("sim time first = ",  manuf_line.sim_time)
    manuf_line.takt_time = eval(str(configuration["takt_time"]))

    manuf_line.references_config = references_config
    manuf_line.machine_config_data = line_data

    print("buffer_sizes", buffer_sizes)
    if buffer_sizes != []:
        for i in range(len(manuf_line.machine_config_data)):
            manuf_line.machine_config_data[i][3] = buffer_sizes[i]

    manuf_line.create_machines(manuf_line.machine_config_data)

def buffer_optim_costfunction(buffer_sizes, configuration, references_config, line_data):
    buffer_sizes = [max(int(100), 1) for b in range(len(line_data))]
    tasks = []
    config_file = 'config.yaml'
    env = simpy.Environment()
    manuf_line = ManufLine(env, tasks, config_file=config_file)

    save_global_settings(manuf_line, configuration, references_config, line_data, buffer_sizes)
    waiting_times, cycle_time, breakdowns= run(manuf_line)

    return waiting_times, cycle_time, breakdowns



def function_to_optimize(buffer_capacities, configuration, references_config, line_data, invent_cost = 10, unit_revenue=100):
    """
    Define the function that you want to optimize.
    """
    sim_results, cycle_time, breakdowns = buffer_optim_costfunction(buffer_capacities, configuration, references_config, line_data)
    result_values = []

    for i in range(len(sim_results)):
        if i == len(sim_results) - 1:  # If it's the last element, just take its current value
            result_values.append(sim_results[i])
        else:
            result_values.append(sim_results[i] + sim_results[i+1])
    

    if buffer_capacities == []:
        buffer_capacities = [1 for _ in range(len(sim_results))]
    inventory_cost = [invent_cost*i for i in buffer_capacities]
    results_values_cost = [-unit_revenue * a_i + b_i for a_i, b_i in zip(result_values, inventory_cost)]

    #total_cost = np.sum(results_values_cost)
    total_cost = results_values_cost
    print("Total Cost : ", total_cost)

    return total_cost, result_values