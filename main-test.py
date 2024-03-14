import random
import simpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim



def known_tasks(list):
    tasks = []
    for task_id, task_mt in enumerate(list):
        machine_time = task_mt  # Adjust the range as needed
        manual_time = 3 # Adjust the range as needed
        task = Task(task_id, machine_time, manual_time)
        tasks.append(task)
    return tasks


def upload_config_test(assembly_line, buffer_size_list=[]):
    # Ask the user to select a file
    
    if os.path.exists("./LineData.xlsx"):
        file_path = "./LineData.xlsx"
    else:
        file_path = filedialog.askopenfilename(
            title="Select a file",
            filetypes=[("Excel files", "*.xlsx;*.xls"), ("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not file_path:
            print("No file selected. Exiting.")
            return None

    # Check the file extension to determine the file type
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        # Read Excel file using pandas
        #try:
        config_data = pd.read_excel(file_path, sheet_name="Line Data")
        config_line_globa_data = pd.read_excel(file_path, sheet_name="Config")
        config_data_gloabl = config_line_globa_data.values.tolist()

        assembly_line.stock_capacity = float(config_data_gloabl[2][2])
        assembly_line.stock_initial = float(config_data_gloabl[3][2])
        assembly_line.refill_time = float(config_data_gloabl[4][2])
        assembly_line.safety_stock = float(config_data_gloabl[5][2])
        assembly_line.refill_size = float(config_data_gloabl[6][2])

        assembly_line.supermarket_in = simpy.Container(assembly_line.env, capacity=assembly_line.stock_capacity, init=assembly_line.stock_initial)
        assembly_line.shop_stock_out = simpy.Container(assembly_line.env, capacity=float(assembly_line.config["shopstock"]["capacity"]), init=float(assembly_line.config["shopstock"]["initial"]))
        
        
        machine_data = config_data.values.tolist()
        if buffer_size_list != []:
            for i in range(len(machine_data)):
                machine_data[i][5] = buffer_size_list[i]
        

        assembly_line.create_machines(machine_data)
        try:
            assembly_line.sim_time = eval(str(config_data_gloabl[0][2]))
            assembly_line.yearly_volume_obj = eval(str(config_data_gloabl[1][2]))
        except:
            assembly_line.sim_time = int(config_data_gloabl[0][2])
            assembly_line.yearly_volume_obj = int(config_data_gloabl[1][2])
    
        return config_data
        # except Exception as e:
        #     print(f"Error reading Excel file: {e}")
        #     return None
    elif file_path.endswith('.json'):
        # Read JSON file
        try:
            with open(file_path, 'r') as json_file:
                config_data = json.load(json_file)
            print("JSON file uploaded and read successfully.")
            return config_data
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return None
    else:
        print("Unsupported file type. Please upload an Excel (.xlsx or .xls) or JSON file.")
        return None


def change_buffers_size(size_list, assembly_line):
    '''
    Takes a list of proposed sizes of buffer [1, 1, 1, 1, 1, 1] and reconfigure the line based on that.  
    '''

    for i, machine in enumerate(assembly_line.list_machines):
        machine.buffer_out = simpy.Container(assembly_line.env, capacity=float(size_list[i]), init=0)


def run(assembly_line, experiment_number=1, save=False, track=False):
    assembly_line.run()

    waiting_times,cycle_time, breakdowns  = assembly_line.get_results(save=save, track=track, experiment_number=experiment_number)
    return waiting_times, cycle_time, breakdowns
   

def buffer_optim_costfunction(variables):
    variables = [max(int(b), 1) for b in variables]

    env = simpy.Environment()
    assembly_line = ManufLine(env, tasks, config_file=config_file)
    upload_config_test(assembly_line, buffer_size_list=variables)
    waiting_times, cycle_time, breakdowns= run(assembly_line)

    return waiting_times, cycle_time, breakdowns

def function_to_optimize(buffer_capacities):
    """
    Define the function that you want to optimize.
    """
    sim_results, cycle_time, breakdowns = buffer_optim_costfunction(buffer_capacities)
    result_values = []

    for i in range(len(sim_results)):
        if i == len(sim_results) - 1:  # If it's the last element, just take its current value
            result_values.append(sim_results[i])
        else:
            result_values.append(sim_results[i] + sim_results[i+1])

    inventory_cost = [10*i for i in buffer_capacities]
    results_values_cost = [-100 * a_i / 250 + b_i for a_i, b_i in zip(result_values, inventory_cost)]

    #total_cost = np.sum(results_values_cost)
    total_cost = results_values_cost
    print("Total Cost : ", total_cost)

    return total_cost


def function_to_optimize_torch(buffer_capacities_tensors):

    buffer_capacities_tensor = torch.stack(buffer_capacities_tensors)
    
    # Calculate waiting times per machine
    buffer_capacities = [buffer_capacity.detach().numpy().astype(int) for buffer_capacity in buffer_capacities_tensors]
    sim_results, cycletime, _ = buffer_optim_costfunction(buffer_capacities)
    
    result_values = [(sim_results[i] + sim_results[i+1]) / cycletime if i < len(sim_results) - 1 else sim_results[i] / cycletime for i in range(len(sim_results))]
    result_values_tensor = torch.tensor(result_values, dtype=torch.float32)
    
    waiting_times_per_buffer = [result_values_tensor[i]/buffer_capacities_tensor[i] for i in range(len(result_values_tensor))]
    total_waiting_time = torch.sum(torch.stack(waiting_times_per_buffer))

    total_cost = torch.sum(torch.mul(buffer_capacities_tensor, 10))
    print("total cost = ", total_cost)
    print("total waiting time = ", total_waiting_time)
    total_loss = total_cost + total_waiting_time    
    return total_loss


def function_to_optimize_movingavg(buffer_capacities):
    """
    Define the function that you want to optimize.
    """
    sim_results, cycle_time, breakdowns = buffer_optim_costfunction(buffer_capacities)
    result_values = []

    for i in range(len(sim_results)):
        if i == len(sim_results) - 1:  
            result_values.append(sim_results[i]/(breakdowns[i]+1))
        else:
            result_values.append((sim_results[i] + sim_results[i+1])/(breakdowns[i]+breakdowns[i+1]+1))

    
    inventory_cost = [10 * i for i in buffer_capacities]
    results_values_cost = [100 * a_i / cycle_time + b_i for a_i, b_i in zip(result_values, inventory_cost)]

    inventory_cost = torch.tensor([10*i for i in buffer_capacities], dtype=torch.float32)
    result_values_tensor = torch.tensor(results_values_cost, dtype=torch.float32)
    
    # Compute gradients for inventory_cost and result_values_tensor
    inventory_cost.requires_grad_(True)
    result_values_tensor.requires_grad_(True)
    total_cost = torch.sum(results_values_cost)
    total_cost.requires_grad_(True)
    return np.sum(results_values_cost)

def finite_perturbation_analysis(function, buffer_capacities, perturbation_value=10):
    """
    Perform Finite Perturbation Analysis to estimate the gradient of the function with respect to buffer capacities.
    """
    gradient_estimate = np.zeros_like(buffer_capacities)

    for i in range(len(buffer_capacities)):
        buffer_capacities_plus = buffer_capacities.copy()
        buffer_capacities_plus[i] += perturbation_value
        buffer_capacities_minus = buffer_capacities.copy()
        buffer_capacities_minus[i] = max(buffer_capacities_minus[i]-perturbation_value, 1)

        function_plus = function(buffer_capacities_plus)
        function_minus = function(buffer_capacities_minus)

        gradient_estimate[i] = (function_plus[i] - function_minus[i]) / (2 * perturbation_value)
        if gradient_estimate[i] > 200:
            gradient_estimate[i]= 500
        elif gradient_estimate[i] < -200:
            gradient_estimate[i]= -200

    return gradient_estimate, min(function_minus, function_plus)

def optimize_buffer_capacities(initial_buffer_capacities, iterations=100, learning_rate=0.1):
    """
    Optimize buffer capacities using Finite Perturbation Analysis.
    """
    buffer_capacities = initial_buffer_capacities.copy()
    gradient_tracks = []
    costfunction_tracks = []
    print("Starting the Optimization")
    for i in range(iterations):
        start =  time.time() 
        gradient, cost = finite_perturbation_analysis(function_to_optimize, [max(int(b), 1) for b in buffer_capacities])
        buffer_capacities -= learning_rate * gradient
        print("Iteration -- " + str(i) + " -- Current capacities = ", [max(int(b), 1) for b in buffer_capacities])
        print("Time required per iteration = ",  time.time() -start)
        gradient_tracks.append(gradient)
        costfunction_tracks.append(cost)
        

    return [max(int(b), 1) for b in buffer_capacities], costfunction_tracks, gradient_tracks


if __name__ == "__main__":

    
    df_tasks = pd.read_xml('./workplan_TestIsostatique_modified.xml', xpath=".//weldings//welding")
    tasks = known_tasks(df_tasks["cycleTime"].astype(int).tolist())
    config_file = 'config.yaml'
    #task_assignement = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3,  3, 3, 3, 3, 3, 3, 3 ]
    
    
    # for i in range(100):
    #     start = time.time()
    
    
    variables = [500, 500, 500, 500, 500, 500, 500]

    env = simpy.Environment()
    assembly_line = ManufLine(env, tasks, config_file=config_file)
    upload_config_test(assembly_line, variables)
    #run(assembly_line, save=True, track=True)
    assembly_line.initialize()

    
    actions = [
        [assembly_line.supermarket_in, assembly_line.list_machines[0]],
        [assembly_line.supermarket_in, assembly_line.list_machines[1]],
        [assembly_line.list_machines[0], assembly_line.list_machines[2]],
        [assembly_line.list_machines[0], assembly_line.list_machines[2]],
        [assembly_line.supermarket_in, assembly_line.list_machines[0]],
        [assembly_line.supermarket_in, assembly_line.list_machines[0]],
        [assembly_line.list_machines[2], assembly_line.list_machines[3]],
        [assembly_line.list_machines[0], assembly_line.list_machines[2]],
        [assembly_line.list_machines[3], assembly_line.list_machines[4]],
    ]

    for action in actions:
        # Run each action
        assembly_line.run_action(action)
        for m in assembly_line.list_machines:
            print(m.ID + " - " + str(m.buffer_in.level) + " | " + str(m.buffer_out.level) + "   -- " + str(m.waiting_time))


    # cycle_times = []
    # for i in range(100):
    #     start = time.time()
        
    #     #variables = [80, 100, 1, 1, 1, 1, 1]
    #     variables2 = [206, 40, 1, 1, 1, 1, 1]
    #     variables3 = [170, 47, 21, 1, 1, 1, 1]
    #     env = simpy.Environment()
    #     assembly_line = ManufLine(env, tasks, config_file=config_file)
    #     upload_config_test(assembly_line, variables2)
    #     print("start sim -- ")
    #     waiting_times, cycle_time, breakdowns= run(assembly_line, save=True, track=True)
    #     print("Time required per iteration = ",  time.time() -start)
    #     print("Waiting time = ", waiting_times)
    #     print("Cycle Time = ", cycle_time)
    #     cycle_times.append(cycle_time)

    # plt.boxplot(cycle_times)
    # plt.show()
    # buffer_cap = [1, 1, 1, 1, 1, 1, 1]
    # print(function_to_optimize(buffer_cap))
    
    # buffer_cap = [49, 165, 1, 18, 1, 16, 1]
    # print(function_to_optimize(buffer_cap))

    #TODO: Use sample to estimate the gradient instead of one simulation.

    # optimized_buffer_capacities_list = []
    # for i, lr in enumerate([0.1, 0.01]):
    #     for j in range(40): ##### 
    #         initial_buffer_capacities = [1 for _ in range(7)]
    #         optimized_buffer_capacities, costfunction_tracks, gradient_tracks = optimize_buffer_capacities(initial_buffer_capacities,iterations=100, learning_rate=lr)
    #         optimized_buffer_capacities_list.append(optimized_buffer_capacities)
    #         csv_file_path = './results/buffer_capacities_optim.csv'

    #         with open(csv_file_path, 'a', newline='') as csvfile:
    #             writer = csv.writer(csvfile)
    #             if i == 0 and j == 0:
    #                 writer.writerow(["Learning Rate", "Best Buffer Capacities", "Cost Function Tracks", "Gradient Tracks"])
    #             writer.writerow([lr, optimized_buffer_capacities, costfunction_tracks, gradient_tracks])
                
    # print("Optimized Buffer Capacities:", optimized_buffer_capacities_list)
    
   

"""
    Ideas for optimization modeling:
        - Variables: 
            -- buffers capacity and how they can affect the cycle time at the end.
            -- priority to be chosen: state [buffers states of each machine] => actions [which one to take] => reward [CT evaluation and also if the flow is blocked]
            -- 
        - 

 """


