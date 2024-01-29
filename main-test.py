import random
import simpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import os
import time




def known_tasks(list):
    tasks = []
    for task_id, task_mt in enumerate(list):
        machine_time = task_mt  # Adjust the range as needed
        manual_time = 3 # Adjust the range as needed
        task = Task(task_id, machine_time, manual_time)
        tasks.append(task)
    return tasks


def upload_config_test(assembly_line):
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
        try:
            config_data = pd.read_excel(file_path, sheet_name="Line Data")
            config_line_globa_data = pd.read_excel(file_path, sheet_name="Config")
            print("Excel file uploaded and read successfully.")
            config_data_gloabl = config_line_globa_data.values.tolist()
            print(config_data_gloabl)

            assembly_line.stock_capacity = float(config_data_gloabl[2][2])
            assembly_line.stock_initial = float(config_data_gloabl[3][2])
            assembly_line.refill_time = float(config_data_gloabl[4][2])
            assembly_line.safety_stock = float(config_data_gloabl[5][2])
            assembly_line.refill_size = float(config_data_gloabl[6][2])

            assembly_line.supermarket_in = simpy.Container(env, capacity=assembly_line.stock_capacity, init=assembly_line.stock_initial)
            assembly_line.shop_stock_out = simpy.Container(env, capacity=float(assembly_line.config["shopstock"]["capacity"]), init=float(assembly_line.config["shopstock"]["initial"]))
            
            assembly_line.create_machines(config_data.values.tolist())
            
            try:
                assembly_line.sim_time = eval(str(config_data_gloabl[0][2]))
                assembly_line.yearly_volume_obj = eval(str(config_data_gloabl[1][2]))
            except:
                assembly_line.sim_time = int(config_data_gloabl[0][2])
                assembly_line.yearly_volume_obj = int(config_data_gloabl[1][2])
           
            return config_data
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return None
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
def run(assembly_line, experiment_number=1):
    assembly_line.run()
    assembly_line.get_results(save=True, experiment_number=experiment_number)
    #list_machines = assembly_line.get_track()

if __name__ == "__main__":

    
    df_tasks = pd.read_xml('./workplan_TestIsostatique_modified.xml', xpath=".//weldings//welding")
    tasks = known_tasks(df_tasks["cycleTime"].astype(int).tolist())
    config_file = 'config.yaml'
    #task_assignement = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3,  3, 3, 3, 3, 3, 3, 3 ]

    for i in range(1000):
        env = simpy.Environment()
        assembly_line = ManufLine(env, tasks, config_file=config_file)
        start_time = time.time()
        upload_config_test(assembly_line)
        run(assembly_line, i+1)
        
        print("--- %s seconds ---" % (time.time() - start_time))
