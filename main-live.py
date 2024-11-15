import random
import simpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from matplotlib.animation import FuncAnimation
import tkinter as tk
import threading
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from collections import deque
import tkinter
import tkinter.messagebox
import customtkinter as ctk
import customtkinter
import re
import multiprocessing
from time import time
from datetime import datetime
#from chart_studio.widgets import GraphWidget


env = simpy.Environment()
global simulation_number
simulation_number = 1
customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class SupportWindow(customtkinter.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("500x300")
        self.resizable(width=False, height=False)
        self.title("Support") 
        self.textbox = customtkinter.CTkTextbox(master=self, width=450, height=250, corner_radius=0, font=("Arial", 18), wrap="word")
        self.textbox.grid(row=1, column=1, padx=(25, 25), pady=(25,25), sticky="nsew")
        self.textbox.tag_config("center", justify="center")
        self.textbox.insert("0.0", "\n This work was completed within the context of a CIFRE PhD Thesis by Anass ELHOUD. \n \n  If you have any concerns, feedback, or have come across errors, please feel free to reach out: \n\n anass.elhoud@forvia.com \n www.elhoud.me", "center")
        self.textbox.configure(state="disabled")





class SettingWindow(customtkinter.CTkToplevel):
    def __init__(self, manuf_line):
        super().__init__()
        self.geometry("1024x576")
        self.title("Settings") 
        self.grid_columnconfigure((1), weight=2)
        self.grid_columnconfigure((0), weight=1)
        self.grid_rowconfigure((1), weight=1)
        self.resizable(width=False, height=False)
        self.manuf_line = manuf_line
      
        self.machine_data = [["Machine", "Type", "OP Time", "MT", "WC", "Link", "B-Capacity", "MTTF", "MTTR"], ["M1", "Type1", 20, 0, 0, "M2", 1, "3600*10", 3600*3],["M2", "Type2", 20, 0, 0, "M3", 1, "3600*10", 3600*3], ["M3", "Type3", 20, 0, 0, "M4", 1, "3600*10", 3600*3], ["M4", "Type4", 20, 0, 0, "END", 1, "3600*10", 3600*3]]
        # Simulation Data Frame
        self.frame = customtkinter.CTkFrame(master=self, corner_radius=20, width = 300)
        self.frame.grid(rowspan=2, column=0, pady = 10, padx=10, sticky="nsew")

        self.sim_time_label = customtkinter.CTkLabel(self.frame, text="Simulation Time (s)")
        self.sim_time_label.grid(row=0,column=0, padx=(10,10), pady=(10,10))
        self.sim_time_input = customtkinter.CTkEntry(self.frame, placeholder_text="Simulation Time (s)", width=200)
        self.sim_time_input.grid(row=1,column=0, padx=(10,10), pady=(10,10))
        self.takt_time_label = customtkinter.CTkLabel(self.frame, text="Expected Takt Time")
        self.takt_time_label.grid(row=2,column=0, padx=(10,10), pady=(10,10))
        self.takt_time_input = customtkinter.CTkEntry(self.frame, placeholder_text="Expected Takt Time", width=200)
        self.takt_time_input.grid(row=3,column=0, padx=(10,10), pady=(10,10))
        


        self.n_robots_label = customtkinter.CTkLabel(self.frame, text="Number of Robots")
        self.n_robots_label.grid(row=4,column=0, padx=(10,10), pady=(10,10))
        self.n_robots_input = customtkinter.CTkEntry(self.frame, placeholder_text="Input number of robots", width=200)
        self.n_robots_input.grid(row=5,column=0, padx=(10,10), pady=(10,10))

        self.strategy_dropdown_label = customtkinter.CTkLabel(self.frame, text="Robot's Load/Unload Strategy")
        self.strategy_dropdown_label.grid(row=6,column=0, padx=(10,10), pady=(10,10))
        self.strategy_dropdown = customtkinter.CTkComboBox(master=self.frame,values=["Balanced Strategy", "Greedy Strategy"], width=200)
        self.strategy_dropdown.grid(row=7,column=0, padx=(10,10), pady=(10,10))
        
        
        self.save_setting_btn = customtkinter.CTkButton(self.frame, text="Save", font=('Arial bold', 16), fg_color="Green", command=self.save_setting)
        self.save_setting_btn.grid(row=8, column=0, padx=20, pady=10)
        # Resource Data Frame (Machine/Buffers...)
        self.tabview = customtkinter.CTkTabview(self, corner_radius=20, width=500, height=250)
        self.tabview.grid(row=0, column=1, padx=(10, 10), pady=(10, 0), sticky="nsew")
        self.tabview.add("Machines")
        self.tabview.add("Stock")
        
        # self.tabview.tab("CTkTabview").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        # self.tabview.tab("Tab 2").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Machines").grid_columnconfigure((0,1), weight=1)  # configure grid of individual tabs
        self.tabview.tab("Machines").grid_rowconfigure((1), weight=1)
        #
        self.frame_machine_list = customtkinter.CTkScrollableFrame(master=self.tabview.tab("Machines"), height=100, width=700)
        self.frame_machine_list.grid(row=1, column=0, columnspan=3, padx=0, pady=0)
        self.table_machines = CTkTable(self.frame_machine_list, row=8, column=8, values=self.machine_data, header_color="deepskyblue4")
        # self.frame_machine_list.grid_rowconfigure((0), weight=1)
        # self.frame_machine_list.grid_columnconfigure((0), weight=1)
        #table.grid(row=0, column=0)
        self.table_machines.pack(expand=True)

        #self.frametest = customtkinter.CTkFrame(self.frame_machine_list, corner_radius=2, bg_color='red')
        #self.frametest.grid(row=0, column=0, pady = 10, padx=10, sticky="nsew")
    
        self.add_machine = customtkinter.CTkButton(master=self.tabview.tab("Machines"), text="+ Add new", width=150, command=self.add_machine)
        self.add_machine.grid(row = 0, column=0, pady=(0,10), padx=(0, 10))
        self.delete_machine = customtkinter.CTkButton(master=self.tabview.tab("Machines"), text="- Delete last", fg_color="brown3", width=150, command=self.delete_machine)
        self.delete_machine.grid(row = 0, column=1, padx=(0, 10), pady=(0,10))  
        upload_img =  tk.PhotoImage(file="./assets/icons/upload_icon.png")
    

        self.upload_data_m = customtkinter.CTkButton(master=self.tabview.tab("Machines"), text="Upload Config", font=('Arial', 15), width = 150, image=upload_img, command=self.upload_config, compound="left")
        self.upload_data_m.grid(row = 0, column=2, padx=(0, 10), pady=(0,10))   
        

        self.stock_capacity_label = customtkinter.CTkLabel(self.tabview.tab("Stock"), text="Input Sock Capacity")
        self.stock_capacity_label.grid(row=0,column=0, padx=(10,10), pady=(10,10))
        self.stock_capacity_input = customtkinter.CTkEntry(self.tabview.tab("Stock"), placeholder_text="Input Stock Capacity", width=200)
        self.stock_capacity_input.grid(row=1,column=0, padx=(10,10), pady=(10,10))

        self.initial_stock_label = customtkinter.CTkLabel(self.tabview.tab("Stock"), text="Initial Input Stock")
        self.initial_stock_label.grid(row=0,column=1, padx=(10,10), pady=(10,10))
        self.initial_stock_input = customtkinter.CTkEntry(self.tabview.tab("Stock"), placeholder_text="Initial Input Stock", width=200)
        self.initial_stock_input.grid(row=1,column=1, padx=(10,10), pady=(10,10))

        self.refill_time_label = customtkinter.CTkLabel(self.tabview.tab("Stock"), text="Refill Time (s)")
        self.refill_time_label.grid(row=0,column=2, padx=(10,10), pady=(10,10))
        self.refill_time_input = customtkinter.CTkEntry(self.tabview.tab("Stock"), placeholder_text="Refill Time (s)", width=200)
        self.refill_time_input.grid(row=1,column=2, padx=(10,10), pady=(10,10))

        self.safety_stock_label = customtkinter.CTkLabel(self.tabview.tab("Stock"), text="Safey Stock")
        self.safety_stock_label.grid(row=2,column=0, padx=(10,10), pady=(10,10))
        self.safety_stock_input = customtkinter.CTkEntry(self.tabview.tab("Stock"), placeholder_text="Safey Stock", width=200)
        self.safety_stock_input.grid(row=3,column=0, padx=(10,10), pady=(10,10))

        self.refill_size_label = customtkinter.CTkLabel(self.tabview.tab("Stock"), text="Refill Size")
        self.refill_size_label.grid(row=2,column=1, padx=(10,10), pady=(10,10))
        self.refill_size_input = customtkinter.CTkEntry(self.tabview.tab("Stock"), placeholder_text="Refill Size", width=200)
        self.refill_size_input.grid(row=3,column=1, padx=(10,10), pady=(10,10))


        self.tabview_footer = customtkinter.CTkTabview(self, corner_radius=20, width=500, height=250)
        self.tabview_footer.grid(row=1, column=1, padx=10, pady = 5, sticky="nsew" )
        self.tabview_footer.add("Breakdowns")
        self.tabview_footer.add("Delays")
        self.tabview_footer.add("Manual Models")

        self.switch_var = customtkinter.StringVar(value="on")
        self.label_break1 = customtkinter.CTkLabel(self.tabview_footer.tab("Breakdowns"), text="Machine Breakdown")
        self.label_break1.grid(row=0, column=0, padx=20, pady=20)
        self.switch = customtkinter.CTkSwitch(self.tabview_footer.tab("Breakdowns"), text="Enabled",
                                 variable=self.switch_var, onvalue="on", offvalue="off")
        
        self.switch.grid(row=1, column=0, padx=20, pady=20)

        self.n_repairmen_label = customtkinter.CTkLabel(self.tabview_footer.tab("Breakdowns"), text="Number of Repairmen")
        self.n_repairmen_label.grid(row=0,column=1, padx=(10,10), pady=(10,10))
        self.n_repairmen_input = customtkinter.CTkEntry(self.tabview_footer.tab("Breakdowns"), placeholder_text="Input number of repairmen", width=100)
        self.n_repairmen_input.grid(row=1,column=1, padx=(10,10), pady=(10,10))

        self.switch_var_rand = customtkinter.StringVar(value="on")
        self.label_break1_rand = customtkinter.CTkLabel(self.tabview_footer.tab("Breakdowns"), text="Random Seed")
        self.label_break1_rand.grid(row=0, column=2, padx=20, pady=20)
        self.switch_rand = customtkinter.CTkSwitch(self.tabview_footer.tab("Breakdowns"), text="Enabled",
                                 variable=self.switch_var_rand, onvalue="on", offvalue="off")

        self.switch_rand.grid(row=1, column=2, padx=20, pady=20)

        self.switch_delays_var = customtkinter.StringVar(value="on")
        self.label_delays1 = customtkinter.CTkLabel(self.tabview_footer.tab("Delays"), text="Hazardous delays")
        self.label_delays1.grid(row=0, column=0, padx=20, pady=20)
        self.switch_delays = customtkinter.CTkSwitch(self.tabview_footer.tab("Delays"), text="Enabled",
                                 variable=self.switch_delays_var, onvalue="on", offvalue="off")
        self.switch_delays.grid(row=1, column=0, padx=20, pady=20)
        self.label_delays2 = customtkinter.CTkLabel(self.tabview_footer.tab("Delays"), text="Probability Distribution")
        self.label_delays2.grid(row=0, column=3, padx=20, pady=20)

        self.choices_delay_dist = ["Weibull Distribution"]
        self.choices_delays_menu = customtkinter.CTkOptionMenu(self.tabview_footer.tab("Delays"), dynamic_resizing=False,
                                                        values=self.choices_delay_dist)
        self.choices_delays_menu.grid(row=1, column=3, padx=20, pady=20)

        ## Default
        self.sim_time_input.insert(0, "3600*24*200")
        self.takt_time_input.insert(0, "100000")
        self.stock_capacity_input.insert(0, "100")
        self.initial_stock_input.insert(0, "100")
        self.safety_stock_input.insert(0, "20")
        self.refill_time_input.insert(0, "120")
        self.refill_size_input.insert(0, "100")
        self.n_robots_input.insert(0, "1")
        self.n_repairmen_input.insert(0, "3")



    
    def add_func(self, current_list , current_table, inputs_labels = ["Machine", ["MT", "WC"], ["OP Time", "Link"], "Buffer Capacity", ["MTTF", "MTTR"]]):
        dialog = CTkInputDialogSetting(text=inputs_labels, title="CTkInputDialog")
        # m_ct = customtkinter.CTkEntry(master=dialog)
        # m_ct.grid(row=4, column=0)
        inputs = dialog.get_input()
        
        if inputs != []:
            temp = []
            for v in inputs:
                try:
                    temp.append(float(v))
                except:
                    temp.append(str(v))
            current_list.append(temp)
            current_table.update_values(current_list)
            return True
        else:
            return False

    def delete_func(self, current_list , current_table):
        if len(current_list) >1:
            del current_list[-1]
            current_table.update_values(current_list)
        else:
            pass
    
    def add_machine(self):
        done = self.add_func(self.machine_data,self.table_machines)
        #self.add_func(self.buffers_data,self.table_buffers, ["Buffer",["Capacity","Initial"]])
        if done:
            machine_names = [sublist[0] for sublist in self.machine_data[1:]]
            self.choices_breakdowns = ["All Machines"] + machine_names
            self.choices_breakdowns_menu.configure(values=self.choices_breakdowns)
        else:
            pass
    def delete_machine(self):
        self.delete_func(self.machine_data,self.table_machines)

    def add_buffer(self):
        self.add_func(self.buffers_data,self.table_buffers, ["Buffer",["Capacity","Initial"]])
    def delete_buffer(self):
        self.delete_buffer(self.buffers_data,self.table_buffers)

    
    def upload_config(self):
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
                config_multi_ref_table = pd.read_excel(file_path, sheet_name="Multi-Ref")
                self.manuf_line.references_config = config_multi_ref_table.set_index('Machine').to_dict(orient='list')

                print("Excel file uploaded and read successfully.")
                config_data_gloabl = config_line_globa_data.values.tolist()
                self.machine_data[1:] = config_data.values.tolist()
                print("Machine Data = ", self.machine_data)
                self.table_machines.update_values(self.machine_data)
                self.sim_time_input.delete(0, END)
                self.sim_time_input.insert(0, str(config_data_gloabl[0][2]))
                self.takt_time_input.delete(0, END)
                self.takt_time_input.insert(0, str(config_data_gloabl[1][2]))

                self.stock_capacity_input.delete(0, END)
                self.stock_capacity_input.insert(0, str(config_data_gloabl[2][2]))

                self.initial_stock_input.delete(0, END)
                self.initial_stock_input.insert(0, str(config_data_gloabl[3][2]))
                self.refill_time_input.delete(0, END)
                
                self.refill_time_input.insert(0, str(config_data_gloabl[4][2]))
                self.safety_stock_input.delete(0, END)
                self.safety_stock_input.insert(0, str(config_data_gloabl[5][2]))
                self.refill_size_input.delete(0, END)
                self.refill_size_input.insert(0, str(config_data_gloabl[6][2]))
                self.n_repairmen_input.delete(0, END)
                self.n_repairmen_input.insert(0, str(config_data_gloabl[11][2]))
                self.n_robots_input.delete(0, END)
                self.n_robots_input.insert(0, str(config_data_gloabl[12][2]))
                
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

    def save_setting(self):
        
        if self.switch_var.get() == "on":
            self.manuf_line.breakdowns_switch = True
        else:
            self.manuf_line.breakdowns_switch = False
        
        if self.switch_var_rand.get() == "on":
            self.manuf_line.randomseed = True
        else:
            self.manuf_line.randomseed = False

        available_strategies = ["Balanced Strategy", "Greedy Strategy"]
        self.manuf_line.stock_capacity = float(self.stock_capacity_input.get())
        self.manuf_line.stock_initial = float(self.initial_stock_input.get())
        # if value1-value2, then the refill time is random between two values
        pattern = r'^(\d+)-(\d+)$'
        match = re.match(pattern, str(self.refill_time_input.get()))
        if match:
            value1 = int(match.group(1))
            value2 = int(match.group(2))
            self.manuf_line.refill_time = [value1, value2]
        else:
            self.manuf_line.refill_time = float(self.refill_time_input.get())

        self.manuf_line.safety_stock = float(self.safety_stock_input.get())
        self.manuf_line.refill_size = float(self.refill_size_input.get())
        self.manuf_line.n_robots = float(self.n_robots_input.get())
        self.manuf_line.n_repairmen = int(self.n_repairmen_input.get())
        self.manuf_line.robot_strategy = int(available_strategies.index(self.strategy_dropdown.get()))
        self.manuf_line.repairmen = simpy.PreemptiveResource(env, capacity=int(self.n_repairmen_input.get()))


        self.manuf_line.supermarket_in = simpy.Container(env, capacity=self.manuf_line.stock_capacity, init=self.manuf_line.stock_initial)
        self.manuf_line.shop_stock_out = simpy.Container(env, capacity=float(self.manuf_line.config["shopstock"]["capacity"]), init=float(self.manuf_line.config["shopstock"]["initial"]))
        
        self.manuf_line.machine_config_data = self.machine_data[1:]
        self.manuf_line.create_machines(self.machine_data[1:])
        
        try:
            self.manuf_line.sim_time = eval(str(self.sim_time_input.get()))
            self.manuf_line.takt_time = eval(str(self.takt_time_input.get()))
        except:
            self.manuf_line.sim_time = int(self.sim_time_input.get())
            self.manuf_line.takt_time = eval(str(self.takt_time_input.get()))

        self.destroy()
        

class ReportingWindow(customtkinter.CTkToplevel):
    

    def __init__(self, parent, manuf_line):
        self.manuf_line = manuf_line
        
       
        super().__init__()
        self.geometry(f"{1920}x{1080}")
        self.loading_window = None
        global simulation_number

        if self.manuf_line.robot_strategy == 0:
            strategy = "Balanced Strategy"
        elif self.manuf_line.robot_strategy == 1:
            strategy = "Greedy Strategy"
        self.title("Simulation {} - {}".format(simulation_number, strategy))
        simulation_number+=1
        self.grid_columnconfigure((1), weight=1)
        #self.grid_columnconfigure((0), weight=1)
        #self.grid_rowconfigure((1,2), weight=1)
        #self.resizable(width=False, height=False)
        self.textbox = customtkinter.CTkTextbox(master=self, width=450, height=250, corner_radius=0, font=("Arial", 18), wrap="word")
        self.textbox.grid_remove()
        self.frame_data = customtkinter.CTkFrame(master=self, corner_radius=20, width = 250, height=1000)
        self.frame_data.grid(rowspan=3, column=0, pady = 10, padx=10, sticky="nsew")
        self.simulated_prod_time_btn = customtkinter.CTkButton(self.frame_data, text="Simulation Time", width = 250, fg_color="grey", text_color_disabled= "white", state="disabled", font=('Arial', 15))
        self.simulated_prod_time_btn.grid(row=0, column=1, padx=20, pady=10)
        self.simulated_prod_time_label = customtkinter.CTkLabel(master=self.frame_data, text="N/A", font=('Arial', 16))
        self.simulated_prod_time_label.grid(row=1, column=1, padx=20, pady=10)

        self.global_cycle_time_btn = customtkinter.CTkButton(self.frame_data, text="Global Cycle Time", width = 250, fg_color="grey", text_color_disabled= "white", state="disabled", font=('Arial', 15))
        self.global_cycle_time_btn.grid(row=2, column=1, padx=20, pady=10)
        self.global_cycle_time_label = customtkinter.CTkLabel(master=self.frame_data, text="N/A", font=('Arial', 16))
        self.global_cycle_time_label.grid(row=3, column=1, padx=20, pady=10)

        self.production_rate_btn = customtkinter.CTkButton(self.frame_data, text="Production Flow Rate", width = 250, fg_color="grey", text_color_disabled= "white", state="disabled", font=('Arial', 15))
        self.production_rate_btn.grid(row=4, column=1, padx=20, pady=10)
        self.production_rate_label = customtkinter.CTkLabel(master=self.frame_data, text="N/A", font=('Arial', 16))
        self.production_rate_label.grid(row=5, column=1, padx=20, pady=10)

        self.efficiency_btn = customtkinter.CTkButton(self.frame_data, text="Efficiency Rate", width = 250, fg_color="grey", text_color_disabled= "white", state="disabled", font=('Arial', 15))
        self.efficiency_btn.grid(row=6, column=1, padx=20, pady=10)
        self.efficiency_label = customtkinter.CTkLabel(master=self.frame_data, text="N/A", font=('Arial', 16))
        self.efficiency_label.grid(row=7, column=1, padx=20, pady=10)

        self.oee_btn = customtkinter.CTkButton(self.frame_data, text="OEE / TRS", width = 250, fg_color="grey", text_color_disabled= "white", state="disabled", font=('Arial', 15))
        self.oee_btn.grid(row=8, column=1, padx=20, pady=10)
        self.oee_label = customtkinter.CTkLabel(master=self.frame_data, text="N/A", font=('Arial', 16))
        self.oee_label.grid(row=9, column=1, padx=20, pady=10)
        self.save_setting_btn = customtkinter.CTkButton(self.frame_data, text="Save Sequence", font=('Arial bold', 16), fg_color="Green", command=self.save_robot_sequence)
        self.save_setting_btn.grid(row=10, column=1, padx=20, pady=10)
        self.frame = customtkinter.CTkFrame(master=self, corner_radius=20, width = 300)
        self.frame.grid(row=0, column=1, pady = 10, padx=10, sticky="nsew")
        self.frame_br = customtkinter.CTkFrame(master=self, corner_radius=20)
        self.frame_br.grid(row=1, column=1, pady = 5, padx=10, sticky="nsew")
        
        # Prep Plot of Machine Avg. Cycle Time
        fig_m, ax_m = plt.subplots(nrows=1, ncols=1, figsize=(9, 4), sharex=True, facecolor="#282C34")
        fig_m.set_facecolor('#282C34' if app.appearence_mode == "Dark" else 'white')
        fg_color = 'white' if app.appearence_mode == "Dark" else 'black'
        ax_m.clear()
        ax_m.set_ylabel('Percentage (%)', color=fg_color)
        ax_m.set_title('Machine Utilization Rate', color=fg_color)
        ax_m.set_facecolor('#282C34' if app.appearence_mode == "Dark" else 'white')
        ax_m.yaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
        ax_m.xaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
        canvas_fig_m = FigureCanvasTkAgg(fig_m, master=self.frame)
        canvas_fig_widget_m = canvas_fig_m.get_tk_widget()
        canvas_fig_widget_m.pack(expand=True, fill=tk.BOTH)

        # Prep Plot of Machine Breakdowns

        fig_br, ax_br = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), sharex=True, facecolor="#282C34")
        fig_br.set_facecolor('#282C34' if app.appearence_mode == "Dark" else 'white')
        fg_color = 'white' if app.appearence_mode == "Dark" else 'black'
        for i in range(len(ax_br)):
            ax_br[i].clear()
            ax_br[i].set_ylabel('Idle Time (s)', color=fg_color)
            ax_br[i].set_title('Idle Time of Machines per type of waiting', color=fg_color)
            ax_br[i].set_facecolor('#282C34' if app.appearence_mode == "Dark" else 'white')
            ax_br[i].yaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
            ax_br[i].xaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
        canvas_fig_br = FigureCanvasTkAgg(fig_br, master=self.frame_br)
        canvas_fig_widget__br = canvas_fig_br.get_tk_widget()
        canvas_fig_widget__br.pack(expand=True, fill=tk.BOTH)

        
        # parent.toplevel_window.destroy()
        # parent.toplevel_window = self
        

        # Set up KPIs
        print("Machins products = ", [m.parts_done for m in manuf_line.list_machines])
        print("Shop_stock_out = ", manuf_line.shop_stock_out.level)
        CT_line = manuf_line.sim_time/manuf_line.shop_stock_out.level
        efficiency_rate = 100*(manuf_line.takt_time/CT_line)

        simulated_prod_time_str = format_time(manuf_line.sim_time)
        self.simulated_prod_time_label.configure(text=simulated_prod_time_str)
        self.global_cycle_time_label.configure(text = f'{CT_line:.1f} s')
        self.efficiency_label.configure(text = f'{efficiency_rate:.1f} %')
        self.production_rate_label.configure(text = f'{3600*manuf_line.shop_stock_out.level/manuf_line.sim_time:.1f} part/h')
        # Plot of Machine Avg. Cycle Time
        machines_names = [m.ID for m in manuf_line.list_machines]
        idle_times = []
        machines_CT = []
        idle_times_sum = []
        
        for i, machine in enumerate(manuf_line.list_machines):
            idle_times_machine = []
            
            #for entry, exit in zip(machine.entry_times, machine.exit_times):
            ct_machine = []
            for finished in machine.finished_times:
                if finished is None:
                    ct_machine.append(0)
                else:
                    ct_machine.append(finished)
            machines_CT.append(np.sum(ct_machine))

            for time in machine.exit_times:
                idle_times_machine.append(time)
            
            idle_times.append(np.mean(idle_times_machine))
            idle_times_sum.append(np.sum(idle_times_machine)/manuf_line.sim_time)

        print("CT Machines = ",machines_CT )
        print("CT Machines 2 = ",[manuf_line.sim_time / m.parts_done if m.parts_done != 0 else 0 for m in manuf_line.list_machines])
        print("Waiting times = ", [np.sum(m.waiting_time) for m in manuf_line.list_machines])
        for ri in range(len(manuf_line.robots_list)):
            print("Waiting time Robot = ", 100*manuf_line.robots_list[ri].waiting_time/manuf_line.sim_time)
        machines_prod_rate = [manuf_line.sim_time / m.parts_done if m.parts_done != 0 else 0 for m in manuf_line.list_machines]

        #machine_efficiency_rate =[int(100*m.parts_done/(manuf_line.sim_time/m.ct)) for m in manuf_line.list_machines]
        machine_efficiency_rate =[100*np.sum([m.ref_produced.count(item)*manuf_line.references_config[item][manuf_line.list_machines.index(m)+1] for item in list(manuf_line.references_config.keys())])/manuf_line.sim_time for m in manuf_line.list_machines]
        machines_util = [m.ct* m.parts_done / manuf_line.sim_time for i,m in enumerate(manuf_line.list_machines)]

       # machine_available_percentage = [100*m.ct* m.parts_done / manuf_line.sim_time for m in manuf_line.list_machines]
        machine_available_percentage = [100*ct / manuf_line.sim_time for m, ct in zip(manuf_line.list_machines,machines_CT) ]
        #waiting_time_percentage = [100*(m.waiting_time[0] + m.waiting_time[1])/manuf_line.sim_time   for m in manuf_line.list_machines]
        breakdown_percentage = [100*float(m.MTTR * float(m.n_breakdowns)) / manuf_line.sim_time for m in manuf_line.list_machines]

        # Calculate machine available percentage, breakdown percentage, and waiting time percentage
        waiting_time_percentage = [100 - available_percentage - breakdown_percentage for available_percentage, breakdown_percentage in zip(machine_available_percentage, breakdown_percentage)]
        #machine_available_percentage = [100 - waiting_percentage - breakdown_percentage for waiting_percentage, breakdown_percentage in zip(waiting_time_percentage, breakdown_percentage)] 
        print("Refernces = ", self.manuf_line.references_config)
        print('Efficiency = ', machine_efficiency_rate)
        print('Availability 1 = ', machines_util)
        print('Utilization 2 = ', machine_available_percentage)
        print("Breakdowns = ", np.sum([ float(m.n_breakdowns) for m in manuf_line.list_machines]))
        for item in list(manuf_line.references_config.keys()):
            print("Items of  = ", [m.ref_produced.count(item)  for m in manuf_line.list_machines])
            print("Ref = " + item + " - " + str(manuf_line.inventory_out.items.count(item)))
        #oee_100quality = 100*np.mean([(m.ct + 2*abs(m.move_robot_time))/ct for m, ct in zip(manuf_line.list_machines, machines_CT)])
        oee_100quality = 100*np.mean([ct/manuf_line.sim_time for m, ct in zip(manuf_line.list_machines, machines_CT)])

        self.oee_label.configure(text = f'{oee_100quality:.1f} %')

        bars1 = ax_m.bar(machines_names, machine_available_percentage, label='Operating', color="green")
        bars2 = ax_m.bar(machines_names, breakdown_percentage, bottom=machine_available_percentage, label='Breakdown', color="red")
        bars3 = ax_m.bar(machines_names, waiting_time_percentage, bottom=np.array(machine_available_percentage) + np.array(breakdown_percentage), label='Waiting', color="Orange")
        ax_m.plot(machines_names, machine_efficiency_rate, '--x', color='white', label="Efficiency")
        ax_m.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=4)
        
        fig_m.tight_layout()

        
        # Plot of Machine Breakdowns
        breakdown_values = [m.n_breakdowns for m in manuf_line.list_machines]
        starvation_times = [m.waiting_time[0] for m in manuf_line.list_machines]
        blockage_times = [m.waiting_time[1] for m in manuf_line.list_machines]

        num_machines = len( manuf_line.list_machines)
        bar_width = 0.35
        index = range(num_machines)

        bar_starv = ax_br[0].bar(index, starvation_times, bar_width, label='Starvation Time')
        bar_block = ax_br[0].bar([i + bar_width for i in index], blockage_times,bar_width, label='Blockage Time')

        ax_br[0].set_xticks([i + bar_width / 2 for i in index])  # Set machine names as x-tick positions
        ax_br[0].set_xticklabels(machines_names)

        ax_br[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=2)
        
        for idx, item in enumerate(list(manuf_line.references_config.keys())):
            items = [m.ref_produced.count(item)  for m in manuf_line.list_machines]
            ax_br[1].bar([x + bar_width * idx for x,_ in enumerate(manuf_line.list_machines)], items, width=0.4, label=item, align='center')

        ax_br[1].set_xlabel('Machine IDs')
        ax_br[1].set_ylabel('Parts Produced')
        ax_br[1].set_title('Parts Passed through Machine per Reference')
        ax_br[1].legend()
        fig_br.tight_layout()

    def longest_repetitive_pattern(self, sequence):
        longest_pattern = None
        max_length = 0

        # Iterate through each possible start index
        for i in range(len(sequence)):
            # Iterate through each possible end index
            for j in range(i + 1, len(sequence)):
                pattern_length = j - i
                # Check if the pattern repeats throughout the sequence
                if all(sequence[x % pattern_length + i] == sequence[x % pattern_length + i - pattern_length] for x in range(j, len(sequence))):
                    if pattern_length > max_length:
                        max_length = pattern_length
                        longest_pattern = sequence[i:j]
        return longest_pattern
    
    def save_robot_sequence(self):
        """
        Save the robot sequence as CSV file to be analyzed or used.
        """
        folder_path = "./Robot Sequence/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Generate current datetime string
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Specify the file name
        csv_file = f"{folder_path}sequence_{datetime_str}_data.csv"

        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.manuf_line.robot_states)

        self.manuf_line.robot_states

class LoadingScreen(customtkinter.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("500x300")
        self.resizable(width=False, height=False)
        self.title("Support") 
        self.textbox = customtkinter.CTkTextbox(master=self, width=450, height=250, corner_radius=0, font=("Arial", 18), wrap="word")
        self.textbox.grid(row=1, column=1, padx=(25, 25), pady=(25,25), sticky="nsew")
        self.textbox.tag_config("center", justify="center")
        self.textbox.insert("0.0", "\n This work was completed within the context of a CIFRE PhD Thesis by Anass ELHOUD. \n \n  If you have any concerns, feedback, or have come across errors, please feel free to reach out: \n\n anass.elhoud@forvia.com \n www.elhoud.me", "center")
        self.textbox.configure(state="disabled")

class App(customtkinter.CTk):
    def __init__(self, manuf_line):
        super().__init__()

        # configure window
        self.title("PRODynamics") 
        self.geometry(f"{1920}x{1080}") #1280x720
        mainicon = tk.PhotoImage(file="./assets/icons/mainicon2.png")
        self.iconphoto(False, mainicon)
        self.appearence_mode = "Dark"
        self.toplevel_window = None
        # configure grid layout (4x4)
        self.grid_columnconfigure((1,2), weight=1)
        
        self.grid_rowconfigure((1, 2, 3, 4), weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.manuf_line = manuf_line
        

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=5, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure((4), weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="PRODynamics", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        setting_img =  tk.PhotoImage(file="./assets/icons/setng_icon.png")
        self.settings_btn = customtkinter.CTkButton(self.sidebar_frame, text="Settings", font=('Arial', 15), command=self.open_setting_window, image=setting_img, compound="left")
        self.settings_btn.grid(row=1, column=0, padx=20, pady=10)
        start_img =  tk.PhotoImage(file="./assets/icons/start_icon.png")
        self.Start = customtkinter.CTkButton(self.sidebar_frame, text="Start", font=('Arial', 15), command=self.start_sim, image=start_img, compound="left")
        self.Start.grid(row=2, column=0, padx=20, pady=10)
        self.report_btn = customtkinter.CTkButton(self.sidebar_frame, text="Reports", font=('Arial', 15), command=self.start_reporting)
        self.report_btn.grid(row=3, column=0, padx=20, pady=10)
        
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%", "140%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))
        self.support_btn = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_event)
        self.support_btn.grid(row=9, column=0, padx=20, pady=10)

        # create scrollable frame
        # self.setting_frame_scroll = customtkinter.CTkScrollableFrame(self, label_text="Settings")
        # self.setting_frame_scroll.grid(row=1, column=3, padx=(10, 10), pady=(10, 0), sticky="nsew")
        # self.setting_frame_scroll.grid_columnconfigure(0, weight=1)
        # self.setting_frame_scroll_switches = []
        # setting_titles = ["Breakdowns", "Hazard Delays", "Buffer Capacity", "Manual Fatigue", "Learning Model"]
    
        # for i in range(len(setting_titles)):
        #     switch = customtkinter.CTkSwitch(master=self.setting_frame_scroll, text=setting_titles[i])
        #     switch.grid(row=i, column=0, padx=10, pady=(0, 20))
        #     self.setting_frame_scroll_switches.append(switch)

        # create checkbox and switch frame
        self.checkbox_slider_frame = customtkinter.CTkFrame(self)
        self.checkbox_slider_frame.grid(row=2, column=2,columnspan=2,  padx=(10, 10), pady=(10, 0), sticky="nsew")
        # self.checkbox_1 = customtkinter.CTkCheckBox(master=self.checkbox_slider_frame)
        # self.checkbox_1.grid(row=2, column=0, pady=(20, 0), padx=20, sticky="n")
        # self.checkbox_2 = customtkinter.CTkCheckBox(master=self.checkbox_slider_frame)
        # self.checkbox_2.grid(row=3, column=0, pady=(20, 0), padx=20, sticky="n")
        # self.checkbox_3 = customtkinter.CTkCheckBox(master=self.checkbox_slider_frame)
        # self.checkbox_3.grid(row=4, column=0, pady=20, padx=20, sticky="n")
        
        # KPI Dynamics Frame
        self.kpi_up_dynamics = customtkinter.CTkFrame(self, height=100)
        self.kpi_up_dynamics.grid(row=0, column=1, columnspan=3, padx=(10, 10), pady=(10, 0), sticky="nsew")
        self.kpi_up_dynamics.grid_columnconfigure((6), weight=1)
        self.increase_speed_btn = customtkinter.CTkButton(self.kpi_up_dynamics, text="+ Speed Up", fg_color="green", font=('Arial', 15), command=increase_timeout)
        self.increase_speed_btn.grid(row=0, column=0, padx=20, pady=10)
        self.decrease_speed_btn = customtkinter.CTkButton(self.kpi_up_dynamics, text="- Slow Down",fg_color="red", font=('Arial', 15), command=decrease_timeout)
        self.decrease_speed_btn.grid(row=1, column=0, padx=20, pady=10)
        self.sim_time_btn = customtkinter.CTkButton(self.kpi_up_dynamics, text="Simulation Time", width = 250, fg_color="grey", text_color_disabled= "white", state="disabled", font=('Arial', 15), command=self.sidebar_button_event)
        self.sim_time_btn.grid(row=0, column=1, padx=20, pady=10)
        self.sim_time_label = customtkinter.CTkLabel(master=self.kpi_up_dynamics, text="N/A", font=('Arial', 16))
        self.sim_time_label.grid(row=1, column=1, padx=20, pady=10)
        self.shift_ct_btn = customtkinter.CTkButton(self.kpi_up_dynamics, text="Shift Cycle Time", width = 150, fg_color="grey", text_color_disabled= "white", state="disabled", font=('Arial', 15), command=self.sidebar_button_event)
        self.shift_ct_btn.grid(row=0, column=2, padx=20, pady=10)
        self.shift_ct_label = customtkinter.CTkLabel(master=self.kpi_up_dynamics, text="N/A", font=('Arial bold', 18))
        self.shift_ct_label.grid(row=1, column=2, padx=20, pady=10)
        self.annual_ct_btn = customtkinter.CTkButton(self.kpi_up_dynamics, text="Annual Cycle Time", width = 150, fg_color="grey", text_color_disabled= "white", state="disabled", font=('Arial', 15), command=self.sidebar_button_event)
        self.annual_ct_btn.grid(row=0, column=3, padx=20, pady=10)
        self.annual_ct_label = customtkinter.CTkLabel(master=self.kpi_up_dynamics, text="N/A", font=('Arial bold', 18))
        self.annual_ct_label.grid(row=1, column=3, padx=20, pady=10)

        self.partsdone_btn = customtkinter.CTkButton(self.kpi_up_dynamics, text="Total Parts", width = 150, fg_color="grey", text_color_disabled= "white", state="disabled", font=('Arial', 15), command=self.sidebar_button_event)
        self.partsdone_btn.grid(row=0, column=4, padx=20, pady=10)
        self.partsdone_label = customtkinter.CTkLabel(master=self.kpi_up_dynamics, text="N/A", font=('Arial bold', 18))
        self.partsdone_label.grid(row=1, column=4, padx=20, pady=10)
        self.oee_btn = customtkinter.CTkButton(self.kpi_up_dynamics, text="Efficiency Rate", width = 150, fg_color="grey", text_color_disabled= "white", state="disabled", font=('Arial', 15), command=self.sidebar_button_event)
        self.oee_btn.grid(row=0, column=5, padx=20, pady=10)
        self.oee_label = customtkinter.CTkLabel(master=self.kpi_up_dynamics, text="N/A", font=('Arial bold', 18))
        self.oee_label.grid(row=1, column=5, padx=20, pady=10)
        save_img =  tk.PhotoImage(file="./assets/icons/save_icon.png")
        self.save_btn = customtkinter.CTkButton(self.kpi_up_dynamics, text="Save", width = 150, text_color_disabled= "white", font=('Arial', 15), image=save_img, command=self.sidebar_button_event, compound="left")
        self.save_btn.grid(row=0, column=7, padx=20, pady=10)

        # Viz CT Frame
        self.viz_ct_frame = customtkinter.CTkFrame(self, width=450)
        self.viz_ct_frame.grid(row=1, column=1, rowspan=3, columnspan=2, padx=(10, 5), pady=(10, 0), sticky="nsew")
        # self.viz_ct_frame1 = customtkinter.CTkFrame(self)
        # self.viz_ct_frame1.grid(row=2, column=1, padx=(10, 5), pady=(10, 0), sticky="nsew")

        # Buffer states Frame

        self.buffer_state_frame = customtkinter.CTkScrollableFrame(self, label_text="Inventory State", label_font=('Arial', 15))
        self.buffer_state_frame.grid(row=1, column=3, columnspan=2, padx=(10, 10), pady=(10, 0), sticky="nsew")
        #self.buffer_state_frame.grid_columnconfigure(0, weight=1)
        self.buffer_state_btn = []
        self.buffer_capacities = []
        self.buffer_levels = []
        self.machine_idles = []
        self.machine_downtimes = []
        self.supermarket_btn = None
        self.supermarket_capacity= None
        self.supermarket_level = None
        self.refill_label = None
        # setting_titles = ["Breakdowns", "Hazard Delays", "Buffer Capacity", "Manual Fatigue", "Learning Model"]

        #self.buffer_state_frame2 = customtkinter.CTkFrame(self)
        #self.buffer_state_frame2.grid(row=2, column=2, padx=(5, 5), pady=(10, 0), sticky="nsew")

        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=3, column=3, padx=(10, 10), pady=(10, 0), sticky="nsew")
        self.tabview.add("Robot")
        self.tabview.add("Tab 2")
        self.tabview.add("Tab 3")
        self.tabview.tab("Robot").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview.tab("Tab 2").grid_columnconfigure(0, weight=1)

        self.robot_waiting_time_btn = customtkinter.CTkButton(self.tabview.tab("Robot"), text="Waiting Rate", width = 150, fg_color="grey", text_color_disabled= "white", state="disabled" )
        self.robot_waiting_time_btn.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.robot_waiting_time = customtkinter.CTkLabel(self.tabview.tab("Robot"), text="N/A",  font=('Arial bold', 16))
        self.robot_waiting_time.grid(row=1, column=0, padx=20, pady=(10, 10))
        # self.string_input_button = customtkinter.CTkButton(self.tabview.tab("Robot"), text="Open CTkInputDialog",
        #                                                    command=self.open_input_dialog_event)
        # self.string_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))
        # self.label_tab_2 = customtkinter.CTkLabel(self.tabview.tab("Tab 2"), text="CTkLabel on Tab 2")
        # self.label_tab_2.grid(row=0, column=0, padx=20, pady=20)

        # Footer App Main

        self.main_footer = customtkinter.CTkFrame(self, height=300, width=250)
        self.main_footer.grid(row=2, column=3, padx=(10, 10), pady=(10, 0), sticky="nsew")
       

        # set default values
        self.support_btn.configure(text="Support", command=self.open_toplevel)
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")


    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
    

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)
        self.appearence_mode = new_appearance_mode

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        print("sidebar_button click")

    def open_toplevel(self):
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = SupportWindow(self)  # create window if its None or destroyed
        else:
            self.toplevel_window.focus()  # if window exists focus it
    
    def open_setting_window(self):
        self.toplevel_window = SettingWindow(self.manuf_line)
        # if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
        #     self.toplevel_window = SettingWindow(self.manuf_line)  # create window if its None or destroyed
        # else:
        #     self.toplevel_window.focus()  # if window exists focus it


    def update_machine_util_viz(self, machine_names, machines_util):
        ax_m.clear()
        fig_m.set_facecolor('#282C34' if app.appearence_mode == "Dark" else 'white')
        fg_color = 'white' if app.appearence_mode == "Dark" else 'black'
        ax_m.set_ylabel('Cycle Time (s)', color=fg_color)
        ax_m.set_title('Machine Avg. Cycle Time', color=fg_color)
        ax_m.set_facecolor('#282C34' if app.appearence_mode == "Dark" else 'white')
        ax_m.yaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
        ax_m.xaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
        bars = ax_m.bar(machine_names, machines_util)  # Set your desired bar color
        #ax_m.set_ylim(0, 110)
        ax_m.axhline(y=100, color='green', linestyle='--')
        
     
        for bar, util in zip(bars, machines_util):
            height = bar.get_height()
            ax_m.annotate(f'{util:.2f}',  
                        xy=(bar.get_x() + bar.get_width() / 2, height/2),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom',
                        color='white', 
                        fontsize=10) 


        canvas_fig_m.draw()

    def start_sim(self):
        self.Start.configure(state="disabled")

        # Draw Supermarket
        self.supermarket_btn = customtkinter.CTkButton(master=self.buffer_state_frame, text=f"SM", width = 50, height=50, fg_color="green", text_color_disabled= "white", state="disabled", font=('Arial', 15))
        self.supermarket_btn.grid(row=0, column=0, rowspan=2, padx=10, pady=(0, 5))
    
        self.supermarket_capacity = customtkinter.CTkLabel(master=self.buffer_state_frame, text="Capacity = N/A", font=('Arial', 14), justify="right")
        self.supermarket_capacity.grid(row=1, column=1, padx=10, pady=(0, 5))

        self.supermarket_level = customtkinter.CTkLabel(master=self.buffer_state_frame, text="Level = N/A", font=('Arial', 14), justify="left")
        self.supermarket_level.grid(row=0, column=1, padx=10, pady=(0, 5))
        self.supermarket_level = self.supermarket_level

        self.refill_label = customtkinter.CTkLabel(master=self.buffer_state_frame, text="N. of Refills = 0", font=('Arial', 14), justify="left")
        self.refill_label.grid(row=0, column=2, padx=5, pady=(0, 5))

        #indices_first_machines = [[m.ID for m in self.manuf_line.list_machines].index(id) for id in [m.ID for m in self.manuf_line.list_machines] if m.first == True]
        
        indices_first_machines = [[machine.ID for machine in self.manuf_line.list_machines].index(id)+1 for id, machine in [(m.ID, m) for m in self.manuf_line.list_machines if m.first]]
        
       

        for m1 in self.manuf_line.list_machines:
            for m2 in self.manuf_line.list_machines:
                # Check if list of next machines is same
                if m1.next_machines == m2.next_machines and m1 != m2:
                    m1.identical_machines.append(m2)

        
        passed_machines = []
        for i,m in enumerate(self.manuf_line.list_machines):
            if m.identical_machines != []:
                if m not in passed_machines:
                    indices_identical_machines = [i+1]
                    indices_identical_machines.extend([[machine.ID for machine in self.manuf_line.list_machines].index(id)+1 for id, machine in [(m.ID, m) for m in m.identical_machines]])
                    buffer_btn = customtkinter.CTkButton(master=self.buffer_state_frame, text=f"B{'+'.join(map(str, indices_identical_machines))}", width = 50, height=50, fg_color="green", text_color_disabled= "white", state="disabled", font=('Arial', 15))
                    buffer_btn.grid(row=2*(i+1), column=0, rowspan=2, padx=10, pady=(0, 5))
                    self.buffer_state_btn.append(buffer_btn)
                    
                    passed_machines.extend(m.identical_machines)

            else:
                buffer_btn = customtkinter.CTkButton(master=self.buffer_state_frame, text=f"B{i+1}", width = 50, height=50, fg_color="green", text_color_disabled= "white", state="disabled", font=('Arial', 15))
                buffer_btn.grid(row=2*(i+1), column=0, rowspan=2, padx=10, pady=(0, 5))
                self.buffer_state_btn.append(buffer_btn)
            
            buffer_capacity = customtkinter.CTkLabel(master=self.buffer_state_frame, text="Capacity = N/A", font=('Arial', 14), justify="right")
            buffer_capacity.grid(row=2*(i+1)+1, column=1, padx=10, pady=(0, 5))
            self.buffer_capacities.append(buffer_capacity)
            buffer_level= customtkinter.CTkLabel(master=self.buffer_state_frame, text="Level = N/A", font=('Arial', 14), justify="left")
            buffer_level.grid(row=2*(i+1), column=1, padx=10, pady=(0, 5))
            self.buffer_levels.append(buffer_level)
            machine_idle = customtkinter.CTkLabel(master=self.buffer_state_frame, text="Avg. Cycle Time = N/A", font=('Arial', 14), justify="left")
            machine_idle.grid(row=2*(i+1), column=2, padx=5, pady=(0, 5))
            self.machine_idles.append(machine_idle)
            machine_downtime = customtkinter.CTkLabel(master=self.buffer_state_frame, text="Total Downtime = N/A", font=('Arial', 14), justify="left")
            machine_downtime.grid(row=2*(i+1)+1, column=2, padx=5, pady=(0, 5))
            self.machine_downtimes.append(machine_downtime)
            try:
                m.buffer_btn= [buffer_btn, buffer_capacity,buffer_level,machine_idle, machine_downtime]
            except: 
                pass
            
        
        env.process(clock(self.manuf_line.env, self.manuf_line, self))
        thread = threading.Thread(target=run, args={self.manuf_line,})
        thread.start()


    def start_loading(self):
        loading_window = LoadingScreen(self) 
        # while True: 
        #     print("waiting")
        #     if not isinstance(self.toplevel_window, ReportingWindow):
        #         self.toplevel_window = loading_window
        #     else:
        #         print("Destroying the loading window")
        #         loading_window.destroy()
        #         break


    def start_reporting(self):
        # self.thread = threading.Thread(target=self.start_loading)
        # self.thread.start()
        # self.create_reporting_window()

        #self.toplevel_window = LoadingScreen(self) 
        self.launch_sim_whileloading()
        self.toplevel_window = ReportingWindow(self, self.manuf_line)
        

    def launch_sim_whileloading(self):
        try:
            run(self.manuf_line)
        except Exception as e:
            print("-- Resetting the environment to avoid multi-sim problems. --")
            self.manuf_line.reset()
            run(self.manuf_line)


def generate_random_tasks(num_tasks):
    tasks = []
    for task_id in range(1, num_tasks + 1):
        machine_time = random.randint(1, 50)  # Adjust the range as needed
        #manual_time = random.randint(1, 50)  # Adjust the range as needed
        manual_time = 0
        task = Task(task_id, machine_time, manual_time)
        tasks.append(task)
    return tasks

def known_tasks(list):
    tasks = []
    for task_id, task_mt in enumerate(list):
        machine_time = task_mt  # Adjust the range as needed
        manual_time = 3 # Adjust the range as needed
        task = Task(task_id, machine_time, manual_time)
        tasks.append(task)
    return tasks

def run(assembly_line):
    assembly_line.run()
    #assembly_line.get_results()
    #list_machines = assembly_line.get_track()

def clock(env, assembly_line, app):
    shift_ct = []
    global_ct = []
    sim_time = []
    oee_list = []
    constant_x_values = []
    d_sim_time = deque(maxlen=250)
    d_shift_ct = deque(maxlen=250)
    d_global_ct = deque(maxlen=250)
    d_constant_x_values = deque(maxlen=250)
    d_oee_list = deque(maxlen=250)
    global last_breakdown 
    last_breakdown = 0
    
    
    global timeout
    timeout = 100

    def update_display():
        elapsed_seconds = env.now
        
        elapsed_seconds_shift = elapsed_seconds % (3600 * 8)
        elapsed_time_str = format_time(elapsed_seconds)
        app.sim_time_label.configure(text=elapsed_time_str)
        app.partsdone_label.configure(text='%d' % assembly_line.shop_stock_out.level)
        
        new_breakdown = None
        global last_breakdown
        for m in assembly_line.list_machines:
            # print("Input = " + m.ID + " --- " +  str(m.previous_machine))
            # print("Output = " + m.ID + " --- " +  str(m.next_machine))
            if (m.broken and env.now-last_breakdown>float(assembly_line.breakdowns["mttr"])) or (m.broken and last_breakdown<2):
                new_breakdown = env.now
                last_breakdown = new_breakdown

        if elapsed_seconds > 0 and assembly_line.shop_stock_out.level > 0  and all([machine.parts_done_shift > 0 for machine in assembly_line.list_machines]):
            if elapsed_seconds_shift < 1000:
                elapsed_seconds_shift = 100
            
            
            #print("Elapsed time = " + str(elapsed_seconds_shift) + " -  Level Shop Stock  = "  + str(assembly_line.list_machines[0].parts_done_shift))
            shift_cycle_time = elapsed_seconds_shift / assembly_line.list_machines[-1].parts_done_shift 
            
            if assembly_line.robot is not None:
                waiting_rate = 100*assembly_line.robot.waiting_time/env.now
                app.robot_waiting_time.configure(text='{:.1f}%'.format(waiting_rate))

            app.shift_ct_label.configure(text='%.2f s' % shift_cycle_time)
            cycle_time = elapsed_seconds / assembly_line.shop_stock_out.level
            app.annual_ct_label.configure(text='%.2f s' % cycle_time)
            #oee = 100*max([m.ct for m in assembly_line.list_machines])/cycle_time
            
            oee =assembly_line.takt_time/cycle_time
            app.oee_label.configure(text='%.2f' % oee)
            
            
            if elapsed_seconds > 500: #avoid warm-up
                draw_buffers(app, assembly_line)
                
                if new_breakdown != None:
                    update_plot_CT((shift_cycle_time, env.now), (cycle_time, env.now), oee, new_breakdown)
                    new_breakdown = None
                else:
                    update_plot_CT((shift_cycle_time, env.now), (cycle_time, env.now), oee)
               
    

    def update_plot_CT(new_y, new_y2, oee_val, new_breakdown=None):
    
        warm_up_period = 1000
        warm_up_passsed = False
        fg_color = 'white' if app.appearence_mode == "Dark" else 'black'

        shift_ct.append(new_y[0])
        global_ct.append(new_y2[0])
        sim_time.append(new_y[1])
        oee_list.append(oee_val)
        d_sim_time.append(new_y[1])
        d_shift_ct.append(new_y[0])
        d_global_ct.append(new_y2[0])
        d_oee_list.append(oee_val)
        index_warm_up = find_closest_index(d_sim_time, warm_up_period)
        axs[0].clear()
        axs[1].clear()
        if new_breakdown != None: 
            constant_x_values.append(new_y[1])
            d_constant_x_values.append(new_y[1])

        # Plot the data
        axs[0].plot(list(d_sim_time), list(d_shift_ct), label='Shift Cycle Time')
        axs[1].plot(list(d_sim_time), list(d_global_ct), color='orange', label='Global Cycle Time')
        axs[2].plot(list(d_sim_time), list(d_oee_list), color='green', label='Overall Equipment Effectiveness (OEE/TRS)')

        #fig.text(0.5, 0.5, 'Duration (s)', ha='center', va='center', color=fg_color)
        
       
        for x_value in list(d_constant_x_values):
            axs[0].axvline(x=x_value, color='red', linestyle='--')
            axs[1].axvline(x=x_value, color='red', linestyle='--')

        if max(sim_time) > warm_up_period and not warm_up_passsed:
            for ax in axs:
                ax.set_xlim(warm_up_period, max(list(d_sim_time)))
            axs[0].set_ylim(0.5*min(shift_ct[index_warm_up:]), 1.5*max(shift_ct[index_warm_up:]))
            axs[1].set_ylim(0.5*min(global_ct[index_warm_up:]), 1.5*max(global_ct[index_warm_up:]))
            axs[2].set_ylim(-10, 110)
            warm_up_passsed = True

        if  len(d_sim_time) == d_sim_time.maxlen:
            for ax in axs:
                ax.set_xlim(min(list(d_sim_time)), max(list(d_sim_time)))
            axs[0].set_ylim(0.5*min(list(d_shift_ct)), 1.5*max(list(d_shift_ct)))
            axs[1].set_ylim(0.5*min(list(d_global_ct)), 1.5*max(list(d_global_ct)))
            axs[2].set_ylim(-10, 110)
        
            

        fig.set_facecolor('#282C34' if app.appearence_mode == "Dark" else 'white')

        axs[0].set_ylabel('Shift Cycle Time (s)', color=fg_color)
        axs[0].set_title('Shift Cycle Time (s)', color=fg_color)
        axs[1].set_title('Avg. Annual Cycle Time (s)', color=fg_color)
        axs[2].set_title('Overall Equipment Effectiveness (%)', color=fg_color)
        axs[2].set_xlabel('Duration (s)', color=fg_color)
        for ax in axs:
            ax.set_facecolor('#282C34' if app.appearence_mode == "Dark" else 'white')
            ax.yaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
            ax.xaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
        
        fig.tight_layout()
        
        canvas_fig.draw()

    def clock_generator():
        while True:
            global timeout
            yield env.timeout(timeout)
            update_display()

    return clock_generator()


def increase_timeout():
    global timeout
    timeout = timeout + 100

def decrease_timeout():
    global timeout
    if timeout > 100:
        timeout = timeout - 100
    if timeout < 10:
        timeout = 1

if __name__ == "__main__":
    
    #df_tasks = pd.read_xml('./workplan_TestIsostatique_modified.xml', xpath=".//weldings//welding")
    #tasks = known_tasks(df_tasks["cycleTime"].astype(int).tolist())
    tasks = []
    config_file = 'config.yaml'
    #task_assignement = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3,  3, 3, 3, 3, 3, 3, 3 ]
    assembly_line = ManufLine(env, tasks, config_file=config_file)
    #assembly_line.set_CT_machines([20, 20, 20, 20])

    ## Compile with OEE diagram modifs + parallel machines + robots transport 
    
    random.seed(10)
    app = App(assembly_line)
    
    #app.open_setting_window()
    ctk.set_appearance_mode("dark")
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 6), sharex=True, facecolor="#282C34")
    fig.set_facecolor('#282C34' if app.appearence_mode == "Dark" else 'white')
    fg_color = 'white' if app.appearence_mode == "Dark" else 'black'
    axs[0].set_ylabel('Shift Cycle Time (s)', color=fg_color)
    axs[0].set_title('Shift Cycle Time (s)', color=fg_color)
    axs[1].set_ylabel('Global Cycle Time (s)', color=fg_color)
    axs[1].set_title('Avg. Annual Cycle Time(s)', color=fg_color)
    axs[1].set_xlabel('Duration (s)', color=fg_color)
    for ax in axs:
        ax.set_facecolor('#282C34' if app.appearence_mode == "Dark" else 'white')
        ax.yaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
        ax.xaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
    canvas_fig = FigureCanvasTkAgg(fig, master=app.viz_ct_frame)
    canvas_fig_widget = canvas_fig.get_tk_widget()
    canvas_fig_widget.pack(expand=True, fill=tk.BOTH)

    fig_m, ax_m = plt.subplots(nrows=1, ncols=1, figsize=(7, 3), sharex=True, facecolor="#282C34")
    fig_m.set_facecolor('#282C34' if app.appearence_mode == "Dark" else 'white')
    fg_color = 'white' if app.appearence_mode == "Dark" else 'black'
    ax_m.set_ylabel('Utilization (%)', color=fg_color)
    ax_m.set_title('Machine Uptime Rate', color=fg_color)
    ax_m.set_facecolor('#282C34' if app.appearence_mode == "Dark" else 'white')
    ax_m.yaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
    ax_m.xaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
    canvas_fig_m = FigureCanvasTkAgg(fig_m, master=app.main_footer)
    canvas_fig_widget_m = canvas_fig_m.get_tk_widget()
    #canvas_fig_widget_m.grid(row=0, column=0, padx=(10, 10), pady=(10, 0), sticky="nsew")
    canvas_fig_widget_m.pack(expand=True, fill=tk.BOTH)
    # window = ctk.CTk()
    # window.geometry("1280x720")
    # canvas = tk.Canvas(window, width=320, height=100, bg="#3D59AB")
    # canvas.pack(side=tk.TOP, anchor=tk.NW) 
    
    # env.process(clock(env, assembly_line, app))
    # thread = threading.Thread(target=run, args={assembly_line,})
    # thread.start()

    app.mainloop()