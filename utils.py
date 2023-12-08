import yaml
import simpy
import random
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
import tkinter as tk
import customtkinter
from customtkinter import *
from typing import Union, Tuple, Optional
import copy

class ManufLine:
    def __init__(self, env, n_machines, tasks, operators_assignement, tasks_assignement, config_file):

        """
        -- operators_assignement = [[1,2], [3], [4]] => 3 operators: first one operates on machine 1 and machine 2,
            second one operates on machine 3 and third one operates on machine 4.

        -- tasks_assignement = [1, 1, 2, 3, 3, 4, 4, 4 ....]

         """
        
        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        self.env = env
        self.parts_done = 0
        self.config = config
        self.sim_time = eval(self.config['sim_time'])
        self.breakdowns = self.config['breakdowns']
        self.first_machine = None
        self.supermarket_in = simpy.Container(env, capacity=float(config["supermarket"]["capacity"]), init=float(config["supermarket"]["initial"]))
        self.shop_stock_out = simpy.Container(env, capacity=float(config["shopstock"]["capacity"]), init=float(config["shopstock"]["initial"]))

        self.num_cycles = 0
        self.supermarket_n_refills = 0
        self.yearly_volume_obj = int(self.config['project_data']["yearly_volume"])
        self.tasks = tasks
        self.tasks_assignement = tasks_assignement
        self.operators_assignement = operators_assignement

        
        
        # machine_indices = [[] for _ in range(n_machines)]
            
        # for index, machine in enumerate(tasks_assignement):
        #         machine_indices[machine - 1].append(index)
            
        
        # if n_machines != max(tasks_assignement):
        #     raise ValueError('No match between assignement of tasks and number of machines.')

        # if n_machines != max(max(operators_assignement)):
        #     raise ValueError('No match between assignement of operators and number of machines.')
        
        self.list_machines = []
        # self.list_operators = [Operator(operators_assignement[i]) for i in range(len(operators_assignement))]

        # previous_machine = None
        # for i in range(n_machines):
           
        #     assigned_tasks = list(np.array(tasks)[machine_indices[i]])
            
        #     first, last = False, False
        #     if i == n_machines-1:
        #         last = True
        #     if i == 0:
        #         first = True

        #     assigned_operator_index = self.get_index_of_item([operator.assigned_machines for operator in self.list_operators], i+1)
            
        #     machine = Machine(self, env, "M"+str(i+1), assigned_tasks, self.config, operator=self.list_operators[assigned_operator_index], previous_machine=previous_machine, first=first, last=last, breakdowns=self.breakdowns['enabled'], mean_time_to_failure=eval(config['breakdowns']['mttf']), hazard_delays=config['hazard_delays']['enabled'])
        #     if machine.first:
        #         self.first_machine = machine
        #     previous_machine = machine
        #     self.list_machines.append(machine)    


    def get_index_of_item(self, list_of_lists, item):
        for index, sublist in enumerate(list_of_lists):
            if item in sublist:
                return index
        return None  
    
    def generate(self):
        return self.env
    
    def get_results(self):
        idle_times = []
        CTs = []
        
        for i, machine in enumerate(self.list_machines):
            idle_times_machine = []
            ct_machine = []
            for entry, exit in zip(machine.entry_times, machine.exit_times):
                idle_times_machine.append(exit-entry)
            idle_times.append(np.mean(idle_times_machine))

            for entry, finished in zip(machine.entry_times, machine.finished_times):
                ct_machine.append(finished-entry)
            CTs.append(np.mean(ct_machine))

        print("Machine Idle Times = ", idle_times)
        print("Downtime of Machines = ", [1000*machine.n_breakdowns for machine in self.list_machines])
        print("Parts done --", self.list_machines[-1].buffer_out.level)
        print("Cycle Time --", self.sim_time/self.list_machines[-1].buffer_out.level)

        return idle_times
    def get_track(self):
        for i, machine in enumerate(self.list_machines):
            if not machine.last:
                plt.plot([t[0] for t in machine.buffer_tracks], [t[1] for t in machine.buffer_tracks], label='Buffer M '+str(i+1))
                plt.legend()
            else:
                machine_last = machine
        plt.show()
        return self.list_machines


    def run(self):
        print(f"Current simulation time at the start: {self.env.now}")
        self.generate()
        for m in self.list_machines:
            self.process = self.env.process(m.machine_process())
            self.env.process(self.break_down(m))
        self.env.process(self.refill_market())
        self.env.run(until=self.sim_time)
        print(f"Current simulation time at the end: {self.env.now}")

    def reset_shift(self):
        print("Reset started")
        for i, machine in enumerate(self.list_machines):
            if machine.first:
                machine.buffer_out.items = []
            if machine.last:
                machine.buffer_in.items = []
                machine.buffer_out.items = []

            if not machine.first and not machine.last:
                machine.buffer_in.items = []
                machine.buffer_out.items = []

        machine.parts_done_shift = 1
        print('Machine ' + machine.ID + ' - Parts shift now = ', machine.parts_done_shift)
        print("Reset finished")
    
    def break_down(self, machine):
        """Break the machine every now and then."""
        if self.breakdowns["enabled"]:
            while True:
                yield self.env.timeout(machine.time_to_failure())
                if not machine.broken:
                    machine.n_breakdowns += 1
                    print("Machine " + machine.ID + " Broken :) ")
                    self.process.interrupt()    
    
    def set_CT_machines(self, CTs):

        if len(self.list_machines) != len(CTs):
            raise ValueError('No matching! You have chosen ' + str(len(self.list_machines)) + ' machines and have given ' + str(len(CTs)))

        for i, machine in enumerate(self.list_machines):
            machine.ct = CTs[i]

    def refill_market(self):
        
        while True:
            yield self.env.timeout(1)
            if self.supermarket_in.level < float(self.config["supermarket"]["refill-threshold"]):
                print("Start Refilling raw supermarket - Stock")
                time_refill_start= self.env.now
                try:
                    yield self.env.timeout(int(self.config["supermarket"]["refill-time"]))
                    print("Time to refill = ", self.env.now-time_refill_start)
                    self.supermarket_in.put(self.supermarket_in.capacity-self.supermarket_in.level)
                    self.supermarket_n_refills =self.supermarket_n_refills + 1
                except: 
                    print("Stock Refilling Interruped by machine breakdown")
                    pass
                
                print("Finished Refilling raw supermarket - Stock")
                yield self.env.timeout(0)
            else:
                yield self.env.timeout(0)


    def deplete_shopstock(self):
        pass

    def deliver_to_client(self):
        pass

    def create_machines(self, list_machines_config):
        """
        
        list_machines_config = [
        "Name",
        "MT",
        "CT",
        "Link",
        "BufferCapacity",
        "MTTF",
        "MTTR"]
            
        """

        machine_indices = [[] for _ in range(len(list_machines_config))]

        for index, n_machine in enumerate(self.tasks_assignement):
                machine_indices[n_machine - 1].append(index)
        

        if len(list_machines_config) != max(self.tasks_assignement):
            raise ValueError('No match between assignement of tasks and number of machines.')

        if len(list_machines_config) != max(max(self.operators_assignement)):
            raise ValueError('No match between assignement of operators and number of machines.')


        self.list_machines = []
        next_machine = None
        
        for i in range(len(list_machines_config)):
            assigned_tasks =  list(np.array(self.tasks)[machine_indices[i]])
            first, last = False, False
            if i == len(list_machines_config) - 1 :
                last = True
            if i == 0 or i == 1:
                first = True    


            try: 
                mttf = eval(list_machines_config[i][6])
                mttr = list_machines_config[i][7]
                buffer_capacity = list_machines_config[i][5]

            except:
                mttf = float(list_machines_config[i][6])
                mttr = float(list_machines_config[i][7])
                buffer_capacity = list_machines_config[i][5]
            
            print(list_machines_config[i][0] + ' mttf = ' + str(mttf) + ' - mttr = ' + str(mttr) + " - capac = " + str(buffer_capacity))
            machine = Machine(self, self.env, list_machines_config[i][0], assigned_tasks, self.config, first=first, last=last, breakdowns=self.breakdowns['enabled'], mttf=mttf, mttr=mttr, buffer_capacity=buffer_capacity ,hazard_delays=self.config['hazard_delays']['enabled'])
            if machine.first:
                self.first_machine = machine
            self.list_machines.append(machine)

        self.set_CT_machines([ct[3] for ct in list_machines_config])
        l = [m.ID for m in self.list_machines]
        for i, machine in enumerate(self.list_machines):
            try: 
                index = l.index(list_machines_config[i][4])
                machine.next_machine = self.list_machines[index]
                self.list_machines[index].buffer_in = simpy.Container(self.env, capacity=float(machine.buffer_capacity), init=0)
                machine.buffer_out = self.list_machines[index].buffer_in
         
                print("capacity == " + str(machine.buffer_capacity) + " --- " + str(machine.buffer_out.capacity))
            except:
                pass 

class Machine:
    def __init__(self, manuf_line, env, machine_id, assigned_tasks, config, operator=None, previous_machine = None, first = False, last=False, breakdowns=True, mttf=3600*24*7, mttr=3600, buffer_capacity=100, hazard_delays=False):
        self.mt = 0
        self.ID = machine_id
        self.env = env
        self.manuf_line = manuf_line
        self.entry_times = []  # List to store entry times of parts
        self.exit_times = []   # List to store exit times of parts
        self.finished_times = []
        self.n_breakdowns = 0
        self.buffer_tracks = []
        self.parts_done = 0
        self.parts_done_shift = 0
        self.ct = 0
        self.config = config
        self.buffer_btn = None
        self.buffer_capacity = buffer_capacity
        if first:
            self.buffer_in = manuf_line.supermarket_in
            self.buffer_out = simpy.Container(env, capacity=float(buffer_capacity), init=0)
           
        
        if last:
            self.buffer_in = simpy.Container(env, capacity=float(buffer_capacity), init=0)
            self.buffer_out = manuf_line.shop_stock_out
        
        if not first and not last:
            self.buffer_in = simpy.Container(env, capacity=float(buffer_capacity), init=0)
            self.buffer_out = simpy.Container(env, capacity=float(buffer_capacity), init=0)
        self.MTTF = mttf #Mean time to failure in seconds
        self.MTTR = mttr
        self.assigned_tasks = assigned_tasks
        
        self.operator = operator  # Assign the operator to the machine

        # Events to signal when the machine is loaded and when the operator is free
    
        

        self.last = last
        self.first = first
        self.broken = False
        self.breakdowns = breakdowns
        self.hazard_delays = 1 if hazard_delays else 0
        self.op_fatigue = config["fatigue_model"]["enabled"]
       
        #self.process = self.env.process(self.machine_process())
        #env.process(self.break_down())
        
        #self.env.process(self.manual_process())
        # if previous_machine:
        #     self.buffer_in = previous_machine.buffer_out
    
    def time_to_failure(self):
        """Return time until next failure for a machine."""
        #deterioration_factor = 1 + (self.env.now / self.simulation_duration)  # Adjust this factor as needed
        #adjusted_MTTF = self.MTTF / deterioration_factor
        val = random.expovariate(1/self.MTTF)
        return val
    
    def fatigue_model(self, elapsed_time, base_time):
        """
        Model operator fatigue based on elapsed time using a sigmoid function.

        Parameters:
        - elapsed_time (float): Elapsed time in hours.
        - base_time (float): Base time needed for a manual action without fatigue.
        - min_rate (float): Minimum fatigue rate.
        - max_rate (float): Maximum fatigue rate.

        Returns:
        - adjusted_time (float): Adjusted time for a manual action based on fatigue.
        """
        fatigue_rate =  float(self.config["fatigue_model"]["max-fatigue-rate"]) * sigmoid(elapsed_time, eval(self.config["fatigue_model"]["tau-fatigue"]))
        adjusted_time = (1+fatigue_rate) * base_time
        return adjusted_time

    def machine_process(self):
        bias_shape = 2  # shape parameter
        bias_scale = 1  # scale parameter
        num_samples = len(self.assigned_tasks)
        print(self.ID + " - Buffer capacity = " + str(self.buffer_out.capacity))
        while True: 
            
            # Reseting every shift   
            if self.env.now // (3600 * 8) == self.manuf_line.num_cycles +1:
                print("Reseting Shift Enter")
                self.manuf_line.num_cycles = self.manuf_line.num_cycles +1
                print("N of cycle = ", self.manuf_line.num_cycles)
                self.manuf_line.reset_shift()
                print("Reseting Shift out")


            if any([m.ct != 0 for m in self.manuf_line.list_machines]):
                if self.op_fatigue:
                    deterministic_time = self.fatigue_model(self.env.now/(self.manuf_line.num_cycles+1), self.ct)
                else:
                    deterministic_time = self.ct
            else:
                if self.op_fatigue:
                    deterministic_time = np.sum([task.machine_time+self.fatigue_model((self.manuf_line.num_cycles+1), task.manual_time) for task in self.assigned_tasks])
                else:
                    deterministic_time = np.sum([task.machine_time+task.manual_time for task in self.assigned_tasks])

            done_in =  deterministic_time + self.hazard_delays*np.mean(weibull_min.rvs(bias_shape, scale=bias_scale, size=num_samples))
            start = self.env.now
            self.buffer_tracks.append((self.env.now, self.buffer_out.level))
            
    
            while done_in > 0:
                try:
                    self.entry_times.append(self.env.now)
                    yield self.buffer_in.get(1)
                    self.exit_times.append(self.env.now)
                    yield self.env.timeout(done_in)
                    finish_time = self.env.now
                    while self.buffer_out.level == self.buffer_out.capacity:
                        yield self.env.timeout(1)
                    self.finished_times.append(finish_time)
                    self.loaded = False
                    yield self.buffer_out.put(1)
                    done_in = 0
                except simpy.Interrupt:
                    self.broken = True
                    done_in -= self.env.now - start
                    try:
                        yield self.env.timeout(self.MTTR) #Time to repair
                    except: 
                         pass
                    self.broken = False
           
            self.parts_done = self.parts_done +1
            
            self.parts_done_shift = self.parts_done_shift+1
            yield self.env.timeout(0)
                




class Task:
    def __init__(self, ID, machine_time, manual_time):
        self.ID = ID
        self.machine_time = machine_time
        self.manual_time = manual_time

class Operator:
    def __init__(self, assigned_machines):
        self.assigned_machines = assigned_machines
        self.wc = 0
        self.free = True


def create_image_canvas(parent, image_paths):
    canvas = tk.Canvas(parent, width=400, height=400)
    canvas.pack()

    for i, image_path in enumerate(image_paths):
        image = tk.PhotoImage(file=image_path)
        canvas.create_image(50 + i * 100, 50, anchor=tk.NW, image=image)


def format_time(seconds):
    years, seconds = divmod(seconds, 31536000)  # 60 seconds/minute * 60 minutes/hour * 24 hours/day * 365.25 days/year
    months, seconds = divmod(seconds, 2592000)   # 60 seconds/minute * 60 minutes/hour * 24 hours/day * 30.44 days/month
    days, seconds = divmod(seconds, 86400)      # 60 seconds/minute * 60 minutes/hour * 24 hours/day
    hours, seconds = divmod(seconds, 3600)       # 60 seconds/minute * 60 minutes/hour
    minutes, seconds = divmod(seconds, 60)       # 60 seconds/minute

    time_str = ""
    non_zero_parts = 0

    if years > 0:
        time_str += f"{int(years)} years, "
        non_zero_parts += 1
    if months > 0 and non_zero_parts < 3:
        time_str += f"{int(months)} months, "
        non_zero_parts += 1
    if days > 0 and non_zero_parts < 3:
        time_str += f"{int(days)} days, "
        non_zero_parts += 1
    if hours > 0 and non_zero_parts < 3:
        time_str += f"{int(hours)} hours, "
        non_zero_parts += 1
    if minutes > 0 and non_zero_parts < 3:
        time_str += f"{int(minutes)} minutes, "
        non_zero_parts += 1
    # if non_zero_parts < 3:
    #     time_str += f"{seconds:.2f} seconds"

    return time_str.rstrip(", ")


def draw_buffers(app, assembly_line):
    machine_util = []
    machine_names = []

    # Draw supermarket of the whole line
    if assembly_line.supermarket_in.level < assembly_line.supermarket_in.capacity * 0.3:
        app.supermarket_btn.configure(fg_color="red")
    elif assembly_line.supermarket_in.level < assembly_line.supermarket_in.capacity * 0.5:
        app.supermarket_btn.configure(fg_color="orange")
    else:
        app.supermarket_btn.configure(fg_color="green")
    app.supermarket_capacity.configure(text=f"Capacity = {assembly_line.supermarket_in.capacity}")
    app.supermarket_level.configure(text=f"Level = {assembly_line.supermarket_in.level}")
    app.refill_label.configure(text="N. of Refills = %d" % int(assembly_line.supermarket_n_refills))

    only_one = False
    
    for i, m in enumerate(assembly_line.list_machines):
        ### Raw supermarket (raw stock)
        machine_names.append(m.ID)
        
        if m.first:
            if not only_one:
                only_one = True
                if m.buffer_out.level < m.buffer_out.capacity * 0.2:
                    m.buffer_btn[0].configure(fg_color="green")
                elif m.buffer_out.level < m.buffer_out.capacity * 0.8:
                    m.buffer_btn[0].configure(fg_color="orange")
                else:
                    m.buffer_btn[0].configure(fg_color="red")
            idle_times_machine = []
            for entry, exit in zip(m.entry_times, m.exit_times):
                idle_times_machine.append(exit-entry)
            idle_time = np.sum(idle_times_machine)
            try:
                m.buffer_btn[1].configure(text=f"Capacity = {m.buffer_out.capacity}")
                m.buffer_btn[2].configure(text=f"Level = {m.buffer_out.level}")
                m.buffer_btn[3].configure( text="Waiting/Idle Time = %.2f" % idle_time)
                m.buffer_btn[4].configure(text="Total Downtime = %.2f" % float(m.MTTR)*m.n_breakdowns)
            except:
                pass
        else:
            if m.buffer_out.level < m.buffer_out.capacity * 0.2:
                m.buffer_btn[0].configure(fg_color="green")
            elif m.buffer_out.level < m.buffer_out.capacity * 0.8:
                m.buffer_btn[0].configure(fg_color="orange")
            else:
                m.buffer_btn[0].configure(fg_color="red")

            idle_times_machine = []
            for entry, exit in zip(m.entry_times, m.exit_times):
                idle_times_machine.append(exit-entry)
            idle_time = np.sum(idle_times_machine)
            
            m.buffer_btn[1].configure(text=f"Capacity = {m.buffer_out.capacity}")
            m.buffer_btn[2].configure(text=f"Level = {m.buffer_out.level}")
            m.buffer_btn[3].configure( text="Waiting/Idle Time = %.2f" % idle_time)
            m.buffer_btn[4].configure(text="Total Downtime = %.2f" % float(float(assembly_line.breakdowns['mttr'])*m.n_breakdowns))
        uptime_m = 100*(1-(float(float(m.MTTR)*m.n_breakdowns)+np.sum(idle_times_machine))/assembly_line.env.now)
        machine_util.append(max(0, uptime_m))
    app.update_machine_util_viz(machine_names, machine_util)



def update_buffer_viz(canvas, assembly_line):
    #canvas.delete("all")

    # Define colors based on buffer status
    for i, m in enumerate(assembly_line.list_machines):
        if m.buffer_out.level < m.buffer_out.capacity * 0.2:
            color = "green"
        elif m.buffer_out.level < m.buffer_out.capacity * 0.8:
            color = "orange"
        else:
            color = "red"

        # Adjust vertical separation between rectangles
        vertical_separation = 30
        idle_times_machine = []
        for entry, exit in zip(m.entry_times, m.exit_times):
            idle_times_machine.append(exit-entry)
        idle_time = np.mean(idle_times_machine)
        # Draw the rectangle representing the buffer
        buffer_width = 50
        buffer_height = 20
        x_start = 10
        y_start = 10 + i * (buffer_height + vertical_separation)
        x_end = x_start + buffer_width
        y_end = y_start + buffer_height
        canvas.create_rectangle(x_start, y_start, x_end, y_end, fill=color)

        # Display text with buffer level and capacity

        text_x = 80
        text_y = 10 + i * (buffer_height + vertical_separation)
        canvas.create_text(text_x, text_y, anchor="w", text=f"Level: {m.buffer_out.level}")
        canvas.create_text(text_x, text_y + 20, anchor="w", text=f"Capacity: {m.buffer_out.capacity}")
        canvas.create_text(text_x + 90, text_y, anchor="w", text="Avg. Idle Time: %.2f" % idle_time)
        canvas.create_text(text_x + 90, text_y +20, anchor="w", text="Total Downtime: %.2f" % float(float(assembly_line.breakdowns['mttr'])*m.n_breakdowns))
        label_x = (x_start + x_end) / 2
        label_y = (y_start + y_end) / 2
        canvas.create_text(label_x, label_y, fill="white", text=f"B{i + 1}")

def find_closest_index(lst, target):
    return min(range(len(lst)), key=lambda i: abs(lst[i] - target))

class CTkTable(customtkinter.CTkFrame):
    """ CTkTable Widget """
    
    def __init__(
        self,
        master: any,
        row: int = None,
        column: int = None,
        padx: int = 1, 
        pady: int = 0,
        values: list = [[None]],
        colors: list = [None, None],
        orientation: str = "horizontal",
        color_phase: str = "horizontal",
        border_width: int = 0,
        text_color: str = None,
        border_color: str = None,
        font: tuple = None,
        header_color: str = None,
        corner_radius: int = 25,
        write: str = False,
        command = None,
        anchor = "c",
        hover_color = None,
        hover = False,
        justify = "center",
        wraplength: int = 1000,
        **kwargs):
        
        super().__init__(master, fg_color="transparent")

        self.master = master # parent widget
        self.rows = row if row else len(values) # number of default rows
        self.columns = column if column else len(values[0])# number of default columns
        self.padx = padx # internal padding between the rows/columns
        self.pady = pady
        self.command = command
        self.values = values # the default values of the table
        self.colors = colors # colors of the table if required
        self.header_color = header_color # specify the topmost row color
        self.phase = color_phase
        self.corner = corner_radius
        self.write = write
        self.justify = justify
        if self.write:
            border_width = border_width=+1
        if hover_color is not None:
            hover=True
        else:
            hover=False
        self.anchor = anchor
        self.wraplength = wraplength
        self.hover = hover 
        self.border_width = border_width
        self.hover_color = customtkinter.ThemeManager.theme["CTkButton"]["hover_color"] if hover_color is None else hover_color
        self.orient = orientation
        self.border_color = customtkinter.ThemeManager.theme["CTkButton"]["border_color"] if border_color is None else border_color
        self.text_color = customtkinter.ThemeManager.theme["CTkLabel"]["text_color"] if text_color is None else text_color
        self.font = font
        # if colors are None then use the default frame colors:
        self.data = {}
        self.fg_color = customtkinter.ThemeManager.theme["CTkFrame"]["fg_color"] if not self.colors[0] else self.colors[0]
        self.fg_color2 = customtkinter.ThemeManager.theme["CTkFrame"]["top_fg_color"] if not self.colors[1] else self.colors[1]

        if self.colors[0] is None and self.colors[1] is None:
            if self.fg_color==self.master.cget("fg_color"):
                self.fg_color = customtkinter.ThemeManager.theme["CTk"]["fg_color"]
            if self.fg_color2==self.master.cget("fg_color"):
                self.fg_color2 = customtkinter.ThemeManager.theme["CTk"]["fg_color"]
            
        self.frame = {}
        self.draw_table(**kwargs)
        
    def draw_table(self, **kwargs):
        
        """ draw the table """
        for i in range(self.rows):
            for j in range(self.columns):
                if self.phase=="horizontal":
                    if i%2==0:
                        fg = self.fg_color
                    else:
                        fg = self.fg_color2
                else:
                    if j%2==0:
                        fg = self.fg_color
                    else:
                        fg = self.fg_color2
                        
                if self.header_color:
                    if self.orient=="horizontal":
                        if i==0:
                            fg = self.header_color
                    else:
                        if j==0:
                            fg = self.header_color

                corner_radius = self.corner    
                if i==0 and j==0:
                    corners = ["", fg, fg, fg]
                elif i==self.rows-1 and j==self.columns-1:
                    corners = [fg ,fg, "", fg]
                elif i==self.rows-1 and j==0:
                    corners = [fg ,fg, fg, ""]
                elif i==0 and j==self.columns-1:
                    corners = [fg , "", fg, fg]
                else:
                    corners = [fg, fg, fg, fg]
                    corner_radius = 0
 
                if self.values:
                    try:
                        if self.orient=="horizontal":
                            value = self.values[i][j]
                        else:
                            value = self.values[j][i]
                    except IndexError: value = " "
                else:
                    value = " "
                    
                if value=="":
                    value = " "
                    
                if (i,j) in self.data.keys():
                    if self.data[i,j]["args"]: 
                        args = self.data[i,j]["args"]
                    else:
                        args = copy.deepcopy(kwargs)
                else:
                    args = copy.deepcopy(kwargs)
                
                self.data[i,j] = {"row": i, "column" : j, "value" : value, "args": args}
                
                args = self.data[i,j]["args"]
                
                if "text_color" not in args:
                    args["text_color"] = self.text_color
                if "border_width" not in args:
                    args["border_width"] = self.border_width
                if "border_color" not in args:
                    args["border_color"] = self.border_color
                if "fg_color" not in args:
                    args["fg_color"] = fg

                if self.write:
                    if "justify" not in args:
                        args["justify"] = self.justify
                    if self.padx==1: self.padx=0
                    self.frame[i,j] = customtkinter.CTkEntry(self,
                                                             font=self.font,
                                                             corner_radius=0,
                                                             **args)
                    self.frame[i,j].insert("0", value)
                    self.frame[i,j].bind("<Key>", lambda e, row=i, column=j, data=self.data: self.after(100, lambda: self.manipulate_data(row, column)))
                    self.frame[i,j].grid(column=j, row=i, padx=self.padx, pady=self.pady, sticky="nsew")
                    if self.header_color:
                        if i==0:
                            self.frame[i,j].configure(state="readonly")
    
                else:
                    if "anchor" not in args:
                        args["anchor"] = self.anchor
                    if "hover_color" not in args:
                        args["hover_color"] = self.hover_color
                    if "hover" not in args:
                        args["hover"] = self.hover
                    self.frame[i,j] = customtkinter.CTkButton(self, background_corner_colors=corners,
                                                              font=self.font, 
                                                              corner_radius=corner_radius,
                                                              text=value, 
                                                              command=(lambda e=self.data[i,j]: self.command(e)) if self.command else None, **args)
                    self.frame[i,j].grid(column=j, row=i, padx=self.padx, pady=self.pady, sticky="nsew")
                    self.frame[i,j]._text_label.config(wraplength=self.wraplength)
                self.rowconfigure(i, weight=1)
                self.columnconfigure(j, weight=1)
                
    def manipulate_data(self, row, column):
        """ entry callback """
        self.update_data()
        data = self.data[row,column]
        if self.command: self.command(data)
        
    def update_data(self):
        """ update the data when values are changes """
        for i in self.frame:
            if self.write:
                self.data[i]["value"]=self.frame[i].get()
            else:
                self.data[i]["value"]=self.frame[i].cget("text")

        self.values = []
        for i in range(self.rows):
            row_data = []
            for j in range(self.columns):
                row_data.append(self.data[i,j]["value"])
            self.values.append(row_data)
            
    def edit_row(self, row, value=None, **kwargs):
        """ edit all parameters of a single row """
        for i in range(self.columns):
            self.frame[row, i].configure(**kwargs)
            self.data[row, i]["args"].update(kwargs)
            if value:
                self.insert(row, i, value)
        self.update_data()
        
    def edit_column(self, column, value=None, **kwargs):
        """ edit all parameters of a single column """
        for i in range(self.rows):
            self.frame[i, column].configure(**kwargs)
            self.data[i, column]["args"].update(kwargs)
            if value:
                self.insert(i, column, value)
        self.update_data()
        
    def update_values(self, values, **kwargs):
        """ update all values at once """
        for i in self.frame.values():
            i.destroy()
        self.frame = {}
        self.values = values
        self.draw_table(**kwargs)
        self.update_data()
        
    def add_row(self, values, index=None, **kwargs):
        """ add a new row """
        for i in self.frame.values():
            i.destroy()
        self.frame = {}
        if index is None:
            index = len(self.values)      
        try:
            self.values.insert(index, values)
            self.rows+=1
        except IndexError: pass
 
        self.draw_table(**kwargs)
        self.update_data()
        
    def add_column(self, values, index=None, **kwargs):
        """ add a new column """
        for i in self.frame.values():
            i.destroy()
        self.frame = {}
        if index is None:
            index = len(self.values[0])
        x = 0
        for i in self.values:
            try:
                i.insert(index, values[x])
                x+=1
            except IndexError: pass
        self.columns+=1
        self.draw_table(**kwargs)
        self.update_data()
        
    def delete_row(self, index=None):
        """ delete a particular row """
        if index is None or index>=len(self.values):
            index = len(self.values)-1
        self.values.pop(index)
        for i in self.frame.values():
            i.destroy()
        self.rows-=1
        self.frame = {}
        self.draw_table()
        self.update_data()
        
    def delete_column(self, index=None):
        """ delete a particular column """
        if index is None or index>=len(self.values[0]):
            index = len(self.values)-1
        for i in self.values:
            i.pop(index)
        for i in self.frame.values():
            i.destroy()
        self.columns-=1
        self.frame = {}
        self.draw_table()
        self.update_data()
        
    def delete_rows(self, indices=[]):
        """ delete a particular row """
        if len(indices)==0:
            return
        self.values = [v for i, v in enumerate(self.values) if i not in indices]
        for i in indices:
            for j in range(self.columns):
                self.data[i, j]["args"] = ""
        for i in self.frame.values():
            i.destroy()
        self.rows -= len(set(indices))
        self.frame = {}
        self.draw_table()
        self.update_data()
        
    def delete_columns(self, indices=[]):
        """ delete a particular column """
        if len(indices)==0:
            return
        x = 0
        
        for k in self.values:
            self.values[x] = [v for i, v in enumerate(k) if i not in indices]
            x+=1
        for i in indices:
            for j in range(self.rows):
                self.data[j, i]["args"] = ""
                
        for i in self.frame.values():
            i.destroy()
        self.columns -= len(set(indices))
        self.frame = {}
        self.draw_table()
        self.update_data()
        
    def get_row(self, row):
        return self.values[row]
    
    def get_column(self, column):
        column_list = []
        for i in self.values:
            column_list.append(i[column])
        return column_list

    def select_row(self, row):
        self.edit_row(row, fg_color=self.hover_color)
        if self.orient!="horizontal":
            if self.header_color:
                self.edit_column(0, fg_color=self.header_color)
        else:
            if self.header_color:
                self.edit_row(0, fg_color=self.header_color)
        return self.get_row(row)
    
    def select_column(self, column):
        self.edit_column(column, fg_color=self.hover_color)
        if self.orient!="horizontal":
            if self.header_color:
                self.edit_column(0, fg_color=self.header_color)
        else:
            if self.header_color:
                self.edit_row(0, fg_color=self.header_color)
        return self.get_column(column)
    
    def deselect_row(self, row):
        self.edit_row(row, fg_color=self.fg_color if row%2==0 else self.fg_color2)
        if self.orient!="horizontal":
            if self.header_color:
                self.edit_column(0, fg_color=self.header_color)
        else:
            if self.header_color:
                self.edit_row(0, fg_color=self.header_color)
                
    def deselect_column(self, column):
        for i in range(self.rows):
            self.frame[i,column].configure(fg_color=self.fg_color if i%2==0 else self.fg_color2)
        if self.orient!="horizontal":
            if self.header_color:
                self.edit_column(0, fg_color=self.header_color)
        else:
            if self.header_color:
                self.edit_row(0, fg_color=self.header_color)

    def select(self, row, column):
        self.frame[row,column].configure(fg_color=self.hover_color)

    def deselect(self, row, column):
        self.frame[row,column].configure(fg_color=self.fg_color if row%2==0 else self.fg_color2)
        
    def insert(self, row, column, value, **kwargs):
        """ insert value in a specific block [row, column] """
        if self.write:
            self.frame[row,column].delete(0, customtkinter.END)
            self.frame[row,column].insert(0, value)
            self.frame[row,column].configure(**kwargs)
        else:        
            self.frame[row,column].configure(text=value, **kwargs)
        if kwargs: self.data[row,column]["args"].update(kwargs)
        self.update_data()
        
    def delete(self, row, column, **kwargs):
        """ delete a value from a specific block [row, column] """
        if self.write:
            self.frame[row,column].delete(0, customtkinter.END)
            self.frame[row,column].configure(**kwargs)
        else:     
            self.frame[row,column].configure(text="", **kwargs)
        if kwargs: self.data[row,column]["args"].update(kwargs)
        self.update_data()
        
    def get(self, row=None, column=None):
        if row and column:
            return self.data[row,column]["value"]
        else:
            return self.values
    
    def configure(self, **kwargs):
        """ configure table widget attributes"""
        
        if "colors" in kwargs:
            self.colors = kwargs.pop("colors")
            self.fg_color = self.colors[0]
            self.fg_color2 = self.colors[1]
        if "header_color" in kwargs:
            self.header_color = kwargs.pop("header_color")
        if "rows" in kwargs:
            self.rows = kwargs.pop("rows")
        if "columns" in kwargs:
            self.columns = kwargs.pop("columns")
        if "values" in kwargs:
            self.values = kwargs.pop("values")
        if "padx" in kwargs:
            self.padx = kwargs.pop("padx")
        if "padx" in kwargs:
            self.pady = kwargs.pop("pady")
        if "wraplength" in kwargs:
            self.wraplength = kwargs.pop("wraplength")
            
        self.update_values(self.values, **kwargs)




class CTkInputDialogSetting(CTkToplevel):
    """
    Dialog with extra window, message, entry widget, cancel and ok button.
    For detailed information check out the documentation.
    """

    def __init__(self,
                 fg_color: Optional[Union[str, Tuple[str, str]]] = None,
                 text_color: Optional[Union[str, Tuple[str, str]]] = None,
                 button_fg_color: Optional[Union[str, Tuple[str, str]]] = None,
                 button_hover_color: Optional[Union[str, Tuple[str, str]]] = None,
                 button_text_color: Optional[Union[str, Tuple[str, str]]] = None,
                 entry_fg_color: Optional[Union[str, Tuple[str, str]]] = None,
                 entry_border_color: Optional[Union[str, Tuple[str, str]]] = None,
                 entry_text_color: Optional[Union[str, Tuple[str, str]]] = None,
                 title: str = "CTkDialog",
                 font: Optional[Union[tuple, CTkFont]] = None,
                 text: Optional[Union[str, Tuple[str, str]]] = "CTkDialog"):

        super().__init__(fg_color=fg_color)

        self._fg_color = ThemeManager.theme["CTkToplevel"]["fg_color"] if fg_color is None else self._check_color_type(fg_color)
        self._text_color = ThemeManager.theme["CTkLabel"]["text_color"] if text_color is None else self._check_color_type(button_hover_color)
        self._button_fg_color = ThemeManager.theme["CTkButton"]["fg_color"] if button_fg_color is None else self._check_color_type(button_fg_color)
        self._button_hover_color = ThemeManager.theme["CTkButton"]["hover_color"] if button_hover_color is None else self._check_color_type(button_hover_color)
        self._button_text_color = ThemeManager.theme["CTkButton"]["text_color"] if button_text_color is None else self._check_color_type(button_text_color)
        self._entry_fg_color = ThemeManager.theme["CTkEntry"]["fg_color"] if entry_fg_color is None else self._check_color_type(entry_fg_color)
        self._entry_border_color = ThemeManager.theme["CTkEntry"]["border_color"] if entry_border_color is None else self._check_color_type(entry_border_color)
        self._entry_text_color = ThemeManager.theme["CTkEntry"]["text_color"] if entry_text_color is None else self._check_color_type(entry_text_color)

        self._user_input = []
        self._running: bool = False
        self._title = title
        self._text = text
        self._font = font
        self._entries = []

        self.title(self._title)
        self.lift()  # lift window on top
        self.attributes("-topmost", True)  # stay on top
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.after(10, self._create_widgets)  # create widgets with slight delay, to avoid white flickering of background
        self.resizable(False, False)
        self.grab_set()  # make other windows not clickable

    def _create_widgets(self):
        j = 0
        self.grid_columnconfigure((0, 1), weight=1)
        #self.rowconfigure(0, weight=1)
        for i in range(len(self._text)):
            if type(self._text[i]) == str:
                self._label = CTkLabel(master=self,
                                    width=300,
                                    wraplength=300,
                                    fg_color="transparent",
                                    text_color=self._text_color,
                                    text=self._text[i],
                                    font=self._font)
                self._label.grid(row=2*i, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

                self._entry = CTkEntry(master=self,
                                    width=230,
                                    fg_color=self._entry_fg_color,
                                    border_color=self._entry_border_color,
                                    text_color=self._entry_text_color,
                                    font=self._font)
                self._entry.grid(row=2*i+1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")
                self._entries.append(self._entry)
            else:
                for j in range(len(self._text[i])):
                    self._label = CTkLabel(master=self,
                                        width=150,
                                        wraplength=150,
                                        fg_color="transparent",
                                        text_color=self._text_color,
                                        text=self._text[i][j],
                                        font=self._font)
                    self._label.grid(row=2*i, column=j,  columnspan=1, padx=10, pady=10, sticky="ew")

                    self._entry = CTkEntry(master=self,
                                        width=230,
                                        fg_color=self._entry_fg_color,
                                        border_color=self._entry_border_color,
                                        text_color=self._entry_text_color,
                                        font=self._font)
                    self._entry.grid(row=2*i+1, column=j,  columnspan=1, padx=10, pady=(0, 10), sticky="ew")
                    self._entries.append(self._entry)
               
        self._ok_button = CTkButton(master=self,
                                    width=100,
                                    border_width=0,
                                    fg_color=self._button_fg_color,
                                    hover_color=self._button_hover_color,
                                    text_color=self._button_text_color,
                                    text='Ok',
                                    font=self._font,
                                    command=self._ok_event)
        self._ok_button.grid(row=2*(len(self._text)+1), column=0, columnspan=1, padx=(10, 10), pady=(20, 20), sticky="ew")

        self._cancel_button = CTkButton(master=self,
                                        width=100,
                                        border_width=0,
                                        fg_color=self._button_fg_color,
                                        hover_color=self._button_hover_color,
                                        text_color=self._button_text_color,
                                        text='Cancel',
                                        font=self._font,
                                        command=self._cancel_event)
        self._cancel_button.grid(row=2*(len(self._text)+1), column=1, columnspan=1, padx=(10, 10), pady=(20, 20), sticky="ew")

        self.after(150, lambda: self._entry.focus())  # set focus to entry with slight delay, otherwise it won't work
        self._entry.bind("<Return>", self._ok_event)

    def _ok_event(self, event=None):
        for i in range(len(self._entries)):
            self._user_input.append(self._entries[i].get())
        self.grab_release()
        self.destroy()

    def _on_closing(self):
        self.grab_release()
        self.destroy()

    def _cancel_event(self):
        self.grab_release()
        self.destroy()

    def get_input(self):
        self.master.wait_window(self)
        return self._user_input
    

def sigmoid(x, tau):
    """
    Sigmoid function.

    Parameters:
    - x (float): Input value.

    Returns:
    - float: Sigmoid of x.
    """
    return 1 / (1 + np.exp(-x/tau))