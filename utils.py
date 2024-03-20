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
import ast
import csv

class ManufLine:
    def __init__(self, env, tasks, operators_assignement=None, tasks_assignement=None, config_file=None):

        """
        -- operators_assignement = [[1,2], [3], [4]] => 3 operators: first one operates on machine 1 and machine 2,
            second one operates on machine 3 and third one operates on machine 4.

        -- tasks_assignement = [1, 1, 2, 3, 3, 4, 4, 4 ....]

         """
        try:
            with open(config_file, 'r') as stream:
                config = yaml.safe_load(stream)
        except:
            config_file = os.path.join(os.path.dirname(sys.executable), 'config.yaml')
            with open(config_file, 'r') as stream:
                config = yaml.safe_load(stream)
        self.env = env
        self.parts_done = 0
        self.config = config
        self.sim_time = eval(self.config['sim_time'])
        self.machine_config_data = []
        self.breakdowns = self.config['breakdowns']
        self.breakdowns_switch = self.config['breakdowns']["enabled"]
        self.repairmen = simpy.PreemptiveResource(env, capacity=1)
        self.first_machine = None
        self.stock_capacity = float(config["supermarket"]["capacity"])
        self.stock_initial = float(config["supermarket"]["initial"])
        self.safety_stock = 0
        self.refill_time = None
        self.refill_size = 1
        self.reset_shift_bool = False

        self.supermarket_in = simpy.Container(env, capacity=self.stock_capacity, init=self.stock_initial)
        self.shop_stock_out = simpy.Container(env, capacity=float(config["shopstock"]["capacity"]), init=float(config["shopstock"]["initial"]))

        self.num_cycles = 0
        self.supermarket_n_refills = 0
        self.takt_time = int(self.config['project_data']["takt_time"])
        self.tasks = tasks
        self.tasks_assignement = tasks_assignement
        self.operators_assignement = operators_assignement
        self.robot = None
        self.robots_list = []
        self.n_robots = 1
        self.robot_strategy = 0
        self.local = True
        #self.ADAS_robot = simpy.Resource(env, capacity=1)

        self.buffer_tracks = []
        self.robot_states = []
        self.machines_states = []
        self.machines_CT = []
        self.machines_idle_times = []
        self.list_machines = []
        self.machines_breakdown = []
        self.sim_times_track = []
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

    # def set_up_input_output():
    #     self.supermarket_in.capacity = self.manuf_line.stock_capacity
    #     self.supermarket_in.init = self.manuf_line.stock_initial

    def get_index_of_item(self, list_of_lists, item):
        for index, sublist in enumerate(list_of_lists):
            if item in sublist:
                return index
        return None  
    
    def generate(self):
        return self.env
    
    def get_results(self, save=False, experiment_number=1, track=False):
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

        # print("Mean Machine Idle Times = ", idle_times)
        # print("Waiting Time of Machines", [machine.waiting_time for machine in self.list_machines])
        # print("Downtime of Machines = ", [machine.n_breakdowns for machine in self.list_machines])
        # print("Mean CT of Machines = ", CTs)
        # print("Parts done --", self.shop_stock_out.level)
        # print("Cycle Time --", self.sim_time/self.shop_stock_out.level)

        if save and track:
            csv_file_path = "./results/forecast_data.csv"
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)

                sim_time = list(self.sim_times_track)
                print("Sim time = ", sim_time)
                waiting_times = list(self.machines_idle_times)
                breakdowns =  list(self.machines_breakdown)
                buffer_tracks = self.buffer_tracks
                robots_states = self.robot_states
                machines_state = self.machines_states
                machines_ct = self.machines_CT
                tracksim = False
                cycle_time = self.sim_time/self.shop_stock_out.level
                if buffer_tracks != [] or tracksim:
                    print("Printed track sim")
                    if experiment_number == 1:
                        writer.writerow(["Sim Instant", "Robot State", "Machines State", "Machine CT", "Machines Breakdowns", "Machines Idle Time", "Buffers State"])
                
                    for i in range(len(sim_time)):
                        writer.writerow([sim_time[i], robots_states[i], machines_state[i], machines_ct[i], breakdowns[i], waiting_times[i], buffer_tracks[i]])
                else:
                 
                    writer.writerow(["Machine ID", "Buffer Tracks"])
                    sim_time = []
                    buffer_tracks = []
                    for machine in self.list_machines:
                        sim_time.append([t[0] for t in machine.buffer_tracks])
                        buffer_tracks.append([t[1] for t in machine.buffer_tracks])

                    for i in range(len(buffer_tracks)):
                        writer.writerow([self.list_machines[i].ID, buffer_tracks[i]])

            return waiting_times, cycle_time, breakdowns

        if save and not track:
            csv_file_path = './results/results.csv'
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)

                if experiment_number == 1:
                    writer.writerow(["Experiment", "Mean Idle Times", "Sum Idle Times", "Downtime", "Mean CT", "Parts Done", "Cycle Time"])

                waiting_times = [machine.waiting_time for machine in self.list_machines]
                breakdowns =  [machine.n_breakdowns for machine in self.list_machines]
                mean_ct = [self.sim_time/machine.parts_done for machine in self.list_machines]
                parts_done = self.shop_stock_out.level
                cycle_time = self.sim_time/self.shop_stock_out.level
                print("buffer tracks  = ", [len(machine.buffer_tracks) for machine in self.list_machines])
                writer.writerow([experiment_number, idle_times, waiting_times, breakdowns, mean_ct, parts_done, cycle_time])
            return waiting_times, cycle_time, breakdowns
        if not save:
            waiting_times = [machine.waiting_time for machine in self.list_machines]
            breakdowns =  [machine.n_breakdowns for machine in self.list_machines]
            if self.shop_stock_out.level != 0:
                cycle_time = self.sim_time/self.shop_stock_out.level
            else:
                cycle_time = 100000000000

            return waiting_times, cycle_time, breakdowns


        
    def get_track(self):
        for i, machine in enumerate(self.list_machines):
            if not machine.last:
                plt.plot([t[0] for t in machine.buffer_tracks], [t[1] for t in machine.buffer_tracks], label='Buffer M '+str(i+1))
                plt.legend()
            else:
                machine_last = machine
        plt.show()
        return self.list_machines

    def track_sim(self, robot_state):
        """
        Tracks the simulation by recording various states and metrics.

        Parameters:
        - robot_state: The state of the robot.

        This function appends the following information to their respective lists:
        - Buffer levels for each machine in the `list_machines`.
        - Robot states.
        - Operating states of each machine in the `list_machines`.
        - Idle times of each machine in the `list_machines`.
        - Cycle times of each machine in the `list_machines`.
        - Number of breakdowns for each machine in the `list_machines`.
        - Simulation times.

        Returns:
        None
        """
        self.buffer_tracks.append([(m.buffer_in.level, m.buffer_out.level) for m in self.list_machines])
        self.robot_states.append(robot_state)
        self.machines_states.append([m.operating for m in self.list_machines])
        self.machines_idle_times.append([m.waiting_time for m in self.list_machines])
        self.machines_CT.append([self.env.now/(m.parts_done+1) for m in self.list_machines])
        self.machines_breakdown.append([m.n_breakdowns for m in self.list_machines])
        self.sim_times_track.append(self.env.now)

    def run(self):
        #print(f"Current simulation time at the start: {self.env.now}")
        self.generate()
        order_process = [None for _ in range(len(self.list_machines))]
        #order_process[0] = self.supermarket_in
        for i, m in enumerate(self.list_machines):
            m.prio = self.full_order[i]
            m.process = self.env.process(m.machine_process())
            self.env.process(self.break_down(m))
        #order_process[-1] = self.shop_stock_out

        self.env.process(self.refill_market())
        print(str(len(self.robots_list)) + "  -- Robot Included")
        for i in range(len(self.robots_list)):
            order_process = [self.list_machines[j-1] for j in self.robots_list[i].order]
            self.robots_list[i].entities_order = order_process
            print(str(self.n_robots) + "  -- Robot Included")

            if i ==0:
                self.robots_list[i].process = self.env.process(self.robots_list[i].robot_process())
            else:
                self.robots_list[i].process = self.env.process(self.robots_list[i].robot_process(False))

        self.env.process(self.reset_shift())
        self.env.run(until=self.sim_time)
        #print(f"Current simulation time at the end: {self.env.now}")

    def reset(self):
        """
        Careful my friend! This function resets the full production line. Used mainly when wanting to start 
        a new simulation without interupting the code. 

        """
        self.env = simpy.Environment()
        self.supermarket_in = simpy.Container(self.env, capacity=self.stock_capacity, init=self.stock_initial)
        self.shop_stock_out = simpy.Container(self.env, capacity=float(self.config["shopstock"]["capacity"]), init=float(self.config["shopstock"]["initial"]))
        self.robots_list = []
        
        
        self.create_machines(self.machine_config_data)

        for robot in self.robots_list:
            robot.env = self.env

        self.reset_shift()


    def initialize(self):
        self.generate()
        for i, m in enumerate(self.list_machines):
            m.process = self.env.process(m.machine_process())
            self.env.process(self.break_down(m))
        
        self.env.process(self.refill_market())
        self.env.process(self.reset_shift())

    def run_action(self, action):
        """
        action is list of two elements: [from_machine, to_machine]
        """
        print('Going from ' + str(action[0]) + " " + str(action[1]) )
        self.robot.process = self.env.process(self.robot.robot_process_local(action[0], action[1]))
        self.env.run(until=self.env.now + 30)


    def reset_shift(self):
        """
        Resets the shift by performing various actions on machines and buffers.

        This function is called to reset the shift after a certain time interval (8 hours).
        It performs the following actions:
        - Prints a message indicating that the shift is being reset.
        - Increments the number of cycles.
        - Resets the shift boolean flag.
        - Performs specific actions on each machine and buffer based on their properties.

        Note: This function assumes the existence of certain variables and objects such as `self.env`,
        `self.num_cycles`, `self.list_machines`, `self.robot`, `machine.buffer_out`, `machine.buffer_in`,
        `machine.first`, `machine.last`, `machine.parts_done_shift`, and `self.shop_stock_out`.

        Yields:
            The function is a generator and yields a timeout of 8 hours.

        """
        while True:
            yield self.env.timeout(3600 * 8)
            print("Resetting Shift")
            
            self.num_cycles = self.num_cycles + 1
            self.reset_shift_bool = True
            for i, machine in enumerate(self.list_machines):
                
                if self.robot is None:
                    if machine.first:
                        while machine.buffer_out.level > 0:
                            machine.buffer_out.get(1)
                    if machine.last:
                        while machine.buffer_in.level > 0:
                            machine.buffer_in.get(1)

                    if not machine.first and not machine.last:
                        while machine.buffer_out.level > 0:
                            machine.buffer_in.get(1)
                            machine.buffer_out.get(1)
                else:
                    while machine.buffer_out.level > 0:
                        if machine.last:
                            machine.buffer_out.get(1)
                            self.shop_stock_out.put(1)
                        else:
                            machine.buffer_out.get(1)

                machine.parts_done_shift = 1

            

    def break_down(self, machine):
        """
        Break the machine every now and then.

        This function is responsible for simulating machine breakdowns. It uses the `time_to_failure` method of the machine
        to determine when the machine will break down. Once the machine breaks down, it increments the `n_breakdowns` attribute
        of the machine and interrupts the ongoing process.

        Args:
            machine: The machine object to be broken down.

        Yields:
            The function is a generator and yields the timeout until the next breakdown occurs.
        """
        if self.breakdowns_switch:
            while True:
                yield self.env.timeout(machine.time_to_failure())
                if not machine.broken:
                    ##print("Machine " + machine.ID + " - Broken")
                    machine.n_breakdowns += 1
                    machine.process.interrupt()
    

    
    def monitor_waiting_time(self, machine):
        while True:
            if not machine.operating:
                print(machine.ID + " Not Operating")
                yield self.env.timeout(10)
                machine.waiting_time_rl += 10


    def set_CT_machines(self, CTs):

        if len(self.list_machines) != len(CTs):
            raise ValueError('No matching! You have chosen ' + str(len(self.list_machines)) + ' machines and have given ' + str(len(CTs)))

        for i, machine in enumerate(self.list_machines):
            machine.ct = CTs[i]

    def refill_market(self):
        """
        Refills the input market with products.

        This function is responsible for refilling the input market with products. It runs in an infinite loop and refills the market
        based on the specified refill time and refill size. The refill time can be either a single value or a range of values.
        If it is a range, a random value within the range is chosen for each refill. The refill size determines the number of
        products to be added to the market during each refill.

        Note: This function uses the `self.env` attribute as the simulation environement.
        """
        while True:
            time_refill_start = self.env.now
            if isinstance(self.refill_time, list):
                refill_time = int(random.uniform(self.refill_time[0], self.refill_time[1]))
            elif isinstance(self.refill_time, float):
                refill_time = self.refill_time
            try:
                yield self.env.timeout(refill_time)
                self.supermarket_in.put(self.refill_size)
                self.supermarket_n_refills += 1
            except:
                pass
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
        if self.operators_assignement or self.tasks_assignement:

            machine_indices = [[] for _ in range(len(list_machines_config))]

            for index, n_machine in enumerate(self.tasks_assignement):
                    machine_indices[n_machine - 1].append(index)
            

            if len(list_machines_config) != max(self.tasks_assignement):
                raise ValueError('No match between assignement of tasks and number of machines.')

            if len(list_machines_config) != max(max(self.operators_assignement)):
                raise ValueError('No match between assignement of operators and number of machines.')


        self.list_machines = []

        # If only one cell of "Robot Transport Time" is NaN (empty), means we are not using a robot to transport components.
        print("HEre 1 : ", all([not np.isnan(list_machines_config[i][10])  for i in range(len(list_machines_config))]) )
        print("Here 2 : ", all([not np.isnan(list_machines_config[i][12])  for i in range(len(list_machines_config))]))
        if all([not np.isnan(list_machines_config[i][10])  for i in range(len(list_machines_config))]) and all([not np.isnan(list_machines_config[i][12])  for i in range(len(list_machines_config))]):
            print("List of robots = ", [list_machines_config[j][12] for j in  range(len(list_machines_config))])
            for i in range(int(max([list_machines_config[j][12] for j in  range(len(list_machines_config))]))):
                self.robot = Robot(self, self.env)
                #print("ORder inside = ", [list_machines_config[j][11]  for j in range(len(list_machines_config)) if list_machines_config[j][11] == int(i+1)])
                self.robot.order = [list_machines_config[j][11] for j in range(len(list_machines_config)) if (list_machines_config[j][12] == int(i+1))]
                self.robot.in_transport_times = [list_machines_config[j][10] for j in range(len(list_machines_config)) if (list_machines_config[j][12] == int(i+1))]
                self.robots_list.append(self.robot)
        
        notfirst_list = [list_machines_config[i][5] for i in range(len(list_machines_config))]
        notfirst_list1 = [item.strip("'")  for item in notfirst_list]

        self.full_transport_times = [list_machines_config[j][10] for j in range(len(list_machines_config))]
        self.full_order = [list_machines_config[j][11] for j in range(len(list_machines_config))]
        # if isinstance(item, str) else item for sublist in notfirst_list
        lists = [eval(item) if item.startswith('[') else [item] for item in notfirst_list1]

        # Flatten the lists
        flattened_list = [item for sublist in lists for item in sublist]

        #  Remove duplicates
        unique_list = list(set(flattened_list))

        for i in range(len(list_machines_config)):
            try:
                assigned_tasks =  list(np.array(self.tasks)[machine_indices[i]])
            except:
                pass
            first, last = False, False
            # if i == len(list_machines_config) - 1 :
            #     last = True
            if list_machines_config[i][5] == "END":
                last = True
            if list_machines_config[i][0] not in unique_list:
                first = True    

            try: 
                mttf = eval(str(list_machines_config[i][8]))
                mttr = eval(str(list_machines_config[i][9]))
                buffer_capacity = list_machines_config[i][6]
                initial_buffer = list_machines_config[i][7]

            except:
                mttf = float(list_machines_config[i][8])
                mttr = float(list_machines_config[i][9])
                buffer_capacity = list_machines_config[i][6]
                initial_buffer = list_machines_config[i][7]
                        
            try:
                machine = Machine(self, self.env, list_machines_config[i][0], list_machines_config[i][1], self.config, assigned_tasks=assigned_tasks, first=first, last=last, breakdowns=self.breakdowns['enabled'], mttf=mttf, mttr=mttr, buffer_capacity=buffer_capacity, initial_buffer=initial_buffer ,hazard_delays=self.config['hazard_delays']['enabled'])
            except:
                machine = Machine(self, self.env, list_machines_config[i][0], list_machines_config[i][1], self.config, first=first, last=last, breakdowns=self.breakdowns['enabled'], mttf=mttf, mttr=mttr, buffer_capacity=buffer_capacity , initial_buffer=initial_buffer, hazard_delays=self.config['hazard_delays']['enabled'])

            
            if machine.first:
                self.first_machine = machine
            self.list_machines.append(machine)
        
        index_next = []
        self.set_CT_machines([ct[2] for ct in list_machines_config])
        l = [m.ID for m in self.list_machines]
        for i, machine in enumerate(self.list_machines):
            
            if machine.first:
                machine.previous_machine = self.supermarket_in
                machine.previous_machines.append(self.supermarket_in)
            if machine.last:
                machine.next_machine = self.shop_stock_out
                machine.next_machines.append(self.shop_stock_out)
            if len(self.robots_list) == 0:
                # Process if robot is NOT used
                try:
                    # if l.index(list_machines_config[i][4]) not in index_next:
                    #     index_next.append(l.index(list_machines_config[i][4]))
                    #     machine.next_machine = self.list_machines[index_next[-1]]
                    #     self.list_machines[index_next[-1]].buffer_in = machine.buffer_out
                    # else:
                    #     machine.next_machine = self.list_machines[l.index(list_machines_config[i][4])]
                    #     machine.buffer_out = machine.next_machine.buffer_in
                    try:
                        link_cell_data = eval(list_machines_config[i][5])
                    except:
                        link_cell_data = list_machines_config[i][5]

                    # If the machine delivers to many machine 
                    if type(link_cell_data) is list:
                        for m_i in link_cell_data:
                            # Machine that never appeared before
                            if l.index(m_i) not in index_next:
                                index_next.append(l.index(m_i))
                                machine.next_machine = self.list_machines[index_next[-1]]
                                self.list_machines[index_next[-1]].previous_machine = machine
                                self.list_machines[index_next[-1]].previous_machines.append(machine)
                                machine.next_machines.append(self.list_machines[index_next[-1]])
                                self.list_machines[index_next[-1]].buffer_in = machine.buffer_out
                            else:
                                machine.next_machine = self.list_machines[l.index(m_i)]
                                self.list_machines[l.index(m_i)].previous_machine = machine
                                machine.next_machines.append(self.list_machines[l.index(m_i)])
                                self.list_machines[l.index(m_i)].previous_machines.append(machine)
                                machine.buffer_out = machine.next_machine.buffer_in
                
                    # If the machine delivers to only one machine 
                    else:
                        try:
                            if l.index(link_cell_data) not in index_next:
                                index_next.append(l.index(link_cell_data))
                                machine.next_machine = self.list_machines[index_next[-1]]
                                self.list_machines[index_next[-1]].previous_machine = machine
                                self.list_machines[index_next[-1]].previous_machines.append(machine)
                                machine.next_machines.append(self.list_machines[index_next[-1]])
                                self.list_machines[index_next[-1]].buffer_in = machine.buffer_out
                            else:
                                machine.next_machine = self.list_machines[l.index(link_cell_data)]
                                self.list_machines[l.index(link_cell_data)].previous_machine = machine
                                machine.next_machines.append(self.list_machines[l.index(link_cell_data)])
                                self.list_machines[l.index(link_cell_data)].previous_machines.append(machine)
                                machine.buffer_out = machine.next_machine.buffer_in
                            
                        except:
                            pass
                    # if l.index(list_machines_config[i][4]) not in index_next:
                    #     index_next.append(l.index(list_machines_config[i][4]))
                    #     machine.next_machine = self.list_machines[index_next[-1]]
                    #     self.list_machines[index_next[-1]].buffer_in = machine.buffer_out
                    # else:
                    #     machine.next_machine = self.list_machines[l.index(list_machines_config[i][4])]
                    #     machine.buffer_out = machine.next_machine.buffer_in
                    try:
                        link_cell_data = eval(list_machines_config[i][5])
                    except:
                        link_cell_data = list_machines_config[i][5]

                    # If the machine delivers to many machine 
                    if type(link_cell_data) is list:
                        for m_i in link_cell_data:
                            # Machine that never appeared before
                            if l.index(m_i) not in index_next:
                                index_next.append(l.index(m_i))
                                machine.next_machine = self.list_machines[index_next[-1]]
                                self.list_machines[index_next[-1]].previous_machine = machine
                                self.list_machines[index_next[-1]].previous_machines.append(machine)
                                machine.next_machines.append(self.list_machines[index_next[-1]])
                                self.list_machines[index_next[-1]].buffer_in = machine.buffer_out
                            else:
                                machine.next_machine = self.list_machines[l.index(m_i)]
                                self.list_machines[l.index(m_i)].previous_machine = machine
                                machine.next_machines.append(self.list_machines[l.index(m_i)])
                                self.list_machines[l.index(m_i)].previous_machines.append(machine)
                                machine.buffer_out = machine.next_machine.buffer_in
                
                    # If the machine delivers to only one machine 
                    else:
                        try:
                            if l.index(link_cell_data) not in index_next:
                                index_next.append(l.index(link_cell_data))
                                machine.next_machine = self.list_machines[index_next[-1]]
                                self.list_machines[index_next[-1]].previous_machine = machine
                                self.list_machines[index_next[-1]].previous_machines.append(machine)
                                machine.next_machines.append(self.list_machines[index_next[-1]])
                                self.list_machines[index_next[-1]].buffer_in = machine.buffer_out
                            else:
                                machine.next_machine = self.list_machines[l.index(link_cell_data)]
                                self.list_machines[l.index(link_cell_data)].previous_machine = machine
                                machine.next_machines.append(self.list_machines[l.index(link_cell_data)])
                                self.list_machines[l.index(link_cell_data)].previous_machines.append(machine)
                                machine.buffer_out = machine.next_machine.buffer_in
                            
                        except:
                            pass
                except:
                    pass
            else:
                machine.move_robot_time = self.full_transport_times[i]
                # Process if robot is used                
                try:
                    link_cell_data = eval(list_machines_config[i][5])
                except:
                    link_cell_data = list_machines_config[i][5]
                # If the machine delivers to many machine 
                if type(link_cell_data) is list:
                    for m_i in link_cell_data:
                        if l.index(m_i) not in index_next:
                            index_next.append(l.index(m_i))
                            machine.next_machine = self.list_machines[index_next[-1]]
                            self.list_machines[index_next[-1]].previous_machine = machine
                            self.list_machines[index_next[-1]].previous_machines.append(machine)
                            machine.next_machines.append(self.list_machines[index_next[-1]])
                        else:
                            machine.next_machine = self.list_machines[l.index(m_i)]
                            self.list_machines[l.index(m_i)].previous_machine = machine
                            machine.next_machines.append(self.list_machines[l.index(m_i)])
                            self.list_machines[l.index(m_i)].previous_machines.append(machine)
                
                # If the machine delivers to only one machine 
                else:
                    try:
                        if l.index(link_cell_data) not in index_next:
                            index_next.append(l.index(link_cell_data))
                            machine.next_machine = self.list_machines[index_next[-1]]
                            self.list_machines[index_next[-1]].previous_machine = machine
                            self.list_machines[index_next[-1]].previous_machines.append(machine)
                            machine.next_machines.append(self.list_machines[index_next[-1]])
                        else:
                            machine.next_machine = self.list_machines[l.index(link_cell_data)]
                            self.list_machines[l.index(link_cell_data)].previous_machine = machine
                            machine.next_machines.append(self.list_machines[l.index(link_cell_data)])
                            self.list_machines[l.index(link_cell_data)].previous_machines.append(machine)
                        
                    except:
                        pass
        
        

class Machine:
    def __init__(self, manuf_line, env, machine_id, machine_name, config,  assigned_tasks = None, robot=None, operator=None, previous_machine = None, first = False, last=False, breakdowns=True, mttf=3600*24*7, mttr=3600, buffer_capacity=100, initial_buffer =0, hazard_delays=False):
        self.mt = 0
        self.ID = machine_id
        self.Name = machine_name
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
        self.initial_buffer = initial_buffer
        self.process = None
        self.next_machine = None
        self.robot = manuf_line.robot
        self.previous_machine = None
        self.operating_state = []

        self.next_machines = []
        self.previous_machines = []
        self.waiting_time = [0, 0] #Stavation # Blockage
        self.waiting_time_rl = 0 #real time waiting time
        self.operating = False
        self.identical_machines = []
        self.move_robot_time = 0


        if self.robot is None:
            if first:
                self.buffer_in = manuf_line.supermarket_in
                self.buffer_out = simpy.Container(env, capacity=float(buffer_capacity), init=self.initial_buffer)
            
            if last:
                self.buffer_in = simpy.Container(env, capacity=float(buffer_capacity), init=self.initial_buffer)
                self.buffer_out = manuf_line.shop_stock_out
            
            if not first and not last:
                self.buffer_in = simpy.Container(env, capacity=float(buffer_capacity), init=self.initial_buffer)
                self.buffer_out = simpy.Container(env, capacity=float(buffer_capacity), init=self.initial_buffer)
        else:
            self.buffer_in = simpy.Container(env, capacity=float(buffer_capacity), init=self.initial_buffer)
            self.buffer_out = simpy.Container(env, capacity=float(buffer_capacity), init=self.initial_buffer)

        
        self.MTTF = mttf #Mean time to failure in seconds
        self.MTTR = mttr
        self.assigned_tasks = assigned_tasks
        
        self.operator = None  # Assign the operator to the machine
        
        # Events to signal when the machine is loaded and when the operator is free
    
        self.loaded = 0

        self.last = last
        self.first = first
        self.prio = 1
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
        
        while True: 
            # Reseting every shift
            
            # if self.env.now // (3600 * 8) == self.manuf_line.num_cycles +1:
            #     print("Reseting Shift Enter")
            #     self.manuf_line.num_cycles = self.manuf_line.num_cycles +1
            #     print("N of cycle = ", self.manuf_line.num_cycles)
            #     self.manuf_line.reset_shift()
            #     print("Reseting Shift out")


            if any([m.ct != 0 for m in self.manuf_line.list_machines]):
                if self.op_fatigue:
                    deterministic_time = self.fatigue_model(self.env.now/(self.manuf_line.num_cycles+1), self.ct)
                else:
                    deterministic_time = self.ct
            elif self.assigned_tasks is not None:
                num_samples = len(self.assigned_tasks)
                if self.op_fatigue:
                    deterministic_time = np.sum([task.machine_time+self.fatigue_model((self.manuf_line.num_cycles+1), task.manual_time) for task in self.assigned_tasks])
                else:
                    deterministic_time = np.sum([task.machine_time+task.manual_time for task in self.assigned_tasks])

            num_samples = int(1/float(self.config["hazard_delays"]["probability"]))
            done_in =  deterministic_time + self.hazard_delays*np.mean(weibull_min.rvs(bias_shape, scale=bias_scale, size=num_samples))
            start = self.env.now
            self.buffer_tracks.append((self.env.now, self.buffer_out.level))
            

            # Get best machine to get from

            
            
            while done_in > 0:
                try:
                    #self.waiting_time += 1
                    entry = self.env.now
                    self.entry_times.append(entry)
                    
                    if self.manuf_line.local:
                        while self.buffer_in.level == 0 :
                            yield self.env.timeout(10)
                            self.waiting_time = [self.waiting_time[0] + 10 , self.waiting_time[1]]

                    yield self.buffer_in.get(1)
                    exit_t = self.env.now
                    self.exit_times.append(exit_t-entry)
                    #self.waiting_time = [self.waiting_time[0] + exit_t-entry , self.waiting_time[1]]
                    

                    self.operating = True
                    yield self.env.timeout(done_in)
                    entry2 = self.env.now
                    if self.manuf_line.local:
                        while self.buffer_out.level == self.buffer_out.capacity:
                            yield self.env.timeout(10)
                            self.waiting_time = [self.waiting_time[0] , self.waiting_time[1] + 10]
                    
                    yield self.buffer_out.put(1)
                    self.finished_times.append(self.env.now-entry)
                    #self.waiting_time = [self.waiting_time[0] , self.waiting_time[1] + self.env.now-entry2]

                    # if self.last:
                    #     self.manuf_line.shop_stock_out.put(1)
                    done_in = 0
                    self.operating = False
                    # self.parts_done = self.parts_done +1
                    # self.parts_done_shift = self.parts_done_shift+1
                    
                    
                    yield self.env.timeout(0)
                except simpy.Interrupt:
                    self.broken = True
                    done_in -= self.env.now - start
                    self.buffer_in.put(1)
                    try:
                        #with self.manuf_line.repairmen.request(priority=self.prio) as req:
                            #yield req
                        yield self.env.timeout(self.MTTR) #Time to repair
                    except: 
                        pass
                    self.broken = False
                self.parts_done = self.parts_done +1
                self.parts_done_shift = self.parts_done_shift+1
    # def monitor_waiting_time(self):
    #     while True:
    #         if not self.operating:
    #             yield self.env.timeout(1)  
    #             self.waiting_time_rl += 1

class Robot:
    """
    Transport Robot between machines
    """
    def __init__(self, manuf_line, env, maxlimit=1):
        #self.assigned_machines = assigned_machines
        
        self.tt = 0
        self.busy = False
        self.manuf_line = manuf_line
        self.env = env
        self.schedule = []
        self.buffer = simpy.Container(self.env, capacity=float(maxlimit), init=0)
        self.robots_res = simpy.PriorityResource(self.env, capacity=1)
        self.waiting_time = 0
        self.in_transport_times = []
        self.process = None
        self.entities_order = None
        self.loadunload_time = 50

    def schedule_transport(self, from_machine, to_machine, transport_time):
        """Schedule a transport task if the robot is available."""

        transport_task = self.env.process(self.transport(from_machine, to_machine, transport_time))
        self.schedule.append(transport_task)
            

    def transport(self, from_entity, to_entity, time=10):
        if isinstance(from_entity, Machine) and isinstance(to_entity, Machine):
            entry = self.env.now
            yield from_entity.buffer_out.get(1)
            to_entity.loaded +=1
            self.waiting_time += self.env.now-entry
            yield self.env.timeout(abs(to_entity.move_robot_time - from_entity.move_robot_time)+self.loadunload_time)
            entry = self.env.now
            yield to_entity.buffer_in.put(1)
            self.waiting_time += self.env.now-entry
            self.manuf_line.track_sim((from_entity.Name, to_entity.Name))
            self.busy = False
            yield self.env.timeout(0)

        if not isinstance(from_entity, Machine) and isinstance(to_entity, Machine):
            entry = self.env.now
            yield from_entity.get(1)
            to_entity.loaded +=1
            self.waiting_time += self.env.now-entry
            yield self.env.timeout(abs(to_entity.move_robot_time)+self.loadunload_time)
            entry = self.env.now
            yield to_entity.buffer_in.put(1)
            self.waiting_time += self.env.now-entry
            self.manuf_line.track_sim(("InputStock", to_entity.Name))
            self.busy = False 
            yield self.env.timeout(0) 
            

        if  isinstance(from_entity, Machine) and not isinstance(to_entity, Machine):
            #print("Transporting from " + from_entity.ID + " to " + str(to_entity))
            ##print([(m.buffer_in.level, m.buffer_out.level) for m in self.manuf_line.list_machines])
            #self.manuf_line.track_sim((from_entity.ID, "OutputStock"))
            entry = self.env.now
            yield from_entity.buffer_out.get(1)
            self.waiting_time += self.env.now-entry
            yield self.env.timeout(abs(from_entity.move_robot_time)+self.loadunload_time)
            entry = self.env.now
            yield to_entity.put(1)
            self.waiting_time += self.env.now-entry
            self.manuf_line.track_sim((from_entity.Name, "OutputStock"))
            self.busy = False 
            yield self.env.timeout(0)

    def handle_empty_buffer_new(self, from_entity, to_entity, i):
        
        
        from_entity.previous_machine = self.which_machine_to_getfrom(from_entity)

        if from_entity.previous_machine == True:
            yield from self.transport(from_entity, to_entity) 
        else:
            
            if not from_entity.previous_machine.operating and from_entity.previous_machine.buffer_out.level == 0:
                yield from self.handle_empty_buffer(from_entity.previous_machine, from_entity, i)
            elif from_entity.previous_machine.operating or from_entity.previous_machine.buffer_out.level>0:
                yield from self.transport(from_entity.previous_machine, from_entity) 
            

        if from_entity.previous_machine == True:
            yield from self.transport(from_entity, to_entity) 
        else:
            try:
                if not from_entity.previous_machine.operating and from_entity.previous_machine.buffer_out.level == 0:
                    yield from self.handle_empty_buffer(from_entity.previous_machine, from_entity, i)
                elif from_entity.previous_machine.operating or from_entity.previous_machine.buffer_out.level>0:
                    yield from self.transport(from_entity.previous_machine, from_entity) 
            except:
                if from_entity.previous_machine.level == 0:
                    yield from self.handle_empty_buffer(from_entity.previous_machine, from_entity, i)
                elif from_entity.previous_machine.level>0:
                    yield from self.transport(from_entity.previous_machine, from_entity) 

    def handle_empty_buffer(self, from_entity, to_entity):

        from_entity.previous_machine = self.which_machine_to_getfrom(from_entity)
        if from_entity.previous_machine == True:
            yield from self.transport(from_entity, to_entity) 
        else:
            try:
                if not from_entity.previous_machine.operating and from_entity.previous_machine.buffer_out.level == 0:
                    yield from self.handle_empty_buffer(from_entity.previous_machine, from_entity)
                elif from_entity.previous_machine.operating or from_entity.previous_machine.buffer_out.level>0:
                    yield from self.transport(from_entity.previous_machine, from_entity) 
            except:
                if from_entity.previous_machine.level == 0:
                    yield from self.handle_empty_buffer(from_entity.previous_machine, from_entity)
                elif from_entity.previous_machine.level>0:
                    yield from self.transport(from_entity.previous_machine, from_entity) 

        # if from_entity.previous_machine == True:
        #     yield from self.transport(from_entity, to_entity) 
        # else:
        #     try:
        #         if not from_entity.previous_machine.operating and from_entity.previous_machine.buffer_out.level == 0:
        #             yield from self.handle_empty_buffer(from_entity.previous_machine, from_entity, i)
        #         elif from_entity.previous_machine.operating or from_entity.previous_machine.buffer_out.level>0:
        #             yield from self.transport(from_entity.previous_machine, from_entity) 
        #     except:
        #         if from_entity.previous_machine.level == 0:
        #             yield from self.handle_empty_buffer(from_entity.previous_machine, from_entity, i)
        #         elif from_entity.previous_machine.level>0:
        #             yield from self.transport(from_entity.previous_machine, from_entity) 


    def which_machine_to_feed(self, current_machine):
        """
        Return the best machine to feed based on a given strategy
        TODO: Upgrade to python 3.10 to use match case 

        """
        if self.manuf_line.robot_strategy == 0:
            empty_buffers_machines = [m.loaded if isinstance(m, Machine) else m.level for m in current_machine.next_machines]
            next_machine = current_machine.next_machines[empty_buffers_machines.index(min(empty_buffers_machines))]
            return next_machine
        
        elif self.manuf_line.robot_strategy == 1:
            empty_buffers_machines = [m.buffer_in.level if isinstance(m, Machine) else m.level for m in current_machine.next_machines]
            next_machine = current_machine.next_machines[empty_buffers_machines.index(min(empty_buffers_machines))]
            return next_machine
    

        elif self.manuf_line.robot_strategy == 2:
            empty_buffers_machines = [m.loaded if isinstance(m, Machine) else m.level for m in current_machine.next_machines]
            next_machine = current_machine.next_machines[empty_buffers_machines.index(min(empty_buffers_machines))]
            return next_machine


    
    def which_machine_to_getfrom(self, current_machine):

        """
        Return the best machine to get from based on a given strategy.
        """
        if self.manuf_line.robot_strategy == 0:
            full_buffers_machines = [m.loaded if isinstance(m, Machine) else m.level for m in current_machine.previous_machines]
            previous_machine = current_machine.previous_machines[full_buffers_machines.index(min(full_buffers_machines))]

        elif self.manuf_line.robot_strategy == 1:
            full_buffers_machines = [m.buffer_out.level if isinstance(m, Machine) else m.level for m in current_machine.previous_machines]
            previous_machine = current_machine.previous_machines[full_buffers_machines.index(max(full_buffers_machines))]
    

        try:
            return previous_machine 
        except:
            # No machine
            return True



    def robot_process_local(self, from_entity, to_entity, transport_time=10):
        yield from self.transport(from_entity, to_entity, transport_time)




    def robot_process(self, first=True):
        """
        entities_order = [input_entity, machine1, machine2, machine3, output_entity]
        times = [10, 10, 10, 10, 10] len(entities_order)
        """
        print("Machines final =", [m.last for m in self.manuf_line.list_machines])
        while True:
            
            if first:
            # Policy 1 => Follow everytime the same order, if not feasible pass to next 
                for  i, m in enumerate([m for m in self.manuf_line.list_machines if m.first]):
                    if m.buffer_in.level < m.buffer_in.capacity:
                        yield from self.transport(m.previous_machine, m)
                    else:
                        if m.next_machine.buffer_in.level < m.next_machine.buffer_in.capacity:
                            yield from self.transport(m, m.next_machine)
                        else: 
                            pass


            for i in range(len(self.entities_order)):
                from_entity = self.entities_order[i]
                to_entity = self.which_machine_to_feed(from_entity)

                if self.manuf_line.reset_shift_bool:
                    self.manuf_line.reset_shift_bool = False
                    break
                
                
                try:
                    if to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level > 0:
                        yield from self.transport(from_entity, to_entity)
                    elif to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level == 0 and from_entity.operating:
                        yield from self.transport(from_entity, to_entity)
                    elif to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level == 0 and not from_entity.operating:
                        yield from self.handle_empty_buffer(from_entity, to_entity)
                    else:
                        continue
            
                except Exception as e:
                    
                    try:
                        if to_entity.level < to_entity.capacity and from_entity.buffer_out.level > 0:
                            yield from self.transport(from_entity, to_entity)
                        elif to_entity.level < to_entity.capacity and from_entity.buffer_out.level == 0 and from_entity.operating:
                            yield from self.transport(from_entity, to_entity)
                        elif to_entity.level < to_entity.capacity and from_entity.buffer_out.level == 0 and not from_entity.operating:
                            yield from self.handle_empty_buffer(from_entity, to_entity)
                        else:
                            continue
                    except Exception as e2:
                        if to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level > 0:
                            yield from self.transport(from_entity, to_entity)
                        elif to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level == 0 and from_entity.operating:
                            yield from self.transport(from_entity, to_entity)
                        elif to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level == 0 and not from_entity.operating:
                            yield from self.handle_empty_buffer(from_entity, to_entity)
                        else:
                            continue


    def find_next_entity(self, machine, entities_order):
        """
        Find the next entity to feed the given machine based on the entities_order.
        """
        # Get the index of the given machine in the entities_order list
        machine_index = entities_order.index(machine)
        
        # Iterate over entities in the order list starting from the machine's index
        for i in range(machine_index + 1, len(entities_order)):
            next_entity = entities_order[i]
            # Check if the next entity has output buffer with available space
            if next_entity.buffer_out.level > 0:
                return next_entity
        return None
   

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
            for entry, exi in zip(m.entry_times, m.exit_times):
                idle_times_machine.append(exi-entry)
            avg_idle_time = np.mean(idle_times_machine)
            #idle_time = idle_times_machine[-1]
            try:
                m.buffer_btn[1].configure(text=f"Capacity = {m.buffer_out.capacity}")
                m.buffer_btn[2].configure(text=f"Level = {m.buffer_out.level}")
                m.buffer_btn[3].configure( text="Waiting/Idle Time = %.2f" % avg_idle_time)
                m.buffer_btn[4].configure(text="Total Downtime = %.2f" % float(m.MTTR*float(m.n_breakdowns)))
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
            for entry, exi in zip(m.entry_times, m.exit_times):
                idle_times_machine.append(exi-entry)
            avg_idle_time = np.mean(idle_times_machine)
            #idle_time = idle_times_machine[-1]
            m.buffer_btn[1].configure(text=f"Capacity = {m.buffer_out.capacity}")
            m.buffer_btn[2].configure(text=f"Level = {m.buffer_out.level}")
            m.buffer_btn[3].configure( text="Avg. Cycle Time = %.2f" % avg_idle_time)
            m.buffer_btn[4].configure(text="Total Downtime = %.2f" % float(float(assembly_line.breakdowns['mttr'])*m.n_breakdowns))
        #uptime_m = 100*(1-(float(float(m.MTTR)*m.n_breakdowns)+m.waiting_time)/assembly_line.env.now)
        #uptime_m = avg_idle_time
        uptime_m=  assembly_line.env.now/m.parts_done
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