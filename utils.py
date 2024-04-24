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
import os
import sys

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
        self.randomseed = False
        self.parts_done = 0
        self.config = config
        self.sim_time = eval(self.config['sim_time'])
        self.machine_config_data = []
        self.breakdowns = self.config['breakdowns']
        self.breakdowns_switch = self.config['breakdowns']["enabled"]
        self.breakdown_law = ""
        self.n_repairmen = 3
        self.repairmen = simpy.PreemptiveResource(env, capacity=self.n_repairmen)
        self.first_machine = None
        self.stock_capacity = float(config["supermarket"]["capacity"])
        self.stock_initial = float(config["supermarket"]["initial"])
        self.safety_stock = 0
        self.refill_time = None
        self.refill_size = 1
        self.reset_shift_dec = False

        ### Multi reference
        self.references_config = None
        self.inventory_in = simpy.Store(env)
        self.inventory_out = simpy.Store(env)

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
        self.reset_shift_bool = False
        self.local = True

        self.buffer_tracks = []
        self.machines_output = []
        self.robot_states = []
        self.machines_states = []
        self.machines_CT = []
        self.machines_idle_times = []
        self.list_machines = []
        self.machines_breakdown = []
        self.sim_times_track = []
        self.output_tracks = []

        self.expected_refill_time = None


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
        
        if not save and not track:
            waiting_times = [machine.parts_done for machine in self.list_machines]
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


    def track_output(self):
        self.machines_output = [[] for _ in range(len(self.list_machines))]
        self.output_tracks_per_ref = [[] for _ in range(len(self.references_config.keys()))]
        while True:
            yield self.env.timeout(self.sim_time/100)
            self.output_tracks.append((self.env.now,self.shop_stock_out.level))
            for i, ref in enumerate(self.references_config.keys()):
                self.output_tracks_per_ref[i].append((self.env.now,self.inventory_out.items.count(ref)))
            for i, m in enumerate(self.list_machines):
                self.machines_output[i].append((self.env.now,m.parts_done))

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
        self.generate()
        order_process = [None for _ in range(len(self.list_machines))]
        for i, m in enumerate(self.list_machines):
            m.prio = self.full_order[i]
            m.process = self.env.process(m.machine_process())
            self.env.process(self.break_down(m))
        
        
        self.expected_refill_time = [0 for _ in range(len(self.references_config.keys()))]
        for ref in list(self.references_config.keys()):
            print("Reference confirmed = ", ref)
            self.env.process(self.refill_market(ref))

        print(str(len(self.robots_list)) + "  -- Robot Included")
        print("first machine = " , [m.first for m in self.list_machines])
        for i in range(len(self.robots_list)):
            order_process = [self.list_machines[j-1] for j in self.robots_list[i].order]
            
            self.robots_list[i].entities_order = order_process
            print("Inclued in robot = ", [self.list_machines[j-1].ID for j in self.robots_list[i].order])
            print("Inclued in robot = ", [self.list_machines[j-1].first for j in self.robots_list[i].order])

            if any([self.list_machines[j-1].first for j in self.robots_list[i].order]):
                self.robots_list[i].entities_order.insert(0, self.supermarket_in)
                self.robots_list[i].process = self.env.process(self.robots_list[i].robot_process())
                #self.robots_list[i].process = self.env.process(self.robots_list[i].robot_process_unique())
            else:
                #self.robots_list[i].process = self.env.process(self.robots_list[i].robot_process_unique(False))
                self.robots_list[i].process = self.env.process(self.robots_list[i].robot_process(False))
        #TODO: fix problem of reset shift with multi references
        print("Resetting shift = ", self.reset_shift_dec)
        if self.reset_shift_dec:
            self.env.process(self.reset_shift())
        self.env.process(self.track_output())
        print("Starting the sim now.")

        for i, m in enumerate(self.list_machines):
            print(m.ID, m.buffer_in.capacity)
            print(m.ID, m.buffer_out.capacity)
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
        self.inventory_in = simpy.Store(self.env)
        self.inventory_out = simpy.Store(self.env)
        self.repairmen = simpy.PreemptiveResource(self.env, capacity=int(self.n_repairmen))
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
                   
                if self.robots_list == []:
                    print("Reseting with no robot")
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
                    machine.store_out.items = [] 
                    machine.store_in.items = [] 
                    if machine.buffer_out.level > 0:
                        if machine.last:
                            machine.buffer_out.get(1)
                            self.shop_stock_out.put(1)                     
                        else:
                            machine.buffer_out.get(1)

                machine.parts_done_shift = 0

            

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
                time_to_break = machine.time_to_failure()
                # if time_to_break < machine.MTTF:
                #     time_to_break = 2*machine.MTTF
                yield self.env.timeout(time_to_break)
                if not machine.broken:
                    time_to_break = machine.time_to_failure()
                    yield self.env.timeout(time_to_break)
                    self.operationg = False
                    print("Machine " + machine.ID + " - Broken at = " + str(self.env.now) + " after time : " + str(time_to_break))
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

    def refill_market(self, ref="Ref A"):
        """
        Refills the input market with products.

        This function is responsible for refilling the input market with products. It runs in an infinite loop and refills the market
        based on the specified refill time and refill size. The refill time can be either a single value or a range of values.
        If it is a range, a random value within the range is chosen for each refill. The refill size determines the number of
        products to be added to the market during each refill.

        Note: This function uses the `self.env` attribute as the simulation environement.
        """
        pattern = r'^(\d+)-(\d+)$'
        match = re.match(pattern, str(self.references_config[ref][0]))
        if match:
            value1 = int(match.group(1))
            value2 = int(match.group(2))
            refill_time_ref = [value1, value2]
        else:
            refill_time_ref = float(self.references_config[ref][0])

        
        while True:
            if isinstance(refill_time_ref, list):
                refill_time = int(random.uniform(refill_time_ref[0], refill_time_ref[1]))
                self.expected_refill_time[list(self.references_config.keys()).index(ref)] = self.env.now + refill_time
            elif isinstance(refill_time_ref, float):
                refill_time = refill_time_ref
                self.expected_refill_time[list(self.references_config.keys()).index(ref)] = self.env.now + refill_time
            try:
                yield self.env.timeout(refill_time)
                yield self.supermarket_in.put(self.refill_size) 
                print("Refilled at = " + str(self.env.now) + "  with " + ref)         
                yield self.inventory_in.put(ref)

                self.supermarket_n_refills += 1
            except:
                pass
            yield self.env.timeout(0)


    def deplete_shopstock(self):
        pass

    def deliver_to_client(self):
        pass

    def repairmen_process(self):
        while True:
            # Start a new job
            done_in = 1000
            while done_in:
                # Retry the job until it is done.
                # Its priority is lower than that of machine repairs.
                with self.repairmen.request(priority=3, preempt=False) as req:
                    yield req
                    start = self.env.now
                    try:
                        yield self.env.timeout(done_in)
                        done_in = 0
                    except simpy.Interrupt:
                        done_in -= self.env.now - start


    
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
  
        self.robots_list = []
        if all([not np.isnan(list_machines_config[i][7])  for i in range(len(list_machines_config))]) and all([not np.isnan(list_machines_config[i][9])  for i in range(len(list_machines_config))]):

            for i in range(int(max([list_machines_config[j][9] for j in  range(len(list_machines_config))]))):
                self.robot = Robot(self, self.env)
                #print("ORder inside = ", [list_machines_config[j][11]  for j in range(len(list_machines_config)) if list_machines_config[j][11] == int(i+1)])
                self.robot.order = [list_machines_config[j][8] for j in range(len(list_machines_config)) if (list_machines_config[j][9] == int(i+1))]
                self.robot.in_transport_times = [list_machines_config[j][7] for j in range(len(list_machines_config)) if (list_machines_config[j][9] == int(i+1))]
                self.robots_list.append(self.robot)
        
        notfirst_list = [list_machines_config[i][2] for i in range(len(list_machines_config))]
        notfirst_list1 = [item.strip("'")  for item in notfirst_list]

        self.full_transport_times = [list_machines_config[j][7] for j in range(len(list_machines_config))]
        self.full_order = [list_machines_config[j][8] for j in range(len(list_machines_config))]
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
            if list_machines_config[i][2] == "END":
                last = True
            if list_machines_config[i][0] not in unique_list:
                first = True    

            try: 
                mttf = eval(str(list_machines_config[i][5]))
                mttr = eval(str(list_machines_config[i][6]))
                buffer_capacity = list_machines_config[i][3]
                initial_buffer = list_machines_config[i][4]

            except:
                mttf = float(list_machines_config[i][5])
                mttr = float(list_machines_config[i][6])
                buffer_capacity = list_machines_config[i][3]
                initial_buffer = list_machines_config[i][4]
                        
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
            if not str(list_machines_config[i][10]) =="nan":
                indexmachine = [m.ID for m in self.list_machines].index(str(list_machines_config[i][7]))
                machine.same_machine = self.list_machines[indexmachine]
            if machine.first:
                machine.previous_machine = self.supermarket_in
                machine.previous_machines.append(self.supermarket_in)
            if machine.last:
                machine.next_machine = self.shop_stock_out
                machine.next_machines.append(self.shop_stock_out)
            if len(self.robots_list) == 0:
                # Process if robot is NOT used
                try:
                    try:
                        link_cell_data = eval(list_machines_config[i][2])
                    except:
                        link_cell_data = list_machines_config[i][2]

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
                                self.list_machines[index_next[-1]].store_in = machine.store_out
                            else:
                                machine.next_machine = self.list_machines[l.index(m_i)]
                                self.list_machines[l.index(m_i)].previous_machine = machine
                                machine.next_machines.append(self.list_machines[l.index(m_i)])
                                self.list_machines[l.index(m_i)].previous_machines.append(machine)
                                machine.buffer_out = machine.next_machine.buffer_in
                                machine.store_out = machine.next_machine.store_in
                
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
                                self.list_machines[index_next[-1]].store_in = machine.store_out
                            else:
                                machine.next_machine = self.list_machines[l.index(link_cell_data)]
                                self.list_machines[l.index(link_cell_data)].previous_machine = machine
                                machine.next_machines.append(self.list_machines[l.index(link_cell_data)])
                                self.list_machines[l.index(link_cell_data)].previous_machines.append(machine)
                                machine.buffer_out = machine.next_machine.buffer_in
                                machine.store_out = machine.next_machine.store_in
                        except:
                            pass
                except:
                    pass
            else:
                machine.move_robot_time = self.full_transport_times[i]
                # Process if robot is used                
                try:
                    link_cell_data = eval(list_machines_config[i][2])
                except:
                    link_cell_data = list_machines_config[i][2]
                
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
        self.same_machine = None
        self.ref_produced = []
        self.loaded_bol = False
        self.current_product = None
        

        if self.manuf_line.robots_list == []:
            if first:
                self.buffer_in = manuf_line.supermarket_in
                self.buffer_out = simpy.Container(env, capacity=float(buffer_capacity), init=self.initial_buffer)
                self.store_in = manuf_line.inventory_in
                self.store_out = simpy.Store(env)
            if last:
                self.buffer_in = simpy.Container(env, capacity=float(buffer_capacity), init=self.initial_buffer)
                self.buffer_out = manuf_line.shop_stock_out
                self.store_in = simpy.Store(env)
                self.store_out = manuf_line.inventory_out
            
            if not first and not last:
                self.buffer_in = simpy.Container(env, capacity=float(buffer_capacity), init=self.initial_buffer)
                self.buffer_out = simpy.Container(env, capacity=float(buffer_capacity), init=self.initial_buffer)
                self.store_in = simpy.Store(env)
                self.store_out = simpy.Store(env)
        else:
            self.buffer_in = simpy.Container(env, capacity=float(buffer_capacity), init=self.initial_buffer)
            self.buffer_out = simpy.Container(env, capacity=float(buffer_capacity), init=self.initial_buffer)
            self.store_in = simpy.Store(env)
            self.store_out = simpy.Store(env)

        
        self.MTTF = mttf #Mean time to failure in seconds
        self.MTTR = mttr
        self.assigned_tasks = assigned_tasks
        
        self.operator = None  # Assign the operator to the machine
    
        self.loaded = 0
        self.done_bool = False
        self.last = last
        self.first = first
        self.prio = 1
        self.broken = False
        self.breakdowns = breakdowns
        self.real_repair_time = []
        self.hazard_delays = 1 if hazard_delays else 0
        self.op_fatigue = config["fatigue_model"]["enabled"]

        


    
    def time_to_failure(self):
        """Return time until next failure for a machine.
        """
        #deterioration_factor = 1 + (self.env.now / self.simulation_duration)  # Adjust this factor as needed
        #adjusted_MTTF = self.MTTF / deterioration_factor
        
        if self.manuf_line.randomseed:
            random.seed(10+int(self.manuf_line.list_machines.index(self)))
        
        if self.manuf_line.breakdown_law == "Weibull Distribution":
            self.shape_parameter = 5
            val = random.weibullvariate(self.MTTF, self.shape_parameter)

        elif self.manuf_line.breakdown_law == "Exponential Distribution":
            val = random.expovariate(1/self.MTTF)

        elif self.manuf_line.breakdown_law == "Normal Distribution":     
            val = random.normalvariate(self.MTTF, self.MTTF/2)
           # print("Val Breakdown generated = " +  str(val) + " - Law "+ self.manuf_line.breakdown_law)

        elif self.manuf_line.breakdown_law == "Gamma Distribution":

            rate_of_decrease = 0.1
            machine_age = self.env.now/self.MTTF
            initial_k = 0.5
            if initial_k !=1:
                k = max(0.1, initial_k - rate_of_decrease * machine_age)
            else:
                k = initial_k

            val = random.gammavariate(0.3, self.MTTF)
        
        else:
            print("No law given. ")
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
            # If the machine has two processes, check of the other process is operating
            if self.same_machine is not None:
                other_process_operating = self.same_machine.operating
            else:
                other_process_operating = False
            if not self.operating and not other_process_operating:
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
                # + self.hazard_delays*np.mean(weibull_min.rvs(bias_shape, scale=bias_scale, size=num_samples))
                self.buffer_tracks.append((self.env.now, self.buffer_out.level))
                entry0 = self.env.now
                self.entry_times.append(entry0)
                self.loaded_bol = False
                self.done_bool = False
                entry_wait = self.env.now
                #if self.store_in.items != []:
                    #print("Here we are")
                    #if float(self.manuf_line.references_config[self.store_in.items[0]][self.manuf_line.list_machines.index(self)+1]) !=0 :
                self.to_be_passed = False
                while not self.loaded_bol :
                    try: 
                        product = None

                        if len(self.store_in.items) !=0:
                            ## The product should not be processed in this machine and to be passed to the next
                            if float(self.manuf_line.references_config[self.store_in.items[0]][self.manuf_line.list_machines.index(self)+1]) ==0:
                                self.to_be_passed = True
                                yield self.buffer_in.get(1)
                                product = yield self.store_in.get()
                                self.current_product = product
                                break

                        yield self.buffer_in.get(1)
                        print("before in " + self.ID + "= " +str(self.store_in.items) + " - PROD " )
                        product = yield self.store_in.get()
                        self.current_product = product
                        print("after in " + self.ID + "= " +str(self.store_in.items) + " - PROD " + product)

                        print("Product " + product + " passed in " + self.ID + " at " + str(self.env.now))
                        done_in = float(self.manuf_line.references_config[product][self.manuf_line.list_machines.index(self)+1])
                        if done_in == 0:
                            ## The product should not be processed in this machine and to be passed to the next
                            self.to_be_passed = True
                            break
                        self.waiting_time = [self.waiting_time[0] + self.env.now - entry_wait , self.waiting_time[1]]  
                        #done_in = deterministic_time 
                        #start = self.env.now
                        self.loaded_bol = True

                    except simpy.Interrupt:                        
                        self.broken = True
                        print(self.ID +" broken at loading at  = "+ str(self.env.now))

                        # try:
                        repair_in = self.env.now
                        with self.manuf_line.repairmen.request(priority=1) as req:
                            yield req
                            yield self.env.timeout(self.MTTR) #Time to repair
                            repair_end = self.env.now
                            self.real_repair_time.append(float(repair_end - repair_in))
                        print(self.ID +" repaired at loading at  = "+ str(self.env.now))
                        #done_in = 0
                        
                        if self.buffer_in.level == 0:
                            self.buffer_in.put(1)
                            #self.loaded_bol = False
                            self.loaded_bol = True
                        if product is not None:
                            self.store_in.put(product)
                            self.loaded_bol = True
                            done_in = float(self.manuf_line.references_config[product][self.manuf_line.list_machines.index(self)+1])
                            
                        else:
                            self.loaded_bol = False

                        self.broken = False
                        self.operating = False
                
                start = self.env.now
                #TODO: Skip robot when part not processed in the machine
                while done_in >0 and not self.to_be_passed:
                    try:
                        entry = self.env.now
                        self.current_product = product
                        exit_t = self.env.now
                        self.exit_times.append(exit_t-entry)
                        self.operating = True
                        yield self.env.timeout(done_in)
                        entry_wait = self.env.now
                        yield self.buffer_out.put(1) and self.store_out.put(product)
                        self.waiting_time = [self.waiting_time[0] , self.waiting_time[1] + self.env.now - entry_wait]  
                        done_in = 0
                        self.loaded_bol = False
                        self.done_bool = True
                        yield self.env.timeout(0)

                    except simpy.Interrupt:                        
                        self.broken = True
                        done_in -= self.env.now - start
                        # try:
                        repair_in = self.env.now

                        with self.manuf_line.repairmen.request(priority=1) as req:
                            yield req
                            repair_end = self.env.now
                            yield self.env.timeout(self.MTTR) #Time to repair
                            self.real_repair_time.append(float(repair_end - repair_in + float(self.MTTR)))
                        print(self.ID +" repaired at operating  buffer in level at " + str(self.env.now))
                        self.broken = False
                        self.operating = False

                if not self.to_be_passed and self.done_bool:
                    self.ref_produced.append(product)
                    self.current_product = None
                    self.finished_times.append(self.env.now-entry0)
                    self.operating = False
                    self.parts_done = self.parts_done +1
                    self.parts_done_shift = self.parts_done_shift+1
                    self.loaded_bol = False
                    self.done_bool = False
                    print("Part " + product + " produced by " + self.ID  + " at " +  str(self.env.now))

                else:
                    real_done_in = done_in
                    start = self.env.now
                    if real_done_in == 0:
                        try:
                            print("Passed zero no process = " + product + " In " + self.ID + " at " + str(self.env.now))
                            entry_wait = self.env.now
                            yield self.buffer_out.put(1) and self.store_out.put(product)
                            self.waiting_time = [self.waiting_time[0] , self.waiting_time[1] + self.env.now-entry_wait]
                            #yield self.store_out.put(product)
                            self.passed_to_next = True
                            done_in = 0
                            self.loaded_bol = False
                            self.current_product = None
                            yield self.env.timeout(0)
                        except simpy.Interrupt: 
                            
                            if not self.passed_to_next:
                                print("Passed zero no process = " + product + " In " + self.ID)
                                entry_wait = self.env.now
                                yield self.buffer_out.put(1) and self.store_out.put(product)
                                self.waiting_time = [self.waiting_time[0] , self.waiting_time[1] + self.env.now-entry_wait]
                                done_in = 0
                                self.loaded_bol = False
                                self.current_product = None
                                yield self.env.timeout(0) 
                            else:
                                self.current_product = None
                                yield self.env.timeout(0)   
                
                    



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
            

    # TODO: Fix the problem of the robot not being able to transport from a machine to another after shift reset
        # DO we need shift reset of buffers??
    def transport(self, from_entity, to_entity, time=10):
        if isinstance(from_entity, Machine) and isinstance(to_entity, Machine):
            if not self.busy:
                print("Transporting from " + from_entity.ID + " to " + to_entity.ID +" at time = " + str(self.env.now) )
                entry = self.env.now
                self.busy = True

                if to_entity.broken or from_entity.broken:
                    print("To entity is broken, skipping remaining instructions")
                    self.busy = False
                    yield self.env.timeout(10) 
                    return

                while from_entity.buffer_out.level == 0:
                    yield self.env.timeout(10)
                    if from_entity.broken or to_entity.broken:
                        print("From entity is broken, skipping remaining instructions")
                        self.busy = False
                        yield self.env.timeout(10)
                        return

                
                yield from_entity.buffer_out.get(1)
                product = yield from_entity.store_out.get()
                to_entity.loaded +=1
                self.waiting_time += self.env.now-entry
                yield self.env.timeout(abs(to_entity.move_robot_time - from_entity.move_robot_time)+self.loadunload_time)
                entry_2 = self.env.now

                while to_entity.buffer_in.level >= to_entity.buffer_capacity:
                    yield self.env.timeout(10)
                    if to_entity.broken:
                        print("From entity is broken, skipping remaining instructions")
                        self.busy = False
                        yield self.env.timeout(10)
                        return 

                yield to_entity.buffer_in.put(1)
                to_entity.store_in.put(product)
                self.waiting_time += self.env.now-entry_2
                self.manuf_line.track_sim((from_entity.Name, to_entity.Name, self.env.now))
                self.busy = False
                yield self.env.timeout(0)
            else:
                print("Busy")
                yield self.env.timeout(0)

        if not isinstance(from_entity, Machine) and isinstance(to_entity, Machine):
            if not self.busy:
                print("Transporting from " + str(from_entity) + " to " + to_entity.ID + " at time = " + str(self.env.now))
                self.busy = True
                entry = self.env.now
            
                if to_entity.broken:
                    print("To entity is broken, skipping remaining instructions")
                    self.busy = False
                    yield self.env.timeout(10) 
                    return
                
                yield from_entity.get(1)

                product = yield self.manuf_line.inventory_in.get() 
                
                to_entity.loaded +=1
                self.waiting_time += self.env.now-entry
                yield self.env.timeout(abs(to_entity.move_robot_time)+self.loadunload_time)
                entry_2 = self.env.now
                
                while to_entity.buffer_in.level >= to_entity.buffer_capacity:
                    yield self.env.timeout(10)
                    if to_entity.broken:
                        print("From entity is broken, skipping remaining instructions")
                        self.busy = False
                        yield self.env.timeout(0) 
                yield to_entity.buffer_in.put(1) and to_entity.store_in.put(product)
                self.waiting_time += self.env.now-entry_2
                self.manuf_line.track_sim(("InputStock", to_entity.Name, self.env.now))
                self.busy = False 
                yield self.env.timeout(0) 
            else:
                print("Busy")
                yield self.env.timeout(0)
                

        if  isinstance(from_entity, Machine) and not isinstance(to_entity, Machine):
            if not self.busy:
                self.busy = True
                print("Transporting from " + from_entity.ID + " to " + str(to_entity) + " at time = " + str(self.env.now))
                ##print([(m.buffer_in.level, m.buffer_out.level) for m in self.manuf_line.list_machines])
                #self.manuf_line.track_sim((from_entity.ID, "OutputStock"))
                entry = self.env.now
                while from_entity.buffer_out.level == 0:
                    yield self.env.timeout(10)
                    if from_entity.broken:
                        print("From entity is broken, skipping remaining instructions")
                        self.busy = False
                        yield self.env.timeout(1)
                        return
                yield from_entity.buffer_out.get(1)
                product = yield  from_entity.store_out.get()
                self.waiting_time += self.env.now-entry
                yield self.env.timeout(abs(from_entity.move_robot_time)+self.loadunload_time)
                entry_2 = self.env.now
                yield to_entity.put(1)
                self.manuf_line.inventory_out.put(product)
                self.waiting_time += self.env.now-entry_2
                self.manuf_line.track_sim((from_entity.Name, "OutputStock", self.env.now))
                self.busy = False 
                yield self.env.timeout(0)
            else:
                print("Busy")
                yield self.env.timeout(0)


    def handle_empty_buffer(self, from_entity, to_entity):

        from_entity.previous_machine = self.which_machine_to_getfrom(from_entity)
        if from_entity.previous_machine == True:
            yield from self.transport(from_entity, to_entity) 
        else:
            if isinstance(from_entity.previous_machine, Machine):
                if not from_entity.previous_machine.operating and from_entity.previous_machine.buffer_out.level == 0:
                    yield from self.handle_empty_buffer(from_entity.previous_machine, from_entity)
                elif from_entity.previous_machine.operating or from_entity.previous_machine.buffer_out.level>0:
                    yield from self.transport(from_entity.previous_machine, from_entity) 
            else:
                if from_entity.previous_machine.level == 0:
                    yield from self.handle_empty_buffer(from_entity.previous_machine, from_entity)
                elif from_entity.previous_machine.level>0:
                    yield from self.transport(from_entity.previous_machine, from_entity) 

    def which_machine_to_feed(self, current_machine):
        """
        Return the best machine to feed based on a given strategy
        TODO: Upgrade to python 3.10 to use match case 
        """


        # If the current element is input stock (not a machine)
        if not isinstance(current_machine, Machine):
            first_machines = [m for m in self.manuf_line.list_machines if m.first]

            if self.manuf_line.robot_strategy == 0:
                empty_buffers_machines = [m.loaded if isinstance(m, Machine) and not m.broken and not m.operating else float('inf') for m in first_machines]
                next_machine = first_machines[empty_buffers_machines.index(min(empty_buffers_machines))]  
                self.manuf_line.expected_refill_time = [abs(i-self.manuf_line.env.now) for i in self.manuf_line.expected_refill_time]
                if  self.manuf_line.inventory_in.items != []:
                    if float(self.manuf_line.references_config[self.manuf_line.inventory_in.items[0]][self.manuf_line.list_machines.index(next_machine)+1]) ==0:
                        next_machine.current_product = self.manuf_line.inventory_in.items[0]
                        return self.which_machine_to_feed(next_machine)
                    else:
                        return next_machine 
                elif float(self.manuf_line.references_config[list(self.manuf_line.references_config.keys())[np.argmin(self.manuf_line.expected_refill_time)]][self.manuf_line.list_machines.index(next_machine)+1]) ==0:
                        next_machine.current_product = list(self.manuf_line.references_config.keys())[np.argmin(self.manuf_line.expected_refill_time)]
                        return self.which_machine_to_feed(next_machine)
                else:
                    return next_machine

            elif self.manuf_line.robot_strategy == 1:
                empty_buffers_machines = [m.buffer_in.level if isinstance(m, Machine) else m.level for m in first_machines]
                next_machine = first_machines[empty_buffers_machines.index(min(empty_buffers_machines))]
                if  self.manuf_line.inventory_in.items != []:
                    if float(self.manuf_line.references_config[self.manuf_line.inventory_in.items[0]][self.manuf_line.list_machines.index(next_machine)+1]) ==0:
                        next_machine.current_product = current_machine.current_product
                        return self.which_machine_to_feed(next_machine)
                    else:
                        return next_machine
                else:
                    return next_machine

        else:
            if self.manuf_line.robot_strategy == 0:
                empty_buffers_machines = [m.loaded if isinstance(m, Machine) and not m.broken and not m.operating else float('inf') for m in current_machine.next_machines]
                next_machine = current_machine.next_machines[empty_buffers_machines.index(min(empty_buffers_machines))]
                
                if  current_machine.store_out.items != [] and isinstance(next_machine, Machine):
                    if float(self.manuf_line.references_config[current_machine.store_out.items[0]][self.manuf_line.list_machines.index(next_machine)+1]) ==0:
                        next_machine.current_product = current_machine.store_out.items[0]
                        return self.which_machine_to_feed(next_machine)
                    else:
                        return next_machine
                    
                elif current_machine.current_product is not None and isinstance(next_machine, Machine):
                    if float(self.manuf_line.references_config[current_machine.current_product][self.manuf_line.list_machines.index(next_machine)+1]) ==0:
                        next_machine.current_product = current_machine.current_product
                        return self.which_machine_to_feed(next_machine)
                    else:
                        return next_machine
                else:
                    return next_machine
            
            elif self.manuf_line.robot_strategy == 1:
                empty_buffers_machines = [m.buffer_in.level if isinstance(m, Machine) else m.level for m in current_machine.next_machines]
                next_machine = current_machine.next_machines[empty_buffers_machines.index(min(empty_buffers_machines))]
                # if float(self.manuf_line.references_config[current_machine.store_out.items[0]][self.manuf_line.list_machines.index(current_machine)+1]) ==0:
                #     return self.which_machine_to_feed(current_machine)
                # else:
                return next_machine
        

            elif self.manuf_line.robot_strategy == 2:
                empty_buffers_machines = [m.loaded if isinstance(m, Machine) else m.level for m in current_machine.next_machines]
                next_machine = current_machine.next_machines[empty_buffers_machines.index(min(empty_buffers_machines))]
                # if float(self.manuf_line.references_config[current_machine.store_out.items[0]][self.manuf_line.list_machines.index(current_machine)+1]) ==0:
                #     return self.which_machine_to_feed(current_machine)
                # else:
                return next_machine

            else:
                return True

    def which_machine_to_getfrom_alone(self, current_machine):

        """
        Return the best machine to get from based on a given strategy.
        """
        if self.manuf_line.robot_strategy == 0:
            if current_machine.first:
                previous_machine = current_machine.previous_machines[0]
            else:
                full_buffers_machines = [m.buffer_out.level if isinstance(m, Machine) and not m.broken and m.operating else float('inf') for m in current_machine.previous_machines]
                previous_machine = current_machine.previous_machines[full_buffers_machines.index(min(full_buffers_machines))]

            

        elif self.manuf_line.robot_strategy == 1:
            if isinstance(current_machine, Machine):
                full_buffers_machines = [m.buffer_out.level if isinstance(m, Machine) else m.level for m in current_machine.previous_machines]
                previous_machine = current_machine.previous_machines[full_buffers_machines.index(max(full_buffers_machines))]
            else:
                return True
        try:
            return previous_machine 
        except:
            # No machine
            return True
    
    def which_machine_to_getfrom(self, current_machine):

        """
        Return the best machine to get from based on a given strategy.
        """
        if self.manuf_line.robot_strategy == 0:
            if isinstance(current_machine, Machine):
                full_buffers_machines = [m.buffer_out.level if isinstance(m, Machine) and not m.broken and m.operating else float('inf') for m in current_machine.previous_machines]
                previous_machine = current_machine.previous_machines[full_buffers_machines.index(min(full_buffers_machines))]
            else:
                return True
            

        elif self.manuf_line.robot_strategy == 1:
            if isinstance(current_machine, Machine):
                full_buffers_machines = [m.buffer_out.level if isinstance(m, Machine) else m.level for m in current_machine.previous_machines]
                previous_machine = current_machine.previous_machines[full_buffers_machines.index(max(full_buffers_machines))]
            else:
                return True
        try:
            return previous_machine 
        except:
            # No machine
            return True


    def robot_process_local(self, from_entity, to_entity, transport_time=10):
        yield from self.transport(from_entity, to_entity, transport_time)

    def load_machine_new(self, current_machine, first=True):
        """
        Find the best machine to get from and load the machine or to transport the component to

        if the current machine output buffer is full, the robot will transport the component to the next machine
        if the current machine output buffer is not full, the robot will load the machine
        if the current machine input buffer is empty, the robot will transport the component to the machine
        """
        if current_machine.first:
            previous_machine = current_machine.previous_machines[0]
            if previous_machine.level > 0:
                yield from self.transport(previous_machine, current_machine)
            elif previous_machine.level  == 0 and  not current_machine.operating:
                yield from self.transport(previous_machine, current_machine)
            elif previous_machine.level == 0 and current_machine.operating:
                yield self.env.timeout(0)
        else:
            for i in range(len(current_machine.previous_machines)):
                previous_machine = current_machine.previous_machines[i]
                if previous_machine.buffer_out.level > 0:
                    yield from self.transport(previous_machine, current_machine)
                    break
                elif previous_machine.buffer_out.level == 0 and  previous_machine.operating:
                    yield self.env.timeout(0)


    def load_machine(self, tobe_loaded_machine, first=True):
        """
        """
        try:
            to_entity = tobe_loaded_machine
            from_entity = self.which_machine_to_getfrom(to_entity)
            try:
                if from_entity == True:
                    yield self.env.timeout(0)
                elif to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level > 0:
                    yield from self.transport(from_entity, to_entity)
                elif to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level == 0 and from_entity.operating:
                    yield from self.transport(from_entity, to_entity)
                else:
                    yield self.env.timeout(0)
            except Exception as e:
                print("Exception 1 :", e)
                if from_entity == True:
                    yield self.env.timeout(0)
                elif to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.level > 0:
                    yield from self.transport(from_entity, to_entity)
                elif to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.level == 0:
                    yield from self.transport(from_entity, to_entity)
                else:
                    yield self.env.timeout(0)
        except Exception as e:
            # If the machine breaks while loading by robot
            print("Machine broke while loading by robot " + tobe_loaded_machine.ID + " - Broken = " + str(from_entity.broken)   + " " + str(tobe_loaded_machine.broken) ) 
            print("Exception 2 :", e)
            yield self.env.timeout(0)

            


    def unload_machine(self, tobe_unloaded_machine, first=True):
        """
        """

        from_entity = tobe_unloaded_machine
        to_entity = self.which_machine_to_feed(from_entity)
        

        if to_entity == True:
            yield self.env.timeout(0)
        elif to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level > 0:
            yield from self.transport(from_entity, to_entity)
        elif to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level == 0 and from_entity.operating:
            yield from self.transport(from_entity, to_entity)
        else:
            yield self.env.timeout(0)

    def robot_process_unique(self, first=True):
        """
        entities_order = [input_entity, machine1, machine2, machine3, output_entity]
        times = [10, 10, 10, 10, 10] len(entities_order)
        """
        while True:
            for i in range(len(self.entities_order)):
                to_entity = self.entities_order[i]
                from_entity = self.which_machine_to_getfrom_alone(to_entity)
                yield from self.transport(from_entity, to_entity)
                    # if self.entities_order[i].first:
                    #     if self.entities_order[i].buffer_in.level < self.entities_order[i].buffer_in.capacity:
                    #         yield from self.transport(self.entities_order[i].previous_machine, self.entities_order[i])
                    # from_entity = self.entities_order[i]
                    # to_entity = self.which_machine_to_feed(from_entity)
                    # yield from self.transport(from_entity, to_entity)

    def robot_process(self, first=True):
        """
        entities_order = [input_entity, machine1, machine2, machine3, output_entity]
        times = [10, 10, 10, 10, 10] len(entities_order)
        """
        while True:
            try:
                # if first:
                # # Policy 1 => Follow everytime the same order, if not feasible pass to next 
                #     for  i, m in enumerate([m for m in self.manuf_line.list_machines if m.first]):
                #         if m.buffer_in.level < m.buffer_in.capacity:
                #             yield from self.transport(m.previous_machine, m)
                #         else:
                #             if m.next_machine.buffer_in.level < m.next_machine.buffer_in.capacity:
                #                 yield from self.transport(m, m.next_machine)
                #             else: 
                #                 pass

                ### Difference push or pull strategy
                for i in range(len(self.entities_order)):
                    from_entity = self.entities_order[i]

                    #Shall we unload this machine?

                    to_entity = self.which_machine_to_feed(from_entity)


                    if self.manuf_line.reset_shift_bool:
                        self.manuf_line.reset_shift_bool = False
                        print("Reseting Robot Process.")
                        break
                    
                    ## When we are feeding the first machines (it comes from a non machine input)
                    if not isinstance(from_entity, Machine):
                        if to_entity.buffer_in.level < to_entity.buffer_in.capacity:
                            yield from self.transport(from_entity, to_entity)
                        # elif to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.level == 0:
                        #   yield from self.transport(from_entity, to_entity)
                        else:
                            pass

                    else:
                        try:
                            if to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level > 0:
                                yield from self.transport(from_entity, to_entity)
                            elif to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level == 0 and (from_entity.operating and not from_entity.broken):
                                yield from self.transport(from_entity, to_entity)
                            else:
                                continue
                    
                        except Exception as e:
                            try:
                                if to_entity.level < to_entity.capacity and from_entity.buffer_out.level > 0:
                                    yield from self.transport(from_entity, to_entity)
                                elif to_entity.level < to_entity.capacity and from_entity.buffer_out.level == 0 and from_entity.operating:
                                    yield from self.transport(from_entity, to_entity)
                                    #continue
                            except Exception as e2:
                                if to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level > 0:
                                    yield from self.transport(from_entity, to_entity)
                                elif to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level == 0 and from_entity.operating:
                                    yield from self.transport(from_entity, to_entity)
            except simpy.Interrupt:
                print("Reseting Robot Process.")
                yield self.env.timeout(0)      

    def robot_process_old(self, first=True):
        """
        entities_order = [input_entity, machine1, machine2, machine3, output_entity]
        times = [10, 10, 10, 10, 10] len(entities_order)
        """

        while True:
            try:
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
                        print("Reseting Robot Process.")
                        break

                    try:
                        if to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level > 0:
                            yield from self.transport(from_entity, to_entity)
                        elif to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level == 0 and (from_entity.operating and not from_entity.broken):
                            yield from self.transport(from_entity, to_entity)
                        elif to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level == 0 and (not from_entity.operating or  from_entity.broken):
                            yield from self.handle_empty_buffer(from_entity, to_entity)
                        elif to_entity.buffer_in.level == to_entity.buffer_in.capacity:
                            yield from self.transport(to_entity, self.which_machine_to_feed(to_entity))
                        else:
                            continue
                
                    except Exception as e:
                        try:
                            if to_entity.level < to_entity.capacity and from_entity.buffer_out.level > 0:
                                yield from self.transport(from_entity, to_entity)
                            elif to_entity.level < to_entity.capacity and from_entity.buffer_out.level == 0 and from_entity.operating:
                                yield from self.transport(from_entity, to_entity)
                                #continue
                            elif to_entity.level < to_entity.capacity and from_entity.buffer_out.level == 0 and (not from_entity.operating or  from_entity.broken):
                                yield from self.handle_empty_buffer(from_entity, to_entity)
                            elif  from_entity.broken or  from_entity.broken:
                                continue
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
            except simpy.Interrupt:
                print("Reseting Robot Process.")
                yield self.env.timeout(0)                      

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
        try:
            uptime_m=  assembly_line.env.now/m.parts_done
        except:
            uptime_m = 0
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






def sigmoid(x, tau):
    """
    Sigmoid function.

    Parameters:
    - x (float): Input value.

    Returns:
    - float: Sigmoid of x.
    """
    return 1 / (1 + np.exp(-x/tau))