from __future__ import annotations
from copy import deepcopy	
import yaml
import simpy
import random
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
import csv
import re
import os
import sys
import time
import itertools


class ManufLine:
    def __init__(self, env, tasks, operators_assignement=None, tasks_assignement=None, config_file=None):

        """

         """
        try:
            with open(config_file, 'r') as stream:
                config = yaml.safe_load(stream)
        except:
            config_file = os.path.join(os.path.dirname(sys.executable), 'config.yaml')
            with open(config_file, 'r') as stream:
                config = yaml.safe_load(stream)

        self.env = env # Why is the SimPy environment a parameter since it's only used in there ? 
        self.randomseed = False
        self.parts_done = 0
        self.config = config
        self.sim_time = eval(self.config['sim_time'])
        self.machine_config_data = []
        self.breakdowns = self.config['breakdowns']
        self.breakdowns_switch = self.config['breakdowns']["enabled"]
        self.breakdown_law = "Weibull Distribution"
        self.n_repairmen = 3
        self.repairmen = simpy.PreemptiveResource(env, capacity=self.n_repairmen)
        self.first_machine = None # Nonsense. To change.
        self.stock_capacity = float(config["supermarket"]["capacity"])
        self.safety_stock = 0
        self.refill_time = None
        self.reset_shift_dec = False
        self.enable_robots = True

        ### Multi reference
        self.references_config = None

        # Set up the supermarket with no stock
        self.supermarket_in = simpy.Store(self.env, capacity=self.stock_capacity)
        self.shop_stock_out = simpy.Store(self.env, capacity=float(self.config["shopstock"]["capacity"]))

        self.pdp = None
        self.pdp_repeat = False
        self.pdp_change_time = 0

        self.num_cycles = 0
        self.supermarket_n_refills = 0
        self.takt_time = int(self.config['project_data']["takt_time"])
        self.tasks = tasks
        self.tasks_assignement = tasks_assignement
        self.operators_assignement = operators_assignement
        self.robot = None # Nonsense. To change.
        self.robots_list = []
        self.robot_strategy = 0
        self.reset_shift_bool = False
        self.local = True
        self.manual_operators = []
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

        self.refill_time_by_ref = {}
        self.expected_refill_time = None

        self.central_storage = None
        self.cs_track = {}

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
                if self.dev_mode:
                    print("Sim time = ", sim_time)
                waiting_times = list(self.machines_idle_times)
                breakdowns =  list(self.machines_breakdown)
                buffer_tracks = self.buffer_tracks
                robots_states = self.robot_states
                machines_state = self.machines_states
                machines_ct = self.machines_CT
                tracksim = False
                cycle_time = self.sim_time/len(self.shop_stock_out.items)
                if tracksim:
                    if self.dev_mode:
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
                parts_done = len(self.shop_stock_out.items)
                cycle_time = self.sim_time/len(self.shop_stock_out.items)
                if self.dev_mode:
                    print("buffer tracks  = ", [len(machine.buffer_tracks) for machine in self.list_machines])
                writer.writerow([experiment_number, idle_times, waiting_times, breakdowns, mean_ct, parts_done, cycle_time])
            return waiting_times, cycle_time, breakdowns
        
        if not save and not track:
            waiting_times = [machine.waiting_time for machine in self.list_machines]
            parts_done_per_machine = [machine.parts_done for machine in self.list_machines]
            breakdowns =  [machine.n_breakdowns for machine in self.list_machines]
            if len(self.shop_stock_out.items) != 0:
                cycle_time = self.sim_time/len(self.shop_stock_out.items)
            else:
                cycle_time = 100000000000

            return parts_done_per_machine, waiting_times, cycle_time, breakdowns

    # get_track can be deleted since only used in "main" files that are outdated
    def get_track(self):
        for i, machine in enumerate(self.list_machines):
            if not machine.last:
                plt.plot([t[0] for t in machine.buffer_tracks], [t[1] for t in machine.buffer_tracks], label='Buffer M '+str(i+1))
                plt.legend()

        plt.show()
        return self.list_machines

    def track_output(self):
        """
        Tracks outputs of each machine and the global outputs of the whole production line .

        Returns:
        None
        """
        self.machines_output = [[] for _ in range(len(self.list_machines))]
        self.output_tracks_per_ref = [[] for _ in range(len(self.references_config.keys()))]

        while True:
            yield self.env.timeout(self.sim_time/1000)
            self.output_tracks.append((self.env.now, len(self.shop_stock_out.items)))
            
            for i, ref in enumerate(self.references_config.keys()):
                self.output_tracks_per_ref[i].append((self.env.now, self.shop_stock_out.items.count(ref)))

            for i, m in enumerate(self.list_machines):
                self.machines_output[i].append((self.env.now, m.parts_done))

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
        self.buffer_tracks.append([(len(m.buffer_in.items), len(m.buffer_out.items)) for m in self.list_machines])
        self.robot_states.append(robot_state)
        self.machines_states.append([m.operating for m in self.list_machines])
        self.machines_idle_times.append([m.waiting_time for m in self.list_machines])
        self.machines_CT.append([self.env.now/(m.parts_done+1) for m in self.list_machines])
        self.machines_breakdown.append([m.n_breakdowns for m in self.list_machines])
        self.sim_times_track.append(self.env.now)

    def update_progress(self, progress_bar):
        """Update the progress bar for the UI"""
        t_start = time.time()
        while True:
            yield self.env.timeout(self.sim_time/100)
            estimated_remaining_time = round((self.sim_time - self.env.now)*(time.time()-t_start)/self.env.now, 2)
            progress_bar.progress(self.env.now/self.sim_time, text=f"Estimated Remaining time = {format_time(estimated_remaining_time, seconds_str=True)}")

    def run(self, progress_bar=None):
        """
        Runs the manufacturing line until the simulation time is over.
        
        Returns:
        None
        """
        self.generate()

        # Process machines and make them break down from time to time
        for i, m in enumerate(self.list_machines):
            m.prio = self.full_order[i] # Unused
            m.process = self.env.process(m.machine_process())
            self.env.process(self.break_down(m))
        
        # Process the refill of each input reference of the supermarket
        if self.pdp is not None:
            self.expected_refill_time = [float('inf')] * len(self.references_config.keys())
            self.env.process(self.refill_market_pdp())

        else:
            self.expected_refill_time = [0] * len(self.references_config.keys())
            for ref in list(self.references_config.keys()):
                if self.dev_mode:
                    print("Reference confirmed = ", ref)
                self.env.process(self.refill_market(ref))

        # Order machines related to each robot and process all robots
        if self.dev_mode:
            print(str(len(self.robots_list)) + "  -- Robot Included")
        if self.dev_mode:
            print("First machine = " , [m.first for m in self.list_machines])
        
        for i in range(len(self.robots_list)):
            self.robots_list[i].order = [int(j) for j in self.robots_list[i].order]
            machines_ordered = [self.list_machines[int(j-1)] for j in self.robots_list[i].order]
            self.robots_list[i].entities_order = machines_ordered
            # print("Inclued in robot = ", [self.list_machines[j-1].ID for j in self.robots_list[i].order])
            # print("Inclued in robot = ", [self.list_machines[j-1].first for j in self.robots_list[i].order])

            # Insert the supermarket as the first entity of the robot if any related machine is the first 
            if any([self.list_machines[j-1].first for j in self.robots_list[i].order]):
                self.robots_list[i].entities_order.insert(0, self.supermarket_in)

            self.robots_list[i].process = self.env.process(self.robots_list[i].robot_process())
            # self.robots_list[i].process = self.env.process(self.robots_list[i].robot_process_unique())

        #TODO: fix problem of reset shift with multi references
        if self.dev_mode:
            print("Resetting shift = ", self.reset_shift_dec)
        if self.reset_shift_dec:
            self.env.process(self.reset_shift())

        # Track outputs 
        self.env.process(self.track_output())
        for m in self.list_machines:
            if self.dev_mode:
                print(m.ID, m.buffer_in.capacity)
                print(m.ID, m.buffer_out.capacity)

        # Progress bar for UI
        if progress_bar is not None:
            self.env.process(self.update_progress(progress_bar))

        # Run the environment
        if self.dev_mode:
            print("Starting the sim now.")
        self.env.run(until=self.sim_time)
        #print(f"Current simulation time at the end: {self.env.now}")

    def reset(self):
        """
        Careful my friend! This function resets the full production line. Used mainly when wanting to start 
        a new simulation without interupting the code. 
        """
        self.env = simpy.Environment()

        # Set up the supermarket with initial stock
        self.supermarket_in = simpy.Store(self.env, capacity=self.stock_capacity)
        for ref in self.references_config.keys():
            for _ in range(int(self.references_config[ref][1])):
                self.supermarket_in.put(ref)

        self.shop_stock_out = simpy.Store(self.env, capacity=float(self.config["shopstock"]["capacity"]))

        self.repairmen = simpy.PreemptiveResource(self.env, capacity=int(self.n_repairmen))

        # Reset robots and machines
        self.robots_list = []
        self.create_machines(self.machine_config_data)
        for robot in self.robots_list: # Useless since robot.env already updated in create_machines ? 
            robot.env = self.env

        self.reset_shift()

    def save_global_settings(self, configuration, references_config, line_data, buffer_sizes=[], pdp:list=None):
        """
        Save global settings and configure the manufacturing line based on the input configuration.
        
        :param configuration: Dictionary with various settings for the manufacturing line.
        :param references_config: Reference configurations.
        :param line_data: Configuration data for machines.
        :param buffer_sizes: Buffer sizes for machines (optional).
        :param pdp: List of (reference, quantity) pairs
        """

        # Enable or disable breakdowns and random seed
        self.breakdowns_switch = configuration.get("enable_breakdowns", False)
        self.randomseed = configuration.get("enable_random_seed", False)

        self.stock_capacity = float(configuration["stock_capacity"]) if pdp is None else float('inf')
        self.reset_shift_dec = bool(configuration["reset_shift"])
        self.dev_mode = bool(configuration["dev_mode"])        
        self.safety_stock = float(configuration["safety_stock"])

        self.breakdown_law = str(configuration["breakdown_dist_distribution"])
        self.n_repairmen = int(configuration["n_repairmen"])
        self.repairmen = simpy.PreemptiveResource(self.env, capacity=int(configuration["n_repairmen"]))

        self.enable_robots = configuration['enable_robots']
        available_strategies = ["Balanced Strategy", "Greedy Strategy"]
        self.robot_strategy = int(available_strategies.index(configuration["strategy"]))

        self.references_config = references_config
        self.machine_config_data = line_data

        self.pdp = pdp
        self.pdp_repeat = configuration['repeat_pdp']
        self.pdp_change_time = configuration['pdp_change_time']

        # Set up the supermarket with initial stock
        self.supermarket_in = simpy.Store(self.env, capacity=self.stock_capacity)
        if self.dev_mode:
            print("supermarket ", self.supermarket_in.capacity)

        if pdp is None:
            for ref in self.references_config.keys():
                for _ in range(int(self.references_config[ref][1])):
                    self.supermarket_in.put(ref)

        # Set up the shop stock
        self.shop_stock_out = simpy.Store(self.env, capacity=float(self.config["shopstock"]["capacity"]))

        # Store refill time for each reference
        for ref in self.references_config.keys():
            pattern = r'^(\d+)-(\d+)$'
            match = re.match(pattern, str(self.references_config[ref][0]))
            if match:
                value1 = int(match.group(1))
                value2 = int(match.group(2))
                self.refill_time_by_ref[ref] = [value1, value2]
            else:
                self.refill_time_by_ref[ref] = float(self.references_config[ref][0])

        # Set simulation time and takt time
        self.sim_time = eval(str(configuration["sim_time"]))
        if self.dev_mode:
            print("sim time first = ",  self.sim_time)
        self.takt_time = eval(str(configuration["takt_time"]))

        # Update buffer sizes if provided
        if buffer_sizes:
            for i in range(len(self.machine_config_data)):
                self.machine_config_data[i][3] = buffer_sizes[i]

        # Initialize central storage tracking
        now = self.env.now
        for ref in self.references_config.keys():
            self.cs_track[ref] = [[now, 0]]

    # Unused (Commented in "run action" button of the app)
    def initialize(self):
        self.generate()
        for i, m in enumerate(self.list_machines):
            m.process = self.env.process(m.machine_process())
            self.env.process(self.break_down(m))

        self.expected_refill_time = [0 for _ in range(len(self.references_config.keys()))]
        for ref in list(self.references_config.keys()):
            if self.dev_mode:
                print("Reference confirmed = ", ref)
            self.env.process(self.refill_market(ref))
        
        self.env.process(self.reset_shift())

    def run_action(self, action):
        """
        Performs an action that is transporting from a machine to another.

        Action is list of two elements: [from_machine, to_machine]

        Returns:
        None
        """
        if self.dev_mode:
            print('Going from ' + str(action[0]) + " " + str(action[1]) )
        self.robot.process = self.env.process(self.robot.robot_process_local(action[0], action[1]))
        self.env.run(until=self.env.now + 1000)

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
            # Shift every 8 hours
            yield self.env.timeout(3600 * 8)
            if self.dev_mode:
                print("Resetting Shift")
            
            self.num_cycles = self.num_cycles + 1
            self.reset_shift_bool = True

            # Empty machine storage (buffers and stores) 
            for machine in self.list_machines:

                # No robots in the manufacturing line -> Be careful about supermarket and shop stock 
                if self.robots_list == []:
                    if self.dev_mode:
                        print("Reseting with no robot")
                    if not machine.last:
                        while len(machine.buffer_out.items) > 0:
                            machine.buffer_out.get()

                    if not machine.first:
                        while len(machine.buffer_in.items) > 0:
                            machine.buffer_in.get()

                # At least one robot -> Empty all buffers
                else:
                    while len(machine.buffer_in.items) > 0:
                        machine.buffer_in.get()

                    while len(machine.buffer_out.items) > 0:
                            product = machine.buffer_out.get()

                            # For last machines linked to the shop stock -> add the product to the shop stock
                            if machine.last:
                                self.shop_stock_out.put(product)

                machine.parts_done_shift = 0

    def break_down(self, machine:Machine):
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
                # Break the machine
                time_to_break = machine.time_to_failure()
                yield self.env.timeout(time_to_break)
                machine.broken = True
                machine.process.interrupt()
                machine.n_breakdowns += 1

                if self.dev_mode:
                    print(f"/!\ Time {round(self.env.now)} :", f"{machine.ID} broke down after {round(time_to_break)}")
    
                # Repair by calling a repairman
                repair_in = self.env.now
                machine.repair_event = self.env.event()
                with self.repairmen.request(priority=1) as req:
                    yield req
                    yield self.env.timeout(machine.MTTR)
                
                machine.broken = False
                machine.repair_event.succeed()
                repair_time = self.env.now - repair_in
                machine.real_repair_time.append(float(repair_time))

                if self.dev_mode:
                    print(f"/!\ Time {round(self.env.now)} :", f"{machine.ID} repaired after {round(repair_time)}")       

    def monitor_waiting_time(self, machine):
        while True:
            if not (machine.operating or machine.broken):
                if self.dev_mode:
                    print(machine.ID + " Not Operating")
                yield self.env.timeout(1)
                machine.waiting_time_rl += 1

    # Outdated (Only used in "main" python files)
    def set_CT_machines(self, CTs):

        if len(self.list_machines) != len(CTs):
            raise ValueError('No matching! You have chosen ' + str(len(self.list_machines)) + ' machines and have given ' + str(len(CTs)))

        for i, machine in enumerate(self.list_machines):
            machine.ct = CTs[i]

    def refill_market(self, ref):
        """
        Refills the input market with products.

        This function is responsible for refilling the input market with products. It runs in an infinite loop and refills the market
        based on the specified refill time and refill size. The refill time can be either a single value or a range of values.
        If it is a range, a random value within the range is chosen for each refill. The refill size determines the number of
        products to be added to the market during each refill.

        Note: This function uses the `self.env` attribute as the simulation environement.
        """
        refill_time_ref = self.refill_time_by_ref[ref]
        amount = int(self.references_config[ref][2])

        while True:
            refill_time = refill_time_ref if isinstance(refill_time_ref, float) else int(random.uniform(refill_time_ref[0], refill_time_ref[1]))
            yield self.env.timeout(refill_time)
            
            for _ in range(amount):
                yield self.supermarket_in.put(ref) 
            self.supermarket_n_refills += 1
            if self.dev_mode:
                print(f"Time {round(self.env.now)} :", f"SUPERMARKET refilled with {amount} {ref}")         

    def refill_market_pdp(self):
        """
        Refills the supermarket using a PDP.

        This function is responsible for refilling the supermarket with products. 
        It is based on a PDP, composed of a finite list of (reference, quantity) pairs.
        The list is repeated in an endltess loop.
       
        For each pair, add the quantity of the reference at hand to the supermarket.
        When moving to the next pair, waits a given time if the new reference is different than the previous one. 
        """
        previous_ref = self.pdp[0][0]
        previous_qty = self.pdp[0][1]

        to_produce = itertools.cycle(self.pdp) if self.pdp_repeat else self.pdp
        for ref, quantity in to_produce:
            if ref != previous_ref:
                # Wait until the previous batch has been completely produced (in shop stock)
                if self.dev_mode:
                    print(f"Time {round(self.env.now)} :", f"Waiting for batch {previous_qty} {previous_ref} to be complete to change tools for {ref}")
                while len(self.shop_stock_out.items) < previous_qty or self.shop_stock_out.items[-previous_qty:] != [previous_ref]*previous_qty:
                    yield self.env.timeout(1)
                
                # Wait some time for tools change
                if self.dev_mode:
                    print(f"Time {round(self.env.now)} :", f"Tools being changed from {previous_ref} to {ref}")
                yield self.env.timeout(self.pdp_change_time)
                if self.dev_mode:
                    print(f"Time {round(self.env.now)} :", f"Tools changed")
            
            else:
                while len(self.supermarket_in.items) > 0:
                    yield self.env.timeout(1)

            # Add the quantity all at once     
            for _ in range(quantity):
                yield self.supermarket_in.put(ref)
            if self.dev_mode:
                print(f"Time {round(self.env.now)} :", f"SUPERMARKET refilled with {quantity} {ref}")         

            # Update the previous reference    
            previous_ref = ref
            previous_qty = quantity
            yield self.env.timeout(1)                

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
        Set up the machines using the production line data.

        list_machines_config = [
        "0 : Name",
        "1 : Description",
        "2 : Link",
        "3 : Buffer Capacity",
        "4 : Initial Buffer",
        "5 : MTTF",
        "6 : MTTR",
        "7 : Transport time",
        "8 : Transport order",
        "9 : Transporter ID", 
        "10 : Operator ID",
        "11 : Manual Time",
        "12 : Identical Station,
        "13 : Fill central storage] 
        """
        # Not understood
        if self.operators_assignement or self.tasks_assignement:
            if len(list_machines_config) != max(self.tasks_assignement):
                raise ValueError('No match between assignement of tasks and number of machines.')
            if len(list_machines_config) != max(max(self.operators_assignement)):
                raise ValueError('No match between assignement of operators and number of machines.')

            machine_indices = [[]] * len(list_machines_config)
            for index, n_machine in enumerate(self.tasks_assignement):
                    machine_indices[n_machine - 1].append(index)

        # Create robots ONLY IF all cells "Robot Transport Time IN" and "Robot Assignment" are not NaN (empty)
        # all_robots_time_defined = all([not np.isnan(list_machines_config[i][7]) for i in range(len(list_machines_config))])
        # all_robots_assigned = all([not np.isnan(list_machines_config[i][9]) for i in range(len(list_machines_config))]) # Conditions required ? 
        ordered_unique_robots = []
        for raw_name in [machine_config[9] for machine_config in list_machines_config]:
            if raw_name not in ordered_unique_robots:
                ordered_unique_robots.append(raw_name)

        self.robots_list = []
        if self.enable_robots:
            for raw_name in ordered_unique_robots:
                # Prevent robots from having a number as its name (causes trouble for plotting graph)
                try:
                    robot_name = f"R{int(raw_name)}"
                except:
                    robot_name = raw_name
                self.robot = Robot(robot_name, self, self.env)

                # Order machines assigned to the robot and their related transport time
                self.robot.order = [machine_config[8] for machine_config in list_machines_config if (machine_config[9] == raw_name)]
                self.robot.in_transport_times = [machine_config[7] for machine_config in list_machines_config if (machine_config[9] == raw_name)]
                self.robots_list.append(self.robot)
        
        # Store order and robot transport times
        self.full_transport_times = [list_machines_config[j][7] for j in range(len(list_machines_config))]
        self.full_order = [list_machines_config[j][8] for j in range(len(list_machines_config))]

        # Order linked machines and remove duplicates
        notfirst_list = [list_machines_config[i][2].strip("'") for i in range(len(list_machines_config))]
        lists = [eval(item) if item.startswith('[') else [item] for item in notfirst_list]
        flattened_list = [item for sublist in lists for item in sublist]
        unique_list = list(set(flattened_list))

        # Create all the "Machine" object from the machines cconfiguration parameters
        self.list_machines = []
        for i, machine_config in enumerate(list_machines_config):
            # May be a first or last machine
            is_first = machine_config[0] not in unique_list
            is_last = machine_config[2] == "END"

            # Evaluate MTTF and MTTR
            try: 
                mttf = eval(str(machine_config[5]))
                mttr = eval(str(machine_config[6]))
            except:
                mttf = float(machine_config[5])
                mttr = float(machine_config[6])

            # Buffer parameters
            buffer_capacity = int(machine_config[3])
            initial_buffer = int(machine_config[4])

            # Allow filling the central storage 
            try:
                fill_central_storage = machine_config[13]
            except:
                fill_central_storage = False

            # TODO : set the operating time of each machine here given the different references in input 

            # Create machine with above parameters whether there were tasks already assigned
            try:
                machine_has_robot = False if machine_config[9]==0 or len(self.robots_list) ==0 else True
                assigned_tasks =  list(np.array(self.tasks)[machine_indices[i]])
                machine = Machine(self, self.env, machine_config[0], machine_config[1], self.config, assigned_tasks=assigned_tasks, first=is_first, last=is_last, breakdowns=self.breakdowns['enabled'], mttf=mttf, mttr=mttr, buffer_capacity=buffer_capacity, initial_buffer=initial_buffer, hazard_delays=self.config['hazard_delays']['enabled'], has_robot=machine_has_robot, fill_central_storage=fill_central_storage)
            except:
                machine_has_robot = False if machine_config[9]==0 or len(self.robots_list) ==0 else True
                machine = Machine(self, self.env, machine_config[0], machine_config[1], self.config, first=is_first, last=is_last, breakdowns=self.breakdowns['enabled'], mttf=mttf, mttr=mttr, buffer_capacity=buffer_capacity , initial_buffer=initial_buffer, hazard_delays=self.config['hazard_delays']['enabled'], has_robot=machine_has_robot, fill_central_storage=fill_central_storage)

            # Store the created machine
            if machine.first:
                self.first_machine = machine
            self.list_machines.append(machine)
        
        # For each machine, retrieve which other machines comes right before, right after and manage their storage
        operators = {}
        index_next = []
        machines_ids = [m.ID for m in self.list_machines]
        for i, machine in enumerate(self.list_machines):
            # TODO : upcoming feature for same machines ?
            if not str(list_machines_config[i][12]) in ["nan", "None", ""]:
                indexmachine = [m.ID for m in self.list_machines].index(str(list_machines_config[i][12]))
                machine.same_machine = self.list_machines[indexmachine]

            
            # Add Manual Operators to machine
            operator_id = list_machines_config[i][10]  # Assigned transporter number
            is_manual = False if list_machines_config[i][10] == 0 else True    # True if manual, else empty (robot)


            # Check if transporter is manual 
            if is_manual is True:
                # If this manual transporter doesn't already have an Operator, create one
                if operator_id not in operators:
                    operators[operator_id] = Operator(operator_id)
                    self.manual_operators.append(operators[operator_id])
                    
                
                # Assign the current machine (i) to the operator
                machine.operator = operators[operator_id]
                machine.manual_time = float(list_machines_config[i][11]) if list_machines_config[i][11] else 0
                #machine.operator.wc = list_machines_config[i][7]
                machine.operator.assign_machine(self.list_machines[i])
            
    
            # Add supermarket and shop stock for first and last machines
            if machine.first:
                machine.previous_machine = self.supermarket_in
                machine.previous_machines.append(self.supermarket_in)
        

            if machine.last:
                machine.next_machine = self.shop_stock_out
                machine.next_machines.append(self.shop_stock_out)
            try:
                # If there is a robot, store the time required to move
                if len(self.robots_list) > 0:
                    machine.move_robot_time = self.full_transport_times[i]

                # Get the machine(s) that comes right after the current machine in the process
                try:
                    linked_machines = eval(list_machines_config[i][2])
                except:
                    linked_machines = list_machines_config[i][2]

                # The machine may deliver to one or many machines, then always use a list as if there were many machines
                linked_machines = linked_machines if type(linked_machines) is list else [linked_machines]

                # Check each following machine
                for linked_machine in linked_machines:
                    i_linked = machines_ids.index(linked_machine)

                    # Machine that never appeared before
                    if i_linked not in index_next:
                        index_next.append(i_linked)

                        # Add previous and following machines
                        machine.next_machine = self.list_machines[i_linked]
                        machine.next_machines.append(self.list_machines[i_linked])

                        self.list_machines[i_linked].previous_machine = machine
                        self.list_machines[i_linked].previous_machines.append(machine)
                        
                        # If no robot, equal storage
                        if len(self.robots_list) == 0 or not machine.has_robot:
                            self.list_machines[i_linked].buffer_in = machine.buffer_out
                    
                    # Machine already encoutered before
                    else:
                        # Add previous and following machines
                        machine.next_machine = self.list_machines[i_linked]
                        machine.next_machines.append(self.list_machines[i_linked])

                        self.list_machines[i_linked].previous_machine = machine
                        self.list_machines[i_linked].previous_machines.append(machine)

                        # If no robot, equal storage
                        if len(self.robots_list) == 0 or not machine.has_robot:
                            machine.buffer_out = machine.next_machine.buffer_in

            except:
                pass

    # Methods below are empty
    def deplete_shopstock(self):
        pass

    def deliver_to_client(self):
        pass


class Machine:
    def __init__(self, manuf_line, env, machine_id, machine_name, config,  assigned_tasks = None, robot=None, operator=None, previous_machine = None, first = False, last=False, breakdowns=True, mttf=3600*24*7, mttr=3600, buffer_capacity=100, initial_buffer=0, hazard_delays=False, has_robot=False, fill_central_storage=False):
        self.mt = 0
        self.ID = machine_id
        self.Name = machine_name
        self.env:simpy.Environment = env
        self.manuf_line:ManufLine = manuf_line
        self.entry_times = []  # List to store entry times of parts
        self.exit_times = []   # List to store exit times of parts
        self.finished_times = []
        self.n_breakdowns = 0
        self.buffer_tracks = []
        self.parts_done = 0
        self.parts_done_shift = 0
        self.ct = 0
        self.wc = [] # Work Content (Manual Op)
        self.manual_time = 0
        self.config = config

        self.buffer_btn = None
        self.buffer_capacity = buffer_capacity
        self.initial_buffer = initial_buffer
        self.process = None
        self.next_machine = None
        self.has_robot = has_robot
        self.previous_machine = None
        self.operating_state = []

        self.next_machines = []
        self.previous_machines = []
        self.waiting_time = [0, 0] #Stavation # Blockage
        self.waiting_time_rl = 0 #real time waiting time
        self.operating = False
        self.move_robot_time = 0
        self.same_machine = None
        self.ref_produced = []
        self.current_product = None

        # Define input & output buffers. Warning ! When the buffer is asked to begin with initial items -> it's filled with the 1st reference to come.
        self.buffer_in = simpy.Store(env, capacity=float(buffer_capacity))
        self.buffer_out = simpy.Store(env, capacity=float(buffer_capacity))

        if manuf_line.pdp is None:
            refilled_times = [refill_time_value if isinstance(refill_time_value, float) else int(random.uniform(refill_time_value[0], refill_time_value[1])) for refill_time_value in self.manuf_line.refill_time_by_ref.values()]
            ref_to_fill = list(self.manuf_line.refill_time_by_ref.keys())[refilled_times.index(min(refilled_times))]
        else:
            ref_to_fill = manuf_line.pdp[0][0]

        for _ in range(self.initial_buffer):
            self.buffer_in.put(ref_to_fill)
            self.buffer_out.put(ref_to_fill)
        
        # When NO robot, directly connect first machine to supermarket and last machine to shop stock 
        if self.manuf_line.robots_list == [] or not self.has_robot:
            if first:
                self.buffer_in = manuf_line.supermarket_in
            
            if last:
                self.buffer_out = manuf_line.shop_stock_out          

        
        self.MTTF = mttf # Mean time to failure in seconds
        self.MTTR = mttr
        self.assigned_tasks = assigned_tasks
        
        self.operator:Operator = None  # Assign the operator to the machine
    
        self.loaded = 0
        self.last = last
        self.first = first
        self.prio = 1
        self.broken = False
        self.breakdowns = breakdowns
        self.real_repair_time = []
        self.hazard_delays = 1 if hazard_delays else 0
        self.op_fatigue = config["fatigue_model"]["enabled"]

        self.fill_central_storage = fill_central_storage

        self.repair_event:simpy.Event = None


    def time_to_failure(self):
        """Return time until next failure for a machine."""
        # deterioration_factor = 1 + (self.env.now / self.simulation_duration)  # Adjust this factor as needed
        # adjusted_MTTF = self.MTTF / deterioration_factor
        
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
            val = random.gammavariate(0.3, self.MTTF)

            # To delete ? 

            # rate_of_decrease = 0.1
            # machine_age = self.env.now/self.MTTF
            # initial_k = 0.5
            # if initial_k !=1:
            #     k = max(0.1, initial_k - rate_of_decrease * machine_age)
            # else:
            #     k = initial_k

        elif self.manuf_line.dev_mode:
            print("No law given. ")

        return val
    
    def fatigue_model(self, elapsed_time, base_time):
        """
        Model operator fatigue based on elapsed time using a sigmoid function.

        Parameters:
        - elapsed_time (float): Elapsed time in hours.
        - base_time (float): Base time needed for a manual action without fatigue.

        Returns:
        - adjusted_time (float): Adjusted time for a manual action based on fatigue.
        """
        fatigue_rate =  float(self.config["fatigue_model"]["max-fatigue-rate"]) * sigmoid(elapsed_time, eval(self.config["fatigue_model"]["tau-fatigue"]))
        adjusted_time = (1+fatigue_rate) * base_time

        return adjusted_time

    def machine_process(self):
        while True: 
            # bias_shape = 2  # shape parameter
            # bias_scale = 1  # scale parameter
            # if any([m.ct != 0 for m in self.manuf_line.list_machines]):
            #     if self.op_fatigue:
            #         deterministic_time = self.fatigue_model(self.env.now/(self.manuf_line.num_cycles+1), self.ct)
            #     else:
            #         deterministic_time = self.ct
            # elif self.assigned_tasks is not None:
            #     num_samples = len(self.assigned_tasks)
            #     if self.op_fatigue:
            #         deterministic_time = np.sum([task.machine_time+self.fatigue_model((self.manuf_line.num_cycles+1), task.manual_time) for task in self.assigned_tasks])
            #     else:
            #         deterministic_time = np.sum([task.machine_time+task.manual_time for task in self.assigned_tasks])

            # num_samples = int(1/float(self.config["hazard_delays"]["probability"]))
            # + self.hazard_delays*np.mean(weibull_min.rvs(bias_shape, scale=bias_scale, size=num_samples))

            entry0 = self.env.now
            self.entry_times.append(entry0)
            self.buffer_tracks.append((entry0, len(self.buffer_out.items)))
            
            self.operating = False
            if self.operator:
                self.operator.busy = False 

            # Load from buffer
            self.current_product = None
            while self.current_product is None:
                try:
                    with self.buffer_in.get() as load_request:
                        self.current_product = yield load_request                   
                except simpy.Interrupt:
                    yield self.repair_event
            self.waiting_time[0] += (self.env.now-entry0)
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f"{self.ID} loaded {self.current_product}")
            
            # Operator
            if self.operator:
                done = False
                while not done:
                    entry_time = self.env.now

                    # Wait for the operator to be free
                    while self.operator.busy:
                        try:
                            yield self.env.timeout(1)
                        except simpy.Interrupt:
                            yield self.repair_event
                            continue
                    self.operator.busy = True

                    # Manual time
                    try:
                        yield self.env.timeout(self.manual_time)
                    except simpy.Interrupt:
                        # Free the operator and start it all again
                        self.operator.busy = False
                        yield self.repair_event
                        continue
                    done = True
                
                self.operator.busy = False
                self.operator.wc += self.manual_time
                self.wc.append(self.env.now-entry_time)

            # Other process ongoing
            if self.same_machine is not None:
                entry_op = self.env.now
                while self.same_machine.operating:
                    try:
                        yield self.env.timeout(1)
                    except simpy.Interrupt:
                        yield self.repair_event
                self.waiting_time[1] += (self.env.now-entry_op)

            # Operate
            done_in = float(self.manuf_line.references_config[self.current_product][self.manuf_line.list_machines.index(self)+3])
            self.operating = True
            while done_in > 0:
                try:
                    start = self.env.now
                    yield self.env.timeout(done_in)
                    done_in = 0
                except simpy.Interrupt:
                    done_in -= (self.env.now-start)
                    yield self.repair_event
            self.operating = False

            # Unload to buffer
            entry_wait = self.env.now
            while self.current_product is not None:
                try:
                    with self.buffer_out.put(self.current_product) as unload_request:
                        yield unload_request
                        self.ref_produced.append(self.current_product)
                        self.current_product = None                   
                except simpy.Interrupt:
                    yield self.repair_event

            self.waiting_time[1] += (self.env.now-entry_wait)
            self.finished_times.append(self.env.now-entry0)
            self.parts_done += 1
            self.parts_done_shift += 1

            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f"{self.ID} produced {self.current_product}")


class Robot:
    """
    Transport Robot between machines
    """
    def __init__(self, id, manuf_line:ManufLine, env:simpy.Environment, maxlimit=1):
        # self.assigned_machines = assigned_machines
        self.ID = id
        self.busy = False
        self.manuf_line = manuf_line
        self.env = env
        self.schedule = [] # Unused yet
        self.buffer = simpy.Store(self.env, capacity=float(maxlimit))
        self.robots_res = simpy.PriorityResource(self.env, capacity=1) # Unused yet
        self.waiting_time = 0
        self.in_transport_times = []
        self.process = None
        self.entities_order = None
        self.loadunload_time = 50
            
    def transport(self, from_entity, to_entity, to_central_storage=False, from_central_storage=False):
        """
        Handle transport between two entities, update storages. Cancel the transport when breakdowns appear.

        Parameters:
         - to_central_storage : True when a product must be sent to the central storage. 'to_entity' is still required for route traceability.
         - from_central_storage : True when a product must be taken from the central storage and sent to 'to_entity'. 'from_entity' is still required for route traceability.

        TODO: Fix the problem of the robot not being able to transport from a machine to another after shift reset
              Do we need shift reset of buffers ?

        Returns:
        None 
        """
    
        # Skip transport if the robot is already busy
        if self.busy:
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f"{self.ID} already busy")
            yield self.env.timeout(1)
            return
        
        self.busy = True
        entry = self.env.now

        # Transport between two machines
        if isinstance(from_entity, Machine) and isinstance(to_entity, Machine) and not to_central_storage and not from_central_storage:
            origin_name, destination_name = from_entity.ID, to_entity.ID
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f"{self.ID} scheduled from {origin_name} to {destination_name}")
       
            # Start by waiting for an available input resource from origin
            while len(from_entity.buffer_out.items) == 0:
                yield self.env.timeout(1)

                # Cancel transport if origin machine breaks down 
                if from_entity.broken:
                    if self.manuf_line.dev_mode:
                        print(f"Time {round(self.manuf_line.env.now)} :", f"{self.ID} transport cancelled because {origin_name} broke down")
                    self.busy = False
                    yield self.env.timeout(1)
                    return
            
            # Get the product
            product = yield from_entity.buffer_out.get()
            self.waiting_time += self.env.now-entry
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f"{self.ID} got {product} from {origin_name}")
            
            # Move robot 
            yield self.env.timeout(abs(to_entity.move_robot_time - from_entity.move_robot_time)+self.loadunload_time)
            
            # Wait for a spot in destination
            entry_2 = self.env.now
            yield to_entity.buffer_in.put(product)
            to_entity.loaded += 1
            self.waiting_time += self.env.now-entry_2
            self.manuf_line.track_sim((origin_name, destination_name, self.env.now))
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f'{self.ID} put {product} in {destination_name}')

        # Transport from supermarket to a machine
        elif isinstance(from_entity, simpy.Store) and isinstance(to_entity, Machine) and not to_central_storage and not from_central_storage:
            origin_name, destination_name = "SUPERMARKET", to_entity.ID
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f"{self.ID} scheduled from {origin_name} to {destination_name}")

            # Get the product
            product = yield from_entity.get()
            self.waiting_time += self.env.now-entry
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f"{self.ID} got {product} from {origin_name}")

            # Move robot 
            yield self.env.timeout(abs(to_entity.move_robot_time)+self.loadunload_time)
            
            # Wait for a spot in destination
            entry_2 = self.env.now
            yield to_entity.buffer_in.put(product) 
            to_entity.loaded += 1
            self.waiting_time += self.env.now-entry_2
            self.manuf_line.track_sim((origin_name, destination_name, self.env.now))
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f'{self.ID} put {product} in {destination_name}')

        # Transport from a machine to the central storage
        elif isinstance(from_entity, Machine) and isinstance(to_entity, Machine) and to_central_storage and not from_central_storage:
            origin_name, destination_name = from_entity.ID, to_entity.ID
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f"{self.ID} scheduled from {origin_name} to {destination_name}")
           
            # Start by waiting for an available input resource from origin
            while len(from_entity.buffer_out.items) == 0:
                yield self.env.timeout(1)

                # Skip transport if origin machine breaks down
                if from_entity.broken:
                    if self.manuf_line.dev_mode:
                        print(f"Time {round(self.manuf_line.env.now)} :", f"{self.ID} transport cancelled because {origin_name} broke down")
                    self.busy = False
                    yield self.env.timeout(1)
                    return

            # Get the product
            product = yield from_entity.buffer_out.get()
            self.waiting_time += self.env.now-entry
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f"{self.ID} got {product} from {origin_name}")

            # Move robot
            yield self.env.timeout(abs(max(self.manuf_line.central_storage.times_to_reach) - from_entity.move_robot_time)+self.loadunload_time)
            
            # Wait for a spot in destination
            entry_2 = self.env.now
            self.manuf_line.central_storage.put(ref_data={'name': product, 'route': (from_entity, to_entity), 'status': 'OK'}) # TODO : Status OK so far.
            self.waiting_time += self.env.now-entry_2 
            self.manuf_line.track_sim((origin_name, destination_name, self.env.now))
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f'{self.ID} put {product} in {destination_name}')

            # Specific tracking
            now = self.env.now
            for ref in self.manuf_line.cs_track:
                self.manuf_line.cs_track[ref].append([now, self.manuf_line.cs_track[ref][-1][1]])
                if ref == product:
                    self.manuf_line.cs_track[ref][-1][1] += 1

        # Transport from the central storage to a machine
        elif isinstance(from_entity, Machine) and isinstance(to_entity, Machine) and not to_central_storage and from_central_storage:
            origin_name, destination_name = from_entity.ID, to_entity.ID
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f"{self.ID} scheduled from {origin_name} to {destination_name}")

            # # Wait the time to go to the central storage
            # yield self.env.timeout(abs(max(self.manuf_line.central_storage.times_to_reach) - from_entity.move_robot_time)+self.loadunload_time)

            # Get the product
            product = self.manuf_line.central_storage.get_by_destination(to_entity)
            self.waiting_time += self.env.now-entry
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f"{self.ID} got {product} from {origin_name}")

            # Move robot 
            yield self.env.timeout(abs(to_entity.move_robot_time - max(self.manuf_line.central_storage.times_to_reach))+self.loadunload_time)
    
            # Wait for a spot in destination
            entry_2 = self.env.now
            yield to_entity.buffer_in.put(product['name'])
            to_entity.loaded += 1
            self.waiting_time += self.env.now-entry_2
            self.manuf_line.track_sim((origin_name, destination_name, self.env.now))
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f'{self.ID} put {product} in {destination_name}')

            # Specific tracking
            now = self.env.now
            for ref in self.manuf_line.cs_track:
                self.manuf_line.cs_track[ref].append([now, self.manuf_line.cs_track[ref][-1][1]])
                if ref == product['name']:
                    self.manuf_line.cs_track[ref][-1][1] -= 1
            
        # Transport from a machine to shop stock 
        elif isinstance(from_entity, Machine) and  isinstance(to_entity, simpy.Store) and not to_central_storage and not from_central_storage:
            origin_name, destination_name = from_entity.ID, "SHOP STOCK"
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f"{self.ID} scheduled from {origin_name} to {destination_name}")

            # Start by waiting for an available input resource from origin
            while len(from_entity.buffer_out.items) == 0:
                yield self.env.timeout(1)

                # Cancel transport if origin machine breaks down 
                if from_entity.broken:
                    if self.manuf_line.dev_mode:
                        print(f"Time {round(self.manuf_line.env.now)} :", f"{self.ID} transport cancelled because {origin_name} broke down")
                    self.busy = False
                    yield self.env.timeout(1)
                    return

            # Get the product
            product = yield from_entity.buffer_out.get()
            self.waiting_time += self.env.now-entry
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f"{self.ID} got {product} from {origin_name}")

            # Move robot 
            yield self.env.timeout(abs(from_entity.move_robot_time)+self.loadunload_time)

            # Wait for a spot in destination
            entry_2 = self.env.now
            yield to_entity.put(product)
            self.waiting_time += self.env.now-entry_2
            self.manuf_line.track_sim((origin_name, destination_name, self.env.now))
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f'{self.ID} put {product} in {destination_name}')

        # Transport from the central storage to shop stock
        elif isinstance(from_entity, Machine) and isinstance(to_entity, simpy.Store) and not to_central_storage and from_central_storage:
            origin_name, destination_name = from_entity.ID, "SHOP STOCK"
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f"{self.ID} scheduled from {origin_name} to {destination_name}")

            # # Wait the time to go to the central storage
            # yield self.env.timeout(abs(max(self.manuf_line.central_storage.times_to_reach) - from_entity.move_robot_time)+self.loadunload_time)

            # Get the product
            product = self.manuf_line.central_storage.get_by_destination(to_entity)
            self.waiting_time += self.env.now-entry
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f"{self.ID} got {product} from {origin_name}")

            # Move robot
            yield self.env.timeout(max(self.manuf_line.central_storage.times_to_reach)+self.loadunload_time)
            
            # Wait for a spot in destination
            entry_2 = self.env.now
            yield to_entity.put(product['name']) # TODO : Status OK so far.
            self.waiting_time += self.env.now-entry_2
            self.manuf_line.track_sim((origin_name, destination_name, self.env.now))
            if self.manuf_line.dev_mode:
                print(f"Time {round(self.manuf_line.env.now)} :", f'{self.ID} put {product} in {destination_name}')

            # Specififc tracking
            now = self.env.now
            for ref in self.manuf_line.cs_track:
                self.manuf_line.cs_track[ref].append([now, self.manuf_line.cs_track[ref][-1][1]])
                if ref == product['name']:
                    self.manuf_line.cs_track[ref][-1][1] -= 1

        yield self.env.timeout(1)
        self.busy = False 

    def which_machine_to_feed(self, current_machine, ref_to_pass=None):
        """
        Select the best machine to feed based on a given strategy.
        
        TODO: Upgrade to python 3.10 to use match case 

        Parameters:
        - ref_to_pass : when using the function in a recursive way, needs to store what was the product reference at hand from the first origin machine.
                        None is the most common usecase, meaning there was no machine skipping before.

        Return:
        Machine
        """
        
        # Feed from the supermarket
        if not isinstance(current_machine, Machine):
            # Update exepected refill time
            if self.manuf_line.pdp is None:
                self.manuf_line.expected_refill_time = [abs(i-self.manuf_line.env.now) for i in self.manuf_line.expected_refill_time]
            
            first_machines = [m for m in self.manuf_line.list_machines if m.first]

            # Balanced-like strategy : try to equally feed machines that are in parallel 
            if self.manuf_line.robot_strategy == 0:
                # Select the machine that has been the less loaded so far
                loads_on_machines = [m.loaded if not m.broken else float('inf') for m in first_machines]
                next_machine = first_machines[loads_on_machines.index(min(loads_on_machines))]  

            # Greedy-like strategy : focus on the machine that has the most space in input
            elif self.manuf_line.robot_strategy == 1:
                # Select the machine that has the fewest items in its input buffer
                buffers_level = [len(m.buffer_in.items) if not m.broken else float('inf') for m in first_machines]
                next_machine = first_machines[buffers_level.index(min(buffers_level))]

            # Skip the next machine if it takes no time to process the next product in the supermarket buffer
            if self.manuf_line.supermarket_in.items != [] and float(self.manuf_line.references_config[self.manuf_line.supermarket_in.items[0]][self.manuf_line.list_machines.index(next_machine)+3]) == 0:
                return self.which_machine_to_feed(next_machine, ref_to_pass=self.manuf_line.supermarket_in.items[0])
                
            # # Skip the next machine if it takes no time to process the next product coming to the supermarket 
            # next_refilled_ref = list(self.manuf_line.references_config.keys())[np.argmin(self.manuf_line.expected_refill_time)]
            # if self.manuf_line.supermarket_in.items == [] and float(self.manuf_line.references_config[next_refilled_ref][self.manuf_line.list_machines.index(next_machine)+3])==0 :
            #     return self.which_machine_to_feed(next_machine, ref_to_pass=next_refilled_ref)
                
            return next_machine

        # Feed from a machine
        else:
            # Balanced-like strategy : try to equally feed machines that are in parallel 
            if self.manuf_line.robot_strategy == 0:
                # Select the machine that has been the less loaded so far
                loads_on_machines = [m.loaded if isinstance(m, Machine) and not m.broken and not m.operating else float('inf') for m in current_machine.next_machines]
                next_machine = current_machine.next_machines[loads_on_machines.index(min(loads_on_machines))]

            # Greedy-like strategy : focus on the machine that has the most space in input
            elif self.manuf_line.robot_strategy == 1:
                buffers_level = [len(m.buffer_in.items) if isinstance(m, Machine) else len(m.items) for m in current_machine.next_machines]
                next_machine = current_machine.next_machines[buffers_level.index(min(buffers_level))]        

            # Skip the next machine if it takes no time to process the product at hand (either a product that already skipped a previous machine, or the next product in the buffer)
            if isinstance(next_machine, Machine) :
                if ref_to_pass is not None and float(self.manuf_line.references_config[ref_to_pass][self.manuf_line.list_machines.index(next_machine)+3]) == 0:
                    return self.which_machine_to_feed(next_machine, ref_to_pass=ref_to_pass)
                
                elif ref_to_pass is None and current_machine.buffer_out.items != [] and float(self.manuf_line.references_config[current_machine.buffer_out.items[0]][self.manuf_line.list_machines.index(next_machine)+3]) == 0:
                    return self.which_machine_to_feed(next_machine, ref_to_pass=current_machine.buffer_out.items[0])
                
            return next_machine

    def robot_process_local(self, from_entity, to_entity, transport_time=10):
        """Wait to tansport from an entity to another."""
        yield from self.transport(from_entity, to_entity, transport_time)

    def robot_process(self):
        """
        Tell the robot what to transport in what order. 

        Return:
        None
        """
        while True:
            try:
                # entities_order = [input_entity, machine1, machine2, machine3, output_entity]
                for i in range(len(self.entities_order)):
                    from_entity = self.entities_order[i]
                    to_entity = self.which_machine_to_feed(from_entity)

                    # "Shall we unload this machine ?"
                    
                    # Handle shift reset
                    if self.manuf_line.reset_shift_bool:
                        self.manuf_line.reset_shift_bool = False
                        if self.manuf_line.dev_mode:
                            print("Reseting Robot Process.")
                        break
                    
                    # Supermarket -> Machine
                    if isinstance(from_entity, simpy.Store):
                        # Available product in supermarket + available space in destination machine
                        if len(from_entity.items) > 0 and len(to_entity.buffer_in.items) < to_entity.buffer_in.capacity:
                            yield from self.transport(from_entity, to_entity)
                        
                        # elif len(to_entity.buffer_in.items) < to_entity.buffer_in.capacity and len(from_entity.items) == 0:
                        #   yield from self.transport(from_entity, to_entity)

                    elif isinstance(from_entity, Machine):
                        # Machine -> Machine
                        if isinstance(to_entity, Machine):
                            # Available product in origin machine + available space in destination machine
                            if len(from_entity.buffer_out.items) > 0 and len(to_entity.buffer_in.items) < to_entity.buffer_in.capacity:
                                yield from self.transport(from_entity, to_entity)

                            # # Under production in orgin machine + vailable space in destination machine
                            # elif (not from_entity.broken and from_entity.operating) and len(to_entity.buffer_in.items) < to_entity.buffer_in.capacity:
                            #     yield from self.transport(from_entity, to_entity)

                            # Full buffer in origin machine + full buffer in destination machine + space in central storage
                            elif (len(to_entity.buffer_in.items) == to_entity.buffer_in.capacity) and (len(from_entity.buffer_out.items) >= from_entity.buffer_out.capacity) and (self.manuf_line.central_storage is not None) and from_entity.fill_central_storage and (self.manuf_line.central_storage.available_spot(ref_name=from_entity.buffer_out.items[0])):
                                yield from self.transport(from_entity, to_entity, to_central_storage=True)

                            # Broken origin machine + available space in destination machine + available product in the central storage
                            elif from_entity.broken and len(to_entity.buffer_in.items) < to_entity.buffer_in.capacity and self.manuf_line.central_storage is not None and self.manuf_line.central_storage.available_ref_by_route(from_entity, to_entity):
                                yield from self.transport(from_entity, to_entity, from_central_storage=True)

                        # Machine -> Shop stock
                        elif isinstance(to_entity, simpy.Store):
                            # Available product in origin machine + available space in shop stock:
                            if len(from_entity.buffer_out.items) > 0 and len(to_entity.items) < to_entity.capacity:
                                yield from self.transport(from_entity, to_entity)

                            # # Under production in orgin machine + available space in shop stock
                            # elif (not from_entity.broken and from_entity.operating) and len(to_entity.items) < to_entity.capacity:
                            #     yield from self.transport(from_entity, to_entity)
                        
                        else:
                            raise Exception(f"Type of {to_entity} is not handled by the code.")
                    else:
                        raise Exception(f"Type of {from_entity} is not handled by the code.")

            except simpy.Interrupt:
                if self.manuf_line.dev_mode:
                    print(f"Resetting process of robot {self.ID}")

            # Pause for other robots to process.
            finally:
                yield self.env.timeout(1)

    # Unused (#Commented in "run" of ManufLine)
    def robot_process_unique(self):
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
                    #     if len(self.entities_order[i].buffer_in.items) < self.entities_order[i].buffer_in.capacity:
                    #         yield from self.transport(self.entities_order[i].previous_machine, self.entities_order[i])
                    # from_entity = self.entities_order[i]
                    # to_entity = self.which_machine_to_feed(from_entity)
                    # yield from self.transport(from_entity, to_entity)

   # All methods above are unused yet
    def which_machine_to_getfrom(self, current_machine):

        """
        Return the best machine to get from based on a given strategy.
        """
        if self.manuf_line.robot_strategy == 0:
            if isinstance(current_machine, Machine):
                full_buffers_machines = [len(m.buffer_out.items) if isinstance(m, Machine) and not m.broken and m.operating else float('inf') for m in current_machine.previous_machines]
                previous_machine = current_machine.previous_machines[full_buffers_machines.index(min(full_buffers_machines))]
            else:
                return True
            

        elif self.manuf_line.robot_strategy == 1:
            if isinstance(current_machine, Machine):
                full_buffers_machines = [len(m.buffer_out.items) if isinstance(m, Machine) else float('inf') for m in current_machine.previous_machines]
                previous_machine = current_machine.previous_machines[full_buffers_machines.index(max(full_buffers_machines))]
            else:
                return True
        try:
            return previous_machine 
        except:
            # No machine
            return True

    def which_machine_to_getfrom_alone(self, current_machine):

        """
        Return the best machine to get from based on a given strategy.
        """
        if self.manuf_line.robot_strategy == 0:
            if current_machine.first:
                previous_machine = current_machine.previous_machines[0]
            else:
                full_buffers_machines = [len(m.buffer_out.items) if isinstance(m, Machine) and not m.broken and m.operating else float('inf') for m in current_machine.previous_machines]
                previous_machine = current_machine.previous_machines[full_buffers_machines.index(min(full_buffers_machines))]

            

        elif self.manuf_line.robot_strategy == 1:
            if isinstance(current_machine, Machine):
                full_buffers_machines = [len(m.buffer_out.items) if isinstance(m, Machine) else float('inf') for m in current_machine.previous_machines]
                previous_machine = current_machine.previous_machines[full_buffers_machines.index(max(full_buffers_machines))]
            else:
                return True
        try:
            return previous_machine 
        except:
            # No machine
            return True
    
    def handle_empty_buffer(self, from_entity, to_entity):

        from_entity.previous_machine = self.which_machine_to_getfrom(from_entity)
        if from_entity.previous_machine == True:
            yield from self.transport(from_entity, to_entity) 
        else:
            if isinstance(from_entity.previous_machine, Machine):
                if not from_entity.previous_machine.operating and len(from_entity.previous_machine.buffer_out.items) == 0:
                    yield from self.handle_empty_buffer(from_entity.previous_machine, from_entity)
                elif from_entity.previous_machine.operating or len(from_entity.previous_machine.buffer_out.items)>0:
                    yield from self.transport(from_entity.previous_machine, from_entity) 
            else:
                if len(from_entity.previous_machine.items) == 0:
                    yield from self.handle_empty_buffer(from_entity.previous_machine, from_entity)
                elif len(from_entity.previous_machine.items)>0:
                    yield from self.transport(from_entity.previous_machine, from_entity) 

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
                        if len(m.buffer_in.items) < m.buffer_in.capacity:
                            yield from self.transport(m.previous_machine, m)
                        else:
                            if len(m.next_machine.buffer_in.items) < m.next_machine.buffer_in.capacity:
                                yield from self.transport(m, m.next_machine)
                            else: 
                                pass
                for i in range(len(self.entities_order)):
                    from_entity = self.entities_order[i]
                    to_entity = self.which_machine_to_feed(from_entity)
                    

                    if self.manuf_line.reset_shift_bool:
                        self.manuf_line.reset_shift_bool = False
                        if self.manuf_line.dev_mode:
                            print("Reseting Robot Process.")
                        break

                    try:
                        if len(to_entity.buffer_in.items) < to_entity.buffer_in.capacity and len(from_entity.buffer_out.items) > 0:
                            yield from self.transport(from_entity, to_entity)
                        elif len(to_entity.buffer_in.items) < to_entity.buffer_in.capacity and len(from_entity.buffer_out.items) == 0 and (from_entity.operating and not from_entity.broken):
                            yield from self.transport(from_entity, to_entity)
                        elif len(to_entity.buffer_in.items) < to_entity.buffer_in.capacity and len(from_entity.buffer_out.items) == 0 and (not from_entity.operating or  from_entity.broken):
                            yield from self.handle_empty_buffer(from_entity, to_entity)
                        elif len(to_entity.buffer_in.items) == to_entity.buffer_in.capacity:
                            yield from self.transport(to_entity, self.which_machine_to_feed(to_entity))
                        else:
                            continue
                
                    except Exception as e:
                        try:
                            if len(to_entity.items) < to_entity.capacity and len(from_entity.buffer_out.items) > 0:
                                yield from self.transport(from_entity, to_entity)
                            elif len(to_entity.items) < to_entity.capacity and len(from_entity.buffer_out.items) == 0 and from_entity.operating:
                                yield from self.transport(from_entity, to_entity)
                                #continue
                            elif len(to_entity.items) < to_entity.capacity and len(from_entity.buffer_out.items) == 0 and (not from_entity.operating or  from_entity.broken):
                                yield from self.handle_empty_buffer(from_entity, to_entity)
                            elif  from_entity.broken or  from_entity.broken:
                                continue
                            else:
                                continue
                        except Exception as e2:
                            if len(to_entity.buffer_in.items) < to_entity.buffer_in.capacity and len(from_entity.buffer_out.items) > 0:
                                yield from self.transport(from_entity, to_entity)
                            elif len(to_entity.buffer_in.items) < to_entity.buffer_in.capacity and len(from_entity.buffer_out.items) == 0 and from_entity.operating:
                                yield from self.transport(from_entity, to_entity)
                            elif len(to_entity.buffer_in.items) < to_entity.buffer_in.capacity and len(from_entity.buffer_out.items) == 0 and not from_entity.operating:
                                yield from self.handle_empty_buffer(from_entity, to_entity)
                            else:
                                continue
            except simpy.Interrupt:
                if self.manuf_line.dev_mode:
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
            if len(next_entity.buffer_out.items) > 0:
                return next_entity
        return None
   
    def schedule_transport(self, from_machine, to_machine, transport_time):
        """Schedule a transport task if the robot is available."""

        transport_task = self.env.process(self.transport(from_machine, to_machine, transport_time))
        self.schedule.append(transport_task)

    def load_machine_new(self, current_machine, first=True):
        """
        Find the best machine to get from and load the machine or to transport the component to

        if the current machine output buffer is full, the robot will transport the component to the next machine
        if the current machine output buffer is not full, the robot will load the machine
        if the current machine input buffer is empty, the robot will transport the component to the machine
        """
        if current_machine.first:
            previous_machine = current_machine.previous_machines[0]
            if len(previous_machine.items) > 0:
                yield from self.transport(previous_machine, current_machine)
            elif len(previous_machine.items)  == 0 and  not current_machine.operating:
                yield from self.transport(previous_machine, current_machine)
            elif len(previous_machine.items) == 0 and current_machine.operating:
                yield self.env.timeout(0)
        else:
            for i in range(len(current_machine.previous_machines)):
                previous_machine = current_machine.previous_machines[i]
                if len(previous_machine.buffer_out.items) > 0:
                    yield from self.transport(previous_machine, current_machine)
                    break
                elif len(previous_machine.buffer_out.items) == 0 and  previous_machine.operating:
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
                elif len(to_entity.buffer_in.items) < to_entity.buffer_in.capacity and len(from_entity.buffer_out.items) > 0:
                    yield from self.transport(from_entity, to_entity)
                elif len(to_entity.buffer_in.items) < to_entity.buffer_in.capacity and len(from_entity.buffer_out.items) == 0 and from_entity.operating:
                    yield from self.transport(from_entity, to_entity)
                else:
                    yield self.env.timeout(0)
            except Exception as e:
                if self.manuf_line.dev_mode:
                    print("Exception 1 :", e)
                if from_entity == True:
                    yield self.env.timeout(0)
                elif len(to_entity.buffer_in.items) < to_entity.buffer_in.capacity and len(from_entity.items) > 0:
                    yield from self.transport(from_entity, to_entity)
                elif len(to_entity.buffer_in.items) < to_entity.buffer_in.capacity and len(from_entity.items) == 0:
                    yield from self.transport(from_entity, to_entity)
                else:
                    yield self.env.timeout(0)
        except Exception as e:
            # If the machine breaks while loading by robot
            if self.manuf_line.dev_mode:
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
        elif len(to_entity.buffer_in.items) < to_entity.buffer_in.capacity and len(from_entity.buffer_out.items) > 0:
            yield from self.transport(from_entity, to_entity)
        elif len(to_entity.buffer_in.items) < to_entity.buffer_in.capacity and len(from_entity.buffer_out.items) == 0 and from_entity.operating:
            yield from self.transport(from_entity, to_entity)
        else:
            yield self.env.timeout(0)


class Task:
    def __init__(self, ID, machine_time, manual_time):
        self.ID = ID
        self.machine_time = machine_time
        self.manual_time = manual_time


class Operator:
    def __init__(self, id):
        self.id = id
        self.assigned_machines = []
        self.wc = 0
        self.free = True
        self.busy = False

    def assign_machine(self, machine):
        self.assigned_machines.append(machine)
    
    def release_machine(self, machine):
        self.assigned_machines.remove(machine)


class CentralStorage:
    def __init__(self, env, manuf_line, central_storage_config, times_to_reach={}, strategy='stack') -> None:
        """Hold two storages : back and front, each having several blocks allowing one or many references.

        Parameters:
        - central_storage_config' : below is an example of its structure.
            {'front': [ {'allowed_ref' : ['Ref A', 'Ref B'],
                         'capacity': 336},
                        {'allowed_ref' : ['Ref A', 'Ref B', 'Ref C'],
                            'capacity': 24}]
                        
             'back': [ {'allowed_ref' : ['Ref A', 'Ref B'],
                        'capacity': 376}]}
        
        - strategy : decide which storage to fill or take from when a new reference is added or removed.
            - 'stack' : Try to fill blocks of the 'front' storage before the 'back'.

        TODO: Implement "fastest" route with times_to_reach. Waiting for more details from Vivi.

        'self.stores' has quite the same structure as 'central_storage_config' :
            - {'front': [ {'allowed_ref' : ['Ref A', 'Ref B'],
                           'store': simpy.FilterStore()},
                        ...

            - where each simpy.FilterStore holds dictionnaries with 3 keys:
                - name
                - route  
                - status           

        """
        self.ID = 'CENTRAL STORAGE'
        self.env = env
        self.manuf_line = manuf_line
        self.strategy = strategy
        self.times_to_reach = list(times_to_reach.values())
        self.stores = deepcopy(central_storage_config)

        self.available_stored_by_ref = {ref: 0 for ref in self.all_allowed_references}
        self.available_spots_by_ref = {ref: 0 for ref in self.all_allowed_references}
        self.available_routes = []

        # Build the stores and initialise counters for available spots
        for side in self.stores.keys():
            for block in self.stores[side]:
                # Stores
                block['store'] = simpy.FilterStore(env, block['capacity'])
                del block['capacity']
                # Available spots
                for ref_name in block['allowed_ref']:
                    self.available_spots_by_ref[ref_name] += block['store'].capacity

    def __str__(self) -> str:
        """Display what is inside the front and back stores.
        
        For each store and block within, display :
        - allowed references
        - capacity
        - level
        - items        
        """

        lines = ["\n=== CENTRAL STORAGE ==="]

        for side in self.stores.keys():
            lines.append(side.upper())
            for block in self.stores[side]:
                lines.append(f"references allowed: {block['allowed_ref']}")
                lines.append(f"capacity : {block['store'].capacity}")
                lines.append(f"level : {len(block['store'].items)}")
                lines.append(f"items : {[(item['name'], (item['route'][0], item['route'][1]), item['status']) for item in block['store'].items]}")
                lines.append('')

        return "\n".join(lines)

    @property
    def all_allowed_references(self) -> list:
        """Build a list of the names of all the references allowed in the central storage no matter where."""

        all_references = []
        for blocks in self.stores.values():
            for block in blocks:
                all_references += block["allowed_ref"]

        # Return a list with uniqueness
        return list(set(all_references))

    def available_spot(self, ref_name=None) -> bool:
        """Check if there is an available spot.
        
        If a reference is specified, check an available spot for this specific reference.
        Otherwise, check if there is any available spot no matter the reference.

        """
        try :
            if ref_name is None:
                is_full = all([n_available == 0 for n_available in self.available_spots_by_ref.values()])
            
            else:
                is_full = (self.available_spots_by_ref[ref_name] == 0)

        # ref_name that does not exist in the entire central storage
        except KeyError:
            is_full = True

        finally:
            if self.manuf_line.dev_mode :
                if is_full:
                    print(f'There is NO available spot for reference "{ref_name}" in the central storage.')
                else:
                    print(f'There is an available spot for reference "{ref_name}" in the central storage.')

            return not is_full

    def available_ref(self, ref_name=None) -> bool:
        """Check if there is an available reference.
        
        If a reference is specified, check this specific reference.
        Otherwise, check if there is at least 1 reference no matter which one.

        """
        try :
            if ref_name is None:
                is_ref_available = any([count > 0 for count in self.available_stored_by_ref.values()])
            
            else:
                is_ref_available = self.available_stored_by_ref[ref_name] > 0

        # ref_name that does not exist in the entire central storage
        except KeyError:
            is_ref_available = True

        finally:
            # Print
            if self.manuf_line.dev_mode :
                if is_ref_available:
                    print(f'There is an available reference "{ref_name}" in the central storage.')
                else:
                    print(f'There is NO available spot for reference "{ref_name}" in the central storage.')

            return is_ref_available
        
    def available_ref_by_route(self, origin, destination) -> bool:
        """Check if there is an available reference with the same route (origin, destination)."""
        is_available = ((origin, destination) in self.available_routes)
        
        if self.manuf_line.dev_mode :
            if is_available:
                print(f'There is an available reference for the specified route "{(origin, destination)}" in the central storage.')
            else:
                print(f'There is NO available reference for the specified route "{(origin, destination)}" in the central storage.')

        return is_available

    def put(self, ref_data):
        """Try to put a reference in the storage determined by the strategy of the central storage.
        
        'ref_data' is a dictionnary that holds :
            - reference 'name' ('Ref A')
            - reference 'route' (tuple with origin and destination entities)
            - reference 'status' (OK / KO / test..)

        """
        
        # Check if the strategy has been implemented.
        if self.strategy not in ['stack']:
            raise Exception(f'Strategy "{self.strategy}" is not yet implemented for the central storage.')

        if self.strategy == 'stack':
            for side in self.stores.keys():
                for block in self.stores[side]:
                    # Check if the ref is allowed in the current block.
                    if ref_data['name'] in block['allowed_ref']:
                        store = block['store']
                        # Check if there is an available spot to put the reference.
                        if len(store.items) < store.capacity:
                            # Put the reference
                            if self.manuf_line.dev_mode:
                                print(f"""Put the reference "{ref_data['name']}" with status "{ref_data['status']}" from entity "{ref_data['route'][0]}" and to entity "{ref_data['route'][1]}" in the central storage.""")
                            store.put(ref_data)

                            # Add the reference route to the stored routes
                            self.available_routes.append(ref_data['route'])
                            # Decrease the number of available spot
                            for ref_name in block['allowed_ref']:
                                self.available_spots_by_ref[ref_name] -= 1
                            # Increase the counter of stored references
                            self.available_stored_by_ref[ref_name] += 1
                            return
                        
        # Haven't found any place to put the reference
        raise Exception(f"Tried to put the reference {ref_data} in the central storage BUT it's FULL.")
    
    def get(self, ref_name=None):
        """Try to get a reference in the storage determined by the strategy of the central storage.
        
        If a reference is specified, get this specific reference.
        Otherwise, get any reference depending on the strategy of the storage.

        """

        # Check if the strategy has been implemented.
        if self.strategy not in ['stack']:
            raise Exception(f'Strategy "{self.strategy}" is not yet implemented for the central storage.')

        if self.strategy == 'stack':
            for side in self.stores.keys():
                for block in self.stores[side]:
                    # When the reference is not specified (None), get as soon as the block is not empty.
                    if ref_name is None and len(block['store'].items) > 0 :
                        # Get the reference
                        ref_got = block['store'].get().value
                        if self.manuf_line.dev_mode:
                            print(f"""Got the first reference "{ref_got['name']}" in the central storage.""")

                        # Remove the reference route to the stored routes
                        self.available_routes.remove(ref_got['route'])
                        # Increase the number of available spots
                        for allowed_ref_name in block['allowed_ref']:
                            self.available_spots_by_ref[allowed_ref_name] += 1
                        # Decrease the counter of stored references
                            self.available_stored_by_ref[ref_got['name']] -= 1

                        return ref_got
                        
                    # When the reference is specified, get as soon as one is present in a block.
                    if ref_name is not None:
                        for ref_data in block['store'].items:
                            if ref_data['name'] == ref_name:
                                # Get the reference
                                ref_got = block['store'].get(lambda x: x['name']==ref_name).value
                                if self.manuf_line.dev_mode:
                                    print(f"""Got the specified reference "{ref_got['name']}" in the central storage.""")

                                # Remove the reference route to the stored routes
                                self.available_routes.remove(ref_got['route'])
                                # Increase the number of available spots
                                for allowed_ref_name in block['allowed_ref']:
                                    self.available_spots_by_ref[allowed_ref_name] += 1
                                # Decrease the counter of stored references
                                self.available_stored_by_ref[ref_got['name']] -= 1

                                return ref_got
                            
        # Haven't found any reference to get
        raise Exception(f"Couldn't get the reference '{ref_name}' in the central storage.")
    
    def get_by_destination(self, destination):
        """Try to get a reference in the storage based on its route destination."""

        # Check if the strategy has been implemented.
        if self.strategy not in ['stack']:
            raise Exception(f'Strategy "{self.strategy}" is not yet implemented for the central storage.')

        if self.strategy == 'stack':
            for side in self.stores.keys():
                for block in self.stores[side]:
                    for ref_data in block['store'].items:
                        if ref_data['route'][1] == destination:
                            # Get the reference
                            ref_got = block['store'].get(lambda x: x['route'][1]==destination).value
                            if self.manuf_line.dev_mode:
                                print(f"""Got the reference "{ref_got['name']}" with specified destination "{destination}" in the central storage.""")

                            # Remove the reference route to the stored routes
                            self.available_routes.remove(ref_got['route'])
                            # Increase the number of available spots
                            for allowed_ref_name in block['allowed_ref']:
                                self.available_spots_by_ref[allowed_ref_name] += 1
                            # Decrease the counter of stored references
                            self.available_stored_by_ref[ref_got['name']] -= 1

                            return ref_got
                            
        # Haven't found any reference to get
        raise Exception(f"Couldn't get a reference with specified destination '{destination}' in the central storage.")
        

def format_time(seconds, seconds_str=False):

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
    if non_zero_parts < 3 and seconds_str:
        time_str += f"{seconds:.2f} seconds"

    return time_str.rstrip(", ")

def draw_buffers(app, assembly_line):
    machine_util = []
    machine_names = []

    # Draw supermarket of the whole line
    if len(assembly_line.supermarket_in) < assembly_line.supermarket_in.capacity * 0.3:
        app.supermarket_btn.configure(fg_color="red")
    elif len(assembly_line.supermarket_in) < assembly_line.supermarket_in.capacity * 0.5:
        app.supermarket_btn.configure(fg_color="orange")
    else:
        app.supermarket_btn.configure(fg_color="green")
    app.supermarket_capacity.configure(text=f"Capacity = {assembly_line.supermarket_in.capacity}")
    app.supermarket_level.configure(text=f"Level = {len(assembly_line.supermarket_in)}")
    app.refill_label.configure(text="N. of Refills = %d" % int(assembly_line.supermarket_n_refills))

    only_one = False
    
    for i, m in enumerate(assembly_line.list_machines):
        ### Raw supermarket (raw stock)
        machine_names.append(m.ID)
        
        if m.first:
            if not only_one:
                only_one = True
                if len(m.buffer_out.items) < m.buffer_out.capacity * 0.2:
                    m.buffer_btn[0].configure(fg_color="green")
                elif len(m.buffer_out.items) < m.buffer_out.capacity * 0.8:
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
                m.buffer_btn[2].configure(text=f"Level = {len(m.buffer_out.items)}")
                m.buffer_btn[3].configure( text="Waiting/Idle Time = %.2f" % avg_idle_time)
                m.buffer_btn[4].configure(text="Total Downtime = %.2f" % float(m.MTTR*float(m.n_breakdowns)))
            except:
                pass
        else:
    
            if len(m.buffer_out.items) < m.buffer_out.capacity * 0.2:
                m.buffer_btn[0].configure(fg_color="green")
            elif len(m.buffer_out.items) < m.buffer_out.capacity * 0.8:
                m.buffer_btn[0].configure(fg_color="orange")
            else:
                m.buffer_btn[0].configure(fg_color="red")

            idle_times_machine = []
            for entry, exi in zip(m.entry_times, m.exit_times):
                idle_times_machine.append(exi-entry)
            avg_idle_time = np.mean(idle_times_machine)
            #idle_time = idle_times_machine[-1]
            m.buffer_btn[1].configure(text=f"Capacity = {m.buffer_out.capacity}")
            m.buffer_btn[2].configure(text=f"Level = {len(m.buffer_out.items)}")
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
        if len(m.buffer_out.items) < m.buffer_out.capacity * 0.2:
            color = "green"
        elif len(m.buffer_out.items) < m.buffer_out.capacity * 0.8:
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
        canvas.create_text(text_x, text_y, anchor="w", text=f"Level: {len(m.buffer_out.items)}")
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

