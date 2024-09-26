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
                if  tracksim:
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
            yield self.env.timeout(self.sim_time/100)
            self.output_tracks.append((self.env.now, self.shop_stock_out.level))
            
            for i, ref in enumerate(self.references_config.keys()):
                self.output_tracks_per_ref[i].append((self.env.now, self.inventory_out.items.count(ref)))

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
        self.buffer_tracks.append([(m.buffer_in.level, m.buffer_out.level) for m in self.list_machines])
        self.robot_states.append(robot_state)
        self.machines_states.append([m.operating for m in self.list_machines])
        self.machines_idle_times.append([m.waiting_time for m in self.list_machines])
        self.machines_CT.append([self.env.now/(m.parts_done+1) for m in self.list_machines])
        self.machines_breakdown.append([m.n_breakdowns for m in self.list_machines])
        self.sim_times_track.append(self.env.now)

    def run(self):
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
        self.expected_refill_time = [0] * len(self.references_config.keys())
        for ref in list(self.references_config.keys()):
            print("Reference confirmed = ", ref)
            self.env.process(self.refill_market(ref))

        # Order machines related to each robot and process all robots
        print(str(len(self.robots_list)) + "  -- Robot Included")
        print("First machine = " , [m.first for m in self.list_machines])
        for i in range(len(self.robots_list)):
            machines_ordered = [self.list_machines[j-1] for j in self.robots_list[i].order]
            self.robots_list[i].entities_order = machines_ordered
            print("Inclued in robot = ", [self.list_machines[j-1].ID for j in self.robots_list[i].order])
            print("Inclued in robot = ", [self.list_machines[j-1].first for j in self.robots_list[i].order])

            # Insert the supermarket as the first entity of the robot if any related machine is the first 
            if any([self.list_machines[j-1].first for j in self.robots_list[i].order]):
                self.robots_list[i].entities_order.insert(0, self.supermarket_in)

            self.robots_list[i].process = self.env.process(self.robots_list[i].robot_process())
            # self.robots_list[i].process = self.env.process(self.robots_list[i].robot_process_unique())

        #TODO: fix problem of reset shift with multi references
        print("Resetting shift = ", self.reset_shift_dec)
        if self.reset_shift_dec:
            self.env.process(self.reset_shift())

        # Track outputs 
        self.env.process(self.track_output())
        for m in self.list_machines:
            print(m.ID, m.buffer_in.capacity)
            print(m.ID, m.buffer_out.capacity)

        # Run the environment
        print("Starting the sim now.")
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

        # Reset robots and machines
        self.robots_list = []
        self.create_machines(self.machine_config_data)
        for robot in self.robots_list: # Useless since robot.env already updated in create_machines ? 
            robot.env = self.env

        self.reset_shift()

    # Unused (Commented in "run action" button of the app)
    def initialize(self):
        self.generate()
        for i, m in enumerate(self.list_machines):
            m.process = self.env.process(m.machine_process())
            self.env.process(self.break_down(m))

        self.expected_refill_time = [0 for _ in range(len(self.references_config.keys()))]
        for ref in list(self.references_config.keys()):
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
            print("Resetting Shift")
            
            self.num_cycles = self.num_cycles + 1
            self.reset_shift_bool = True

            # Empty machine storage (buffers and stores) 
            for machine in self.list_machines:
                
                # No robots in the manufacturing line
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

                # At least one robot, empty stores 
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

    # Outdated (Only used in "main" python files)
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
        "7 : Robot Transport time in",
        "8 : Transport order",
        "9 : Robot asignement", 
        "10 : Same station as"] 
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

        # Don't create robots ANY robot on the whole line if only one cell of "Robot Transport Time IN" or "Robot Assignment" is NaN (empty)
        all_robots_time_defined = all([not np.isnan(list_machines_config[i][7]) for i in range(len(list_machines_config))])
        all_robots_assigned = all([not np.isnan(list_machines_config[i][9]) for i in range(len(list_machines_config))]) # Conditions required ? 

        self.robots_list = []
        if all_robots_time_defined and all_robots_assigned:
            for i in range(int(max([list_machines_config[j][9] for j in  range(len(list_machines_config))]))):
                # Order machines assigned to the robot and their related transport time
                self.robot = Robot(self, self.env)
                # print("Order inside = ", [list_machines_config[j][11]  for j in range(len(list_machines_config)) if list_machines_config[j][11] == int(i+1)])
                self.robot.order = [list_machines_config[j][8] for j in range(len(list_machines_config)) if (list_machines_config[j][9] == int(i+1))]
                self.robot.in_transport_times = [list_machines_config[j][7] for j in range(len(list_machines_config)) if (list_machines_config[j][9] == int(i+1))]
                self.robots_list.append(self.robot)
        
        # Store order and robot trnasport times
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
            buffer_capacity = machine_config[3]
            initial_buffer = machine_config[4]

            # TODO : set the operating time of each machine here given the different references in input 

            # Create machine with above parameters whether there were tasks already assigned
            try:
                assigned_tasks =  list(np.array(self.tasks)[machine_indices[i]])
                machine = Machine(self, self.env, machine_config[0], machine_config[1], self.config, assigned_tasks=assigned_tasks, first=is_first, last=is_last, breakdowns=self.breakdowns['enabled'], mttf=mttf, mttr=mttr, buffer_capacity=buffer_capacity, initial_buffer=initial_buffer ,hazard_delays=self.config['hazard_delays']['enabled'])
            except:
                machine = Machine(self, self.env, machine_config[0], machine_config[1], self.config, first=is_first, last=is_last, breakdowns=self.breakdowns['enabled'], mttf=mttf, mttr=mttr, buffer_capacity=buffer_capacity , initial_buffer=initial_buffer, hazard_delays=self.config['hazard_delays']['enabled'])

            # Store the created machine
            if machine.first:
                self.first_machine = machine
            self.list_machines.append(machine)
        
        # For each machine, retrieve which other machines comes right before, right after and manage their storage
        index_next = []
        machines_ids = [m.ID for m in self.list_machines]
        for i, machine in enumerate(self.list_machines):
            # TODO : upcoming feature for same machines ?
            if not str(list_machines_config[i][10]) =="nan":
                indexmachine = [m.ID for m in self.list_machines].index(str(list_machines_config[i][7]))
                machine.same_machine = self.list_machines[indexmachine]

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
                        if len(self.robots_list) == 0:
                            self.list_machines[i_linked].buffer_in = machine.buffer_out
                            self.list_machines[i_linked].store_in = machine.store_out
                    
                    # Machine already encoutered before
                    else:

                        # Add previous and following machines
                        machine.next_machine = self.list_machines[i_linked]
                        machine.next_machines.append(self.list_machines[i_linked])

                        self.list_machines[i_linked].previous_machine = machine
                        self.list_machines[i_linked].previous_machines.append(machine)

                        # If no robot, equal storage
                        if len(self.robots_list) == 0:
                            machine.buffer_out = machine.next_machine.buffer_in
                            machine.store_out = machine.next_machine.store_in

            except:
                pass

    # Methods below are empty
    def deplete_shopstock(self):
        pass

    def deliver_to_client(self):
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

        # Define storage : buffer and store
        self.buffer_in = simpy.Container(env, capacity=float(buffer_capacity), init=self.initial_buffer)
        self.buffer_out = simpy.Container(env, capacity=float(buffer_capacity), init=self.initial_buffer)
        self.store_in = simpy.Store(env)
        self.store_out = simpy.Store(env)
        
        # When NO robot, directly connect first machine to supermarket and last machine to shop stock 
        if self.manuf_line.robots_list == []:
            if first:
                self.buffer_in = manuf_line.supermarket_in
                self.store_in = manuf_line.inventory_in
            
            if last:
                self.buffer_out = manuf_line.shop_stock_out
                self.store_out = manuf_line.inventory_out          

        
        self.MTTF = mttf # Mean time to failure in seconds
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

        else:
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
        bias_shape = 2  # shape parameter
        bias_scale = 1  # scale parameter
        
        while True: 
            # FHS special case when a machine may have two processes, check of the other process is operating
            other_process_operating = False if self.same_machine is None else self.same_machine.operating
            
            # Do not process if there is already a process ongoing
            if self.operating or other_process_operating:
                return

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
            
            self.buffer_tracks.append((self.env.now, self.buffer_out.level))
            entry0 = self.env.now
            self.entry_times.append(entry0)
            self.loaded_bol = False
            self.done_bool = False
            entry_wait = self.env.now
            self.to_be_passed = False

            # First, needs to have a product loaded
            while not self.loaded_bol :
                try: 
                    product = None

                    if len(self.store_in.items) !=0:
                        # The product should not be processed in this machine and to be passed to the next
                        if float(self.manuf_line.references_config[self.store_in.items[0]][self.manuf_line.list_machines.index(self)+1]) ==0:
                            self.to_be_passed = True
                            yield self.buffer_in.get(1)
                            product = yield self.store_in.get()
                            self.current_product = product
                            break

                    # Take product from input buffer / store
                    yield self.buffer_in.get(1)
                    print("before in " + self.ID + "= " +str(self.store_in.items) + " - PROD " )
                    product = yield self.store_in.get()
                    self.current_product = product
                    print("Product " + product + " passed in " + self.ID + " at " + str(self.env.now))
                    print("after in " + self.ID + "= " +str(self.store_in.items) + " - PROD " + product)

                    # Already handled above ? 
                    done_in = float(self.manuf_line.references_config[product][self.manuf_line.list_machines.index(self)+1])
                    if done_in == 0:
                        ## The product should not be processed in this machine and to be passed to the next
                        self.to_be_passed = True
                        break

                    self.waiting_time = [self.waiting_time[0] + self.env.now - entry_wait , self.waiting_time[1]]  
                    # done_in = deterministic_time 
                    # start = self.env.now
                    self.loaded_bol = True

                # Handle breakdown 
                except simpy.Interrupt:                        
                    self.broken = True
                    print(self.ID +" broken at loading at  = "+ str(self.env.now))

                    # Call a repairman
                    repair_in = self.env.now
                    with self.manuf_line.repairmen.request(priority=1) as req:
                        yield req
                        yield self.env.timeout(self.MTTR) # Time to repair
                        repair_end = self.env.now
                        self.real_repair_time.append(float(repair_end - repair_in))
                    print(self.ID +" repaired at loading at  = "+ str(self.env.now))
                    
                    if self.buffer_in.level == 0:
                        self.buffer_in.put(1)
                        self.loaded_bol = True # Set to true but overidden by next two conditions

                    if product is not None:
                        self.store_in.put(product)
                        self.loaded_bol = True
                        done_in = float(self.manuf_line.references_config[product][self.manuf_line.list_machines.index(self)+1])
                        
                    else:
                        self.loaded_bol = False

                    self.broken = False
                    self.operating = False
            
            
            #TODO: Skip robot when part not processed in the machine

            # Then, make the machine operate
            start = self.env.now
            while done_in > 0 and not self.to_be_passed:
                try:
                    entry = self.env.now
                    self.current_product = product
                    exit_t = self.env.now # Roughly 0 no ? 
                    self.exit_times.append(exit_t-entry)
                    self.operating = True
                    yield self.env.timeout(done_in)
                    entry_wait = self.env.now
                    yield self.buffer_out.put(1) and self.store_out.put(product)
                    self.waiting_time = [self.waiting_time[0] , self.waiting_time[1] + self.env.now-entry_wait]
                    done_in = 0
                    self.loaded_bol = False
                    self.done_bool = True
                    yield self.env.timeout(0)

                # Handle breakdown
                except simpy.Interrupt:                        
                    self.broken = True
                    done_in -= self.env.now - start

                    repair_in = self.env.now
                    with self.manuf_line.repairmen.request(priority=1) as req:
                        yield req
                        repair_end = self.env.now
                        yield self.env.timeout(self.MTTR) # Time to repair
                        self.real_repair_time.append(float(repair_end - repair_in + float(self.MTTR)))
                    print(self.ID +" repaired at operating  buffer in level at " + str(self.env.now))

                    self.broken = False
                    self.operating = False

            # When work is done on the current product
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

            # Product not done yet OR must be passed to the next machine
            else:
                start = self.env.now
                if done_in == 0:
                    try:
                        print("Passed zero no process = " + product + " In " + self.ID + " at " + str(self.env.now))
                        entry_wait = self.env.now
                        yield self.buffer_out.put(1) and self.store_out.put(product)
                        self.waiting_time = [self.waiting_time[0] , self.waiting_time[1] + self.env.now-entry_wait]
                        # yield self.store_out.put(product)
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


class Robot:
    """
    Transport Robot between machines
    """
    def __init__(self, manuf_line, env, maxlimit=1):
        # self.assigned_machines = assigned_machines
        self.busy = False
        self.manuf_line = manuf_line
        self.env = env
        self.schedule = [] # Unused yet
        self.buffer = simpy.Container(self.env, capacity=float(maxlimit), init=0)
        self.robots_res = simpy.PriorityResource(self.env, capacity=1) # Unused yet
        self.waiting_time = 0
        self.in_transport_times = []
        self.process = None
        self.entities_order = None
        self.loadunload_time = 50
            
    def transport(self, from_entity, to_entity, time=10):
        """
        Handle transport between two entities, update storages. Cancel the transport when breakdowns appear.

        TODO: Fix the problem of the robot not being able to transport from a machine to another after shift reset
              Do we need shift reset of buffers ?

        Returns:
        None 
        """
    
        # Skip transport if the robot is already busy
        if self.busy:
            print("Busy")
            yield self.env.timeout(0)
            return

        # Transport between two machines
        if isinstance(from_entity, Machine) and isinstance(to_entity, Machine):
            print("Transporting from " + from_entity.ID + " to " + to_entity.ID +" at time = " + str(self.env.now) )
            
            # Skip transport if any machine is broken 
            if to_entity.broken or from_entity.broken:
                print("A machine is broken, skipping remaining instructions")
                yield self.env.timeout(10) 
                return
        
            # Start by waiting for an available input resource from entity
            entry = self.env.now
            self.busy = True
            while from_entity.buffer_out.level == 0:
                yield self.env.timeout(10)

                # Skip transport if any machine is broken 
                if from_entity.broken or to_entity.broken:
                    print("A machine is broken, skipping remaining instructions")
                    self.busy = False
                    yield self.env.timeout(10)
                    return
            yield from_entity.buffer_out.get(1)
            product = yield from_entity.store_out.get()

            # Move robot and unload / load
            to_entity.loaded +=1
            self.waiting_time += self.env.now-entry
            yield self.env.timeout(abs(to_entity.move_robot_time - from_entity.move_robot_time)+self.loadunload_time)
            
            # Wait for a spot in the input buffer of 'to_entity' 
            entry_2 = self.env.now
            while to_entity.buffer_in.level >= to_entity.buffer_capacity:
                yield self.env.timeout(10)

                # Skip transport if the machine to deliver is broken 
                if to_entity.broken:
                    print("To entity is broken, skipping remaining instructions")
                    self.busy = False
                    yield self.env.timeout(10)
                    return 

            # Update buffer and store now that the transport is complete
            yield to_entity.buffer_in.put(1)
            to_entity.store_in.put(product)
            self.waiting_time += self.env.now-entry_2
            self.manuf_line.track_sim((from_entity.Name, to_entity.Name, self.env.now))
            self.busy = False
            yield self.env.timeout(0)

        # Transport from something to a machine
        if not isinstance(from_entity, Machine) and isinstance(to_entity, Machine):
            print("Transporting from " + str(from_entity) + " to " + to_entity.ID + " at time = " + str(self.env.now))

            # SKip transport if the machine to deliver is broken 
            if to_entity.broken:
                print("To entity is broken, skipping remaining instructions")
                yield self.env.timeout(10) 
                return

            # Start by waiting for an available input resource from entity
            self.busy = True
            entry = self.env.now
            print("Start to wait at - ", entry)
            yield from_entity.get(1)
            product = yield self.manuf_line.inventory_in.get() 

            # Move robot and unload / load
            print("Ready  at - ", self.env.now)
            to_entity.loaded +=1
            self.waiting_time += self.env.now-entry
            yield self.env.timeout(abs(to_entity.move_robot_time)+self.loadunload_time)
            
            # Wait for a spot in the input buffer of 'to_entity' 
            entry_2 = self.env.now
            while to_entity.buffer_in.level >= to_entity.buffer_capacity:
                yield self.env.timeout(10)

                # SKip transport if the machine to deliver breaks down while waiting
                if to_entity.broken:
                    print("From entity is broken, skipping remaining instructions")
                    self.busy = False
                    yield self.env.timeout(0) 
            
            # Update buffer and store now that the transport is complete
            yield to_entity.buffer_in.put(1) and to_entity.store_in.put(product)
            self.waiting_time += self.env.now-entry_2
            self.manuf_line.track_sim(("InputStock", to_entity.Name, self.env.now))
            self.busy = False 
            yield self.env.timeout(0) 
                
        # Transport from a machine to something 
        if  isinstance(from_entity, Machine) and not isinstance(to_entity, Machine):
            print("Transporting from " + from_entity.ID + " to " + str(to_entity) + " at time = " + str(self.env.now))

            # Start by waiting for an available input resource from entity
            entry = self.env.now
            self.busy = True
            while from_entity.buffer_out.level == 0:
                yield self.env.timeout(10)

                # Skip transport if the machine breaks down while waiting for it
                if from_entity.broken:
                    print("From entity is broken, skipping remaining instructions")
                    self.busy = False
                    yield self.env.timeout(1)
                    return
            yield from_entity.buffer_out.get(1)
            product = yield  from_entity.store_out.get()

            # Move robot and unload / load
            self.waiting_time += self.env.now-entry
            yield self.env.timeout(abs(from_entity.move_robot_time)+self.loadunload_time)
            entry_2 = self.env.now

            # Wait for a spot in the input buffer of 'to_entity' and update storage
            yield to_entity.put(1)
            self.manuf_line.inventory_out.put(product)
            self.waiting_time += self.env.now-entry_2
            self.manuf_line.track_sim((from_entity.Name, "OutputStock", self.env.now))
            self.busy = False 
            yield self.env.timeout(0)

    def which_machine_to_feed(self, current_machine):
        """
        Select the best machine to feed based on a given strategy.
        
        TODO: Upgrade to python 3.10 to use match case 

        Return:
        Machine
        """
        
        # Feed a machine from the input stock
        if not isinstance(current_machine, Machine):
            # Update exepected refill time
            self.manuf_line.expected_refill_time = [abs(i-self.manuf_line.env.now) for i in self.manuf_line.expected_refill_time]

            first_machines = [m for m in self.manuf_line.list_machines if m.first]

            # Balanced-like strategy : try to equally feed machines that are in parallel 
            if self.manuf_line.robot_strategy == 0:
                # Select the machine that has been the less loaded so far
                loads_on_machines = [m.loaded if isinstance(m, Machine) and not m.broken and not m.operating else float('inf') for m in first_machines]
                next_machine = first_machines[loads_on_machines.index(min(loads_on_machines))]  

                # Skip the connection with the next machine connection when it takes 0 time to process a reference (arbitrary convention)
                if self.manuf_line.inventory_in.items != [] and float(self.manuf_line.references_config[self.manuf_line.inventory_in.items[0]][self.manuf_line.list_machines.index(next_machine)+1]) == 0:
                    next_machine.current_product = self.manuf_line.inventory_in.items[0]
                    return self.which_machine_to_feed(next_machine)
                    
                # Skip the connection with the next machine connection when it takes 0 time to process a reference (arbitrary convention)
                next_refilled_ref = list(self.manuf_line.references_config.keys())[np.argmin(self.manuf_line.expected_refill_time)]
                skip_connection_2 = float(self.manuf_line.references_config[next_refilled_ref][self.manuf_line.list_machines.index(next_machine)+1])==0
                if self.manuf_line.inventory_in.items == [] and skip_connection_2 :
                    next_machine.current_product = next_refilled_ref
                    return self.which_machine_to_feed(next_machine)
                
                return next_machine

            # Greedy-like strategy : focus on the machine that has the most space in input
            elif self.manuf_line.robot_strategy == 1:
                # Select the machine that has the fewest items in its input buffer
                buffers_level = [m.buffer_in.level if isinstance(m, Machine) else m.level for m in first_machines]
                next_machine = first_machines[buffers_level.index(min(buffers_level))]

                # Feed to a future machine if the next one takes 0 time to process (convention to skip machine connections)
                if self.manuf_line.inventory_in.items != [] and float(self.manuf_line.references_config[self.manuf_line.inventory_in.items[0]][self.manuf_line.list_machines.index(next_machine)+1]) == 0:
                    next_machine.current_product = current_machine.current_product
                    return self.which_machine_to_feed(next_machine)
                
                return next_machine

        # Feed a machine from another machine
        else:
            # Balanced-like strategy : try to equally feed machines that are in parallel 
            if self.manuf_line.robot_strategy == 0:
                # Select the machine that has been the less loaded so far
                loads_on_machines = [m.loaded if isinstance(m, Machine) and not m.broken and not m.operating else float('inf') for m in current_machine.next_machines]
                next_machine = current_machine.next_machines[loads_on_machines.index(min(loads_on_machines))]
                
                # Skip the connection with the next machine connection when it takes 0 time to process a reference (arbitrary convention)
                if isinstance(next_machine, Machine) :
                    if current_machine.store_out.items != [] and float(self.manuf_line.references_config[current_machine.store_out.items[0]][self.manuf_line.list_machines.index(next_machine)+1]) == 0:
                        next_machine.current_product = current_machine.store_out.items[0]
                        return self.which_machine_to_feed(next_machine)
                    
                    if current_machine.current_product is not None and float(self.manuf_line.references_config[current_machine.current_product][self.manuf_line.list_machines.index(next_machine)+1]) ==0 :
                        next_machine.current_product = current_machine.current_product
                        return self.which_machine_to_feed(next_machine)
                    
                return next_machine
            
            # Greedy-like strategy : focus on the machine that has the most space in input
            elif self.manuf_line.robot_strategy == 1:
                buffers_level = [m.buffer_in.level if isinstance(m, Machine) else m.level for m in current_machine.next_machines]
                next_machine = current_machine.next_machines[buffers_level.index(min(buffers_level))]
                # if float(self.manuf_line.references_config[current_machine.store_out.items[0]][self.manuf_line.list_machines.index(current_machine)+1]) ==0:
                #     return self.which_machine_to_feed(current_machine)
                # else:
                return next_machine
        
            # Unused ? 
            elif self.manuf_line.robot_strategy == 2:
                empty_buffers_machines = [m.loaded if isinstance(m, Machine) else m.level for m in current_machine.next_machines]
                next_machine = current_machine.next_machines[empty_buffers_machines.index(min(empty_buffers_machines))]
                # if float(self.manuf_line.references_config[current_machine.store_out.items[0]][self.manuf_line.list_machines.index(current_machine)+1]) ==0:
                #     return self.which_machine_to_feed(current_machine)
                # else:
                return next_machine

            return True

    def robot_process_local(self, from_entity, to_entity, transport_time=10):
        """Wait to tansport from an entity to another."""
        yield from self.transport(from_entity, to_entity, transport_time)

    def robot_process(self):
        """
        Tell the robot what to transport in what order. 

        Is there only the pull strategy implemented ?

        Return:
        None
        """
        while True:
            try:
                # entities_order = [input_entity, machine1, machine2, machine3, output_entity]
                for i in range(len(self.entities_order)):
                    from_entity = self.entities_order[i]
                    to_entity = self.which_machine_to_feed(from_entity)

                    # "Shall we unload this machine ?", why is there this message here ? 
                    
                    # Handle shift reset
                    if self.manuf_line.reset_shift_bool:
                        self.manuf_line.reset_shift_bool = False
                        print("Reseting Robot Process.")
                        break
                    
                    # When we are feeding the first machines, from the supermarket
                    if not isinstance(from_entity, Machine):
                        if to_entity.buffer_in.level < to_entity.buffer_in.capacity:
                            yield from self.transport(from_entity, to_entity)
                        # elif to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.level == 0:
                        #   yield from self.transport(from_entity, to_entity)

                    # When feeding from a machine
                    else:
                        # When feeding a machine
                        try:
                            # Available space in destination and available input
                            if to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level > 0:
                                yield from self.transport(from_entity, to_entity)

                            # Available space in destination but no input, wait if it's under production
                            elif to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level == 0 and (from_entity.operating and not from_entity.broken):
                                yield from self.transport(from_entity, to_entity)


                        except Exception as e:
                            # When feeding the shop stock 
                            try:
                                # Available space in destination and available input
                                if to_entity.level < to_entity.capacity and from_entity.buffer_out.level > 0:
                                    yield from self.transport(from_entity, to_entity)

                                # Available space in destination but no input, wait if it's under production
                                elif to_entity.level < to_entity.capacity and from_entity.buffer_out.level == 0 and from_entity.operating:
                                    yield from self.transport(from_entity, to_entity)

                            # Useless ?     
                            except Exception as e2:
                                if to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level > 0:
                                    yield from self.transport(from_entity, to_entity)

                                elif to_entity.buffer_in.level < to_entity.buffer_in.capacity and from_entity.buffer_out.level == 0 and from_entity.operating:
                                    yield from self.transport(from_entity, to_entity)

            except simpy.Interrupt:
                print("Reseting Robot Process.")
                yield self.env.timeout(0)      

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
                    #     if self.entities_order[i].buffer_in.level < self.entities_order[i].buffer_in.capacity:
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


class CentralStorage:
    def __init__(self, env, central_storage_config, times_to_reach={}, strategy='stack') -> None:
        """Hold two storages : back and front, each having several blocks allowing one or many sizes.

        Parameters:
        - central_storage_config' : below is an example of its structure.
            {'front': [ {'allowed_sizes' : ['XS', 'S'],
                         'capacity': 336},
                        {'allowed_sizes' : ['XS', 'S', 'L'],
                            'capacity': 24}]
                        
             'back': [ {'allowed_sizes' : ['XS', 'S'],
                        'capacity': 376}]}
        
        - strategy : decide which storage to fill or take from when a new reference is added or removed.
            - 'stack' : Try to fill blocks of the 'front' storage before the 'back'.

        TODO: Handle the difference between 'size' and 'ref' from the user.
        TODO: Implement "fastest" route with times_to_reach. Waiting for more details from Vivi.

        """
        self.env = env
        self.strategy = strategy
        self.times_to_reach = times_to_reach
        self.stores = central_storage_config

        # Turn the 'capacity' attributes to 'simpy.Store' objects
        for side in self.stores.keys():
            for block in self.stores[side]:
                block['store'] = simpy.FilterStore(env, block['capacity'])
                del block['capacity']

    def __str__(self) -> str:
        """Display what is inside the front and back stores.
        
        For each store and block within, display :
        - allowed sizes
        - capacity
        - level
        - items        
        """

        lines = ["\n=== CENTRAL STORAGE ==="]

        for side in self.stores.keys():
            lines.append(side.upper())
            for block in self.stores[side]:
                lines.append(f"allowed_sizes : {block['allowed_sizes']}")
                lines.append(f"capacity : {block['store'].capacity}")
                lines.append(f"level : {len(block['store'].items)}")
                lines.append(f"items : {block['store'].items}")
                lines.append('')

        return "\n".join(lines)

    def available_spot(self, ref=None) -> bool:
        """Check if there is an available spot.
        
        If a reference is specified, check an available spot for this specific reference.
        Otherwise, check if there is any available spot no matter the reference.

        """

        for side in self.stores.keys():
            for block in self.stores[side]:
                # Return True if there is any space when the reference is not specified.
                if ref is None and len(block['store'].items) < block['store'].capacity:
                    print(f'The central storage is not full.')
                    return True
                
                # Return True if there is any space and the specified reference is allowed in the block.
                if ref is not None and len(block['store'].items) < block['store'].capacity and ref in block['allowed_sizes']:
                    print(f'There is an available spot for reference "{ref}" in the central storage.')
                    return True

        # No space has been found.
        print(f'There is NO available spot for reference "{ref}" in the central storage.')
        return False

    def available_ref(self, ref=None) -> bool:
        """Check if there is an available reference.
        
        If a reference is specified, check this specific reference.
        Otherwise, check if there is at least 1 reference no matter which one.

        """

        for side in self.stores.keys():
            for block in self.stores[side]:
                # Return True if there is any item when the reference is not specified.
                if ref is None and len(block['store'].items) > 0:
                    print(f'The central storage is empty.')
                    return True
                
                # Return True if there is a specified reference that is allowed and present.
                if ref is not None and ref in block['allowed_sizes'] and ref in block['store'].items:
                    print(f'The specified reference "{ref}" is available in the central storage.')
                    return True

        # No reference has been found.
        print(f'There is no available reference "{ref}" in the central storage.')
        return False

    def put(self, ref):
        """Try to put a reference in the storage determined by the strategy of the central storage."""
        
        # Check if the strategy has been implemented.
        if self.strategy not in ['stack']:
            raise Exception(f'Strategy "{self.strategy}" is not yet implemented for the central storage.')

        if self.strategy == 'stack':
            for side in self.stores.keys():
                for block in self.stores[side]:
                    # Check if the ref is allowed in the current block.
                    if ref in block['allowed_sizes']:
                        store = block['store']
                        # Check if there is an available spot to put the reference.
                        if len(store.items) < store.capacity:
                            print(f'Put the reference "{ref}" in the central storage.')
                            store.put(ref)
                            return
                        
        # Haven't found any place to put the reference
        raise Exception(f"Tried to put the reference {ref} in the central storage that is FULL.")
            
    def get(self, ref=None):
        """Try to get a reference in the storage determined by the strategy of the central storage.
        
        If a reference is specified, get this specific reference.
        Otherwise, get any reference depending on the strategy of the storage.

        TODO: How to get a specific reference ?
        """

        # Check if the strategy has been implemented.
        if self.strategy not in ['stack']:
            raise Exception(f'Strategy "{self.strategy}" is not yet implemented for the central storage.')

        if self.strategy == 'stack':
            for side in self.stores.keys():
                for block in self.stores[side]:
                    # When the reference is not specified (None), get as soon as the block is not empty.
                    if ref is None and len(block['store'].items) > 0 :
                        print(f'Got the first reference "{ref}" in the central storage.')
                        return block['store'].get() 
                        
                    # When the reference is specified, get as soon as one is present in a block.
                    if ref is not None and ref in block['store'].items:
                        print(f'Got the specified reference "{ref}" in the central storage.')
                        return block['store'].get(lambda x: x==ref)
                            
        # Haven't found any reference to get
        raise Exception(f"Tried to get the reference {ref} in the central storage but couldn't.")


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