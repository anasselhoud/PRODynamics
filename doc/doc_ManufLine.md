# ManufLine Class Documentation

The `ManufLine` class is designed to simulate the dynamic behavior of production lines. It manages various aspects of the manufacturing environment such as machine processing, breakdowns, stock management, operator assignments, robot operations, and shift resets. It uses the SimPy library to model real-time event-driven processes. This documentation provides a detailed explanation of the class methods, attributes, and usage.

## Table of Contents
- [Class Initialization](#class-initialization)
- [Attributes](#attributes)
- [Methods](#methods)
  - [run](#run)
  - [reset](#reset)
  - [save_global_settings](#save_global_settings)
  - [initialize](#initialize)
  - [run_action](#run_action)
  - [reset_shift](#reset_shift)
  - [break_down](#break_down)
  - [monitor_waiting_time](#monitor_waiting_time)
  - [refill_market](#refill_market)
  - [repairmen_process](#repairmen_process)
  - [create_machines](#create_machines)
- [Usage](#usage)

---

## Class Initialization

### `__init__(self, env, tasks, operators_assignement=None, tasks_assignement=None, config_file=None)`

**Parameters:**
- `env`: SimPy environment. This manages the simulation's timing and processes.
- `tasks`: List of tasks to be performed in the production line.
- `operators_assignement` (optional): Specifies the assignment of operators to tasks.
- `tasks_assignement` (optional): Specifies the assignment of tasks.
- `config_file` (optional): Configuration file for setting up the production line parameters, typically in YAML format.

**Description:**
The `__init__` method initializes the production line by loading configurations from a provided file, setting up machine processes, repairmen, and managing stock levels in the supermarket. It also handles multiple references and strategies for robot and operator management.

---

## Attributes

### Core Attributes
- `env`: The SimPy environment that controls the simulation time and execution.
- `config`: Configuration loaded from the YAML file specifying production line parameters.
- `tasks`: A list of tasks representing the operations in the production line.
- `operators_assignement`: Operators assigned to specific tasks.
- `tasks_assignement`: Assignment of tasks to machines or operators.

### Simulation Parameters
- `sim_time`: Total time for which the simulation runs.
- `n_repairmen`: Number of repairmen available to fix machine breakdowns.
- `repairmen`: A SimPy PreemptiveResource for managing repairmen processes.
- `breakdowns`: A configuration object determining if machine breakdowns are enabled.
- `breakdowns_switch`: Boolean indicating whether breakdowns are enabled.

### Machine and Stock Management
- `machine_config_data`: Data used to configure individual machines in the production line.
- `list_machines`: A list of all machines in the production line.
- `supermarket_in`: The input supermarket storage to manage stock levels.
- `shop_stock_out`: Output stock storage for completed products.
- `buffer_tracks`: List of buffer tracks for managing intermediate products between machines.
- `stock_capacity`: Capacity of the supermarket stock.
- `num_cycles`: Number of complete production cycles during the simulation.

### Robot Management
- `robots_list`: List of robots assigned to the production line.
- `n_robots`: Number of robots available for operation.
- `robot_strategy`: Strategy to be followed by the robots (e.g., balanced, greedy).
- `robot_states`: List of robot states during the simulation.

---

## Methods

### `run(self, progress_bar=None)`

**Description:**
Starts the production line simulation. It initializes machine processes, handles breakdowns, refills the supermarket, and operates robots. If a progress bar is provided, the simulation progress can be tracked visually.

**Parameters:**
- `progress_bar` (optional): An object to track the progress of the simulation.

**Returns:**
None.

---

### `reset(self)`

**Description:**
Resets the production line, including all machines, robots, and the supermarket. It sets up the environment for a new simulation run without restarting the code. 

**Returns:**
None.

---

### `save_global_settings(self, configuration, references_config, line_data, buffer_sizes=[])`

**Description:**
Configures the global settings for the manufacturing line. This includes enabling breakdowns, setting stock capacities, assigning robot strategies, and configuring machines and buffer sizes.

**Parameters:**
- `configuration`: Dictionary containing global settings such as stock capacity, breakdown laws, and robot strategies.
- `references_config`: Configuration of references used in the simulation.
- `line_data`: Machine configuration data.
- `buffer_sizes` (optional): Buffer sizes for machines.

**Returns:**
None.

---

### `initialize(self)`

**Description:**
Initializes the machine processes and refills the supermarket at the start of the simulation. It sets up robots and starts their operations.

**Returns:**
None.

---

### `run_action(self, action)`

**Description:**
Executes an action to transport products between machines. This simulates the interaction between a robot and machines.

**Parameters:**
- `action`: A list containing two elements `[from_machine, to_machine]`.

**Returns:**
None.

---

### `reset_shift(self)`

**Description:**
Resets the shift after every 8 hours of operation. This involves emptying machine buffers and supermarkets, updating the number of cycles, and resetting production counters.

**Returns:**
None.

---

### `break_down(self, machine)`

**Description:**
Simulates machine breakdowns. The machine breaks down based on its mean time to failure (MTTF) and interrupts its process until repairs are completed.

**Parameters:**
- `machine`: The machine object that will break down.

**Returns:**
None.

---

### `monitor_waiting_time(self, machine)`

**Description:**
Monitors the waiting time of a machine when it is not operating and tracks idle time.

**Parameters:**
- `machine`: The machine being monitored.

**Returns:**
None.

---

### `refill_market(self, ref)`

**Description:**
Refills the input supermarket with products based on specified refill times and sizes. This method runs in a loop, refilling the supermarket stock as required.

**Parameters:**
- `ref`: The reference type being refilled in the market.

**Returns:**
None.

---

### `repairmen_process(self)`

**Description:**
Simulates the operation of repairmen. It handles machine repairs by allocating repair resources and tracking the completion of repair jobs.

**Returns:**
None.

---

### `create_machines(self, list_machines_config)`

**Description:**
Sets up machines in the production line based on configuration data. Each machine is assigned properties such as buffer capacities, mean time to failure (MTTF), and transport time.

**Parameters:**
- `list_machines_config`: Configuration data for the machines in the line.

**Returns:**
None.

---

## Usage

1. **Initialize the Environment**:
   ```python
   import simpy
   env = simpy.Environment()
   ```

2. **Create a ManufLine Object**:
   ```python
   production_line = ManufLine(env, tasks, operators_assignement, tasks_assignement, config_file)
   ```

3. **Run the Simulation**:
   ```python
   production_line.run()
   ```

4. **Reset the Simulation**:
   ```python
   production_line.reset()
   ```

5. **Save Global Settings**:
   ```python
   production_line.save_global_settings(configuration, references_config, line_data, buffer_sizes)
   ``` 

By using these methods, you can create a dynamic and configurable simulation of production lines, allowing for detailed experimentation with machine configurations, breakdowns, and shift management.