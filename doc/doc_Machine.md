### `Machine` Class Documentation

The `Machine` class models the behavior of machines in a manufacturing line simulation. It interacts with other machines, buffers, operators, robots, and handles part processing, breakdowns, and repairs. The simulation environment uses this class to manage the workflow of products as they pass through various machines.

#### **Class Constructor:**

```python
def __init__(self, manuf_line, env, machine_id, machine_name, config, assigned_tasks=None, robot=None, operator=None, previous_machine=None, first=False, last=False, breakdowns=True, mttf=3600*24*7, mttr=3600, buffer_capacity=100, initial_buffer=0, hazard_delays=False, has_robot=False, fill_central_storage=False)
```

##### **Parameters:**
- `manuf_line`: The manufacturing line to which this machine belongs.
- `env`: The simulation environment (e.g., a SimPy environment) that handles event scheduling.
- `machine_id`: Unique identifier for the machine.
- `machine_name`: Human-readable name for the machine.
- `config`: A configuration dictionary that contains various parameters for the machine's operation.
- `assigned_tasks`: Optional list of tasks assigned to this machine.
- `robot`: Optional robot assigned to this machine.
- `operator`: Optional operator assigned to handle manual operations for the machine.
- `previous_machine`: Reference to the previous machine in the manufacturing line.
- `first`: Boolean flag to mark if this is the first machine in the line.
- `last`: Boolean flag to mark if this is the last machine in the line.
- `breakdowns`: Boolean flag indicating whether the machine experiences breakdowns.
- `mttf`: Mean Time to Failure (default is one week in seconds).
- `mttr`: Mean Time to Repair (default is one hour in seconds).
- `buffer_capacity`: Maximum capacity of the machine's input/output buffer.
- `initial_buffer`: Initial number of parts to preload into the buffer.
- `hazard_delays`: Boolean flag indicating if hazard delays affect the machine.
- `has_robot`: Boolean flag indicating if the machine has an associated robot.
- `fill_central_storage`: Boolean flag indicating whether the machine fills a central storage.

##### **Attributes:**
- `self.mt`: Machine's current time tracker.
- `self.ID`: The machine's unique ID.
- `self.Name`: The machine's name.
- `self.env`: The simulation environment.
- `self.manuf_line`: Reference to the manufacturing line.
- `self.entry_times`, `self.exit_times`: Lists that store timestamps of when parts enter and exit the machine.
- `self.finished_times`: List to store the completion times of processed parts.
- `self.n_breakdowns`: Counter for the number of breakdowns.
- `self.buffer_tracks`: Tracks the buffer states over time.
- `self.parts_done`: Total parts processed.
- `self.parts_done_shift`: Parts processed in a shift.
- `self.ct`: Cycle time for part processing.
- `self.wc`: Work content for manual operations.
- `self.manual_time`: Time required for manual operations.
- `self.config`: Configuration dictionary passed during initialization.
- `self.buffer_btn`: Button indicator for buffer interaction.
- `self.buffer_capacity`: Maximum buffer capacity.
- `self.initial_buffer`: Initial parts stored in the buffer.
- `self.process`: Current ongoing process in the machine.
- `self.next_machine`: Reference to the next machine in the line.
- `self.has_robot`: Boolean indicating if the machine is assisted by a robot.
- `self.previous_machine`: Reference to the previous machine.
- `self.operating_state`: Tracks the machine's operating state.
- `self.waiting_time`: [Starvation, Blockage] waiting time metrics.
- `self.waiting_time_rl`: Real-time waiting duration.
- `self.operating`: Boolean indicating if the machine is currently operating.
- `self.identical_machines`: List of identical machines (if any).
- `self.ref_produced`: List of references for produced products.
- `self.loaded_bol`: Flag indicating whether a part is loaded in the machine.
- `self.current_product`: Current product being processed by the machine.
- `self.buffer_in`, `self.buffer_out`: Input and output buffer objects for the machine.
- `self.MTTF`: Mean Time to Failure.
- `self.MTTR`: Mean Time to Repair.
- `self.assigned_tasks`: List of tasks assigned to the machine.
- `self.operator`: Operator assigned to the machine.
- `self.broken`: Boolean flag indicating machine breakdown.
- `self.real_repair_time`: Stores real repair times for each breakdown.
- `self.op_fatigue`: Operator fatigue flag.
- `self.fill_central_storage`: Flag indicating if the machine fills central storage.
  
---

#### **Methods:**

##### **`time_to_failure()`**
```python
def time_to_failure(self)
```
Generates and returns the time until the next machine failure based on the breakdown distribution law configured for the manufacturing line.

- Uses different failure distribution laws (Weibull, Exponential, Normal, Gamma).
- Adjusts MTTF based on the elapsed time or environmental conditions.

##### **`fatigue_model(elapsed_time, base_time)`**
```python
def fatigue_model(self, elapsed_time, base_time)
```
Models operator fatigue using a sigmoid function. As the operator works over time, their efficiency decreases, and this method adjusts the base processing time based on fatigue.

##### **Parameters:**
- `elapsed_time`: The time elapsed in hours for manual operations.
- `base_time`: The base time required for manual tasks without fatigue.

##### **Returns:**
- Adjusted processing time accounting for operator fatigue.

##### **`machine_process()`**
```python
def machine_process(self)
```
Main process method for the machine, which simulates the full operation cycle, including product loading, processing, and output. It handles part movement through the machine and interacts with buffers and other machines.

##### **Features:**
- Handles loading parts from the input buffer (`buffer_in`).
- Calls a repair process if the machine breaks down.
- Simulates the machine's cycle time (`ct`) for processing parts.
- Interacts with an operator if manual work is required.
- Handles breakdowns via `simpy.Interrupt` and manages repairs using the `repairmen` resource.
- Passes processed products to the output buffer (`buffer_out`).
- Tracks time and state changes for each operation cycle.

##### **Breakdown and Repair:**
The method simulates random breakdowns, pauses the machine for repair using `simpy`'s interrupt and request processes. The `MTTR` defines the time for the machine to be repaired before resuming its operations.

##### **Operator Interaction:**
If an operator is assigned, the machine checks if the operator is available and waits if they are busy. Once the operator is free, manual tasks are completed before the machine can proceed.

##### **Buffers:**
Products are stored in input and output buffers (`buffer_in`, `buffer_out`). The machine ensures that buffer capacities are respected and tracks the flow of products.

##### **Flow Example:**
1. Wait for parts in `buffer_in`.
2. Check operator availability (if assigned).
3. Process the product (includes fatigue adjustment).
4. Send processed parts to `buffer_out`.
5. Handle breakdowns and repairs.

---

### **Usage:**
This `Machine` class is designed for a manufacturing simulation. It models the behavior of machines that may operate autonomously or with human or robotic assistance. It supports real-world manufacturing complexities such as fatigue, breakdowns, repairs, and part flow between multiple machines.