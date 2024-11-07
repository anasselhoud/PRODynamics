# `Robot` Object Documentation

The `Robot` class is designed to manage the transportation of items between machines in a manufacturing line. It handles the transportation process, machine selection strategies, and addresses complex scenarios such as shift resets and machine breakdowns. This documentation explains the key attributes, methods, and usage of the `Robot` class.

---

## Class: `Robot`

### Overview
The `Robot` class simulates a transportation unit (either a robot or Automated Guided Vehicle - AGV) that moves items between various entities (machines, storages) in a manufacturing line. The robot is responsible for selecting the best machine to feed, transporting between machines, and handling edge cases like empty buffers and shift resets.

### Constructor

```python
def __init__(self, manuf_line, env, maxlimit=1)
```

**Parameters:**
- `manuf_line`: Reference to the manufacturing line instance that the robot operates within.
- `env`: The simulation environment (typically a SimPy environment).
- `maxlimit` (optional, default=1): Maximum capacity of the robot's buffer.

**Attributes:**
- `busy`: Boolean indicating whether the robot is currently in use.
- `manuf_line`: Manufacturing line object the robot is part of.
- `env`: Simulation environment.
- `schedule`: Planned transport schedule (currently unused).
- `buffer`: A SimPy store that holds items for transport.
- `robots_res`: SimPy priority resource (unused in current implementation).
- `waiting_time`: Tracks the robot's waiting time during operations.
- `in_transport_times`: List storing transport times for performance tracking.
- `process`: Holds the current robot process.
- `entities_order`: List defining the sequence of entities for transport operations.
- `loadunload_time`: Time for loading/unloading items (default=50 time units).

---

## Methods

### `transport(from_entity, to_entity, transport_time=10)`

```python
def transport(self, from_entity, to_entity, transport_time=10)
```

This method handles the actual transportation of items between two entities, simulating the process of moving items from `from_entity` to `to_entity`. The process can be interrupted if breakdowns or shift resets occur.

**Parameters:**
- `from_entity`: Entity from which items are being transported (machine, storage, etc.).
- `to_entity`: Entity to which items are being transported.
- `transport_time` (optional, default=10): Time it takes to transport between entities.

---

### `which_machine_to_feed(current_machine)`

```python
def which_machine_to_feed(self, current_machine)
```

This method selects the next machine that the robot should feed based on a strategy defined within the manufacturing line. It is a placeholder for more advanced machine selection logic.

**Parameters:**
- `current_machine`: The current machine in the production line.

**Returns:**
- `Machine`: The next machine to be fed.

---

### `robot_process_local(from_entity, to_entity, transport_time=10)`

```python
def robot_process_local(self, from_entity, to_entity, transport_time=10)
```

This method initiates a local transport operation between two entities with a specified transport time.

**Parameters:**
- `from_entity`: Entity from which items are being transported.
- `to_entity`: Entity to which items are being transported.
- `transport_time` (optional, default=10): Time it takes to transport between entities.

---

### `robot_process()`

```python
def robot_process(self)
```

The main process of the robot. This method continually checks the `entities_order` list to determine which entities require transport and performs the transport between them. The process handles shift resets, interruptions, and pauses when needed.

---

### `which_machine_to_getfrom(current_machine)`

```python
def which_machine_to_getfrom(self, current_machine)
```

This method selects the best machine from which the robot should retrieve items, based on a strategy defined within the manufacturing line. It ensures that the machine with the fullest buffer (or most urgent need for emptying) is selected.

**Parameters:**
- `current_machine`: The current machine that requires input.

**Returns:**
- `Machine`: The best machine to retrieve items from.

---

### `handle_empty_buffer(from_entity, to_entity)`

```python
def handle_empty_buffer(self, from_entity, to_entity)
```

Handles scenarios where an entity's buffer is empty. This method recursively determines the previous machine in the line to transport from when the current machine cannot supply items.

**Parameters:**
- `from_entity`: The entity with the empty buffer.
- `to_entity`: The entity requiring items to be transported.

---

### Helper Methods

#### `_perform_transport_logic(from_entity, to_entity)`

```python
def _perform_transport_logic(self, from_entity, to_entity)
```

Helper method that performs transport logic based on buffer and machine states. It handles transporting items from both machines and storage locations.

#### `_handle_machine_transport(from_entity, to_entity)`

```python
def _handle_machine_transport(self, from_entity, to_entity)
```

Handles transportation between machines by checking buffer conditions and central storage availability.

#### `_handle_shop_stock_transport(from_entity, to_entity, exception)`

```python
def _handle_shop_stock_transport(self, from_entity, to_entity, exception)
```

Handles transport exceptions related to shop stock, ensuring that the robot continues working under abnormal conditions.

#### `_select_previous_machine(current_machine)`

```python
def _select_previous_machine(self, current_machine)
```

Selects the best machine from the previous machines in the production line, considering buffer fullness and machine status.

---

## Usage Example

Here is a brief example showing how to create a `Robot` instance and initiate its transport process in a SimPy environment:

```python
import simpy

def manufacturing_process(env):
    manuf_line = ManufLine() 
    robot = Robot(manuf_line, env)

    # Set the robot's transport schedule
    robot.entities_order = [entity1, machine1, machine2, output_entity]

    # Start the robot's transport process
    env.process(robot.robot_process())

# Create a SimPy environment and run the manufacturing process
env = simpy.Environment()
env.process(manufacturing_process(env))
env.run(until=100)  # Simulate for 100 time units
```

In this example, a `Robot` is initialized with the manufacturing line and SimPy environment. It is then assigned an `entities_order` that dictates the flow of transport tasks, and the robot's process is started in the simulation environment.

