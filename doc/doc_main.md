# PRODynamics Tool Documentation

### Version Information
- **Tool Name**: PRODynamics
- **Version**: 0.1.0
- **Author**: Anass ELHOUD
- **Contact**: [anass.elhoud@forvia.com](mailto:anass.elhoud@forvia.com)
- **License**: Internal Use

---

### Overview
PRODynamics is an optimization and simulation tool developed at FORVIA Clean Mobility as part of an industrial PhD thesis project, conducted from February 2022 to December 2024. The tool provides advanced capabilities for evaluating and optimizing material flows in production lines, particularly assessing the impact of various hazardous events, such as:
- Machine breakdowns
- Microstops
- Supply chain delays

By modeling these events and their effects on the production flow, PRODynamics enables highly accurate, fast analysis, supporting effective decision-making in complex production environments.

This documentation will guide you through the development and customization of PRODynamics. Before diving into this, basic algorithmic and programming knowledge is recommended. For theoretical insights, please consult Chapter V of the PhD dissertation associated with this tool.

---

### Requirements
To customize and run PRODynamics, you’ll need the following software:
- **Code Editor**: Any will work, though Visual Studio Code is recommended.
- **Anaconda/Miniconda**: Helps build and manage the Python environment.
- **Docker**: Essential for managing the containerization of the application, allowing you to test new features efficiently.

---

### Setting Up

#### Step 1: Create a Conda Environment
1. **Install Miniconda**: If not already installed, visit the [Conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and follow the installation instructions.
2. **Create Environment**: Ensure you’re using Python version 3.9.

    ```bash
    conda create -n prodynamics python=3.9
    ```

#### Step 2: Set Up Environment
After creating the Conda environment, it will initially be empty. To populate it with the required libraries:
1. **Activate the environment**:

    ```bash
    conda activate prodynamics
    ```
2. **Install Dependencies**: A `requirements.txt` file is available in the main directory, listing all required libraries with specified versions. Install all dependencies at once by navigating to the main folder and running:

    ```bash
    pip install -r requirements.txt
    ```

#### Step 3: Launch the Tool
PRODynamics uses the Streamlit framework, a straightforward yet powerful tool for building and deploying Python-based applications.
1. **Run the application locally**:

    ```bash
    streamlit run app.py
    ```
    > **Note**: `app.py` is the primary script for the full application. We'll explore the structure and files within the main folder in subsequent sections.

2. **Deploy Online**: Streamlit offers convenient options for deploying the application on the cloud, making it easy to test and showcase the tool’s capabilities.

---


# UML & Architecture

To understand the technical structure of PRODynamics, let’s delve into its core architecture. The tool comprises a simulation module and an optimization module, each serving to model production flows, generate stochastic events, and optimize production efficiency based on simulated scenarios.

## Simulation Part
The dynamic simulator is built using the **SimPy** framework and follows a modular design, which provides flexibility and room for future expansion. The overall architecture is displayed in the UML diagram below:

![UML and Architecture Diagram](.\images\simulation-usecase.PNG)

The main simulation functions are defined in `utils.py`. Here’s a breakdown of the simulation phases:

1. **Preparation**: This phase involves defining the production line and its components. Users specify:
   - **Production Line**: Includes the number and types of machines, operational sequences for each product, and processing times.
   - **Product Details**: Defines product types, bill of materials, and assembly requirements.
   - **Stochastic Events**: Introduces realistic factors such as machine breakdowns and variable product arrival rates.

   These definitions serve as the basis for generating all simulation models.

2. **Simulation Generation**: Multiple production models are initialized to simulate different scenarios or configurations. This involves:
   - **Production Line Initialization**: Creating instances of machines, buffers, transporters, and maintenance personnel.
   - **Multi-Model Initialization**: Setting up diverse configurations for products and storage (supply and output).
   - **Network Generation**: Establishing a network that links production modules, allowing product flow between machines based on routing rules, transportation times, and buffer capacities.

3. **Simulation Execution**: The core simulation engine handles discrete events and tracks the progress of products through the production line.
   - **One-Time Execution**: Handles discrete events, such as machine breakdowns and product arrivals.
   - **Sequential Execution**: Manages transport actions, machine operations, and event scheduling in real-time.
   - **KPI Visualization**: Captures and visualizes key performance indicators (KPIs) like throughput, lead time, work-in-process, and machine utilization throughout the simulation.

![UML and Architecture Diagram](.\images\sim-framework.PNG)
![UML and Architecture Diagram](.\images\sim-UML.PNG)


### Simulation Classes

The main simulation classes in PRODynamics represent various components of the production line. Each of these classes has a dedicated Markdown file containing detailed documentation on its attributes, methods, and usage. Click the links to learn more about each class. Click on each class name for a detailed documentation. 

- **[ManufLine](.\doc_ManufLine.md)**: This is the core class of the simulation. It manages the SimPy environment and handles the primary simulation configurations, such as breakdown modes, laws, and simulation time. The production line setup includes details like the number of machines, operators, robots/transporters, and production rates. Key attributes include:
  - **Environment**: The SimPy environment where the simulation runs.
  - **Simulation Configurations**: Settings such as breakdown mode and simulation duration.
  - **Production Line Setup**: Information on machines, operators, references, supply/output rates, etc.

- **[Machine](doc_Machine.md)**: Represents an individual machine in the production line. Each machine has several attributes:
  - **ID**: Unique identifier for the machine.
  - **Processing Time**: Time required to process each product.
  - **Buffer**: Storage for products waiting to be processed.
  - **Linked Machines**: Machines connected to this one (input and output).
  - **Operational Flag**: Indicates whether the machine is running.
  - **Loading Flag**: Status indicating if the machine is loading a product.
  - **Current Reference**: The specific product currently being processed.

- **[Robot](doc_Robot.md)**: Represents the transporter in the production line, typically a handling or manipulation robot. Main attributes include:
  - **Transportation Time**: Time taken to transport materials between machines.
  - **Busy Flag**: Indicates if the robot is occupied.
  - **Sequence of Transportation**: Order in which items are moved across the production line.

- **[Central Storage Block](doc_central_storage.md)**: Represents defined to the central storage of the production line. It is mainly present in FHS manufacturing lines when simulating macro-production of the full plant.

### Simulation Functions

Each of the classes includes key functions to simulate different aspects of the production environment:

- **ManufLine - `run()`**: Starts and manages the main simulation loop, coordinating various components of the production line. [Learn more in the ManufLine documentation.](doc_ManufLine.md)
- **Machine - `break_down()`**: Simulates machine breakdowns, impacting the production flow. [Detailed documentation here.](doc_Machine.md)

Refer to the individual class documentation files for more information on additional methods and examples of how to use these functions. 

---

## Optimization Part
The current version of PRODynamics includes two primary optimization objectives:
1. **Intermediate Buffer Sizing** (Functional): 
   - Implemented in `utils_optim.py`, the buffer sizing optimization leverages a method called **SEOD (Sample-Estimate-Optimize-Decide)**. 
   - SEOD is a surrogate model optimization technique that samples possible buffer sizes, evaluates them using the stochastic simulator, approximates the results with an analytical model, and then numerically optimizes buffer sizes to determine the optimal configuration.

2. **Transportation Flow Optimization**:
   - This module addresses the complex challenge of optimizing material transport between machines, often an NP-hard problem. Traditional transporters, such as robots or AGVs, follow rigid programming, which can be inefficient during machine breakdowns.
   - To address this, **Deep Reinforcement Learning (DRL)**, specifically **Deep Q-Networks (DQN)**, is applied. The DRL agent dynamically determines transport actions based on the real-time state of the production line.

![DQN Framework](.\images\statespace.PNG)
   
   - **State Space**: Contains a vector of real-time metrics such as machine states, buffer levels, machine idle times, and transporter positions.
   - **Action Space**: Represents the decision of picking up from a source machine (`M-src`) and delivering to a target machine (`M-tgt`).

![DQN Framework](.\images\dqn-architecture.PNG)

At each timestep, the DRL agent selects an action to maximize future rewards, aiming to reduce idle times and improve transport efficiency. The following diagram outlines the architecture for transportation flow optimization:


---

### Main Folder Structure
The primary directory for PRODynamics contains key files and folders that facilitate development and customization. Here’s an overview of its layout:
- **app.py**: The main entry point of the application.
- **requirements.txt**: Lists all Python packages needed for the environment setup.
- **src/**: Contains source code for PRODynamics’ core functionalities, including modules for optimization, simulation, and hazard modeling.
- **data/**: Stores any input/output data files necessary for the tool’s simulations and analyses.

---

### Customization Guide
To add or modify features:
1. **Familiarize Yourself with the Source Code**: Within the `src` folder, locate modules corresponding to your feature of interest.
2. **Test New Features Locally**: Use Docker to build and test changes in an isolated container.
3. **Consider Dependencies**: Update `requirements.txt` if additional libraries are necessary.

---

### Contact & Support
For assistance or to report issues, contact the author at [anass.elhoud@forvia.com](mailto:anass.elhoud@forvia.com).

---