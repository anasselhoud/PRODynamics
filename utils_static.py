import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import linkage, fcluster
import xml.etree.ElementTree as ET
from scipy.cluster.hierarchy import linkage, dendrogram
from utils import *

import numpy as np

class Part:
    """
    Represents a part in the task system.

    Attributes:
        id (int): Unique identifier for the part.
        ref (str): Reference or name of the part.
        duration (float): Duration to complete the part.
        weight (float): Weight associated with the part.
    """
    def __init__(self, id, ref, duration, weight):
        self.id = id
        self.ref = ref
        self.duration = duration
        self.weight = weight


class Task:
    """
    Represents a task in the task system.

    Attributes:
        id (int): Unique identifier for the task.
        parts (list of Part): Parts associated with the task.
        CT (float): Cycle time required to complete the task.
        preced_list (list): List of tasks that must precede this task.
        forbid_list (list): List of tasks that cannot be scheduled concurrently.
        clusterTasksID (list): Tasks clustered with this task.
    """
    def __init__(self, id, parts, CT, preced_list, forbid_list):
        self.id = id
        self.parts = parts
        self.CT = CT
        self.preced_list = preced_list
        self.forbid_list = forbid_list
        self.clusterTasksID = []


class QLearning:
    """
    Implements Q-learning to optimize task scheduling.

    Attributes:
        Tasks (list of Task): List of tasks to be scheduled.
        target (float): Target cycle time for the scheduling.
        tolerance (float): Allowed tolerance for exceeding the cycle time.
        solution (list): Current solution, i.e., task scheduling order.
        session_rewards (list): Rewards accumulated across episodes.
        n_episodes (int): Number of episodes to run the Q-learning process.
    """
    def __init__(self, n_episodes, Tasks, targetCT, tolerance):
        self.Tasks = Tasks
        self.target = targetCT
        self.tolerance = tolerance
        self.solution = []
        self.session_rewards = []
        self.n_episodes = n_episodes

    def step(self, action):
        """
        Perform a step in the environment (task scheduling).

        Args:
            action (int): The index of the task to schedule next.
        
        Returns:
            new_state (int): The index of the task that was just scheduled.
            done (bool): Whether the scheduling is complete.
        """
        done = False
        new_state = action
        self.solution.append(new_state)
        if len(self.solution) == len(self.Tasks):
            done = True
        return new_state, done

    def split(self, a, n):
        """
        Split a list into n almost equal parts.
        
        Args:
            a (list): List to be split.
            n (int): Number of parts to split into.
        
        Returns:
            Generator: Splits of the list.
        """
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def precedence_graph(self):
        """
        Build the precedence graph based on task constraints.
        """
        preced_graph = []
        # No precedence restrictions
        preced_graph.append([task for task in self.Tasks if not task.preced_list])
        # Handle precedence relationships
        for task in self.Tasks:
            for preced_task in task.preced_list:
                if preced_task in preced_graph:
                    preced_graph.append(task)

    def get_nworkstations(self):
        """
        Calculate the number of workstations required based on task scheduling.
        
        Returns:
            num_workstations (int): Number of workstations needed.
            ct_WS (list): Cycle times per workstation.
            tasks_WS (list): Tasks assigned to each workstation.
        """
        def allocate_tasks(target):
            """
            Helper function to allocate tasks to workstations.
            
            Args:
                target (float): Target cycle time for allocation.
            
            Returns:
                num_workstations (int): Number of workstations.
                ct_WS (list): Cumulative cycle times per workstation.
                tasks_WS (list): List of tasks per workstation.
            """
            total_ct = 0
            ct_WS = [0]  # List to keep cumulative cycle time of each workstation
            tasks_WS = [[]]  # List to keep the tasks assigned to each workstation

            for i in self.solution:
                task_ct = float(self.Tasks[i].CT)
                # If adding the current task exceeds the target with tolerance
                if ct_WS[-1] + task_ct > (1 + self.tolerance) * (target / 2):
                    ct_WS.append(task_ct)  # Start a new workstation
                    tasks_WS.append([self.Tasks[i].id])
                else:
                    ct_WS[-1] += task_ct  # Add the task to the current workstation
                    tasks_WS[-1].append(self.Tasks[i].id)

            return len(ct_WS), ct_WS, tasks_WS

        # Initial allocation of tasks to workstations
        num_workstations, ct_WS, tasks_WS = allocate_tasks(self.target)

        # # Ensure the number of workstations is odd (optional, depending on use case)
        if num_workstations % 2 != 0:
            # Rebalance tasks to adjust for new number of workstations
            num_workstations, ct_WS, tasks_WS = allocate_tasks((num_workstations)*self.target / (num_workstations+1))

            if num_workstations % 2 != 0:
              # Rebalance tasks to adjust for new number of workstations
              num_workstations, ct_WS, tasks_WS = allocate_tasks((num_workstations+1)*self.target / (num_workstations))

          
        return num_workstations, ct_WS, tasks_WS

    def objectiveR2(self):
        """
        Reward function based on the number of workstations and tasks completed.
        
        Returns:
            reward (float): The calculated reward for the current state.
        """
        if len(self.solution) == len(self.Tasks):
            n, CTs, _ = self.get_nworkstations()
            m, CTsWorkers = self.estimate_WC()

            cost_empty_workstation = -100 if n % 2 == 0 else 1000
            reward = -n * np.var(CTs) - m * np.var(CTsWorkers) - cost_empty_workstation
        else:
            reward = -100000  # Sequence infeasible
        return reward

    def estimate_WC(self):
        """
        Estimate the number of workers required and their cycle times.

        Returns:
            n_workers (int): Number of workers needed.
            ct_Workers (list): Cycle times for each worker.
        """
        n_workers = 0
        total_ct = 0
        parts_done = []
        ct_Workers = []

        for i in self.solution:
            total_ct += sum([float(part.duration) for part in self.Tasks[i].parts if part not in parts_done])
            total_ct += sum([4 for part in self.Tasks[i].parts if part in parts_done])  # Extra time for repeated parts

            parts_done.extend(self.Tasks[i].parts)

            if total_ct >= self.target*(1 + self.tolerance):
                n_workers += 1
                ct_Workers.append(total_ct)
                total_ct = 0

        if n_workers == 1:  # If only one worker, append the final cycle time
            ct_Workers.append(total_ct)

        return len(ct_Workers), ct_Workers

    def sequence_to_scenario(self, indiv):
      Tasks_ID = [task.id for task in self.Tasks]

      _,_, groups = self.get_nworkstations()
      
      scenario = [0 for i in range(len(Tasks_ID))]
      for i in range(len(groups)):
        for j in groups[i]:
          scenario[Tasks_ID.index(j)] = i+1
      return scenario

    def sequence_to_scenario2(self, ant, final=False):
      """
      Convert a sequence of tasks to workstation assignments based on cycle time (CT).
      
      :param ant: List of task indices representing the sequence of tasks.
      :param final: Flag indicating whether this is the final scenario (default: False).
      :return: List representing the workstation assignment for each task.
      """
      total_ct = 0
      scenario = [0 for _ in ant]
      n_workstations = 1

      # Assign tasks to workstations based on cycle time
      for i in ant:
          total_ct += float(self.Tasks[i].CT)
          if total_ct >= (self.target / 2):  # Threshold to start a new workstation
              n_workstations += 1
              total_ct = 0
          scenario[i] = n_workstations

      return scenario

    def update_feasible_actions(self, state, actions):
        """
        Update feasible actions based on task precedencies and shared parts.
        
        :param state: Current task state.
        :param actions: List of potential next actions.
        :return: Updated list of feasible actions.
        """
        # Initialize new actions (exclude actions already in the solution)
        new_actions = [i for i in range(len(self.Tasks)) if i not in self.solution]

        # Filter tasks with shared parts
        feasible_actions = [
            action for action in new_actions 
            if set(self.Tasks[state].parts).intersection(self.Tasks[action].parts)
        ]
        
        # If no feasible actions, return the original new_actions
        return feasible_actions if feasible_actions else new_actions

    def cluster_reward(self):
        """
        Calculate a reward based on task clustering preferences.
        
        :return: Reward value (10 for preferred actions, -10 otherwise).
        """
        if self.solution[-1] in self.prefered_actions(self.solution[-2]) or \
          self.solution[-2] in self.prefered_actions(self.solution[-1]):
            return 10
        return -10

    def get_feasible_actions(self, state):
        """
        Get a list of feasible actions (tasks) that can be performed next based on precedence.
        
        :param state: Current task state.
        :return: List of feasible actions.
        """
        Tasks_ID = [task.id for task in self.Tasks]
        # Unlock tasks whose predecessors are completed
        return [
            self.Tasks.index(task) for task in self.Tasks
            if all(Tasks_ID.index(preced_task) in self.solution for preced_task in task.preced_list)
            and self.Tasks.index(task) not in self.solution
        ]

    def prefered_actions(self, state):
        """
        Get the preferred tasks for a given state based on clustering.
        
        :param state: Current task state.
        :return: List of preferred task indices.
        """
        return list(self.Tasks[state].clusterTasksID)

    def check_precedence(self):
        """
        Check if the latest task in the solution respects the precedence constraints.
        
        :return: 100 if valid, -100 if invalid, 0 if no precedence.
        """
        Tasks_ID = [task.id for task in self.Tasks]
        last_task = self.Tasks[self.solution[-1]]

        if last_task.preced_list:
            if all(Tasks_ID.index(p) in self.solution[:-1] for p in last_task.preced_list):
                return 100
            return -100
        return 0

    def check_forbid(self):
        """
        Check if the last task violates any forbidden task combinations within a workstation.
        
        :return: 10 if valid, -10 if invalid, 0 if no restrictions.
        """
        Tasks_ID = [task.id for task in self.Tasks]
        total_ct = 0
        groups, group = [], []

        # Group tasks based on cycle time
        for i in self.solution:
            total_ct += float(self.Tasks[i].CT)
            group.append(i)
            if total_ct >= (self.target / 2):
                groups.append(group.copy())
                total_ct = 0
                group = []

        groups.append(group.copy())  # Add remaining tasks to a group

        # Check forbidden task combinations within the same workstation
        for group in groups:
            if self.solution[-1] in group and any(
                Tasks_ID.index(f) in group for f in self.Tasks[self.solution[-1]].forbid_list
            ):
                return -10
        return 10

    def check_forbid_full(self, indiv):
        """
        Check if tasks done in the same workstation are physically connected 
        (to avoid multiple isolated subassemblies).
        
        :param indiv: List of task indices representing a sequence.
        :return: Error count (higher means more isolated subassemblies).
        """
        total_ct = 0
        groups, group = [], []

        # Group tasks by workstations
        for i in indiv:
            total_ct += float(self.Tasks[i].CT)
            group.append(i)
            if total_ct >= (self.target / 2):
                groups.append(group.copy())
                total_ct = 0
                group = []

        groups.append(group.copy())  # Add remaining tasks to a group
        errors = sum(self.are_tasks_connected([self.Tasks[i] for i in group]) for group in groups if group != [])
        
        return errors

    def check_precedence_final(self, candidate):
        """
        Check precedence constraints for a complete task sequence.
        
        :param candidate: List of task indices representing a complete sequence.
        :return: Error count (higher means more violations).
        """
        Tasks_ID = [task.id for task in self.Tasks]
        errors = 0

        for i in range(len(candidate) - 1):
            if self.Tasks[candidate[i+1]].preced_list:
                if not all(Tasks_ID.index(p) in candidate[:i+1] for p in self.Tasks[candidate[i+1]].preced_list):
                    errors += 1

        return errors

    def find_feasible_tasks(self):
        """
        For each task, find other feasible tasks that can be performed next based on precedence and shared parts.
        
        :return: Dictionary mapping task IDs to feasible task lists.
        """
        feasible_tasks = {}

        # List of all task IDs
        Tasks_ID = [task.id for task in self.Tasks]

        # Compare each task with others to find shared parts
        for task in self.Tasks:
            task_id = task.id
            task_parts = {part.id for part in task.parts}
            feasible_tasks[task_id] = [
                other_task.id for other_task in self.Tasks if other_task.id != task_id and
                task_parts.intersection({part.id for part in other_task.parts})
            ]
        
        return feasible_tasks
    
    def are_tasks_connected(self, task_list):
      # Start with parts of the first task
      errs = 0

      connected_parts = {part.id for part in task_list[0].parts}
      
      # Check every subsequent task if it shares parts with the current connected parts
      for task in task_list[1:]:
          task_parts = {part.id for part in task.parts}
          if not connected_parts.intersection(task_parts):
              errs = errs+1
          # Add task's parts to connected parts
          connected_parts.update(task_parts)
      
      return errs

    



    def train(self, n_episodes=3000, exploration_prob=1, gamma=0.5, lr=0.001, streamlit_UI=None):

        number_workstations_per_episode = []
        exploration_decreasing_decay = 0.01
        min_exploration_prob = 0.1
        max_workstations = np.sum([float(t.CT) for t in self.Tasks])//self.target + 5
        max_workers = max_workstations
        actions = range(len(self.Tasks))
        states = range(len(self.Tasks))

        q_table = np.zeros((len(states), len(actions)))
        reward = np.full((len(states), len(actions)), -1500)

        pbar = tqdm(range(self.n_episodes), desc="QLearning", colour='green')
        ep = 0
        #reward_global = [0,0,0]
        for e in pbar:
            ep += 1
            done = False
            current_state = 0
            self.solution = []
            elapsed = pbar.format_dict["elapsed"]
            rate = pbar.format_dict["rate"] if pbar.format_dict["rate"] else 0
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            streamlit_UI.my_bar_static_optim.progress(ep/self.n_episodes, text="Estimated Remaining time = "+format_time(remaining, seconds_str=True) + "  |  Rate = " + str(int(rate))+ " it/s" )
            #actions = [self.Tasks.index(task) for task in self.Tasks if not task.preced_list]

            actions = [self.Tasks.index(task) for task in self.Tasks]
            while not done:
                if np.random.uniform(0, 1) < exploration_prob or e<0.2*self.n_episodes:
                  action = random.choice(actions)
                else:
                  action = actions[np.argmax([q_table[current_state, i] for i in actions])]

                next_state, done = self.step(action)
                if len(self.solution) > 1:
                  reward[self.solution[-2], self.solution[-1]] = reward[self.solution[-2], self.solution[-1]]+ self.cluster_reward() + self.check_precedence() -100*self.check_forbid_full(self.solution)
                actions = self.update_feasible_actions(next_state, actions)


                if actions == []:
                  done = True

                current_state = next_state

            # TODO: Integrate a reward per state/action related to starting with longer task.
            global_reward = self.objectiveR2()
            for i, state_i in enumerate(self.solution):
                if i < len(self.solution) - 1:
                    reward[state_i, self.solution[i + 1]] = (reward[state_i, self.solution[i + 1]] + global_reward) / 2
                    #reward[state_i, self.solution[i + 1]] = self.objectiveR2() - reward_global[0]/self.target - np.std(reward_global[1]) - np.std(reward_global[2])
                    q_table[state_i, self.solution[i + 1]] = (1 - lr) * q_table[state_i, self.solution[i + 1]] + lr * (
                                reward[state_i, self.solution[i + 1]] + gamma * max(q_table[:, self.solution[i + 1]]))

            exploration_prob = max(min_exploration_prob, np.exp(-exploration_decreasing_decay * e))
            pbar.set_postfix({"Reward": np.mean(reward)})
            # print("Exploration prob = ", exploration_prob)
            self.session_rewards.append(np.mean(reward))
            number_workstations_per_episode.append(self.get_nworkstations()[0])


        # plt.plot(self.session_rewards)
        # plt.show()
        done = False
        current_state = 0

        self.solution = []
        #actions = self.update_feasible_actions(current_state, [self.Tasks.index(task) for task in self.Tasks if not task.preced_list])
        actions = [self.Tasks.index(task) for task in self.Tasks]
        while not done:
            action = actions[np.argmax([q_table[current_state, a] for a in actions])]
            next_state, done = self.step(action)
            actions = self.update_feasible_actions(next_state, actions)
            current_state = next_state

        indiv = self.sequence_to_scenario(self.solution)

        return indiv, self.get_nworkstations(), self.estimate_WC(), self.session_rewards
    

def read_prepare_mbom(file_path, parts_data):

    if isinstance(file_path, str): 
        ebom = pd.read_xml(file_path, xpath=".//weldings//welding")
        df = pd.read_xml(file_path, xpath=".//weldings//welding")
        df_parts = pd.read_xml(file_path, xpath=".//parts//part")
    else:
        ebom = file_path
        df = file_path
        df_parts = parts_data
    
    # set parameters
    precedence_weight = 0.5 # set the weight for precedence relationships
    parts_weight = 1 - precedence_weight # set the weight for parts similarity
    duration_limit = 10 # set the limit for total duration of tasks in each cluster
    flow_weight = 1 - precedence_weight

    label_encoder = LabelEncoder()
    df_parts["Encoded PartFamily"] = label_encoder.fit_transform(df_parts['PartFamily'])
    part_family_mapping = df_parts.set_index('id')['Encoded PartFamily'].to_dict()
    #parts dict
    parts = {}
    for i, row in df_parts.iterrows():
        part_id = row['id']
        ref = row['ref']
        index = None
        # for i, sublist in enumerate(matched_names):
        #   if ref in sublist:
        #       index = i
        #       break
        try:
            level = int(ebom.loc[ebom['Part_Number'] == row['ref'], 'Level_In_BOM'])
        except:
            level = 0 
        duration = row['cycleTime']
        weight = row['weight']
        fam_part = part_family_mapping[ref]
        parts[part_id] = {'id':part_id, 'ref': ref, 'duration': duration, 'fam_part': fam_part,'level': level, 'weight': weight}

    tasks = {}
    for i, row in df.iterrows():
        task_id = row['id']
        parts_ids = row['assy'].split(';')
        ref_fam=np.sum([parts[part_id]['fam_part'] for part_id in parts_ids])
        levels=[parts[part_id]['level'] for part_id in parts_ids]

        level = min(levels)
        duration = row['cycleTime']
        precedence = row['precedency'].split(';') if pd.notnull(row['precedency']) else []
        forbid = row['forbidden'].split(';') if pd.notnull(row['forbidden']) else []
        tasks[task_id] = {'id':task_id, 'parts': set(parts_ids), 'duration': duration, 'ref_fam': ref_fam,'level': level, 'precedency': precedence, 'forbid': forbid}
    
    updated_tasks = {}
    def add_lower_level_precedencies(tasks):
        levels = {task['level'] for task in tasks.values()}
        levels = list(levels)

        for i in range(len(levels) - 1):
            print(i)
            lower_level = levels[i]
            higher_level = levels[i + 1]

            lower_level_tasks = [task['id'] for task in tasks.values() if task['level'] == lower_level]
            higher_level_tasks = [task for task in tasks.values() if task['level'] == higher_level]
            print(len(lower_level_tasks))
            for task in higher_level_tasks:
                task['precedency'].extend(lower_level_tasks)

        return tasks

    updated_tasks = add_lower_level_precedencies(tasks)
    tree = ET.parse('./assets/inputs/L76 Dual Passive MBOM.xml')
    root = tree.getroot()

    # Update the precedencies in the XML file
    for task_id, task_data in tasks.items():
        task_elem = root.find(".//welding[@id='{}']".format(task_id))
        if task_elem is not None:
            task_elem.set('precedency', ';'.join(task_data['precedency']))

    # Save the modified XML file
    tree.write('./assets/inputs/L76 Dual Passive MBOM_modif.xml')

    def task_distance(task1, task2):
        parts_sim = len(tasks[task1]['parts'].intersection(tasks[task2]['parts'])) / len(tasks[task1]['parts'].union(tasks[task2]['parts']))
        prec_weighted = precedence_weight * (task2 in tasks[task1]['precedency']) + precedence_weight * (task1 in tasks[task2]['precedency'])
        return parts_weight * (1 - parts_sim) + prec_weighted

    # compute pairwise distances
    n_tasks = len(tasks)
    dist_matrix = np.zeros((n_tasks, n_tasks))
    for i, task1 in enumerate(tasks.keys()):
        for j, task2 in enumerate(tasks.keys()):
            if i < j:
                dist = task_distance(task1, task2)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist


    # perform hierarchical clustering
    clustering = linkage(dist_matrix, method='complete')
    
    cluster_labels = fcluster(clustering, 0.5, criterion='distance')
    print(list(cluster_labels))
    clusters_ind = list(cluster_labels)
    for task in tasks.values():
        task['target'] = clusters_ind.pop(0)

    grouped_tasks = {}
    for task_name, task in tasks.items():
        precedence_status = "no precedence" if not task["precedency"] else "has precedence"
        target_value = task["target"]

        if precedence_status not in grouped_tasks:
            grouped_tasks[precedence_status] = {}

        if target_value not in grouped_tasks[precedence_status]:
            grouped_tasks[precedence_status][target_value] = []

        grouped_tasks[precedence_status][target_value].append(task_name)

    sorted_tasks = grouped_tasks
    
    Parts = []

    for part_id, part in parts.items():
        part = Part(
            id=part['id'],
            ref=part['ref'],
            duration=part['duration'],
            weight=part['weight']
        )
        Parts.append(part)

    Parts_ID = [part.id for part in Parts]

    Tasks = []

    for machine_id, machine in tasks.items():
        task = Task(
            id=machine['id'],
            parts=[Parts[Parts_ID.index(part)] for part in machine["parts"]],
            CT=machine['duration'],
            preced_list=machine['precedency'],
            forbid_list=machine['forbid']
        )
        Tasks.append(task)

    Tasks_ID = [task.id for task in Tasks]

    for cluster_id, tasks_id in sorted_tasks['no precedence'].items():
        if len(tasks_id) == 1:
            continue

    else:
        for i in range(len(tasks_id)):
            for task in Tasks:
                if task.id == tasks_id[i]:
                    for task_id in tasks_id[:i] + tasks_id[i+1:]:
                        task.clusterTasksID.append(Tasks_ID.index(task_id))



    for cluster_id, tasks_id in sorted_tasks['has precedence'].items():
        if len(tasks_id) == 1:
            continue

    else:
        for i in range(len(tasks_id)):
            for task in Tasks:
                if task.id == tasks_id[i]:
                    for task_id in tasks_id[:i] + tasks_id[i+1:]:
                        task.clusterTasksID.append(Tasks_ID.index(task_id))
    return Tasks

def run_QL(n_episodes, Tasks, targetCT, tolerance=0.1, streamlit_UI = None ):


    ql = QLearning(n_episodes, Tasks, targetCT, tolerance)
    best_solution, ressource_list, operators_list, session_rewards = ql.train(streamlit_UI=streamlit_UI)


    return best_solution, ressource_list, operators_list, session_rewards

def vizualize_QL_results(self):

    plt.plot(self.session_rewards)
    plt.show()
    return True



def prepare_detailed_line_sim(machines_CT, EOL_stations_CT, manu_op_assignments):
    
    """
    This function build the dataframe to convert the assembly sequence optimization to full assembly line that can be simulated and estimated using AMDO dynamics.

    manu_op_assignments = {
    'M1': (1, 20),
    'M2': (2, 25),
    'M3': (3, 15),
    'EOL1': (4, 10),
    'EOL2': (5, 12) }
    """


    # Base data for machines
    machines = [f'S{i}' for i in range(1, len(machines_CT) + 1)]

    # Base data for EOL stations
    eol_stations = [f'EOL{i}' for i in range(1, len(EOL_stations_CT) + 1)]

    
    # Combine both lists to form the final list of all stations
    all_stations = machines + eol_stations

    # Linear links for machines, each machine points to the next one
    links = [f'[\"{machines[i+1]}\"]' if i < len(machines) - 1 else f'[\"{eol_stations[0]}\"]' for i in range(len(machines))]
    
    # Linear links for EOL stations, last one points to "END"
    eol_links = [f'[\"{eol_stations[i+1]}\"]' if i < len(eol_stations) - 1 else 'END' for i in range(len(eol_stations))]
    
    # Combine machine and EOL links
    links.extend(eol_links)
    
    # Descriptions (placeholder names)
    descriptions = [f'Machine {i} OP' for i in range(1, len(machines_CT) + 1)] + \
                   [f'EOL Station {i}' for i in range(1, len(EOL_stations_CT) + 1)]
    
    # Static values based on the image
    buffer_capacity = [1000] * len(all_stations)
    initial_buffer = [0] * len(all_stations)
    mttf = ['3600*24*1'] * len(all_stations)
    mttr = [3600] * len(all_stations)
    transport_time = [np.nan for i in range(len(all_stations))]
    transport_order = list(range(1, len(all_stations) + 1))
    transporter_id = [1] * len(all_stations)
    fill_central_storage = [False] * len(all_stations)
    
    # Identical station logic for machines
    identical_station = []
    for i in range(len(machines)):
        if (i + 1) % 2 != 0:  # If odd index (M1, M3, etc.)
            if i + 1 < len(machines):
                identical_station.append(machines[i + 1])  # Station after
            else:
                identical_station.append('')  # No station after last
        else:
            identical_station.append(machines[i - 1])  # Station before

    # EOL stations have empty identical station
    identical_station.extend([''] * len(eol_stations))
    
    # Operator IDs and manual time based on manu_op_assignments
    operator_id = []
    manual_time = []
    
    for station in all_stations:
        # If the station is in the assignments, extract operator_id and manual_time
        if station in manu_op_assignments:
            operator_id.append(manu_op_assignments[station][0])  # operator ID
            manual_time.append(manu_op_assignments[station][1])  # manual time
        else:
            operator_id.append(0)  # default operator ID
            manual_time.append(0)  # default manual time

    operator_list = [assignment[0] for assignment in manu_op_assignments.values()]  # Extract operator IDs
    manual_time_list = [assignment[1] for assignment in manu_op_assignments.values()]  # Extract manual times

    print("len ops = ", len(operator_list))
    print("len ops 2 = ", len(manual_time_list))

    # Create the DataFrame
    assembly_line = pd.DataFrame({
        'Machine': all_stations,
        'Description': descriptions,
        'Link': links,
        'Buffer Capacity': buffer_capacity,
        'Initial Buffer': initial_buffer,
        'MTTF': mttf,
        'MTTR': mttr,
        'Transport Time': transport_time,
        'Transport Order': transport_order,
        'Transporter ID': transporter_id,
        'Operator ID': operator_list,
        'Manual Time': manual_time_list,
        'Identical Station': identical_station,
        'Fill central storage':fill_central_storage
    })

    # Combine machine and EOL cycle times
    cycle_times = machines_CT + EOL_stations_CT
    
    # Create the reference DataFrame with 'Ref A' and 'Ref B'
    df_ref = pd.DataFrame({
        'Machine': machines + eol_stations,
        'Ref A': cycle_times
    })
    
    # Add Input row
    input_row = pd.DataFrame({
        'Machine': ['Input'],
        'Ref A': ['1500']
    })
    initial_stock = pd.DataFrame({
        'Machine': ['Initial stock'],
        'Ref A': ['0']
    })
    refill_size = pd.DataFrame({
        'Machine': ['Refill size'],
        'Ref A': ['1500']
    })
    
    # Concatenate the Input row with the rest of the table
    references_config = pd.concat([input_row, initial_stock,refill_size, df_ref], ignore_index=True)
    
    return assembly_line, references_config
    
# Tasks = read_prepare_mbom(".\\assets\\inputs\\240426_EBOM_4391567_01.xlsx")
# run_QL(100000, Tasks, 38, 0.1)


def parts_to_workstation(mbom_data, parts_data, best_solution):
    workstations = {}  # To store parts by workstation
    part_to_workstation = {}  # To track the last workstation where each part was assembled

    # Process workstations in order: start with workstation 1, then workstation 2, and so on
    for workstation in sorted(set(best_solution)):
        # Iterate through the best_solution list and process only tasks for the current workstation
        part_refs = []
        for task_idx, assigned_workstation in enumerate(best_solution):
            if assigned_workstation == workstation:
                # Get the 'assy' column value from mbom_data for the current task
                assy_parts = mbom_data.loc[task_idx, "assy"]

                # Split the 'assy' column by ';' to get individual part references
                part_refs = part_refs + assy_parts.split(';')

                # Initialize the set for the workstation if it doesn't exist
        if workstation not in workstations:
            workstations[workstation] = set()  # Use a set to avoid duplicates

        # Iterate over the part references and add them to the workstation's set
        for part_ref in part_refs:
            # Check if the part has already been assembled in a previous workstation
            if part_ref in part_to_workstation:
                previous_workstation = part_to_workstation[part_ref]

                # If the previous workstation is different, add "SubAssy" from the previous workstation
                if previous_workstation != workstation:
                    subassy_label = f"SubAssy OP{previous_workstation}"
                    if subassy_label not in workstations[workstation]:
                        workstations[workstation].add(subassy_label)
            else:
                # If the part hasn't been assembled yet, add it as a raw part
                workstations[workstation].add(part_ref)

            part_to_workstation[part_ref] =workstation
            # Update the part's latest workstatio
            #part_to_workstation[part_ref] = list(workstations.keys())[-1]

        print("Keys = ", part_to_workstation)
        for part in part_to_workstation:
            part_to_workstation[part] =workstation

    # Display the parts being assembled at each workstation
    for workstation, parts in workstations.items():
        print(f"Workstation {workstation}: Assembling parts: {', '.join(parts)}")

    return workstations




def generate_station_ids(num_machines):
    stations = []
    for i in range(1, num_machines + 1):
        stations.append(f"M{i}-S1")  # Station S1 for Machine i
        stations.append(f"M{i}-S2")  # Station S2 for Machine i
    return stations

import xml.etree.ElementTree as ET
from collections import defaultdict, deque

# Function to extract welding data from the XML
def extract_welding_data(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Store welding data here
    welding_data = []
    precedency_dict = {}
    task_cycle_times = {}
    forbidden_dict = {}

    # Loop through all welding elements
    for welding in root.findall(".//welding"):
        task_id = welding.get('id')
        precedency = welding.get('precedency')
        cycle_time = welding.get('cycleTime')
        forbidden = welding.get('forbidden')

        # Parse precedency and forbidden data if they exist
        precedency_list = precedency.split(';') if precedency else []
        forbidden_list = forbidden.split(';') if forbidden else []

        # Build the dictionaries
        precedency_dict[task_id] = precedency_list
        task_cycle_times[task_id] = float(cycle_time)  # Convert cycleTime to integer
        forbidden_dict[task_id] = forbidden_list

    return precedency_dict, task_cycle_times, forbidden_dict

# Function to perform topological sorting
def topological_sort(tasks, precedency):
    in_degree = defaultdict(int)
    adj_list = defaultdict(list)

    # Build the adjacency list and compute in-degrees
    for task, precedes in precedency.items():
        for t in precedes:
            adj_list[t].append(task)
            in_degree[task] += 1

    # Find tasks with zero in-degree
    zero_in_degree = deque([task for task in tasks if task not in in_degree])

    sorted_tasks = []
    while zero_in_degree:
        current_task = zero_in_degree.popleft()
        sorted_tasks.append(current_task)

        for neighbor in adj_list[current_task]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree.append(neighbor)
    
    if len(sorted_tasks) != len(tasks):
        raise ValueError("There exists a cycle in the task precedency!")


    return sorted_tasks

# Function to allocate tasks to machines
def allocate_tasks_to_machines(tasks, sorted_tasks, task_cycle_times, forbidden_pairs, max_cycle_time):
    machines = []
    machine_loads = []

    for task in sorted_tasks:
        cycle_time = task_cycle_times[task]
        assigned = False

        # Try to assign the task to an existing machine
        for i, machine in enumerate(machines):
            if (all(forbid not in machine for forbid in forbidden_pairs.get(task, [])) and
                machine_loads[i] + cycle_time <= max_cycle_time):
                machine.append(task)
                machine_loads[i] += cycle_time
                assigned = True
                break

        # If task couldn't be assigned, create a new machine
        if not assigned:
            machines.append([task])
            machine_loads.append(cycle_time)

    return machines, machine_loads

# Main function to execute topological sorting and allocation
def schedule_tasks(xml_file, max_cycle_time):
    # Extract precedency, cycle times, and forbidden pairs from XML
    precedency_dict, task_cycle_times, forbidden_dict = extract_welding_data(xml_file)

    # Get the list of tasks from the precedency dictionary
    tasks = list(precedency_dict.keys())

    # Perform topological sorting to respect precedency constraints
    sorted_tasks = topological_sort(tasks, precedency_dict)

    # Allocate tasks to machines, ensuring forbidden pairs are not scheduled together
    machines, machine_loads = allocate_tasks_to_machines(tasks, sorted_tasks, task_cycle_times, forbidden_dict, max_cycle_time)

    return machines, machine_loads

# Example usage: Provide the path to your XML file and maximum cycle time per machine
xml_file = '.\assets\inputs\L76 Dual Passive MBOM.xml'
max_cycle_time = 55  # Define the maximum cycle time for each machine
machines, machine_loads = schedule_tasks(xml_file, max_cycle_time)

# Print the task allocation to machines
for i, machine in enumerate(machines):
    print(f"Machine {i+1}: Tasks: {machine}, Total Cycle Time: {machine_loads[i]}")


import xml.etree.ElementTree as ET
from collections import defaultdict, deque

# Function to extract welding data from the XML
def extract_welding_data(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Store welding data here
    precedency_dict = {}
    task_cycle_times = {}
    forbidden_dict = {}
    task_parts = {}

    # Loop through all welding elements
    for welding in root.findall(".//welding"):
        task_id = welding.get('id')
        precedency = welding.get('precedency')
        cycle_time = welding.get('cycleTime')
        forbidden = welding.get('forbidden')
        parts = welding.get('assy')  # Extract parts to be assembled

        # Parse precedency, forbidden, and parts data if they exist
        precedency_list = precedency.split(';') if precedency else []
        forbidden_list = forbidden.split(';') if forbidden else []
        part_list = parts.split(';') if parts else []  # Parts are separated by ;

        # Build the dictionaries
        precedency_dict[task_id] = precedency_list
        task_cycle_times[task_id] = float(cycle_time)  # Convert cycleTime to integer
        forbidden_dict[task_id] = forbidden_list
        task_parts[task_id] = part_list  # Store multiple parts as a list

    return precedency_dict, task_cycle_times, forbidden_dict, task_parts

# Function to perform topological sorting
def topological_sort(tasks, precedency):
    in_degree = defaultdict(int)
    adj_list = defaultdict(list)

    # Build the adjacency list and compute in-degrees
    for task, precedes in precedency.items():
        for t in precedes:
            adj_list[t].append(task)
            in_degree[task] += 1

    # Find tasks with zero in-degree
    zero_in_degree = deque([task for task in tasks if task not in in_degree])
    sorted_tasks = []

    while zero_in_degree:
        current_task = zero_in_degree.popleft()
        sorted_tasks.append(current_task)

        for neighbor in adj_list[current_task]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree.append(neighbor)

    # Check for cycles in the graph
    if len(sorted_tasks) != len(tasks):
        raise ValueError("There exists a cycle in the task precedency!")

    return sorted_tasks

# Function to check if a task can be added to a machine based on shared parts
def can_add_task_to_machine(task, machine, task_parts):
    machine_parts = set()  # Track parts currently on the machine
    for t in machine:
        machine_parts.update(task_parts[t])  # Add all parts from tasks on the machine

    task_parts_set = set(task_parts[task])  # Convert task's parts to a set
    return not machine_parts.isdisjoint(task_parts_set)  # Check for any shared parts

# Function to allocate tasks to machines
def allocate_tasks_to_machines(tasks, sorted_tasks, task_cycle_times, forbidden_pairs, task_parts, max_cycle_time, tolerance):
    machines = []
    machine_loads = []

    for task in sorted_tasks:
        cycle_time = task_cycle_times[task]
        assigned = False

        # Try to assign the task to an existing machine
        for i, machine in enumerate(machines):
            if (all(forbid not in machine for forbid in forbidden_pairs.get(task, [])) and
                machine_loads[i] + cycle_time <= (1+tolerance)*max_cycle_time and
                can_add_task_to_machine(task, machine, task_parts)):  # Ensure part compatibility
                machine.append(task)
                machine_loads[i] += cycle_time
                assigned = True
                break

        # If task couldn't be assigned, create a new machine
        if not assigned:
            machines.append([task])
            machine_loads.append(cycle_time)

    return machines, machine_loads


# Main function to execute topological sorting and allocation
def schedule_tasks(xml_file, max_cycle_time, tolerance):
    # Extract precedency, cycle times, forbidden pairs, and part dependencies from XML
    precedency_dict, task_cycle_times, forbidden_dict, task_parts = extract_welding_data(xml_file)
    
    # Get the list of tasks from the precedency dictionary
    tasks = list(precedency_dict.keys())
    print("tasks = ", tasks)
    # Perform topological sorting to respect precedency constraints
    sorted_tasks = topological_sort(tasks, precedency_dict)

    # Allocate tasks to machines, ensuring forbidden pairs are not scheduled together
    machines, machine_loads = allocate_tasks_to_machines(tasks, sorted_tasks, task_cycle_times, forbidden_dict, task_parts, max_cycle_time, tolerance)

    resource_list = (len(machine_loads), machine_loads, machines)
    operators_results = (2, [112.0, 104.0])

    best_solution = [0 for _ in range(len(tasks))]
    for i, task in enumerate(tasks):
        for j, machine in enumerate(machines):
            if task in machine:
                best_solution[i] = j+1
                break
            
    

    print("Best solution = ", best_solution)
    print("macghine = ", resource_list)

    return best_solution, resource_list, operators_results, [0 for _ in range(10000)]

# # Example usage: Provide the path to your XML file and maximum cycle time per machine
# xml_file = 'assets\inputs\L76 Dual Passive MBOM.xml'
# max_cycle_time = 55  # Define the maximum cycle time for each machine
# machines, machine_loads = schedule_tasks(xml_file, max_cycle_time)

# # Print the task allocation to machines
# for i, machine in enumerate(machines):
#     print(f"Machine {i+1}: Tasks: {machine}, Total Cycle Time: {machine_loads[i]}")

ref = "2204259X"
print(os.path.exists("assets/inputs/INC1114673PNGFiles/"+ref+".CATPart.png"))
