import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


class QLearning:

    def __init__(self, n_episodes, Tasks, targetCT, tolerance):
        self.Tasks = Tasks
        self.target = targetCT
        self.tolerance = tolerance
        self.solution = []
        self.session_rewards = []
        self.n_episodes = n_episodes

    def step(self, action):
        done = False
        new_state = action
        self.solution.append(new_state)

        if len(self.solution) == len(self.Tasks):
            done = True
        return new_state, done

    def split(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def precedence_graph(self):
        preced_graph = []

        # No Precedence Restrictions
        preced_graph.append([task for task in self.Tasks if not task.preced_list])
        for task in self.Tasks:
            # A precedence restriction with one of the first group
            for preced_task in task.preced_list:
                if preced_task in preced_graph:
                    preced_graph.append(task)

    def get_nworkstations(self):
        total_ct = 0
        ct_WS = [0]
        tasks_WS = [[]]

        for i in self.solution:
          if ct_WS[-1]+float(self.Tasks[i].CT) > (1+self.tolerance)*(self.target / 2):
            ct_WS.append(float(self.Tasks[i].CT))
            tasks_WS.append([self.Tasks[i].id])
          else:
            ct_WS[-1]+=float(self.Tasks[i].CT)
            tasks_WS[-1].append(self.Tasks[i].id)

        return len(ct_WS), ct_WS, tasks_WS

    def objectiveR2(self):
        """
        Reward = -1 * (Number of Workstations Used) * (Number of Tasks Completed)
        This reward function takes into account both the number of tasks completed and the number of workstations used up to
        the current time. By multiplying the number of tasks completed with the negative of the number of workstations used,
        the reward function encourages the agent to complete as many tasks as possible while minimizing the number of workstations
        used.
        The partial reward can be calculated at each step of the assembly process, after each task is completed. The agent can
        then use this partial reward to update its policy and choose the next task to be completed based on the updated policy.

        Note that this partial reward function assumes that all tasks have the same complexity and require the same amount of time and resources. If this is not the case, you may need to modify the reward function to take into account the specific characteristics of each task.
        """
        if len(self.solution) == len(self.Tasks):
          n, CTs, _ = self.get_nworkstations()
          m, CTsWorkers = self.estimate_WC()
          cost_empty_workstation = -10 if n % 2 == 0 else 10

          reward =-n*np.var(CTs)-m*np.var(CTsWorkers)-cost_empty_workstation-1000*self.check_forbid_full(self.solution)
          #reward =-n*np.var(CTs)-m*np.var(CTsWorkers)-cost_empty_workstation

        else:
          reward = -100000 #Sequence infeasible

        #reward = (-n-n*np.std(CTs)-m*np.std(CTsWorkers)+self.cluster_reward()+self.check_precedence()+self.check_forbid())*len(self.solution)
        #reward = (-n-m+self.cluster_reward()+self.check_forbid()+self.check_precedence()+cost_empty_workstation)*len(self.solution)
        #reward = (-n-m+self.check_forbid())*len(self.solution)
        return reward
    
    def objectiveR2_final(self):
        """
        Reward = -1 * (Number of Workstations Used) * (Number of Tasks Completed)
        This reward function takes into account both the number of tasks completed and the number of workstations used up to
        the current time. By multiplying the number of tasks completed with the negative of the number of workstations used,
        the reward function encourages the agent to complete as many tasks as possible while minimizing the number of workstations
        used.
        The partial reward can be calculated at each step of the assembly process, after each task is completed. The agent can
        then use this partial reward to update its policy and choose the next task to be completed based on the updated policy.

        Note that this partial reward function assumes that all tasks have the same complexity and require the same amount of time and resources. If this is not the case, you may need to modify the reward function to take into account the specific characteristics of each task.
        """
        n, CTs, _ = self.get_nworkstations()
        m, CTsWorkers = self.estimate_WC()
        if CTsWorkers == []:
          m,CTsWorkers = 0, [0]
        cost_empty_workstation = -100 if len(CTs) % 2 == 0 else 100

        reward = (-n*n-n*np.std(CTs)-n*np.max(CTs)-m*np.std(CTsWorkers)-n*cost_empty_workstation+self.cluster_reward()+ self.check_precedence())*len(self.solution)
        print("CT Machines = " + str(np.std(CTs))  + " - CT workers = " + str(np.std(CTsWorkers)))
        print("Cluster reward = " + str(self.cluster_reward())  + " - Check precedence = " + str(self.check_precedence()))
        #reward = (self.cluster_reward() + self.check_precedence() -n)*len(self.solution)
        return reward

    def estimate_WC(self):
        n_workers = 1
        total_ct = 0
        parts_done = []
        ct_Workers = []
        for i in self.solution:
            total_ct += sum([float(part.duration) for part in self.Tasks[i].parts if part not in parts_done])
            total_ct += sum([3 for part in self.Tasks[i].parts if part in parts_done])
            for p in self.Tasks[i].parts:
                parts_done.append(p)

            if total_ct >= self.target:
                n_workers += 1
                ct_Workers.append(total_ct)
                total_ct = 0

        if n_workers == 1:
            ct_Workers.append(total_ct)
        return n_workers, ct_Workers

    def sequence_to_scenario(self, indiv):
      Tasks_ID = [task.id for task in self.Tasks]

      total_ct = 0
      groups = []
      group = []
      for i in indiv:
          total_ct += float(self.Tasks[i].CT)
          group.append(i)
          if total_ct >= (1+self.tolerance)*(self.target / 2):
              groups.append(group.copy())
              total_ct = 0
              group=[]

      if total_ct > 0:
        if total_ct <= 0.3*(self.target / 2):
          groups[-1] = groups[-1] + group
        else:
          groups.append(group.copy())

      scenario = [0 for i in range(len(Tasks_ID))]
      for i in range(len(groups)):
        for j in groups[i]:
          scenario[j] = i+1
      return scenario

    def sequence_to_scenario2(self, ant, final=False):
      total_ct = 0
      scenario = [0 for i in range(len(ant))]
      n_workstations = 1
      for i in ant:
            total_ct += float(self.Tasks[i].CT)
            if total_ct >= (self.target / 2):
                n_workstations += 1
                total_ct = 0
            scenario[i] = n_workstations

      return scenario
    
    def update_feasible_actions(self, state, actions):
        '''
        The only restriction is that a certain node x is not allowed to be vis-ited unless all the predecessor nodes are visited prior to n.
        '''

        # We remove last action from the possible actions of next step
        #new_actions = list(actions).copy()
        new_actions = list(range(len(self.Tasks)))
        for act in self.solution:
          new_actions.remove(act)
        # We add all new possible actions to it now (tasks that get now unlocked)
        # Tasks_ID = [task.id for task in self.Tasks]

        # new_unlocked = [self.Tasks.index(task) for task in self.Tasks if (
        #             all(Tasks_ID.index(item) in self.solution for item in task.preced_list) and self.Tasks.index(
        #         task) not in self.solution and self.Tasks.index(task) not in new_actions)]
        # for task_ind in new_unlocked:
        #     new_actions.append(task_ind)

        # print("Old actions = ", new_actions)
        # Keep only the one mehcanically feasible
        new_actions_all = []
        for action in new_actions:
            other_task_parts = set(self.Tasks[action].parts)
            # Check if there is any common part
            if set(self.Tasks[state].parts).intersection(other_task_parts):
                new_actions_all.append(action)

        if new_actions_all == []:
          new_actions_all = new_actions.copy()
        return new_actions_all

    def find_feasible_tasks(self, state, actions, tasks, task_id):
        """
        For each task, find the list of other tasks that are feasible to assemble with (based on shared parts).
        
        :param task_dict: Dictionary of task definitions.
        :return: Dictionary where each task points to a list of feasible tasks.
        """
        new_actions = list(actions).copy()
        new_actions.remove(state)
        Tasks_ID = [task.id for task in self.Tasks]

        feasible_tasks = {}

        feasible_tasks[task_id] = []
        
        # Get the parts associated with the current task
        task_parts = set(tasks[task_id]['parts'])
        
        # Compare with all other tasks to find common parts
        for other_task in tasks:
            if other_task.id == task_id:
                continue  # Skip comparing the task with itself
            
            other_task_parts = set(other_task.parts)
            
            # Check if there is any common part
            if task_parts.intersection(other_task_parts):
                feasible_tasks[task_id].append(other_task.id)
        
        return feasible_tasks
    
    def check_precedence(self):
      Tasks_ID = [task.id for task in self.Tasks]
      if self.Tasks[self.solution[-1]].preced_list != []:
        if not all(Tasks_ID.index(item) in self.solution[:-1] for item in self.Tasks[self.solution[-1]].preced_list):

          return -100
        else:
          return 100
      else:
        return 0

    def check_forbid(self):
      Tasks_ID = [task.id for task in self.Tasks]

      total_ct = 0
      groups = []
      group = []
      for i in self.solution:
          total_ct += float(self.Tasks[i].CT)
          group.append(i)
          if total_ct >= (self.target / 2):
              groups.append(group.copy())
              total_ct = 0
              group=[]

      groups.append(group.copy())

      if self.Tasks[self.solution[-1]].forbid_list != []:
        for i in range(len(groups)):
          if self.solution[-1] in groups[i]:
            if all(Tasks_ID.index(item) not in groups[i] for item in self.Tasks[self.solution[-1]].forbid_list):
              return 10
            else:
              return -10
      else:
        return 0
      

    def check_forbid_full(self, indiv):
      Tasks_ID = [task.id for task in self.Tasks]

      total_ct = 0
      groups = []
      group = []
      for i in indiv:
          total_ct += float(self.Tasks[i].CT)
          group.append(i)
          if total_ct >= (self.target / 2):
              groups.append(group.copy())
              total_ct = 0
              group=[]

      groups.append(group.copy())
      j=0
      for i in range(len(groups)):
        if groups[i] != []:
          j = j + self.are_tasks_connected([self.Tasks[ind] for ind in groups[i]])
        else:
          j = j + 1

      return j

    def check_precedence_full_sequence(self):
      Tasks_ID = [task.id for task in self.Tasks]
      j=0
      for i in range(len(self.solution)-1):
        if self.Tasks[self.solution[i+1]].preced_list != []:
          if not all(Tasks_ID.index(item) in self.solution[:i+1] for item in self.Tasks[self.solution[i+1]].preced_list):
            j+=1
      return j

    def check_precedence_final(self, candidate):
      Tasks_ID = [task.id for task in self.Tasks]
      j=0
      for i in range(len(candidate)-1):
        if self.Tasks[candidate[i+1]].preced_list != []:
          if not all(Tasks_ID.index(item) in candidate[:i+1] for item in self.Tasks[candidate[i+1]].preced_list):
            j+=1
      return j
    
    
    
    def find_feasible_tasks(self):
      """
      For each task, find the list of other tasks that are feasible to perform next, 
      based on task dependencies (precedency) and shared parts.
      
      :return: Dictionary where each task points to a list of feasible tasks.
      """
      feasible_tasks = {}
      
      # List of all task IDs
      Tasks_ID = [task.id for task in self.Tasks]
      
      # Iterate over each task in self.Tasks
      for task in self.Tasks:
          task_id = task.id
          feasible_tasks[task_id] = []  # Initialize the list of feasible tasks for this task
          
          # Get the parts associated with the current task
          task_parts = set([part.id for part in task.parts])
          
          # Compare with all other tasks to find shared parts and unlocked tasks
          for other_task in self.Tasks:
              other_task_id = other_task.id
              if other_task_id == task_id:
                  continue  # Skip comparing the task with itself
              
              # Check if all predecessor tasks of the other task are completed
              # if all(Tasks_ID.index(preced_task) in self.solution for preced_task in other_task.preced_list):
              other_task_parts = set([part.id for part in other_task.parts])
              
              # Check if there are common parts between the current task and the other task
              if task_parts.intersection(other_task_parts):
                  feasible_tasks[task_id].append(other_task_id)
      
      ##TODO: multiply number of errors by a negative reward
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

    



    def train(self, n_episodes=3000, exploration_prob=1, gamma=0.5, lr=0.001):

        number_workstations_per_episode = []
        exploration_decreasing_decay = 0.01
        min_exploration_prob = 0.1
        max_workstations = np.sum([float(t.CT) for t in self.Tasks])//self.target + 5
        max_workers = max_workstations
        actions = range(len(self.Tasks))
        states = range(len(self.Tasks))

        q_table = np.zeros((len(states), len(actions)))
        reward = np.full((len(states), len(actions)), -1000)

        pbar = tqdm(range(self.n_episodes), desc="QLearning", colour='green')

        reward_global = [0,0,0]
        for e in pbar:
            done = False
            current_state = 0
            self.solution = []

            #actions = [self.Tasks.index(task) for task in self.Tasks if not task.preced_list]

            actions = [self.Tasks.index(task) for task in self.Tasks]
            while not done:
                if np.random.uniform(0, 1) < exploration_prob or e<0.2*self.n_episodes:
                  action = random.choice(actions)
                else:
                  action = actions[np.argmax([q_table[current_state, i] for i in actions])]

                next_state, done = self.step(action)
                if len(self.solution) > 1:
                  reward[self.solution[-2], self.solution[-1]] = reward[self.solution[-2], self.solution[-1]]+ self.cluster_reward() + self.check_precedence()
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


        plt.plot(self.session_rewards)
        plt.show()
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

        return indiv, 0, self.get_nworkstations()[0], self.get_nworkstations()[2], 0
    

def read_prepare_mbom(file_path):

    ebom = pd.read_excel(file_path, 0, skiprows=5)
    df = pd.read_xml('..\\assets\inputs\L76 Dual Passive MBOM.xml', xpath=".//weldings//welding")
    df_parts = pd.read_xml('..\\assets\inputs\L76 Dual Passive MBOM.xml', xpath=".//parts//part")
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
        print(str(ebom.loc[ebom['Part_Number'] == row['ref'], 'Level_In_BOM']))
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
    
    return True

def run_QL(n_episodes, Tasks, targetCT, tolerance):
    ql = QLearning(n_episodes, Tasks, targetCT=38, tolerance=0.2)
    best_solution, best_ct, n_machines,_,  n_workers = ql.train()

    print("best solution = ", best_solution)
    print("best solution sequence = ", ql.solution)
    print("n machines = ", n_machines)
    print("reward = ", ql.get_nworkstations())
    print("Precedence = ", ql.check_precedence_final(ql.solution))


    return True

def vizualize_QL_results():
    return True