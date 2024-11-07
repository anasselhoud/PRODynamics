import simpy

from utils import CentralStorage, Machine

# Class to create the simpy Environment and set up the central storage. 
class Manuf:
    def __init__(self) -> None:
        self.env = simpy.Environment()
        self.dev_mode = True
        self.config_cs = {
            'front': [
                {'allowed_ref' : ['ref A', 'ref B'],
                'capacity': 336},
                {'allowed_ref' : ['ref A', 'ref B', 'ref C'],
                'capacity': 3}],

            'back': [
                {'allowed_ref' : ['ref A', 'ref B'],
                'capacity': 376}]
            }
        self.central_storage = CentralStorage(self.env, self, self.config_cs)
        self.robots_list = []
        self.machines_list = [
            Machine(self, self.env, 'M1', 'Test_1', {"fatigue_model":{"enabled": False}}),
            Machine(self, self.env, 'M2', 'Test_2', {"fatigue_model":{"enabled": False}}),
        ]
        

# Tests are below. Run the file and look at the terminal (details are printed for each step).
m = Manuf()
print(m.central_storage)
m.central_storage.available_spot('ref A')
m.central_storage.available_spot('ref C')
m.central_storage.available_spot('ref D')

m.central_storage.put({'name': 'ref C', 'route': (m.machines_list[0], m.machines_list[1]), 'status': 'OK'})
print(m.central_storage)

m.central_storage.available_ref_by_route(m.machines_list[0], m.machines_list[0])
m.central_storage.available_ref_by_route(m.machines_list[0], m.machines_list[1])

m.central_storage.available_ref('ref A')
m.central_storage.available_ref('ref C')

m.central_storage.available_spot('ref C')
m.central_storage.put({'name': 'ref C', 'route': (m.machines_list[0], m.machines_list[1]), 'status': 'KO'})
print(m.central_storage)

m.central_storage.available_ref_by_route(m.machines_list[1], m.machines_list[0])
m.central_storage.available_ref_by_route(m.machines_list[0], m.machines_list[1])

m.central_storage.get('ref C')
print(m.central_storage)

print("Add 337 'ref A' to fill the first block and add 1 ref A to the next block.")
for _ in range(337):
    m.central_storage.put({'name': 'ref A', 'route': (m.machines_list[0], m.machines_list[1]), 'status': 'test'})
print(m.central_storage)

m.central_storage.available_spot('ref C')
m.central_storage.put({'name': 'ref C', 'route': (m.machines_list[1], m.machines_list[0]), 'status': 'OK'})
print(m.central_storage)

print("Remove 336 'ref A' to empty the first block but keep 1 ref A in the second block.")
for _ in range(336):
    m.central_storage.get('ref A')
print(m.central_storage)

m.central_storage.available_spot('ref C')

m.central_storage.available_ref_by_route(m.machines_list[1], m.machines_list[0])
m.central_storage.get_by_destination(m.machines_list[0])
print(m.central_storage)

m.central_storage.available_ref_by_route(m.machines_list[0], m.machines_list[1])
m.central_storage.get_by_destination(m.machines_list[1])
m.central_storage.available_ref_by_route(m.machines_list[1], m.machines_list[0])
print(m.central_storage)
