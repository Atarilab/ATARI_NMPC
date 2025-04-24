import numpy as np
from itertools import product
from typing import Tuple, List
import math
try:
    from .utils.abstract import Graph
    
except:
    from utils.abstract import Graph
    
class GraphPhasePatchBase(Graph):
    def __init__(self,
                 n_nodes : int,
                 node_per_phase : int,
                 n_cnt : int,
                 patches : List[int],
                 min_in_cnt : int = 0):
        super().__init__()
        self.n = n_nodes
        self.n_per_phase = node_per_phase
        self.n_cnt = n_cnt
        self.patches = patches
        self.min_in_cnt = min_in_cnt
        self.n_phases = math.ceil(self.n / self.n_per_phase) - 1
        
        # All the possible contact combinations
        self.cnt_neighbors = list(product((0, 1), repeat=self.n_cnt))
        
        # All the patch combinations depending on the number
        # of contacts
        self.patch_neighbors = [
            list(product(self.patches, repeat=i))
            for i in range(self.n_cnt+1)
        ]
        # All possible phases
        self.all_possible_phases = [
            (seq, patch)
            # Get all possible contact status
            for seq in self.cnt_neighbors
            # Get all patches for the given number of contacts
            for patch in self.patch_neighbors[sum(seq)]
            if sum(seq) >= self.min_in_cnt
        ]
    
    @staticmethod
    def are_phases_consistant(cnt_a, cnt_b, patch_a, patch_b) -> bool:
        """
        Check if contact patches do not change if contact is kept
        between two phases. 
        """
        i_patch_a, i_patch_b = 0, 0
        for (a, b) in zip(cnt_a, cnt_b):
            # If both in contacts and if patches are not the same
            if (a and b):
                if patch_a[i_patch_a] != patch_b[i_patch_b]:
                    return False
            
            i_patch_a += a
            i_patch_b += b
        return True
    
class GraphPhasePatchWithPath(GraphPhasePatchBase):
    
    def get_neighbors(self, node):
        i_phase = len(node)
        
        if i_phase >= self.n_phases:
            return []
        
        elif i_phase > 0:

            cnt, patch = node[-1]
            # if phase with at least one contact
            if sum(cnt) > 0:
                # filter out next phases that keep contact but change patch
                return [
                    (*node, phase)                        
                    for phase in self.all_possible_phases
                    if GraphPhasePatchBase.are_phases_consistant(
                        cnt, phase[0], patch, phase[1]
                    )
                    ]
            else:
                # if flight phase, all possible phases are kept
                return [(*node, phase) for phase in self.all_possible_phases]
        
        else:
            return  [((phase), ) for phase in self.all_possible_phases]
        
class GraphPhasePatch(GraphPhasePatchBase):
    
    def get_neighbors(self, node):
        if node:
            i_phase, cnt, patch = node
            
            if i_phase >= self.n_phases:
                return []
            

            # if phase with at least one contact
            if sum(cnt) > 0:
                # filter out next phases that keep contact but change patch
                return [
                    (i_phase+1, *phase)
                    for phase in self.all_possible_phases
                    if GraphPhasePatchBase.are_phases_consistant(
                        cnt, phase[0], patch, phase[1]
                    )
                ]
            else:
                # if flight phase, all possible phases are kept
                return [(i_phase+1, *phase) for phase in self.all_possible_phases]
            
        else:
            # first node
            return  [(0, *phase) for phase in self.all_possible_phases]

        
class GraphPhasePatchWithPos(GraphPhasePatchBase):
    def __init__(self,
                 n_nodes,
                 node_per_phase,
                 n_cnt,
                 goal_patches,
                 center_patches,
                 min_in_cnt : int = 0,
                 ):
        self.pos = np.array(center_patches)
        patches = list(range(len(center_patches)))
        super().__init__(n_nodes, node_per_phase, n_cnt, patches, min_in_cnt)
        self.goal_node = (self.n_phases, (1,) * self.n_cnt, goal_patches)
        
    def is_crossing_legs(self, cnt_b, cnt_a, patch_b, patch_a) -> bool:
        i_patch_a = np.cumsum(cnt_a) - 1
        i_patch_b = np.cumsum(cnt_b) - 1
        
        patch_loc_transition = np.zeros((len(cnt_a), 3))
        cnt_transition = np.zeros(len(cnt_a))
        for i, (a, b, i_a, i_b) in enumerate(zip(cnt_a, cnt_b, i_patch_a, i_patch_b)):
            if a:
                patch_loc_transition[i] = self.pos[patch_a[i_a]]
                cnt_transition[i] = 1
            elif b:
                patch_loc_transition[i] = self.pos[patch_b[i_b]]
                cnt_transition[i] = 1
        
        # eeff_a, eeff_b, axis (x, y)
        rules = [
            [0,1,1],
            [2,3,1],
            [0,2,0],
            [1,3,0],
            ]
        
        for rule in rules:
            cnt_a, cnt_b, axis = rule
            if cnt_transition[cnt_a] and cnt_transition[cnt_b]:
                if patch_loc_transition[cnt_a][axis] < patch_loc_transition[cnt_b][axis]:
                    return True

        return False
    
    def enough_contact_with_goal(self, n_phase_remaining : int, cnt, patch) -> bool:
        if self.min_in_cnt == 0:
            return True
        
        cnt_with_goal = sum(1 for patch_id in patch if patch_id in self.goal_node[2])
        sw = self.n_cnt - sum(cnt)
        if self.n_cnt - (cnt_with_goal + sw) > (n_phase_remaining - 1) * (self.n_cnt - self.min_in_cnt):
            return False
        
        return True

    def get_neighbors(self, node):
        if node:
            i_phase, cnt, patch = node
            n_phase_remaining = self.n_phases - 1 - i_phase
            # Before the last phase, allow only valid transitions from the last node that could lead to goal
            if n_phase_remaining == 1:
                return [
                    (i_phase+1, *phase)
                    for phase in self.all_possible_phases
                    if (
                        GraphPhasePatchBase.are_phases_consistant(
                            cnt, phase[0], patch, phase[1]
                        ) and
                        GraphPhasePatchBase.are_phases_consistant(
                            phase[0], self.goal_node[1], phase[1], self.goal_node[2]
                        ) and
                        not self.is_crossing_legs(
                            cnt, phase[0], patch, phase[1]
                        ) and
                        not self.is_crossing_legs(
                            phase[0], self.goal_node[1], phase[1], self.goal_node[2]
                        ) and
                        self.enough_contact_with_goal(
                        n_phase_remaining, phase[0], phase[1]
                        )
                    )
                ]
            # Last phase has to be goal
            elif n_phase_remaining == 0:
                return [self.goal_node]
            
            # End of the plan reached
            elif n_phase_remaining < 0:
                return []

            # if phase with at least one contact
            if sum(cnt) > 0:
                # filter out next phases that keep contact but change patch
                return [
                    (i_phase+1, *phase)
                    for phase in self.all_possible_phases
                    if GraphPhasePatchBase.are_phases_consistant(
                        cnt, phase[0], patch, phase[1]
                    ) and not self.is_crossing_legs(
                        cnt, phase[0], patch, phase[1]
                    ) and self.enough_contact_with_goal(
                        n_phase_remaining, phase[0], phase[1]
                    )
                ]
            else:
                # if flight phase, all possible phases are kept
                return [(i_phase+1, *phase) for phase in self.all_possible_phases if not self.is_crossing_legs(cnt, phase[0], patch, phase[1])]
            
        else:
            # first node
            return  [(0, *phase) for phase in self.all_possible_phases if not self.is_crossing_legs((0, 0, 0, 0), phase[0], (), phase[1])]


if __name__ == "__main__":
    import sys
    import timeit

    n_nodes = 10
    node_per_phase = 2
    n_cnt = 4
    patches = [0, 1, 2]

    graph_phase_patch = GraphPhasePatch(n_nodes, node_per_phase, n_cnt, patches)
    # cnt_a, cnt_b = (0, 1, 0, 1), (0, 0, 0, 1)
    # patch_a, patch_b = (0, 2), (1, )
    # print("Consistant phases")
    # print(GraphPhasePatchBase.are_phases_consistant(cnt_a, cnt_b, patch_a, patch_b))
    # graph_phase_patch = GraphPhasePatchWithPath(n_nodes, node_per_phase, n_cnt, patches)
    
    print("Number of phases", graph_phase_patch.n_phases)
    print("Contact neighbors")
    print(graph_phase_patch.cnt_neighbors)
    print("Patch neighbors")
    print(graph_phase_patch.patch_neighbors)
    print("All possible neighbors")
    print(graph_phase_patch.all_possible_phases)
    
    neighbors = graph_phase_patch.get_neighbors(())
    
    phase = neighbors[-n_cnt*2]
    print("Neighbors of", phase)
    neighbors = graph_phase_patch.get_neighbors(phase)
    print(neighbors)
    print("Number of neighbors", len(neighbors))
    print("Data size", sys.getsizeof(neighbors))
    N = 100
    exec_time = timeit.timeit(lambda : [graph_phase_patch.get_neighbors(phase) for phase in neighbors], number=N)
    print("Time taken to get all children: {:.6f} ms".format(exec_time * 1000 / N))


    phase = (0, (0, 1, 1, 1), (0, 1, 2))
    print("Neighbors of", phase)
    neighbors = graph_phase_patch.get_neighbors(phase)
    n = (1, (0, 0, 1, 1), (0, 1))
    print(n in neighbors)
    print(neighbors)
    print("Number of neighbors", len(neighbors))
    print("neighbors[2]", neighbors[2])
