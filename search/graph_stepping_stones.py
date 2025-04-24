import numpy as np
from itertools import product
try:
    from .utils.prunning import check_crossing, check_distance_to_center, get_id_positions_within_radius
    from .utils.a_star import WeightedGraph
except:
    from utils.prunning import check_crossing, check_distance_to_center, get_id_positions_within_radius
    from search.utils.a_star import WeightedGraph

type Node = tuple[int, int, int, int]

class SteppingStonesGraph(WeightedGraph):
    def __init__(self,
                    stone_positions : np.ndarray,
                    n_feet : int,
                    max_step_size : float,
                    max_foot_displacement : float,
                    ):
        super().__init__()
        self.positions = stone_positions
        self.n_feet = n_feet
        self.max_step_size = max_step_size
        self.max_foot_displacement = max_foot_displacement
        self.n = len(stone_positions)
        
        # hash constant
        self.digit = len(str(self.n))
        self.base_hash = int(10**(self.digit * self.n_feet + 1))
        self.hash_mult = [int(10**((self.n_feet-i-1) * self.digit)) for i in range(self.n_feet)]

    def hash(self, node : Node) -> int:
        hash = self.base_hash + sum([n * m for (n, m) in zip(node, self.hash_mult)]) 
        return hash
    
    def revert_hash(self, hash: int) -> Node:
        hash -= self.base_hash
        node = []
        for mult in self.hash_mult:
            node.append(hash // mult)
            hash %= mult
        return tuple(node)
    
    def get_pos(self, node : Node) -> np.ndarray:
        return self.positions[[node]][0]

    def cost(self, node_from : Node, node_to : Node):
        return np.mean(np.linalg.norm(self.get_pos(node_from) - self.get_pos(node_to), axis=-1))
    
    def get_neighbors(self, node : Node) -> list[Node]:
        # Get the current feet positions
        node_pos = self.get_pos(node)

        # Find reachable Nodes for each foot
        possible_contact_id = get_id_positions_within_radius(node_pos, self.positions, self.max_foot_displacement)

        # Generate all possible combinations of foot placements
        all_states_id = np.array(list(product(*possible_contact_id)), dtype=np.int16)
        all_states = self.positions[all_states_id]

        # Prune states based on average displacement constraint
        reachable = check_distance_to_center(all_states, self.max_step_size)
        
        # Filter out crossing legs configurations
        valid_states_id = all_states_id[reachable]
        valid_states_w = all_states[reachable]
        not_crossing = check_crossing(valid_states_w)

        # Get legal next states after filtering
        legal_next_states = valid_states_id[not_crossing]

        # Add neighbors to the graph
        hash = self.hash(node)
        neighbors = [self.hash(neighbor.tolist()) for neighbor in legal_next_states]
        self.edges[hash] = neighbors
        
        return neighbors
    

if __name__ == "__main__":
    
    # Test on stepping stones
    import numpy as np
    import time
    from itertools import product

    type Node = tuple[int, int, int, int]
                
    N_STONES = 81
    N_FEET = 4
    MAX_STEP_SIZE = 0.3
    MAX_FOOT_DISPLACEMENT = 0.2

    stone_positions = np.random.rand(N_STONES, 3)
    graph = SteppingStonesGraph(stone_positions, N_FEET, MAX_STEP_SIZE, MAX_FOOT_DISPLACEMENT)
    
    # Test hash state
    state = (0, 10, 20, 0)
    hash = graph.hash(state)
    state_revert = graph.revert_hash(hash)
    
    N = 100000
    start_time = time.time()
    for _ in range(N):
        hash = graph.hash(state)
    end_time = time.time()
    print(f"Hashing time: {(end_time - start_time)*1e6/N} micro seconds")

    start_time = time.time()
    for _ in range(N):
        state_revert = graph.revert_hash(hash)
    end_time = time.time()
    print(f"Reverting hash time: {(end_time - start_time)*1e6/N} micro seconds")
           
    print("hash", hash)
    print("revert", state_revert)

    # Test cost
    stateA = (0, 10, 20, 44)
    stateB = (12, 50, 77, 32)
    posA = graph.get_pos(stateA)
    posB = graph.get_pos(stateB)
    c = graph.cost(stateA, stateB)
    print("Cost A-B", c)
    c = graph.cost(stateB, stateB)
    print("Cost B-B", c)
    
    # Test add_neighbors
    n = graph.get_neighbors(state)
    N = 100
    start_time = time.time()
    for _ in range(N):
        n = graph.get_neighbors(state)
    end_time = time.time()
    print(f"Add neighbors time: {(end_time - start_time)*1e3/N} milli seconds")
    print("Number of neighbors", len(n))
    
    import sys
    total_size = sys.getsizeof(next(iter(graph.edges.items())))
    print(f"Total memory size of all nodes: {total_size:.2f}")