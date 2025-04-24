import numpy as np
from itertools import product

try:
    from .utils.abstract import Graph
    
except:
    from utils.abstract import Graph

type Node = str

class ChunkBinaryTree(Graph):
    def __init__(self, n_nodes : int = 64, allow_all_equal : bool = True):
        super().__init__()
        self.allow_all_equal : bool = allow_all_equal
        
        base_float = np.log(n_nodes) / np.log(2)
        self.base = round(base_float)
        self.resolution = self.base
        if self.base != base_float:
            self.n_nodes = int(2 ** self.base)
            print(f"Number of nodes set to a power of 2: ({self.n_nodes}).")
        else:
            self.n_nodes = n_nodes
        
        self.start_node = '1'
        
    def increase_res(self) -> None:
        self.resolution += 1
        
    def decrease_res(self) -> None:
        self.resolution -= 1   
        
    def set_resolution(self, resolution : int) -> None:
        self.resolution = resolution
        print("tree", self.resolution)
        
    @staticmethod
    def get_resolution(node : Node) -> int:
        return int(np.log2(len(node)))
        
    @staticmethod
    def duplicate(node : Node) -> Node:
        """
        Duplicate all binary in a node.
        """
        return "".join((b * 2 for b in node))
        
    @staticmethod
    def can_be_simplified(node : Node) -> bool:
        """
        True is a node can be written with a lower resolution.
        """
        n = len(node)
        if n == 1:
            return False
        return not any((node[i] != node[i+1] for i in range(0, n, 2)))
    
    @staticmethod
    def simplify(node : Node) -> Node:
        if ChunkBinaryTree.can_be_simplified(node):
            return ChunkBinaryTree.simplify("".join((b for b in node[::2])))
        else:
            return node
    
    @staticmethod
    def all_equal(bin_state : str):
        return bin_state.count(bin_state[0]) == len(bin_state)
    
    def filter_child(self, node) -> bool:
        if not self.allow_all_equal and self.all_equal(node):
            return False
        
        if self.can_be_simplified(node):
            return False
        
        return True
        
    def get_neighbors(self, node):
        
        res = self.get_resolution(node)
        if res > self.base or res >= self.resolution:
            return []
        
        if res == self.base:
            if not self.filter_child(node):
                return []
            
            neighbors = list(
                filter(
                    lambda n : n not in self.edges,
                    filter(
                        self.filter_child,
                        (
                            "".join(
                                (b if j != i else str(1 - int(b))
                                for j, b in enumerate(node))
                            )
                            for i in range(len(node))
                        )
                    )
                )
            )
            return [node] + neighbors
        
        
        
        node_duplicate = self.duplicate(node)
        if not self.allow_all_equal and self.all_equal(node):
            parent = []
        elif not self.can_be_simplified(node):
            parent = [node]
        else:
            parent = [node_duplicate]
            
        neighbors = list(
                filter(
                    self.filter_child,
                    (
                        "".join(
                            (b if j != i else str(1 - int(b))
                            for j, b in enumerate(node))
                        )
                        for i in range(len(node))
                    )
                )
        )
        neighbors_duplicate = list(filter(
            self.filter_child,
            (
                "".join(
                    (b if j != i else str(1 - int(b))
                    for j, b in enumerate(node_duplicate))
                )
                for i in range(len(node_duplicate))
            )
            )
        )
        return parent + neighbors + neighbors_duplicate

    
class GaitParallelGraph(Graph):
    def __init__(self, n_eeff : int, n_nodes_per_eeff : int):
        super().__init__()
        # Create one separate tree for each end effector
        # A node is a tupple of all tree binary state
        self.n_nodes_per_eeff = n_nodes_per_eeff
        self.trees = [ChunkBinaryTree(n_nodes_per_eeff, False) for _ in range(n_eeff)]
        self.base = self.trees[0].base
        self.start_node = tuple(tree.start_node for tree in self.trees)
        
    def increase_res(self) -> None:
        for tree in self.trees:
            tree.decrease_res()
    
    def decrease_res(self) -> None:
        for tree in self.trees:
            tree.increase_res()
            
    def set_resolution(self, resolution : int) -> None:
        for tree in self.trees:
            tree.set_resolution(resolution)
        
    def get_neighbors(self, node : Node) -> list[Node]:
        neighbors = list(product(*(map(lambda children : children if children else n, tree.get_neighbors(n)) for tree, n in zip(self.trees, node))))
        return neighbors
    
if __name__ == "__main__":
    import time
    
    #########
    ######### ChunkBinaryTree
    #########
    N_NODES = 4
    tree = ChunkBinaryTree(N_NODES, allow_all_equal=False)
    
    print("Test neighbors")
    n = [tree.start_node]
    tree.set_resolution(3)
    for i in range(4):
        print("Stage", i)
        node = n[-1]
        print("node", node)
        n = tree.neighbors(node)
        print("neighbors", n)
        
    print("node", '1000')
    n = tree.neighbors('1000')
    print("neighbors", n)
    
        
    print("node", '1100')
    n = tree.neighbors('1100')
    print("neighbors", n)
    print("node", '10')
    n = tree.neighbors('10')
    print("neighbors", n)
    # timings
    
    N_NODES = 8
    tree = ChunkBinaryTree(N_NODES, allow_all_equal=True)
    print("Time add all leaves")
    
    all_leaves = []
    visited = set()
    from collections import deque
    def add_all_leaves(node):

        queue = deque([tree.start_node])

        while queue:
            node = queue.popleft()
            if node in all_leaves or node in visited:
                continue
            
            neighbors = tree.neighbors(node)
            visited.add(node)

            if not neighbors:
                all_leaves.append(node)
            else:
                if node in neighbors:
                    all_leaves.append(node)
                queue.extend(neighbors)
    
    start = time.time()
    add_all_leaves(tree.start_node)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    print(f"Number of leaves: {len(all_leaves)}")
    import sys
    total_size = sys.getsizeof(tree.edges)
    total_size_mb = total_size / (1024)
    print(f"Total memory size of all nodes: {total_size_mb:.2f} KB")