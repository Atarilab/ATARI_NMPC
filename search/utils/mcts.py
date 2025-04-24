import random
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict
from typing import Any
from tqdm import tqdm

from .abstract import Graph

Node = Any

class MCTSBase(ABC):
    def __init__(self, graph : Graph, C: float = 1.0):
        self.graph = graph
        self.value_visit: dict[Node, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
        self.current_search_path: list[Node] = []
        self.C = C
        self.it = 0
        
    def heuristic_bias(self, node: Node) -> float:
        """
        Heuristic bias for the node.
        This can be replaced with a more complex heuristic if needed.
        """
        return 0.0

    def UCB(self, node: Node, parent_visit : int) -> float:
        """
        Upper Confidence Bound (UCB) formula for MCTS.
        UCB = Q/N + C * sqrt(log(N_parent) / N)
        """
        if node not in self.value_visit:
            return float("inf")  # Encourage exploration of unexplored nodes

        q, n = self.value_visit[node]
        if n == 0:
            return float("inf")

        return ((q + self.heuristic_bias(node))/ n) + self.C * np.sqrt(np.log(parent_visit) / (n + 1e-6))
    
    def is_leaf(self, node : Node) -> bool:
        """
        Returns true if a node is terminal.
        """
        # No neighbors
        return len(self.graph.get_neighbors(node)) == 0

    def best_child(self, node: Node) -> Node:
        """
        Selects the best child node based on UCB.
        """
        if self.is_leaf(node):
            return node
        n = self.value_visit[node][1]
        children = self.graph.get_neighbors(node)
        np.random.shuffle(children)
        return max(children, key=lambda child: self.UCB(child, n))

    def select(self, node: Node) -> Node:
        """
        Traverse the tree using UCB until an unexplored node is found.
        """
        self.current_search_path = [node]

        while (self.value_visit[node][1] > 0  and not self.is_leaf(node)):
            best_child = self.best_child(node)
            if best_child is None:  # Ensure there's a valid child to move to
                break
            node = best_child
            self.current_search_path.append(node)

        return node  # Return the first unexplored or leaf node
    
    @abstractmethod
    def evaluate(self, simulation_path : list[Node]) -> float:
        """
        Compute the reward associated with a full simulation path.
        """
        pass
    
    def rollout_policy(self, node: Node) -> Node:
        """
        Selects a random child node during rollout.
        """
        children = self.graph.get_neighbors(node)
        return random.choice(children) if children else node  # Return node itself if terminal

    def simulate(self, node: Node) -> float:
        """
        Perform a random rollout from the selected node to estimate value.
        """
        simulation_path = []
        while not self.is_leaf(node):
            node = self.rollout_policy(node)
            simulation_path.append(node)

        return self.evaluate(self.current_search_path + simulation_path)

    def backpropagate(self, reward: float):
        """
        Backpropagate the reward along the visited path.
        """
        for node in reversed(self.current_search_path):
            q, n = self.value_visit[node]
            self.value_visit[node] = (q + reward, n + 1)

    def run(self, root: Node, iterations: int):
        """
        Runs MCTS for a given number of iterations.
        """
        for self.it in tqdm(range(iterations)):
            # Infeasible -> reward is None
            reward = None
            while reward is None:
                selected_node = self.select(root)
                reward = self.simulate(selected_node)
            self.backpropagate(reward)
            
    def best_path(self, root: Node) -> list[Node]:
        """
        Returns the best path from the root node after the search is done.
        """
        path = [root]
        node = root

        # Take maximum average reward child
        while (node in self.value_visit 
               and self.value_visit[node][1] > 0
               and not self.is_leaf(node)):
            children = self.graph.get_neighbors(node)
            node = max(children, key=lambda child: self.value_visit[child][0] / (self.value_visit[child][1]+1))
            if node == path[-1]:
                break
            path.append(node)

        return path
    
    def best_value_node(self) -> Node:
        return max(self.value_visit.keys(), key=lambda node: self.value_visit[node][0] / (self.value_visit[node][1]+1))
