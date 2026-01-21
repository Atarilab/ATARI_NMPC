import numpy as np
import tqdm
import time
from typing import List, Tuple
from collections import defaultdict
from functools import wraps

State = List[int]

class Action:
    def __init__(self) -> None:
        self.visit = 0
        self.value = 0.

    def increment_visit(self) -> None:
        self.visit += 1

    def update_value(self, reward) -> None:
        self.value += reward

class Node:
    def __init__(self) -> None:
        self.visit = 0
        self.actions = defaultdict(Action)             # { child state (hash): Action }
        self.expanded = False
        
    def increment_visit(self) -> None:
        self.visit += 1

class Tree:
    def __init__(self) -> None:
        self.nodes = defaultdict(Node)               # { state (hash) : Node }

        self.current_search_path = []               # Path of the current search List[state]

    @staticmethod
    def hash_state(state : State) -> str:
        hash_string = '-'.join(map(str, state))
        return hash_string
    
    @staticmethod
    def unhash_state(h_state:str) -> State:
        state = list(map(int, h_state.split('-')))
        return state
    
    def add_node(self, state : State):
        h = self.hash_state(state)
        if not h in self.nodes.keys():
            self.nodes[h] = Node()

    def has_children(self, state : State) -> bool:
        h = self.hash_state(state)
        return bool(self.nodes[h].actions)
    
    def expanded(self, state : State) -> bool:
        h = self.hash_state(state)
        return self.nodes[h].expanded
    
    def get_children(self, state : State) -> List[State]:
        h = self.hash_state(state)
        node = self.nodes[h]
        return list(map(self.unhash_state, node.actions.keys()))

    def add_children_to_node(self, state : State, children_states : List[State]) -> None:
        h = self.hash_state(state)
        node = self.nodes[h] # Create node if not exists
        node.expanded = True
        if not node.actions:
            # Add children nodes to the tree
            h_children = list(map(self.hash_state, children_states))            
            if len(h_children) > 0:
                self.nodes.update({h_child : Node() for h_child in h_children if not h_child in self.nodes.keys()})
                
                # Add actionhas_childrens and children to current state 
                node.actions = {h_child : Action() for h_child in h_children}
            
    def update_value_visit_action(self, stateA : State, stateB : State, reward : float) -> None:
        hA = self.hash_state(stateA)
        hB = self.hash_state(stateB)

        node = self.nodes[hA]
        action = node.actions[hB]

        node.increment_visit()
        action.increment_visit()
        action.update_value(reward)

    def reset_search_path(self) -> None:
        self.current_search_path = []
    
    def get_action(self, stateA : State, stateB : State) -> Action:
        hA = self.hash_state(stateA)

        if hA in self.nodes.keys():
            hB = self.hash_state(stateB)
            if hB in self.nodes[hA].actions.keys():
                return self.nodes[hA].actions[hB]
        return None
    
    def get_actions(self, state : State) -> List[Action]:
        h = self.hash_state(state)

        if h in self.nodes.keys():
            return list(self.nodes[h].actions.values())
        return None
    
    def get_node(self, state : State) -> Node:
        h = self.hash_state(state)
        if h in self.nodes.keys():
            return self.nodes[h]
        return None
    
    def UCB(self, stateA : State, stateB : State, C = 3.0e-2) -> float:
        hA = self.hash_state(stateA)
        hB = self.hash_state(stateB)

        node = self.nodes[hA]
        action = node.actions[hB]

        if action.visit == 0:
            return float("+inf")

        return action.value / action.visit + C * np.sqrt(np.log(node.visit) / action.visit)
    
def timing(method_name):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            N = self.it + 1
            t = (end_time - start_time) * 1000 # ms
            # Update average
            self.timings[method_name] = 1 / N * ((N - 1) * self.timings[method_name] + t)
            return result
        return wrapper
    return decorator

class MCTS():
    PRINT_INFO_STEP = 1
    def __init__(self,
                 simulation_steps:int=10,
                 C:float=np.sqrt(2),
                 **kwargs) -> None:
        """
        MCTS algorithm.
        
        Args:
            - simulation_steps (int)    : Number of simulation steps
            - C (float)                 : Exploration vs Exploitation coefficient in UCB
            - max_depth_selection       : Stop selection phase after <max_depth_selection> steps (to avoid infinite loop)
            - max_solution_search       : Stop search when <max_solution_search> have been found
            - print_info (bool)         : Print current number of nodes in tree and reward
        """
        
        self.simulation_steps = simulation_steps
        self.C = C

        self.tree = Tree()
        self.n_solution_found = 0
        self.it = 0
        self.solutions = []

        optional_args = {
            "max_depth_selection" : 8,
            "max_solution_search" : 10,
            "print_info" : False,
        }

        optional_args.update(kwargs)
        for k, v in optional_args.items(): setattr(self, k, v)
        
        self.timings = defaultdict(float)

        self.statistics = {
            "num_iterations" : 0,
            "num_simulation_calls" : 0,
            "num_solution_found" : 0,
            "num_nodes" : 0,
            "time" : time.time(),
            "solution_statistic" : []
        }

    @staticmethod
    def is_terminal(start_state : State, state_goal : State) -> bool:
        return start_state == state_goal
    
    def get_children(self, state : State) -> List[State]:
        """
        Returns the children of a state.
        To override.
        """
        return [state]
    
    def heuristic(self, states : List[State], state_goal : State) -> State:
        """
        Default heuristic. Select a node randomly from a set of states.
        To override.
        """
        return np.random.choice(states)
    
    def reward(self, state : State, state_goal : State) -> float:
        """
        Default reward. Computes the reward associated to the current state.
        To override.
        """
        return np.random.rand()
    
    @timing("selection")
    def selection(self, state : State, state_goal : State) -> State:
        self.tree.current_search_path = []

        depth = 0
        while True:
            self.tree.current_search_path.append(state)

            if depth >= self.max_depth_selection:
                break
            
            # Select node that haven't been expanded
            if not self.tree.expanded(state) or not self.tree.has_children(state):
                break

            # Select one of the children that haven't been expanded if exists
            children = self.tree.get_children(state)
            unexplored = list(filter(lambda state: not self.tree.expanded(state), children))

            if unexplored:
                state = self.heuristic(unexplored, state_goal)
                self.tree.current_search_path.append(state)
                break

            # Go one level deeper in the tree
            depth += 1
            # If all node have been expanded, select action with maximum UCB score
            state = max(children, key=lambda child_state: self.tree.UCB(state, child_state, self.C))

        return state
    
    @timing("expansion")
    def expansion(self, state : State) -> None:
        # If has no children already
        if not self.tree.has_children(state):
            children_states = self.get_children(state)
            self.tree.add_children_to_node(state, children_states)
    
    @timing("simulation")
    def simulation(self, state : State, goal_state : State) -> Tuple[float, bool]:
        terminal_state = False
        simulation_path = []
        for _ in range(self.simulation_steps):

            # Choose successively one child until goal is reached
            if self.tree.has_children(state) and not self.is_terminal(state, goal_state):
                children = self.tree.get_children(state)
                state = self.heuristic(children, goal_state)
                simulation_path.append(state)
            else:
                break

        if self.is_terminal(state, goal_state):
            terminal_state = True
            full_path = self.tree.current_search_path + simulation_path
            self.solutions.append(full_path)

        reward = self.reward(state, goal_state)
        contact_error = 0.0
        return reward, terminal_state, contact_error
    
    @timing("back_propagation")
    def back_propagation(self, reward : float) -> None:
        child_state = self.tree.current_search_path[-1]
        for state in reversed(self.tree.current_search_path[:-1]):
            self.tree.update_value_visit_action(state, child_state, reward)
            child_state = state
    
    def search(self, state_start, state_goal, num_iterations:int=1000):
        
        progress_bar = tqdm.trange(0, num_iterations)
        self.n_solution_found = 0

        self.tree.reset_search_path()

        for self.it in progress_bar:
            # Selection
            leaf = self.selection(state_start, state_goal)
            # Expansion
            self.expansion(leaf)
            # Simulation
            reward, terminal_state, contact_error = self.simulation(leaf, state_goal)
            # Backpropagation
            self.back_propagation(reward)

            if terminal_state:
                self.n_solution_found += 1
                self.statistics["solution_statistic"].append({
                    "num_nodes" : len(self.tree.nodes),
                    "num_iterations" : self.it,
                    "num_solution_found" : self.n_solution_found,
                    "time": time.time() - self.statistics["time"],
                    "num_simulation_calls" : self.statistics["num_simulation_calls"],
                    "contact_error" : contact_error,
                })

            if self.print_info and self.it % MCTS.PRINT_INFO_STEP == 0:
                progress_bar.set_postfix({
                        "found": self.n_solution_found,
                        "nodes": len(self.tree.nodes),
                        "reward": reward,
                        "sims": self.statistics["num_simulation_calls"],
                        })
                
            if self.n_solution_found >= self.max_solution_search:
                break

        self.statistics["num_iterations"] = self.it
        self.statistics["num_solution_found"] = self.n_solution_found
        self.statistics["num_nodes"] = len(self.tree.nodes)
        self.statistics["time"] = time.time() - self.statistics["time"]

    def get_best_children(self, state_start, state_goal, n_children:int=1, mode:str="visit") -> List:
    
        def value_child(child_state):
            """
            Function to be maximised by the action
            """
            print("Warning. Value function not set.")
            return 0.

        if mode == "visit":
            def value_child(state, child_state):
                """ Maximum visit
                """
                return self.tree.get_action(state, child_state).visit

        elif mode == "value":
            def value_child(state, child_state):
                """ Maximum average value
                """
                action = self.tree.get_action(state, child_state)
                if action.visit > 0:
                    return action.value / action.visit
                else:
                    return float("-inf")
        
        children = []
        state = state_start
        for _ in range(n_children):

            best_child = max(self.tree.get_node(state).children, key=lambda child : value_child(state, child))
            children.append(best_child)
            state = best_child

            if self.is_terminal(state, state_goal):
                break

        return children
    
    def get_timings(self):
        return self.timings