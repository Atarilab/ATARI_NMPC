import copy
import numpy as np
from numpy import sqrt
import tqdm
import time
from typing import List
from itertools import product, chain

from kinematics import QuadrupedKinematicFeasibility
from abstract import MCTS, timing

# State is the current 4 contact locations, referenced by their indices
State = List[int]

class MCTSSteppingStonesKin(MCTS):
    def __init__(self,
                 stepping_stones_sim: SteppingStonesSimulator,
                 simulation_steps: int = 1,
                 C: float = np.sqrt(2),
                 W: float = 10.,
                 alpha_exploration: float = 0.0,
                 **kwargs,
                 ) -> None:
        
        self.sim = stepping_stones_sim
        self.alpha_exploration = alpha_exploration
        self.C = C
        self.W = W

        optional_args = {
            "max_depth_selection" : 12,
            "max_solution_search" : 3,
            "n_threads_kin" : 10,
            "n_threads_sim" : 10,
            "use_inverse_kinematics" : True,
        }
        optional_args.update(kwargs)

        self.performance = {
            "time_first" : 0.,
            "n_nmpc_first" : 0,
            "iteration_first" : 0,
        }
        
        super().__init__(simulation_steps, C, **optional_args)
        
        # Maximum distance between contact locations
        self.d_max = self._compute_max_dist(self.sim.stepping_stones.positions)
        
        # Kinematics feasibility
        self.kinematics = QuadrupedKinematicFeasibility(self.sim.robot, num_threads=self.n_threads_kin)
            
    def _compute_max_dist(self, contact_pos_w) -> float:
        diffs = contact_pos_w[:, np.newaxis, :] - contact_pos_w[np.newaxis, :, :]
        d_squared = np.sum(diffs**2, axis=-1)
        d_max = np.sqrt(np.max(d_squared))
        return d_max

    @staticmethod
    def avg_dist_to_goal(contact_pos_w: np.ndarray,
                        current_states: list[State],
                        goal_state: State) -> float:
        """
        Computes average distance to goal.
        """
        d_to_goal = contact_pos_w[current_states] - contact_pos_w[np.newaxis, goal_state]
        avg_dist_to_goal = np.mean(np.linalg.norm(d_to_goal, axis=-1), axis=-1)
        return avg_dist_to_goal
    
    @timing("heuristic")
    def heuristic(self,
                  states: list[State],
                  goal_state: State) -> State:
        """
        Heuristic function to guide the search computed in a batched way.
        
        Args:
            states (List[State]): Compute the value of the heuristic on those states.
            goal_state (State): Goal state.

        Returns:
            State: State chosen by the heuristic.

        """
        heuristic_values = self.avg_dist_to_goal(
            self.sim.stepping_stones.positions,
            states,
            goal_state)

        # Exploration
        if np.random.rand() < self.alpha_exploration:
            probs = heuristic_values / sum(heuristic_values)
            id = np.random.choice(np.arange(len(states)), p=probs)

        # Exploitation
        else:
            id = np.argmin(heuristic_values)
        
        state = states[id]
        return state
    
    def get_children(self, state: State) -> List[State]:
        """
        Get kinematically reachable states from the current state.

        Args:
            state (State): current state.

        Returns:
            List[State]: Reachable states as a list.
        """
        feet_pos_w = self.sim.stepping_stones.positions[state]

        # Shape [Nr, 4]
        possible_contact_id = [
            self.kinematics.reachable_locations(
            foot_pos,
            self.sim.stepping_stones.positions,
            scale_reach=0.55
            ) for foot_pos in feet_pos_w]

        # Combinaison of feet location [NComb, 4]
        possible_states = np.array(list(product(*possible_contact_id)))

        # exclude current state
        possible_states = possible_states[~np.all(possible_states == state, axis=1)]
        # Bool array [NComb]
        reachable = self.kinematics.is_feasible(
            self.sim.stepping_stones.positions[possible_states],
            allow_crossed_legs=False,
            check_collision=True,
            check_inverse_kinematics=self.use_inverse_kinematics,
            )
        
        legal_next_states = possible_states[reachable]
        return legal_next_states

    @staticmethod
    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    def reward(self, contact_plan: list[list[State]],
               goal_state: State,
               ) -> float:
        
        if contact_plan[-1] != goal_state:
            avg_d_goal = MCTSSteppingStonesKin.avg_dist_to_goal(
                self.sim.stepping_stones.positions,
                contact_plan[-1],
                goal_state,
            )[0]
            r = 1 - avg_d_goal / self.d_max
            r = MCTSSteppingStonesKin.sigmoid(5 * (r - 1))
            return r, False, None
        
        goal_reached = self.sim.run_contact_plan(contact_plan)

        target_contacts = self.sim.stepping_stones.positions[contact_plan]
        target_contacts = np.array(target_contacts)[:, :, :2]
        achieved_contacts = self.sim.data_recorder.record_feet_contact
        achieved_contacts = np.array(achieved_contacts)[:, :, :2]

        if goal_reached:
            mean_contact_error = []
            for target_contact, achieved_contact in zip(target_contacts, achieved_contacts):
                contact_error = np.linalg.norm(target_contact - achieved_contact, axis=-1).mean()
                mean_contact_error.append(contact_error)
            mean_contact_error = np.mean(mean_contact_error)
        
        if goal_reached:
            return self.W, True, mean_contact_error
        else:
            return -1, True, None
    
    @timing("simulation")
    def simulation(self, state, goal_state) -> float:
        
        simulation_path = []
        for _ in range(self.simulation_steps):
            
            # Choose successively one child until goal is reached
            if self.tree.has_children(state) and not self.is_terminal(state, goal_state):
                children = self.tree.get_children(state)
                state = self.heuristic(children, goal_state)

                simulation_path.append(state)
            else:
                break
        contact_plan = self.tree.current_search_path + simulation_path
        reward, simualtion_used, contact_error = self.reward(contact_plan, goal_state)
        if simualtion_used:
            self.statistics["num_simulation_calls"] += 1
        solution_found = reward >= 1
        
        if solution_found:
            self.solutions.append(contact_plan)

        return reward, solution_found, contact_error