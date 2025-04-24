import numpy as np
import mujoco
from collections import defaultdict
from typing import List, Tuple
import os
import time
from functools import wraps

from mpc_controller.mpc_acyclic import AcyclicMPC
from mj_pin.simulator import Simulator
from search.utils.mcts import MCTSBase
from search.utils.save import save_phase_sequence_to_yaml
from search.graph_phase_patch import GraphPhasePatchWithPos
from scene.primitives import Surface
from search.save_data import DataSaver

COLLISION_FREE_NAME = "collision_free"
CLOSE_LOOP_NAME = "close_loop"

def timeit(func):
    @wraps(func)
    def timed(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {(end_time - start_time) * 1e3:.4f} ms")
        return result
    return timed

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class MCTSPhaseLocomotionTask(MCTSBase):
    def __init__(self,
                 C : float,
                 alpha_exploration : float,
                 sim : Simulator,
                 mpc_solver : AcyclicMPC,
                 mpc_close_loop : AcyclicMPC,
                 n_phases : int,
                 surfaces : List[Surface],
                 min_in_cnt : int,
                 goal_surf_id : List[int],
                 min_mpc_log10_prod_res : float = 0.,
                 min_mpc_avg_collision : float = 0.5,
                 save_dir : str = "",
                 ):
        self.sim = sim
        self.q0_mj, self.v0_mj = self.sim.get_initial_state()
        self.q0, self.v0 = mpc_solver.solver.dyn.convert_from_mujoco(self.q0_mj, self.v0_mj)
        self.mpc_solver = mpc_solver
        self.mpc_close_loop = mpc_close_loop
        self.save_dir = save_dir
                
        # Init graph
        self.surfaces = surfaces
        self.solver_nodes = mpc_solver.solver.config_opt.n_nodes
        self.node_per_phase = int(self.solver_nodes / n_phases)
        self.n_phases = n_phases
        self.n_cnt = mpc_solver.n_foot
        self.patch_pos = np.array([surf.center for surf in surfaces])
        self.goal_pos = np.array([self.patch_pos[i] for i in goal_surf_id])
        graph = GraphPhasePatchWithPos(
            self.solver_nodes,
            self.node_per_phase,
            self.n_cnt,
            goal_surf_id,
            self.patch_pos,
            min_in_cnt,
        )

        # For heuristics
        dist_all_patches = np.linalg.norm(self.patch_pos[None, :, :] - self.patch_pos[:, None, :], axis=-1)
        self.max_dist_patches = np.max(dist_all_patches)
        self.mean_pos_patches = np.mean(self.patch_pos.reshape(-1, 3), axis=0, keepdims=True)
        
        # To compute reward
        self.max_reward = 0.
        self.mj_feet_frames = ["FL", "FR", "RL", "RR"]
        self.non_robot_geom_id = [
                mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, obj) if isinstance(obj, str)
                else int(obj)
                for obj
                in ["floor"] + sim.edit.name_allowed_collisions + self.mj_feet_frames
            ]
        self.feet_geom_id = self.non_robot_geom_id[-len(self.mj_feet_frames):]
        
        self.base_body_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.non_base_geom_id = [
            geom_id for geom_id in range(sim.mj_model.ngeom)
            if sim.mj_model.geom_bodyid[geom_id] != self.base_body_id
        ]
        self.min_mpc_log10_prod_res = min_mpc_log10_prod_res
        self.min_mpc_avg_collision = min_mpc_avg_collision
        self.alpha_exploration = alpha_exploration
        self.goal_geom_id = self.get_goal_geom_id()
        
        # Save data
        self.save_data = DataSaver(self.save_dir)
        self.start_time = time.time()
        
        # Init MCTS
        super().__init__(graph, C)

    def heuristic_bias(self, node):
        return self.distance_to_goal(node)
    
    def count_kin_collision(self,
                            q_mj_traj,
                            allowed_collision = [],
                            safe_collision = []
                            ):
        collision_count = defaultdict(int)
        
        for q_mj in q_mj_traj:
            self.sim.mj_data.qpos[:] = q_mj
            mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)
            
            for geom1, geom2 in zip(self.sim.mj_data.contact.geom1, self.sim.mj_data.contact.geom2):
                if geom1 in safe_collision or geom2 in safe_collision:
                    continue
                
                # If collision
                if not(geom1 in allowed_collision and geom2 in allowed_collision):
                    name1 = mujoco.mj_id2name(self.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or "robot"
                    name2 = mujoco.mj_id2name(self.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or "robot"
                    collision_count[name1] += 1
                    collision_count[name2] += 1
                
        return collision_count
    
    def eeff_in_cnt(self, node):
        avg_cnt = sum(node[1]) / len(node[1])
        return avg_cnt
    
    def distance_to_goal(self, node):
        _, cnt, patch = node
        n_in_cnt = sum(cnt)
        
        if n_in_cnt == 0:
            return 0.
        
        else:
            patch_pos = np.take_along_axis(self.patch_pos, np.array(patch).reshape(-1, 1), axis=0).reshape(-1, 3)
            goal_pos = self.goal_pos[np.array(cnt) == 1].reshape(-1, 3)
        
        avg_dist_to_goal = np.linalg.norm(np.mean(patch_pos - goal_pos, axis=0))
        return 1 - avg_dist_to_goal / self.max_dist_patches
    
    def rollout_policy(self, node):
        """
        Selects a random child node during rollout.
        """
        children = self.graph.get_neighbors(node)
        
        if np.random.rand() < self.alpha_exploration:
            child_id = np.random.choice(len(children))
        else:
            biases = np.array([self.distance_to_goal(child) for child in children]) + 1e-6
            s = biases.sum()
            n = len(biases)
            # Add mean to increase lower probabilities
            biases += s / n
            probabilities = biases / (2 * s)
            child_id = np.random.choice(len(children), p=probabilities)
            
        return children[child_id]
    
    def get_sequence_patches_from_path(self, path, node_per_phase : int):
        seq = np.array([list(phase[1]) for phase in path]).T.repeat(node_per_phase, axis=-1)
        seq = np.concatenate([seq, seq[:, None, -1]], axis=-1)
        
        n_eeff = len(path[-1][1])
        patches_dict = {i_eeff : [] for i_eeff in range(n_eeff)}
        last_cnt = {i_eeff : 0 for i_eeff in range(n_eeff)}
        for phase in path:
            _, cnt, patch = phase
            i_patch = 0
            for i_eeff, c in enumerate(cnt):
                if c:
                    if last_cnt[i_eeff] == 0:
                        patches_dict[i_eeff].append(patch[i_patch])
                last_cnt[i_eeff] = c
                i_patch += c
        patches = list(patches_dict.values())
        return seq, patches
    
    def get_goal_geom_id(self) -> int:
        for geom_id in range(self.sim.mj_model.ngeom):
            name = mujoco.mj_id2name(self.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name is not None and "goal" in name:
                return geom_id
        raise ValueError("Geometry named 'goal' not found.")

    def get_contact_patch(
        self,
        contact_seq: np.ndarray,
        contact_patches: List[List[int]],
        surfaces: List[Surface],
    ):
        """
        Determines the contact surface properties (center, normal, size) for each foot at each time node.

        :param contact_seq: (n_feet, n_nodes) binary contact sequence (1 if in contact, 0 otherwise)
        :param contact_patches: List of valid surface indices for each foot
        :param surfaces: List of available surfaces

        :return:
            - surface_centers: (n_feet, n_nodes, 3) array of surface centers per foot per time step
            - surface_normals: (n_feet, n_nodes, 3) array of surface normals per foot per time step
            - surface_sizes: (n_feet, n_nodes, 2) array of surface sizes per foot per time step (size_x, size_y)
        """
        n_feet, n_nodes = contact_seq.shape

        # Initialize output arrays with zeros
        surface_centers = np.zeros((n_feet, n_nodes, 3))
        surface_rot = np.zeros((n_feet, n_nodes, 3, 3))
        surface_sizes = np.zeros((n_feet, n_nodes, 2))
        
        for i_foot in range(n_feet):
            cnt_phase = 0
            last_in_cnt = False
            for i_node in range(n_nodes):
                if contact_seq[i_foot, i_node] == 1:  # Foot is in contact
                    last_in_cnt = True
                else:
                    # If break cnt, go to next phase
                    if last_in_cnt:
                        cnt_phase += 1
                    last_in_cnt = False
                    
                # Select a surface from the valid patches
                i = min(cnt_phase, len(contact_patches[i_foot])-1)
                id_surf = contact_patches[i_foot][i]
                surface = surfaces[id_surf]

                # Assign surface properties
                surface_centers[i_foot, i_node] = surface.center
                surface_rot[i_foot, i_node] = surface.rot
                surface_sizes[i_foot, i_node] = [surface.size_x, surface.size_y]             

        return surface_centers, surface_rot, surface_sizes

    def run_traj_opt(self, simulation_path : list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.mpc_solver.reset(reset_solver=True)
        start_phase = 0 if simulation_path[0] else 1
        
        cnt_sequence, patches = self.get_sequence_patches_from_path(simulation_path[start_phase:], self.node_per_phase)
        cnt_sequence = cnt_sequence[:, :self.solver_nodes+1]

        patch_center, patch_rot, patch_size = self.get_contact_patch(cnt_sequence, patches, self.surfaces)
        self.mpc_solver.set_cnt_plan(
            cnt_sequence,
            patch_center,
            patch_rot,
            patch_size
        )        
        
        try:
            q_sol, v_sol, _, _, dt_sol = self.mpc_solver.optimize(self.q0, self.v0)
            return q_sol, v_sol, dt_sol
        
        except Exception as e:
            print(e)
            return [], [], []
        
    def run_mpc(self,
                simulation_path : list,
                node_per_phase : int,
                record_video : bool = False,
                use_viewer : bool = False,
                ) -> bool:
        self.mpc_close_loop.reset(reset_solver=True)
        
        start_phase = 0 if simulation_path[0] else 1
        cnt_sequence, patches = self.get_sequence_patches_from_path(simulation_path[start_phase:], node_per_phase)

        patch_center, patch_rot, patch_size = self.get_contact_patch(cnt_sequence, patches, self.surfaces)
        self.mpc_close_loop.set_cnt_plan(
            cnt_sequence,
            patch_center,
            patch_rot,
            patch_size
        )
        
        try:
            dt_nodes = self.mpc_close_loop.config_opt.time_horizon / self.mpc_close_loop.config_opt.n_nodes
            duration = len(simulation_path) * node_per_phase * dt_nodes
            # Run close loop in simulator
            # Succes if base doesn't collide
            self.sim.run(
                sim_time=duration + 1.5,
                use_viewer=use_viewer,
                controller=self.mpc_close_loop,
                record_video=record_video,
                allowed_collision=self.non_base_geom_id
                )
            success = not self.sim.collided

            geom_cnt_with_feet = []
            for geom1, geom2 in zip(self.sim.mj_data.contact.geom1, self.sim.mj_data.contact.geom2):
                if geom1 in self.feet_geom_id:
                    geom_cnt_with_feet.append(geom2)
                elif geom2 in self.feet_geom_id:
                    geom_cnt_with_feet.append(geom1)
            # Check all feet are in contact with the same geometry     
            if len(np.unique(geom_cnt_with_feet)) != 1:
                success = False
            else:
                # Should be goal geometry
                if geom_cnt_with_feet[0] != self.goal_geom_id:
                    success = False
                    
            return success
        
        except Exception as e:
            # print(e)
            return 0

    def evaluate(self, simulation_path : list) -> float:
        if simulation_path[-1][-1] != self.graph.goal_node[-1]:
            return None

        q_sol, v_sol, dt_sol = self.run_traj_opt(simulation_path)
        # If diverged
        if len(q_sol) == 0:
            return 0
        
        # Compute reward
        reward = 1.

        # Reward on the residuals
        log10_prod_res = np.log10(np.prod(self.mpc_solver.solver.solver.get_stats("residuals")))
        W_RES_POS = 1/3
        W_RES_NEG = 1/6
        reward *= sigmoid(-(
            W_RES_POS * max(log10_prod_res, 0) +
            W_RES_NEG * min(log10_prod_res, 0)
            ))
        
        # Reward on the number of collisions
        v = np.zeros_like(v_sol[0])
        q_mj_traj = np.stack([self.mpc_solver.solver.dyn.convert_to_mujoco(q, v)[0] for q in q_sol])
        all_collisions = self.count_kin_collision(
                                         q_mj_traj,
                                         self.non_robot_geom_id,
                                         )
        n_robot_collision = sum([all_collisions[k] for k in ["robot"] + self.mj_feet_frames])
        avg_robot_collision = n_robot_collision / len(q_sol)
        W_COLLISION = 0.2
        reward *= np.exp(-W_COLLISION * avg_robot_collision)
        
        if avg_robot_collision == 0:
            run_dir = os.path.join(self.save_dir, f"{COLLISION_FREE_NAME}_iteration_{self.it}")
            save_phase_sequence_to_yaml(run_dir, simulation_path, self.node_per_phase)
            
        success = False
        run_mpc = log10_prod_res < self.min_mpc_log10_prod_res and avg_robot_collision < self.min_mpc_avg_collision
        # If promising solution, run close loop
        if run_mpc:
            # Run with different number of nodes per phase
            # Repeat first one makes the MPC better
            simulation_path_close_loop = [simulation_path[0]] + simulation_path
            for nodes_per_phase in [6, 7, 8, 9, 10]:
                success = self.run_mpc(simulation_path_close_loop, nodes_per_phase)
                if success:
                    # Record video
                    print("SUCCESS")
                    # Save results if run_dir specified
                    if self.save_dir:
                        run_dir = os.path.join(self.save_dir, f"{CLOSE_LOOP_NAME}_iteration_{self.it}")
                        self.sim.vs.video_dir = os.path.join(run_dir, f"{CLOSE_LOOP_NAME}_mpc.mp4")
                        save_phase_sequence_to_yaml(run_dir, simulation_path_close_loop, nodes_per_phase)
                    success = self.run_mpc(simulation_path_close_loop, nodes_per_phase, record_video=True)
                    break
            if success:
                reward = 1.
            else:
                MULT_FAILURE = 0.
                reward *= MULT_FAILURE
                
        # Save log data
        data = {
            "sequence":simulation_path,
            "residuals":[float(r) for r in self.mpc_solver.solver.solver.get_stats("residuals")],
            "log_prod_res":float(log10_prod_res),
            "avg_collision":avg_robot_collision,
            "reward":float(reward),
            "iteration": self.it,
            "run_close_loop" : int(run_mpc),
            "success_close_loop" : int(success),
            "search_time": time.time() - self.start_time,
        }
        self.save_data.append(**data)
            
        return reward