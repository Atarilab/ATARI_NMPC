from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pinocchio as pin
import time

from contact_tamp.traj_opt_acados.interface.problem_formuation import ProblemFormulation
from contact_tamp.traj_opt_acados.interface.acados_helper import AcadosSolverHelper
from contact_tamp.traj_opt_acados.interface.acados_helper import create_model
from ..config.config_abstract import MPCCostConfig, MPCOptConfig
from .dynamics import QuadrupedDynamics, BipedDynamics
from .transform import *
from .profiling import time_fn, print_timings
from mj_pin.utils import pin_frame_pos

class QuadrupedAcadosSolver(AcadosSolverHelper):
    NAME = "quadruped_solver"

    def __init__(self,
                 path_urdf : str,
                 feet_frame_names : List[str],
                 config_opt : MPCOptConfig,
                 config_cost : MPCCostConfig,
                 height_offset : float = 0.,
                 print_info : bool = False,
                 compute_timings : bool = True,
                 ):
        self.feet_frame_names = feet_frame_names
        self.config_opt = config_opt
        self.config_cost = config_cost
        self.height_offset = height_offset
        self.print_info = print_info
        self.restrict_cnt : bool = False

        self.dyn = QuadrupedDynamics(
            path_urdf,
            self.feet_frame_names,
            True,
            mu_contact=0.8
            )

        dt_min, dt_max = self.config_opt.get_dt_bounds()
        self.dt_nodes = self.config_opt.get_dt_nodes()
        self.enable_time_opt = self.config_opt.enable_time_opt
        problem = ProblemFormulation(self.dt_nodes, dt_min, dt_max, self.enable_time_opt)

        self.dyn.setup(problem)

        # Init solver
        super().__init__(
            problem,
            self.config_opt.n_nodes,
            QuadrupedAcadosSolver.NAME,
            self.config_cost.reg_eps,
            self.config_cost.reg_eps_e,
            )
    
        # Solver params
        self.reset()

        self.default_normal = np.array([0., 0., 1.])

        # Setup timings
        self.compute_timings = compute_timings
        self.timings = defaultdict(list)
        
    def reset(self):
        self.last_node = 0
        self.setup(self.config_opt.recompile,
                   self.config_opt.use_cython,
                   self.config_opt.real_time_it,
                   self.config_opt.max_qp_iter,
                   self.config_opt.hpipm_mode)
        self.timings = defaultdict(list)
        
        self.set_max_iter(self.config_opt.max_iter)
        self.set_warm_start_inner_qp(self.config_opt.warm_start_qp)
        self.set_warm_start_nlp(self.config_opt.warm_start_nlp)
        self.set_qp_tol(self.config_opt.qp_tol)
        self.set_nlp_tol(self.config_opt.nlp_tol)

        self.data = self.get_data_template()

        # Init cost
        self.set_cost_weights()
        self.update_cost_weights()

        # Init solution
        self.q_sol_euler = np.zeros_like(self.states[self.dyn.q.name])
        self.v_sol_euler = np.zeros_like(self.states[self.dyn.v.name])
        self.a_sol = np.zeros_like(self.inputs[self.dyn.a.name])
        self.f_sol = np.zeros((self.config_opt.n_nodes, 4, 3))
        self.dt_node_sol = np.zeros((self.config_opt.n_nodes))
        
    def set_contact_restriction(self, restrict : bool = True):
        self.restrict_cnt = restrict
        self.set_cost_weights()
        self.update_cost_weights()

    def update_cost(self, config_cost : MPCCostConfig):
        """
        Update MPC cost.
        """
        self.config_cost = config_cost
        self.set_cost_weights()
        self.update_cost_weights()

    def set_cost_weights(self):
        """
        Set up the running and terminal cost for the solver using weights from the config file.
        """
        # Terminal cost weights (W_e) for base position, orientation, and velocity
        self.data["W_e"][self.dyn.base_cost.name] = np.array(self.config_cost.W_e_base)
        # Running cost weights (W) for base position, orientation, and velocity
        self.data["W"][self.dyn.base_cost.name] = np.array(self.config_cost.W_base)
        # Acceleration cost weights
        self.data["W"][self.dyn.acc_cost.name] = np.array(self.config_cost.W_acc)
        # Swing cost weights (for each foot)
        self.data["W"][self.dyn.swing_cost.name] = np.array(self.config_cost.W_swing)
        # Joint cost to ref
        self.data["W"][self.dyn.joint_cost.name] = np.array(self.config_cost.W_joint)
        self.data["W_e"][self.dyn.joint_cost.name] = np.array(self.config_cost.W_e_joint)
        self.data["W_e"][self.dyn.swing_cost.name] = np.array(self.config_cost.W_swing)
        if self.enable_time_opt:
            self.data["W"]["dt"][0] = np.array(self.config_cost.time_opt)

        # Foot force regularization weights (for each foot)
        for i, foot_cnt in enumerate(self.dyn.feet):
            self.data["W"][foot_cnt.f_reg.name] = np.array(self.config_cost.W_cnt_f_reg[i])

            # Foot displacement penalization
            if self.restrict_cnt:
                self.data["W"][foot_cnt.pos_cost.name][:2] = self.config_cost.W_foot_displacement[0]
                self.data["W_e"][foot_cnt.pos_cost.name][:2] = self.config_cost.W_foot_displacement[0]
            else:
                self.data["W"][foot_cnt.pos_cost.name][:] = 0.
                self.data["W_e"][foot_cnt.pos_cost.name][:] = 0.

        # Apply these costs to the solver
        self.set_cost_weight_constant(self.data["W"])
        self.set_cost_weight_terminal(self.data["W_e"])

    @staticmethod
    def _repeat_last(arr: np.ndarray, n_repeat: int, axis=-1):
        # Get the last values along the specified axis
        last_value = np.take(arr, [-1], axis=axis)
        # Repeat the last values n times along the specified axis
        repeated_values = np.repeat(last_value, n_repeat, axis=axis)
        # Concatenate the original array with the repeated values
        result = np.concatenate([arr, repeated_values], axis=axis)
        return result

    def setup_reference(self,
                        base_ref : np.ndarray,
                        base_ref_e : np.ndarray,
                        joint_ref : np.ndarray,
                        step_height : float,
                        ):
        """
        Set up the reference trajectory (yref).
        """
        if base_ref_e is None:
            base_ref_e = base_ref.copy()
        
        # Set the nominal time step
        if self.enable_time_opt:
            self.cost_ref["dt"][:] = self.dt_nodes
        self.cost_ref[self.dyn.base_cost.name][:] = base_ref[:, None]
        self.cost_ref_terminal[self.dyn.base_cost.name] = base_ref_e

        # Joint reference is nominal position with zero velocities
        joint_ref_vel = np.concatenate((joint_ref, np.zeros_like(joint_ref)))
        self.cost_ref[self.dyn.joint_cost.name] = joint_ref_vel[:, None]
        self.cost_ref_terminal[self.dyn.joint_cost.name] = joint_ref_vel.copy()
        
        if not self.restrict_cnt:
            self.cost_ref[self.dyn.swing_cost.name][:] = self.height_offset + step_height
            self.cost_ref_terminal[self.dyn.swing_cost.name][:] = self.height_offset + step_height


    def setup_initial_state(self,
                            q_euler : np.ndarray,
                            v_global : np.ndarray,):
        """
        Initialize the state (x) of the robot in the solver.
        """
        self.data["x"][self.dyn.q.name] = q_euler
        self.data["x"][self.dyn.v.name] = v_global
        self.data["x"][self.dyn.h.name] = self.dyn.pin_data.hg.np

        if self.config_opt.enable_time_opt:
            self.inputs["dt"][:] = self.dt_nodes

        self.set_initial_state(self.data["x"])

    def setup_initial_feet_pos(self, i_node : int = 0):
        """
        Set up the initial position of the feet based on the current
        contact mode and robot configuration.
        """
        feet_pos = self.dyn.get_feet_position_w()

        for (foot_cnt, pos) in zip(self.dyn.feet, feet_pos):
            if i_node == 0:
                self.params[foot_cnt.active.name][0, 0] = 1
            is_cnt = self.params[foot_cnt.active.name][0, 0]

            # If contact, setup initial contact location and normal
            if is_cnt:
                next_swing = np.argmin(self.params[foot_cnt.active.name][0, :])
                self.params[foot_cnt.plane_point.name][:, :next_swing] = pos[:, None]
                if self.print_info: print(f'Reset foot contact {foot_cnt.frame_name}, height {pos[2]}')
    
    def init_contacts_parameters(self):
        # Fill contact parameters, will be overriden by gait planner
        for i_foot, foot_cnt in enumerate(self.dyn.feet):
            self.params[foot_cnt.active.name][:] = 1.
            self.params[foot_cnt.plane_normal.name][:] = self.default_normal[:, None]
            self.params[foot_cnt.plane_point.name][:] = np.zeros((3,1))
            self.params[foot_cnt.plane_point.name][-1, :] = self.height_offset
            self.params[foot_cnt.p_gain.name][:] = self.config_cost.W_foot_pos_constr_stab[i_foot]
            self.params[foot_cnt.restrict.name][:] = 0.
            if self.restrict_cnt:
                self.params[foot_cnt.range_radius.name][:] = self.config_cost.cnt_radius
            else:
                self.params[foot_cnt.range_radius.name][:] = 1.0e10

    def setup_cnt_status(self,
                        cnt_sequence : np.ndarray,
                        peak_plan : Optional[np.ndarray] = None):
        """
        Setup contact status in the optimization nodes.
        """
        assert np.shape(cnt_sequence)[-1] == self.config_opt.n_nodes + 1, \
            f"""Invalid contact plan shape. Wrong number of optimization nodes.
            ({np.shape(cnt_sequence)[-1]} vs {self.config_opt.n_nodes + 1})"""
        
        assert np.shape(cnt_sequence)[0] == len(self.feet_frame_names), \
            "Invalid contact plan shape. Wrong number of end effectors."
                
        # Set reference for terminal contact
        for i_foot, foot_cnt in enumerate(self.dyn.feet):
            self.params[foot_cnt.active.name][0, :] = cnt_sequence[i_foot]

            # Set peak constraints
            if self.config_opt.opt_peak and peak_plan is not None:
                self.params[foot_cnt.peak.name][0, :] = peak_plan[i_foot]

            # Set restriction
            if self.restrict_cnt:
                restrict = np.diff(cnt_sequence[i_foot], prepend=cnt_sequence[i_foot, 0])
                restrict[restrict == -1] = 0
                self.params[foot_cnt.restrict.name][0, :] = restrict

    @time_fn("setup_contact_plan")
    def setup_contact_loc(self, contact_loc_plan : np.ndarray, step_height : float):
        """
        Set the contact locations 

        Args:
            contact_loc_plan (np.ndarray): _description_
        """
        assert np.shape(contact_loc_plan)[1] == self.config_opt.n_nodes + 1, \
            "Invalid contact plan shape. Wrong number of optimization nodes."
        
        assert np.shape(contact_loc_plan)[0] == len(self.feet_frame_names), \
            "Invalid contact plan shape. Wrong number of end effectors."
        
        assert np.shape(contact_loc_plan)[-1] == 3, \
            "Invalid contact plan shape. 3D points required."

        # Set reference for terminal contact
        for i_foot, foot_cnt in enumerate(self.dyn.feet):
            self.params[foot_cnt.plane_point.name] = contact_loc_plan[i_foot,].T
            
            # Reference cost distance to foothold
            self.cost_ref[foot_cnt.pos_cost.name] = contact_loc_plan[i_foot, 1:, :].T
            self.cost_ref_terminal[foot_cnt.pos_cost.name] = self.params[foot_cnt.plane_point.name][:, -1].T
            
            # Reference cost step height
            self.cost_ref[self.dyn.swing_cost.name][i_foot, :] = contact_loc_plan[i_foot, 1:, -1] + step_height
            self.cost_ref_terminal[self.dyn.swing_cost.name][i_foot, :] = contact_loc_plan[i_foot, -1, -1] + step_height

    def print_contact_constraints(self):
        print()
        print("-"*10, "Contacts", "-"*10)
        for foot_cnt in self.dyn.feet:
            print("\n--- ", foot_cnt.frame_name, "contact")
            print(np.int8(self.params[foot_cnt.active.name]))
            print(np.unique(self.params[foot_cnt.plane_point.name], axis=1).T)
            print(np.int8(self.params[foot_cnt.peak.name]))

            if self.restrict_cnt:
                print("Restriction")
                print(np.int8(self.params[foot_cnt.restrict.name]))

    @time_fn("warm_start_solver")
    def warm_start_solver(self,
                        i_node: int,
                        repeat_last : bool = False,
                        ):
        """
        Warm start solver with the solution starting after
        start_node of the last solution.

        Args:
            i_node (int): Current optimization node.
            repeat_last (bool): Repeat last values for the last non warm started nodes.
        """

        # Warm start first values with last solution
        # q, v, a, forces, dt
        start_node = i_node - self.last_node
        n_warm_start = self.config_opt.n_nodes - start_node
        if self.print_info: print(f"Warm start size: {n_warm_start}, start_node {start_node}")

        # States [n_nodes + 1]
        self.states[self.dyn.q.name][:, 1:n_warm_start+1] = self.q_sol_euler[start_node+1:].T
        self.states[self.dyn.v.name][:, 1:n_warm_start+1] = self.v_sol_euler[start_node+1:].T
        self.states[self.dyn.h.name][:, 1:n_warm_start+1] = self.h_sol[start_node+1:].T
        
        # Inputs [n_nodes]
        self.inputs[self.dyn.a.name][:, :n_warm_start] = self.a_sol[start_node:].T
        
        for i, foot_name in enumerate(self.feet_frame_names):
            self.inputs[f"f_{foot_name}_{self.dyn.name}"][:, :n_warm_start] = self.f_sol[start_node:, i, :].T
            self.inputs[f"f_{foot_name}_{self.dyn.name}"][:, n_warm_start:] = 0.
            if repeat_last:
                self.inputs[f"f_{foot_name}_{self.dyn.name}"][:, n_warm_start:] = self.f_sol[None, -1, i].T

        # dt [n_nodes]
        if self.enable_time_opt:
            self.inputs["dt"][:, :n_warm_start] = self.dt_node_sol[start_node:]

        if repeat_last and n_warm_start < self.config_opt.n_nodes:
            self.states[self.dyn.q.name][:, n_warm_start:] = self.q_sol_euler[None, -1].T
            self.states[self.dyn.v.name][:, n_warm_start:] = self.v_sol_euler[None, -1].T
            self.states[self.dyn.h.name][:, n_warm_start:] = self.h_sol[None, -1].T
            self.inputs[self.dyn.a.name][:, n_warm_start:] = self.a_sol[None, -1].T
            
            # set to nominal dt
            if self.enable_time_opt:
                self.inputs["dt"][:, n_warm_start:] = self.dt_nodes

        # Dual variables
        self.warm_start_multipliers(start_node, n_warm_start, repeat_last=repeat_last)

        # Update last opt node
        self.last_node = i_node

    @time_fn("update_solver")
    def update_solver(self) -> None:
        """
        Set data in the solver.
        """
        self.update_states()
        self.update_inputs()
        self.update_parameters()
        self.update_ref()
        self.set_ref_terminal()

    @time_fn("init_solver")
    def init(self,
              i_node : int,
              q : np.ndarray,
              v : np.ndarray,
              base_ref : np.ndarray,
              base_ref_e : np.ndarray,
              joint_ref : np.ndarray,
              step_height : float,
              cnt_sequence : np.ndarray,
              cnt_locations : Optional[np.ndarray] = None,
              swing_peak : Optional[np.ndarray] = None,
              ):
        """
        Setup solver depending on the current configuration and the
        current optimization node.
        """
        # Base reference position and velocities in world frame
        self.setup_reference(base_ref, base_ref_e, joint_ref, step_height)

        self.setup_initial_state(q, v)

        # Set contact parameters
        self.init_contacts_parameters()
        self.setup_cnt_status(cnt_sequence, swing_peak)

        if self.restrict_cnt:
            assert not(cnt_locations is None), "Contact plan not provided"
            self.setup_contact_loc(cnt_locations, step_height)

        self.setup_initial_feet_pos(i_node)

        # Warm start solver
        if (i_node > 0 and
            self.config_opt.warm_start_sol):
            self.warm_start_solver(i_node, repeat_last=False)

        self.update_solver()

        if self.print_info: self.print_contact_constraints()

    @time_fn("solve")
    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the optimization problem and parse solution.
        returns pinocchio states with FreeFlyer
        """
        super().solve(print_stats=self.print_info, print_time=self.print_info)
        self.parse_sol()

        self.q_sol_euler = self.states[self.dyn.q.name].T.copy()

        # positions, [n_nodes + 1, 19]
        # q_sol = ypr_to_quat_state_batched(self.q_sol_euler)
        
        # velocities, [n_nodes + 1, 18]
        self.v_sol_euler = self.states[self.dyn.v.name].T.copy()
        # v_sol = v_glob_to_local_batched(self.q_sol_euler, self.v_sol_euler)

        # centroidal momentum, [n_nodes + 1, 6]
        self.h_sol = self.states[self.dyn.h.name].T.copy()

        # acceleration, [n_nodes, 18]
        self.a_sol = self.inputs[self.dyn.a.name].T.copy()

        # end effector forces, [n_nodes, 4, 3]
        self.f_sol = np.array([
            self.inputs[f"f_{foot_cnt.frame_name}_{self.dyn.name}"]
            for foot_cnt in self.dyn.feet
        ]).transpose(2, 0, 1).copy()

        # dt time, [n_nodes, ]
        if self.enable_time_opt:
            self.dt_node_sol = self.inputs["dt"].flatten().copy()
        else:
            self.dt_node_sol = np.full((len(self.a_sol),), self.dt_nodes)

        return self.q_sol_euler, self.v_sol_euler, self.a_sol, self.f_sol, self.dt_node_sol

    def print_timings(self):
        print_timings(self.timings)

class BipedAcadosSolver(AcadosSolverHelper):
    NAME = "biped_solver"

    def __init__(self,
                 path_urdf: str,
                 feet_frame_names: List[str],
                 config_opt: MPCOptConfig,
                 config_cost: MPCCostConfig,
                 height_offset: float = 0.,
                 print_info: bool = True,
                 compute_timings: bool = True):

        self.feet_frame_names = feet_frame_names
        self.config_opt = config_opt
        self.config_cost = config_cost
        self.height_offset = height_offset
        self.print_info = print_info
        self.compute_timings = compute_timings

        self.restrict_cnt = False


        self.dyn = BipedDynamics(
            urdf_path=path_urdf,
            feet_frame_names=self.feet_frame_names,
            mu_contact=0.8,            
            cnt_patch_restriction=True 
        )

      
        dt_min, dt_max = self.config_opt.get_dt_bounds()
        self.dt_nodes = self.config_opt.get_dt_nodes()
        self.enable_time_opt = self.config_opt.enable_time_opt

        problem = ProblemFormulation(
            self.dt_nodes, dt_min, dt_max, self.enable_time_opt
        )
       
        self.dyn.setup(problem)

        
        super().__init__(
            problem,
            self.config_opt.n_nodes,
            BipedAcadosSolver.NAME,    
            self.config_cost.reg_eps,
            self.config_cost.reg_eps_e,
        )

    
        self.default_normal = np.array([0., 0., 1.])

      
        self.timings = defaultdict(list)

        self.reset()

    def reset(self):
        self.last_node = 0

       
        self.setup(
            self.config_opt.recompile,
            self.config_opt.use_cython,
            self.config_opt.real_time_it,
            self.config_opt.max_qp_iter,
            self.config_opt.hpipm_mode
        )
        self.timings = defaultdict(list)

      
        self.set_max_iter(self.config_opt.max_iter)
        self.set_warm_start_inner_qp(self.config_opt.warm_start_qp)
        self.set_warm_start_nlp(self.config_opt.warm_start_nlp)
        self.set_qp_tol(self.config_opt.qp_tol)
        self.set_nlp_tol(self.config_opt.nlp_tol)

    
        self.data = self.get_data_template()

     
        self.set_cost_weights()
        self.update_cost_weights()


        self.q_sol_euler = np.zeros_like(self.states[self.dyn.q.name])
        self.v_sol_euler = np.zeros_like(self.states[self.dyn.v.name])
        self.a_sol       = np.zeros_like(self.inputs[self.dyn.a.name])


        n_feet = len(self.dyn.feet)
        self.f_sol = np.zeros((self.config_opt.n_nodes, n_feet, 3))
        self.dt_node_sol = np.zeros((self.config_opt.n_nodes))

    def set_cost_weights(self):
            """
            Set up the running and terminal cost for the solver using weights from the config file.
            """
            # Terminal cost weights (W_e) for base position, orientation, and velocity
            self.data["W_e"][self.dyn.base_cost.name] = np.array(self.config_cost.W_e_base)
            # Running cost weights (W) for base position, orientation, and velocity
            self.data["W"][self.dyn.base_cost.name] = np.array(self.config_cost.W_base)
            # Acceleration cost weights
            self.data["W"][self.dyn.acc_cost.name] = np.array(self.config_cost.W_acc)
            # Swing cost weights (for each foot)
            self.data["W"][self.dyn.swing_cost.name] = np.array(self.config_cost.W_swing)
            # Joint cost to ref 
            self.data["W"][self.dyn.joint_cost.name] = np.array(self.config_cost.W_joint)
            self.data["W_e"][self.dyn.joint_cost.name] = np.array(self.config_cost.W_e_joint)
            self.data["W_e"][self.dyn.swing_cost.name] = np.array(self.config_cost.W_swing)
            if self.enable_time_opt:
                self.data["W"]["dt"][0] = np.array(self.config_cost.time_opt)

            # Foot force regularization weights (for each foot)
            for i, foot_cnt in enumerate(self.dyn.feet):
                self.data["W"][foot_cnt.f_reg.name] = np.array(self.config_cost.W_cnt_f_reg[i])

                # Foot displacement penalization
                if self.restrict_cnt:
                    self.data["W"][foot_cnt.pos_cost.name][:2] = self.config_cost.W_foot_displacement[0]
                    self.data["W_e"][foot_cnt.pos_cost.name][:2] = self.config_cost.W_foot_displacement[0]
                else:
                    self.data["W"][foot_cnt.pos_cost.name][:] = 0.
                    self.data["W_e"][foot_cnt.pos_cost.name][:] = 0.

            # Apply these costs to the solver
            self.set_cost_weight_constant(self.data["W"])
            self.set_cost_weight_terminal(self.data["W_e"])
    def set_contact_restriction(self, restrict: bool = True):

        self.restrict_cnt = restrict
        self.set_cost_weights()
        self.update_cost_weights()

    def update_cost(self, config_cost: MPCCostConfig):
        self.config_cost = config_cost
        self.set_cost_weights()
        self.update_cost_weights()

    def setup_reference(self,
                        base_ref: np.ndarray,
                        base_ref_e: np.ndarray,
                        joint_ref: np.ndarray,
                        step_height: float):

        if base_ref_e is None:
            base_ref_e = base_ref.copy()

        if self.enable_time_opt:
            self.cost_ref["dt"][:] = self.dt_nodes

        # base cost
        self.cost_ref[self.dyn.base_cost.name][:] = base_ref[:, None]
        self.cost_ref_terminal[self.dyn.base_cost.name] = base_ref_e

        # joint cost
        joint_ref_vel = np.concatenate((joint_ref, np.zeros_like(joint_ref)))
        self.cost_ref[self.dyn.joint_cost.name] = joint_ref_vel[:, None]
        self.cost_ref_terminal[self.dyn.joint_cost.name] = joint_ref_vel.copy()

        if not self.restrict_cnt:
            self.cost_ref[self.dyn.swing_cost.name][:]         = self.height_offset + step_height
            self.cost_ref_terminal[self.dyn.swing_cost.name][:] = self.height_offset + step_height

    def setup_initial_state(self, q_euler: np.ndarray, v_global: np.ndarray):

        self.data["x"][self.dyn.q.name] = q_euler
        self.data["x"][self.dyn.v.name] = v_global
        self.data["x"][self.dyn.h.name] = self.dyn.pin_data.hg.np  # centroid

        if self.enable_time_opt:
            self.inputs["dt"][:] = self.dt_nodes

        self.set_initial_state(self.data["x"])

    def init_contacts_parameters(self):

        for i_foot, foot_cnt in enumerate(self.dyn.feet):
            self.params[foot_cnt.active.name][:] = 1.
            self.params[foot_cnt.plane_normal.name][:] = self.default_normal[:, None]
            self.params[foot_cnt.plane_point.name][:] = np.zeros((3,1))
            self.params[foot_cnt.plane_point.name][-1, :] = self.height_offset
            self.params[foot_cnt.p_gain.name][:] = self.config_cost.W_foot_pos_constr_stab[i_foot]
            self.params[foot_cnt.restrict.name][:] = 0.
            if self.restrict_cnt:
                self.params[foot_cnt.range_radius.name][:] = self.config_cost.cnt_radius
            else:
                self.params[foot_cnt.range_radius.name][:] = 1.0e10

    def setup_initial_feet_pos(self, i_node: int = 0):
        feet_pos = self.dyn.get_feet_position_w()
        for (foot_cnt, pos) in zip(self.dyn.feet, feet_pos):
            if i_node == 0:
                self.params[foot_cnt.active.name][0, 0] = 1
            is_cnt = self.params[foot_cnt.active.name][0, 0]
            if is_cnt:
                next_swing = np.argmin(self.params[foot_cnt.active.name][0, :])
                self.params[foot_cnt.plane_point.name][:, :next_swing] = pos[:, None]
                if self.print_info:
                    print(f"[init_feet_pos] foot {foot_cnt.frame_name}, set plane_point to {pos}")

    def setup_cnt_status(self, cnt_sequence: np.ndarray, peak_plan: Optional[np.ndarray] = None):
        """
        cnt_sequence shape: (n_feet, n_nodes+1)
        """
        assert cnt_sequence.shape[-1] == self.config_opt.n_nodes + 1
        assert cnt_sequence.shape[0] == len(self.feet_frame_names)

        for i_foot, foot_cnt in enumerate(self.dyn.feet):
            self.params[foot_cnt.active.name][0, :] = cnt_sequence[i_foot]
            if self.config_opt.opt_peak and peak_plan is not None:
                self.params[foot_cnt.peak.name][0, :] = peak_plan[i_foot]

            if self.restrict_cnt:
                restrict = np.diff(cnt_sequence[i_foot], prepend=cnt_sequence[i_foot, 0])
                restrict[restrict == -1] = 0
                self.params[foot_cnt.restrict.name][0, :] = restrict

    @time_fn("setup_contact_plan")
    def setup_contact_loc(self, contact_loc_plan: np.ndarray, step_height: float):
        """
        contact_loc_plan shape: (n_feet, n_nodes+1, 3)
        """
        assert contact_loc_plan.shape[0] == len(self.feet_frame_names)
        assert contact_loc_plan.shape[1] == self.config_opt.n_nodes + 1
        assert contact_loc_plan.shape[2] == 3

        for i_foot, foot_cnt in enumerate(self.dyn.feet):
            # plane point
            self.params[foot_cnt.plane_point.name] = contact_loc_plan[i_foot].T
            # cost ref => pos cost
            self.cost_ref[foot_cnt.pos_cost.name] = contact_loc_plan[i_foot, 1:, :].T
            self.cost_ref_terminal[foot_cnt.pos_cost.name] = contact_loc_plan[i_foot, -1].T

            # cost ref => swing cost
            self.cost_ref[self.dyn.swing_cost.name][i_foot, :] = contact_loc_plan[i_foot, 1:, -1] + step_height
            self.cost_ref_terminal[self.dyn.swing_cost.name][i_foot, :] = contact_loc_plan[i_foot, -1, -1] + step_height

    def print_contact_constraints(self):
        print("\n--- Contacts Info ---")
        for foot_cnt in self.dyn.feet:
            active_ = np.int8(self.params[foot_cnt.active.name])
            plane_p_ = self.params[foot_cnt.plane_point.name]
            peak_ = self.params[foot_cnt.peak.name]
            print(f"Foot {foot_cnt.frame_name} active = {active_}")
            print("plane_point unique:\n", np.unique(plane_p_, axis=1).T)
            print("peak:\n", np.int8(peak_))
            if self.restrict_cnt:
                print("restrict:\n", np.int8(self.params[foot_cnt.restrict.name]))


    @time_fn("warm_start_solver")
    def warm_start_solver(self, i_node: int, repeat_last: bool = False):

        start_node = i_node - self.last_node
        n_warm_start = self.config_opt.n_nodes - start_node
        if self.print_info:
            print(f"[warm_start_solver] start_node={start_node}, size={n_warm_start}")

        # states
        self.states[self.dyn.q.name][:, 1:n_warm_start+1] = self.q_sol_euler[start_node+1:].T
        self.states[self.dyn.v.name][:, 1:n_warm_start+1] = self.v_sol_euler[start_node+1:].T
        self.states[self.dyn.h.name][:, 1:n_warm_start+1] = self.h_sol[start_node+1:].T

        # inputs
        self.inputs[self.dyn.a.name][:, :n_warm_start] = self.a_sol[start_node:].T
        for i, foot_cnt in enumerate(self.dyn.feet):
            self.inputs[f"f_{foot_cnt.frame_name}_{self.dyn.name}"][:, :n_warm_start] = \
                self.f_sol[start_node:, i, :].T
  
            self.inputs[f"f_{foot_cnt.frame_name}_{self.dyn.name}"][:, n_warm_start:] = 0.
            if repeat_last:
                self.inputs[f"f_{foot_cnt.frame_name}_{self.dyn.name}"][:, n_warm_start:] = \
                    self.f_sol[-1, i, :].reshape(3,1)

        if self.enable_time_opt:
            self.inputs["dt"][:, :n_warm_start] = self.dt_node_sol[start_node:]

        if repeat_last and n_warm_start < self.config_opt.n_nodes:
            # states
            self.states[self.dyn.q.name][:, n_warm_start:] = self.q_sol_euler[-1].reshape(-1,1)
            self.states[self.dyn.v.name][:, n_warm_start:] = self.v_sol_euler[-1].reshape(-1,1)
            self.states[self.dyn.h.name][:, n_warm_start:] = self.h_sol[-1].reshape(-1,1)
            # inputs
            self.inputs[self.dyn.a.name][:, n_warm_start:] = self.a_sol[-1].reshape(-1,1)
            if self.enable_time_opt:
                self.inputs["dt"][:, n_warm_start:] = self.dt_nodes

        self.warm_start_multipliers(start_node, n_warm_start, repeat_last)
        self.last_node = i_node

    @time_fn("update_solver")
    def update_solver(self):
        """把 data/params/ref 等写进 solver"""
        self.update_states()
        self.update_inputs()
        self.update_parameters()
        self.update_ref()
        self.set_ref_terminal()

    @time_fn("init_solver")
    def init(self,
             i_node: int,
             q: np.ndarray,
             v: np.ndarray,
             base_ref: np.ndarray,
             base_ref_e: np.ndarray,
             joint_ref: np.ndarray,
             step_height: float,
             cnt_sequence: np.ndarray,
             cnt_locations: Optional[np.ndarray] = None,
             swing_peak: Optional[np.ndarray] = None):

        self.setup_reference(base_ref, base_ref_e, joint_ref, step_height)
        self.setup_initial_state(q, v)

        self.init_contacts_parameters()
        self.setup_cnt_status(cnt_sequence, swing_peak)

        if self.restrict_cnt:
            if cnt_locations is None:
                raise ValueError("Contact plan not provided, restrict_cnt=True requires it.")
            self.setup_contact_loc(cnt_locations, step_height)

        self.setup_initial_feet_pos(i_node)

        # warm start
        if i_node > 0 and self.config_opt.warm_start_sol:
            self.warm_start_solver(i_node, repeat_last=False)

        self.update_solver()

        if self.print_info:
            self.print_contact_constraints()

    @time_fn("solve")
    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        super().solve(print_stats=self.print_info, print_time=self.print_info)
        self.parse_sol()

        self.q_sol_euler = self.states[self.dyn.q.name].T.copy()
        self.v_sol_euler = self.states[self.dyn.v.name].T.copy()
        self.h_sol       = self.states[self.dyn.h.name].T.copy()
        self.a_sol       = self.inputs[self.dyn.a.name].T.copy()

        # forces => shape(n_nodes, n_feet, 3)
        n_feet = len(self.dyn.feet)
        f_array = []
        for foot_cnt in self.dyn.feet:
            arr = self.inputs[f"f_{foot_cnt.frame_name}_{self.dyn.name}"]
            f_array.append(arr)
        self.f_sol = np.array(f_array).transpose(2,0,1).copy()

        # dt
        if self.enable_time_opt:
            self.dt_node_sol = self.inputs["dt"].flatten().copy()
        else:
            self.dt_node_sol = np.full((len(self.a_sol),), self.dt_nodes)

        return self.q_sol_euler, self.v_sol_euler, self.a_sol, self.f_sol, self.dt_node_sol

    def print_timings(self):
        print_timings(self.timings)