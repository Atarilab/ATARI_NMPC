from typing import Tuple
import numpy as np

from .utils.contact_planner import ContactPlannerAcyclic
from .mpc import LocomotionMPC
from .utils.profiling import time_fn

class AcyclicMPC(LocomotionMPC):
    """
    Abstract base class for an MPC controller.
    This class defines the structure for an MPC controller
    where specific optimization methods can be implemented by inheriting classes.
    """
    def __init__(self, path_urdf, feet_frame_names, config_opt, config_cost, config_gait, joint_ref = None, interactive_goal = False, sim_dt = 0.001, height_offset = 0, contact_planner = "", print_info = True, compute_timings = True, solve_async = True):
        super().__init__(path_urdf, feet_frame_names, config_opt, config_cost, config_gait, joint_ref, interactive_goal, sim_dt, height_offset, contact_planner, print_info, compute_timings, solve_async)
        
        self.contact_planner = ContactPlannerAcyclic()
        self.solver.set_contact_restriction(True)
        
        self.base_pos_ref_traj = None
        self.base_vel_ref_traj = None
        self.joint_pos_ref_traj = None
        self.joint_vel_ref_traj = None

    def set_cnt_plan(
        self,
        cnt_sequence,
        cnt_center = None,
        cnt_rot = None,
        cnt_size = None,
    ) -> None:
        self.contact_planner.set_sequence(cnt_sequence)
        if (
            cnt_center is not None and 
            cnt_rot is not None and 
            cnt_size is not None
            ):
            self.contact_planner.set_center_rot_size(cnt_center, cnt_rot, cnt_size)
            
    def set_convergence_on_first_iter(self):
        N_SQP_FIRST = 60
        if self.first_solve:
            self.solver.set_max_iter(N_SQP_FIRST)
            self.solver.set_nlp_tol(self.solver.config_opt.nlp_tol)
            self.solver.set_qp_tol(self.solver.config_opt.qp_tol)
        elif self.sim_step <= self.replanning_steps:
            self.solver.set_max_iter(1)
            
    def keep_solution_as_reference(self):        
        self.base_pos_ref_traj, self.joint_pos_ref_traj = np.split(self.solver.q_sol_euler.copy(), [6,], axis=-1)
        self.base_vel_ref_traj, self.joint_vel_ref_traj = np.split(self.solver.v_sol_euler.copy(), [6,], axis=-1)
    
    @time_fn("optimize")
    def optimize(self,
                 q : np.ndarray,
                 v : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        return optimized trajectories.
        """
        # Update model state based on current MuJoCo state
        self.solver.dyn.update_pin(q, v)

        # Contact parameters
        cnt_sequence = self.contact_planner.get_sequence(self.current_opt_node, self.config_opt.n_nodes+1)
        swing_peak = self.contact_planner.get_peak(self.current_opt_node, self.config_opt.n_nodes+1)
        cnt_center, cnt_rot, cnt_size = self.contact_planner.get_center_rot_size_patch(self.current_opt_node, self.config_opt.n_nodes+1)
        base_ref, base_ref_e = self.compute_base_ref_vel_tracking(q)

        self.solver.init(
            self.current_opt_node,
            q,
            v,
            base_ref,
            base_ref_e,
            self.joint_ref,
            self.config_gait.step_height,
            cnt_sequence,
            cnt_center,
            cnt_rot,
            cnt_size,
            swing_peak,
            )
            
        q_sol, v_sol, a_sol, f_sol, dt_sol = self.solver.solve()

        return q_sol, v_sol, a_sol, f_sol, dt_sol
