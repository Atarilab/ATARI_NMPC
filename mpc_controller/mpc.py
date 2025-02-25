from collections import defaultdict
import math
import time
from typing import Any, Dict, List, Tuple
from matplotlib import pyplot as plt
import numpy as np
from bisect import bisect_left, bisect_right
from scipy.interpolate import interp1d, CubicHermiteSpline
from concurrent.futures import ThreadPoolExecutor, Future
import pinocchio as pin
import traceback

from mj_pin.abstract import PinController
from .utils.interactive import SetVelocityGoal
from .utils.contact_planner import RaiberContactPlanner, CustomContactPlanner, ContactPlanner
from .utils.solver import QuadrupedAcadosSolver, BipedAcadosSolver
from .utils.profiling import time_fn, print_timings
from .config.quadruped.utils import get_quadruped_config,get_biped_config

class LocomotionMPC(PinController):
    CONTACT_PLANNERS = ["raibert", "custom"]
    """
    Abstract base class for an MPC controller.
    This class defines the structure for an MPC controller
    where specific optimization methods can be implemented by inheriting classes.
    """
    def __init__(self,
                 path_urdf : str,
                 feet_frame_names : List[str],
                 robot_name : str,
                 gait_name: str = "still_stand",
                 joint_ref : np.ndarray = None,
                 interactive_goal : bool = False,
                 sim_dt : float = 1.0e-3,
                 height_offset : float = 0.,
                 contact_planner : str = "",
                 print_info : bool = True,
                 compute_timings : bool = True,
                 solve_async : bool = True,
                 ) -> None:

        self.gait_name = gait_name
        self.print_info = print_info
        self.height_offset = height_offset
        # Solver
        self.config_gait, self.config_opt, self.config_cost = get_biped_config(gait_name, robot_name)
        self.solver = BipedAcadosSolver(
            path_urdf,
            feet_frame_names,
            self.config_opt,
            self.config_cost,
            height_offset,
            print_info,
            compute_timings)

        super().__init__(pin_model=self.solver.dyn.pin_model)

        # Set joint reference
        nu = self.solver.dyn.pin_model.nv - 6
        if joint_ref is None:
            if self.solver.dyn.pin_model.referenceConfigurations["home"]:
                joint_ref = self.solver.dyn.pin_model.referenceConfigurations["home"][-nu:]
            else:
                print("Joint reference not found in pinocchio model. Set to zero.")
                joint_ref = np.zeros(nu)
        self.joint_ref = joint_ref[-nu:]
               
        # Contact planner
        q0, v0 = np.zeros(self.solver.dyn.pin_model.nq), np.zeros(self.solver.dyn.pin_model.nv)
        q0[-nu:] = self.joint_ref
        self.solver.dyn.update_pin(q0, v0)
        self.base_ref_vel_tracking = np.zeros(6)

        self.n_foot = len(feet_frame_names)
        self._contact_planner_str = contact_planner 

        if contact_planner.lower() == "raibert":
            offset_hip_b = self.solver.dyn.get_feet_position_w()
            offset_hip_b[:, -1] = 0.
            self.contact_planner = RaiberContactPlanner(
                feet_frame_names,
                self.solver.dt_nodes,
                self.config_gait,
                offset_hip_b,
                y_offset=0.02,#TODO
                x_offset=0.04,
                foot_size=-0.03,
                cache_cnt=False
                )
            self.restrict_cnt = True  
            
        elif contact_planner.lower() == "custom":
            self.contact_planner = CustomContactPlanner(
                feet_frame_names,
                self.solver.dt_nodes,
                self.config_gait,
                )
            self.restrict_cnt = True
            
        else:
            self.contact_planner = ContactPlanner(feet_frame_names, self.solver.dt_nodes, self.config_gait)
            self.restrict_cnt = False
        
        self.solver.set_contact_restriction(self.restrict_cnt)
        
        # Set params
        self.Kp = self.solver.config_opt.Kp
        self.Kd = self.solver.config_opt.Kd
        self.com_z_traj = [] # Norm Height Tuning
        self.sim_dt = sim_dt
        self.dt_nodes : float = self.solver.dt_nodes
        self.replanning_freq : int = self.config_opt.replanning_freq
        self.replanning_steps : int = int(1 / (self.replanning_freq * sim_dt))
        self.sim_step : int = 0
        self.plan_step : int = 0
        self.current_opt_node : int = 0
        self.node_since_last_opt : int = 0
        self.solve_async : bool = solve_async
        self.delay : int = 0
        self.last : int = 0

        self.v_des : np.ndarray = np.zeros(3)
        self.w_des : np.ndarray = np.zeros(3)
        self.base_ref_vel_tracking : np.ndarray = np.zeros(12)
        self.q_plan : np.ndarray = None
        self.v_plan : np.ndarray = None
        self.a_plan : np.ndarray = None
        self.w_plan : np.ndarray = None
        self.q_opt : np.ndarray = None
        self.v_opt : np.ndarray = None
        self.time_traj : np.ndarray = np.array([])

        # For plots
        self.q_full = []
        self.v_full = []
        self.a_full = []
        # self.f_full = []
        self.w_full = []
        self.tau_full = []
        self.dt_full = []

        self.diverged : bool = False

        # Setup timings
        self.compute_timings = compute_timings
        self.timings = defaultdict(list)

        # Multiprocessing
        self.executor = ThreadPoolExecutor(max_workers=1)  # One thread for asynchronous optimization
        self.optimize_future: Future = None                # Store the future result of optimize
        self.plan_submitted = False                        # Flag to indicate if a new plan is ready

        self.velocity_goal = SetVelocityGoal() if interactive_goal else None

    def _replan(self) -> bool:
        """
        Returns true if replanning step.
        Record trajectory of the last 
        """
        replan = self.sim_step % self.replanning_steps == 0

        if self.solve_async:
            replan = replan and (self.optimize_future is None or self.optimize_future.done())

        return replan
    
    def _step(self) -> None:
        self.increment_base_ref_position()
        self.sim_step += 1
        self.plan_step += 1

    def _record_plan(self) -> None:
        """
        Record trajectory of the last plan until self.plan_step.
        """
        self.q_full.append(self.q_plan[self.delay : self.plan_step])
        self.v_full.append(self.v_plan[self.delay : self.plan_step])
        self.a_full.append(self.a_plan[self.delay : self.plan_step])
        # self.f_full.append(self.f_plan[self.delay : self.plan_step])
        self.w_full.append(self.w_plan[self.delay : self.plan_step])

    def set_command(self, v_des: np.ndarray = np.zeros((3,)), w_yaw: float = 0.) -> None:
        """
        Set velocity commands for the MPC.
        """
        self.v_des = v_des
        self.w_des[2] = w_yaw

    def increment_base_ref_position(self):
        R_WB = pin.rpy.rpyToMatrix(self.base_ref_vel_tracking[3:6][::-1])
        v_des_glob = np.round(R_WB @ self.v_des, 1)
        self.base_ref_vel_tracking[:2] += v_des_glob[:2] * self.sim_dt
        self.base_ref_vel_tracking[3] += self.w_des[-1] * self.sim_dt

    def compute_base_ref_vel_tracking(self, q_mj : np.ndarray) -> np.ndarray:
        """
        Compute base reference for the solver.
        """
        t_horizon = self.solver.config_opt.time_horizon

        # Set position
        base_ref = np.zeros(12)
        base_ref[:2] = np.round(q_mj[:2], 2)
        # Height to config
        base_ref[2] = self.config_gait.nom_height + self.height_offset
        # Set yaw
        qw, qx, qy, qz = q_mj[3:7]
        yaw = math.atan2(2.0*(qy*qx + qw*qz), -1. + 2. * (qw*qw + qx*qx))
        base_ref[3] = round(yaw, 1)

        # Setup reference velocities in global frame
        # v_des is in local frame
        # w_yaw in global frame
        R_WB = pin.rpy.rpyToMatrix(self.base_ref_vel_tracking[3:6][::-1])
        v_des_glob = np.round(R_WB @ self.v_des, 1)

        base_ref[6:9] = v_des_glob
        base_ref[-3:] = self.w_des[::-1]

        # Terminal reference, copy base ref
        base_ref_e = base_ref.copy()

        # Compute velocity in global frame
        # Apply angular velocity
        R_yaw = pin.rpy.rpyToMatrix(self.w_des * t_horizon)
        base_ref_e[6:9] = R_yaw @ base_ref[6:9]

        if self.velocity_goal:
            pos_ref = np.round(q_mj[:3], 2)
            yaw_ref = yaw
        else:
            pos_ref = self.base_ref_vel_tracking[:3]
            yaw_ref = self.base_ref_vel_tracking[3]

        base_ref_e[:2] = pos_ref[:2] + v_des_glob[:2] * t_horizon
        # Clip base ref in direction of the motion
        # (don't go too far if the robot is too slow)
        base_ref_e[:2] = np.clip(base_ref_e[:2],
                -base_ref[:2] + v_des_glob[:2] * t_horizon * 1.2,
                 base_ref[:2] + v_des_glob[:2] * t_horizon * 1.2,
                )
        
        base_ref_e[3] = yaw_ref + self.w_des[-1] * t_horizon
        base_ref_e[3] = np.clip(base_ref_e[3],
                -yaw_ref + self.w_des[-1] * t_horizon * 1.5,
                 yaw_ref + self.w_des[-1] * t_horizon * 1.5,
                )
        # Set the base ref inbetween
        base_ref[:2] += (base_ref_e[:2] - base_ref[:2]) * 0.75
        base_ref[3] += (base_ref_e[3] - base_ref[3]) * 0.75
        # Base vertical vel
        base_ref_e[8] = 0.
        # Base pitch roll
        base_ref_e[4:6] = 0.
        base_ref[4:6] = 0.
        # Base pitch roll vel
        base_ref_e[-2:] = 0.

        return base_ref, base_ref_e
    
    def compute_base_ref_cnt_restricted(self,
                                        q_mj : np.ndarray,
                                        contact_locations : np.ndarray) -> None:
        """
        Compute base reference and base terminal reference
        for a given contact plan.
        """
        # Center of first and last set of contact locations
        # That are non zero (default location to [0., 0., 0.])
        cnt_loc = np.unique(contact_locations, axis=1)
        id_non_zero = np.argwhere(
            np.all(cnt_loc != np.zeros(3), axis=-1)
        )
        bin_count = np.bincount(id_non_zero[:, 1])
        # If some set of locations are all zeros(3)
        if len(bin_count) > 0:
            id_first_all_non_zero = np.argmax(bin_count)
            id_last_all_non_zero = len(bin_count) - np.argmax(bin_count[::-1]) - 1
            center_first_cnt = np.mean(cnt_loc[:, id_first_all_non_zero, :], axis=0)
            center_last_cnt = np.mean(cnt_loc[:, id_last_all_non_zero, :], axis=0)
        # All non zero
        else:
            center_first_cnt = np.mean(contact_locations[:, 0, :], axis=0)
            center_last_cnt = np.mean(contact_locations[:, -1, :], axis=0)
            
        # Base references
        base_ref = np.zeros(12)
        base_ref_e = np.zeros(12)
        # Set position
        alpha = 0.35
        base_ref[:2] = alpha * center_first_cnt[:2] + (1-alpha) * center_last_cnt[:2]
        base_ref_e[:2] = center_last_cnt[:2]
        # Height to config
        base_ref[2] = self.config_gait.nom_height + self.height_offset
        base_ref_e[2] = self.config_gait.nom_height + self.height_offset

        # Linear velocity
        # t_plan = self.config_gait.nominal_period
        # v_ref = (center_last_cnt - center_first_cnt) / t_plan
        # base_ref[6:8] = v_ref[:2]

        return base_ref, base_ref_e

    def reset(self) -> None:
        """
        Reset the controller state and reinitialize parameters.
        """
        self.solver.reset()
        self.executor = ThreadPoolExecutor(max_workers=1)  # One thread for asynchronous optimization
        self.optimize_future: Future = None                # Store the future result of optimize
        self.plan_submitted = False                        # Flag to indicate if a new plan is ready

        self.sim_step : int = 0
        self.plan_step : int = 0
        self.current_opt_node : int = 0
        self.node_since_last_opt : int = 0
        self.delay : int = 0
        self.last : int = 0
        
        self.v_des : np.ndarray = np.zeros(3)
        self.w_des : float = np.zeros(3)
        #self.w_des: np.ndarray = np.zeros(3)
        

        self.diverged : bool = False
    
    @time_fn("optimize")
    def optimize(self,
                 q_mj : np.ndarray,
                 v_mj : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        return optimized trajectories.
        """
        # Update model state based on current MuJoCo state
        q, v = self.solver.dyn.convert_from_mujoco(q_mj, v_mj)
        print(f"[DEBUG] In optimize: q {q}, v {v}")
        self.solver.dyn.update_pin(q, v)

        # Update goal
        if self.velocity_goal:
            self.v_des, self.w_des[2] = self.velocity_goal.get_velocity()

        # Contact parameters
        cnt_sequence = self.contact_planner.get_contacts(self.current_opt_node, self.config_opt.n_nodes+1)
        swing_peak = None
        com_xyz = pin.centerOfMass(self.solver.dyn.pin_model, self.solver.dyn.pin_data)
        self.com_z_traj.append(com_xyz[2])
        if self.config_opt.opt_peak:
            swing_peak = self.contact_planner.get_peaks(self.current_opt_node, self.config_opt.n_nodes+1)
        cnt_locations = None
        if self.restrict_cnt:
            if self._contact_planner_str.lower() == "raibert":
                com_xyz = pin.centerOfMass(self.solver.dyn.pin_model, self.solver.dyn.pin_data)
                self.contact_planner.set_state(q[:3], v[:3], q[3:6][::-1], com_xyz, self.v_des, self.w_des[-1])
            cnt_locations = self.contact_planner.get_locations(self.current_opt_node, self.config_opt.n_nodes+1)
        
        # Base reference
            base_ref, base_ref_e = self.compute_base_ref_cnt_restricted(q_mj, cnt_locations)
        else:
            base_ref, base_ref_e = self.compute_base_ref_vel_tracking(q_mj)
        

        self.solver.init(
            self.current_opt_node,
            q,
            v,
            base_ref,
            base_ref_e,
            self.joint_ref,
            self.config_gait.step_height,
            cnt_sequence,
            cnt_locations,
            swing_peak,
            )
        # q_sol, v_sol, a_sol, f_sol, dt_sol = self.solver.solve()
        q_sol, v_sol, a_sol, w_sol, dt_sol = self.solver.solve()
        
        

        return q_sol, v_sol, a_sol, w_sol, dt_sol

    @time_fn("interpolate_trajectory")
    def interpolate_trajectory(
        self,
        traj : np.ndarray,
        time_traj : np.ndarray,
        kind: str = "",
        ) -> np.ndarray:
        """
        Interpolate traj at a sim_dt period.

        Args:
            traj (np.ndarray): Trajectory to interpolate.
            time_traj (np.ndarray): Time at each trajectory elements.

        Returns:
            np.ndarray: trajectory interpolated at a 1/sim_freq
        """
        # Create an interpolation object that supports multi-dimensional input
        interp_func = interp1d(
            time_traj,
            traj,
            axis=0,
            kind = kind,
            fill_value="extrapolate",
            bounds_error=False,
            assume_sorted=True,
            )
        
        # Interpolate the trajectory for all dimensions at once
        t_interpolated = np.linspace(0., time_traj[-1], int(time_traj[-1]/self.sim_dt)+1)
        interpolated_traj = interp_func(t_interpolated)

        return interpolated_traj
    
    def interpolate_sol(self,
                        q_sol : np.ndarray,
                        v_sol : np.ndarray,
                        a_sol : np.ndarray,
                        w_sol : np.ndarray,
                        dt_sol : np.ndarray,
                        ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        Interpolate solution found by the solver at sim dt time intervals.
        Repeat for inputs.
        Linear interpolation for states.
        """
        # Interpolate plan at sim_dt intervals
        input_full = np.concatenate((
            a_sol,
            w_sol.reshape(-1, self.n_foot * 6),
        ), axis=-1)

        time_traj = np.cumsum(dt_sol)
        time_traj = np.concatenate(([0.], time_traj))

        q_plan, v_plan = self.interpolate_trajectory_with_derivatives(time_traj, q_sol, v_sol, a_sol)
        print(f"[DEBUG] After interpolate_trajectory_with_derivatives: q_plan shape: {q_plan.shape}, v_plan shape: {v_plan.shape}")
        input_full_interp = self.interpolate_trajectory(input_full, time_traj[:-1], kind='zero')
        a_plan, w_plan = np.split(
            input_full_interp,
            [a_sol.shape[-1]],
                axis=-1
        )
        w_plan = w_plan.reshape(-1, self.n_foot, 6)
        print(f"[DEBUG] Interpolated a_plan shape: {a_plan.shape}, w_plan shape: {w_plan.shape}")
        return q_plan, v_plan, a_plan, w_plan, time_traj
            
    def interpolate_trajectory_with_derivatives(
        self,
        time_traj: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate trajectory using polynomial interpolation with derivative constraints.

        Args:
            time_traj (np.ndarray): Time at each trajectory element. Shape: (N,).
            positions (np.ndarray): Position trajectory. Shape: (N, d).
            velocities (np.ndarray): Velocity trajectory. Shape: (N, d).

        Returns:
            np.ndarray: Interpolated trajectory at 1/sim_dt frequency. Shape: (T, d).
        """
        t_interpolated = np.arange(0., time_traj[-1], self.sim_dt)
        poly_pos = CubicHermiteSpline(time_traj, positions, velocities)
        interpolated_pos = poly_pos(t_interpolated)
        a0 = (velocities[1] - velocities[0]).reshape(1, -1) / self.dt_nodes
        accelerations = np.concatenate((a0, accelerations))
        poly_vel = CubicHermiteSpline(time_traj, velocities, accelerations)
        interpolated_vel = poly_vel(t_interpolated)

        return interpolated_pos, interpolated_vel

    def open_loop(self,
                  q_mj : np.ndarray,
                  v_mj : np.ndarray,
                  trajectory_time : float) -> Tuple[np.ndarray]:
        """
        Computes trajectory in a MPC fashion starting at q0

        Args:
            q0 (np.ndarray): Initial state
            v0 (np.ndarray): Initial velocities
            trajectory_time (float): Total trajectory time

        Returns:
            np.ndarray: _description_
        """
        q_full_traj = []
        sim_time = 0.
        time_traj = []

        while sim_time <= trajectory_time:

            # Replan trajectory    
            if self._replan():

                # Record trajectory
                if self.sim_step > 0:
                    self._record_plan()

                self.set_convergence_on_first_iter()
                
                # Find the corresponding optimization node
                self.current_opt_node += bisect_right(time_traj, sim_time - self.sim_dt)

                q_sol, v_sol, a_sol, w_sol, dt_sol = self.optimize(q_mj, v_mj)

                (
                self.q_plan,
                self.v_plan,
                self.a_plan,
                self.w_plan,
                time_traj,
                ) = self.interpolate_sol(q_sol, v_sol, a_sol, w_sol, dt_sol)

                time_traj += sim_time
                self.plan_step = 1
                self.delay = 0
            
            # Simulation step
            q_mj, v_mj = self.solver.dyn.convert_to_mujoco(self.q_plan[self.plan_step], self.v_plan[self.plan_step])
            q_full_traj.append(q_mj)
            self._step()
            sim_time = sim_time + self.sim_dt

        q_full_traj_arr = np.array(q_full_traj)
        return q_full_traj_arr
    
    def set_convergence_on_first_iter(self):
        if self.sim_step == 0:
            self.solver.set_max_iter(50)
            self.solver.set_nlp_tol(self.solver.config_opt.nlp_tol / 10.)
            self.solver.set_qp_tol(self.solver.config_opt.qp_tol / 10.)
        elif self.sim_step <= self.replanning_steps:
            self.solver.set_max_iter(self.solver.config_opt.max_iter)
            self.solver.set_nlp_tol(self.solver.config_opt.nlp_tol)
            self.solver.set_qp_tol(self.solver.config_opt.qp_tol)
    
    def get_torques(self, sim_step : int, mj_data: Any) -> Dict[str, float]:
        """
        Compute torques based on robot state in the MuJoCo simulation.
        """
        # Get state in pinocchio format
        q_mj, v_mj = mj_data.qpos, mj_data.qvel
        
        
        # Increment the optimization node every dt_nodes
        # TODO: This may be changed in case of dt time optimization
        # One may update the opt node according to the last dt results
        sim_time = round(mj_data.time, 4)
        print(f"[DEBUG] sim_time: {sim_time}, current_opt_node: {self.current_opt_node}, sim_step: {self.sim_step}")
        if sim_time >= (self.current_opt_node+1) * self.dt_nodes:
            self.current_opt_node += 1
            print(f"[DEBUG] Updated current_opt_node: {self.current_opt_node}")

        # Start a new optimization asynchronously if it's time to replan
        if self._replan():

            # Compute replanning time
            self.start_time = sim_time
            # Set solver parameters on first iteration
            self.set_convergence_on_first_iter()

            # Set up asynchronous optimize call
            self.time_start_plan = sim_time
            self.optimize_future = self.executor.submit(self.optimize, q_mj, v_mj)
            self.plan_submitted = True
            print(f"[DEBUG] Optimize future submitted at sim_time: {sim_time}")
            if self.print_info:
                print()
                print("#"*10, "Replan", "#"*10)
                print("Current node:", self.current_opt_node,
                      "Sim time:", sim_time,
                      "Sim step:", self.sim_step)
                print()

            # Wait for the solver if no delay
            while not self.solve_async and not self.optimize_future.done():
                time.sleep(5.0e-4)

        # Check if the future is done and if the new plan is ready to be used
        if (self.plan_submitted and self.optimize_future.done() or
            self.sim_step == 0):
            try:
                # Retrieve new plan from future
                q_sol, v_sol, a_sol, w_sol, dt_sol = self.optimize_future.result()
                print(f"[DEBUG] Optimize future result shapes: q_sol {q_sol.shape}, v_sol {v_sol.shape}, a_sol {a_sol.shape}, w_sol {w_sol.shape}, dt_sol {dt_sol.shape}")
                # Record trajectory
                if self.sim_step > 0:
                    self._record_plan()

                # Interpolate plan at sim_dt interval
                self.q_plan, self.v_plan, self.a_plan, self.w_plan, self.time_traj = self.interpolate_sol(q_sol, v_sol, a_sol, w_sol, dt_sol)
                print(f"[DEBUG] Interpolated plan shapes: q_plan {self.q_plan.shape}, v_plan {self.v_plan.shape}, a_plan {self.a_plan.shape}, w_plan {self.w_plan.shape}")
                # Apply delay
                if (self.solve_async and self.sim_step != 0):
                    replanning_time = sim_time - self.start_time
                    self.delay = math.ceil(replanning_time / self.sim_dt)
                else:
                    self.delay = 0

                self.plan_step = self.delay
                self.plan_submitted = False
                
                # Plot current state vs optimization plan
                # self.plot_current_vs_plan(q_mj, v_mj)

            except Exception as e:
                print("Optimization error:\n")
                print(traceback.format_exc())
                self.optimize_future: Future = None
                self.diverged = True
                self.plan_submitted = False
                self.executor.shutdown(wait=False, cancel_futures=True)
                time.sleep(0.1)
        
        q, v = self.solver.dyn.convert_from_mujoco(q_mj, v_mj)
        
        idx = min(self.plan_step, self.a_plan.shape[0] - 1)
        if idx != self.plan_step:
            print(f"[DEBUG] Clamping plan_step from {self.plan_step} to {idx}")
        
        
        torques = self.solver.dyn.id_torques(
            q,
            v,
            self.a_plan[idx],
            self.w_plan[idx],
        )
        print(f"[DEBUG] Computed torques: {torques}")

        self.torques_dof[-self.nu:] = (torques +
                      self.Kp * (self.q_plan[self.plan_step, -self.nu:] - q_mj[-self.nu:]) +
                      self.Kd * (self.v_plan[self.plan_step, -self.nu:] - v_mj[-self.nu:]))
        
        torque_map = self.get_torque_map()

        # Record trajectories
        self.tau_full.append(self.torques_dof[-self.nu:].copy())
        
        self._step()

        return torque_map
        
    def plot_current_vs_plan(self, q_mj: np.ndarray, v_mj: np.ndarray):
        """
        Plot the current state vs the optimization plan.
        """
        time_points = np.linspace(0, len(self.q_plan) * self.sim_dt, len(self.q_plan))

        fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        # Plot positions
        axs[0].plot(time_points, self.q_plan[:, -3:], label="Planned Position")
        axs[0].scatter([self.plan_step * self.sim_dt]*3, [q_mj[-3:]], color="red", label="Current Position")
        axs[0].set_title("Position Comparison")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Position")
        axs[0].grid()
        axs[0].legend()

        # Plot velocities
        axs[1].plot(time_points, self.v_plan[:, -3:], label="Planned Velocity")
        axs[1].scatter([self.plan_step * self.sim_dt]*3, [v_mj[-3:]], color="red", label="Current Velocity")
        axs[1].set_title("Velocity Comparison")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Velocity")
        axs[1].grid()
        axs[1].legend()

        plt.tight_layout()
        plt.show()
    
    def plot_traj(self, var_name: str):
        """
        Plot one of the recorded plans using time as the x-axis in a subplot with 3 columns.

        Args:
            var_name (str): Name of the plan to plot. Should be one of:
                            'q', 'v', 'a', 
                            'f', 'dt', 'tau'.
        """
        # 兼容处理：如果传入 "f"（忽略大小写和前后空白），自动映射为 "w"
        if var_name.strip().lower() == "f":
            print("[DEBUG] Mapping 'f' to 'w' for compatibility")
            var_name = "w"
            
        valid_vars = ['q', 'v', 'a', 'w', 'dt', 'tau']
        if var_name not in valid_vars:
            raise ValueError(f"Plan '{var_name}' does not exist. Choose from: {valid_vars}.")
        var_name_full = var_name + "_full"
        if not hasattr(self, var_name_full):
            raise ValueError(f"Plan '{var_name_full}' does not exist. Choose from: {valid_vars}.")
        
        # Get the selected plan and the time intervals (dt)
        plan = getattr(self, var_name_full)
        plan = np.vstack(plan)

        N = len(plan)
        plan = plan.reshape(N, -1)
        time = np.linspace(start=0., stop=(N+1)*self.sim_dt, num=N)

        # Number of dimensions in the plan (columns)
        num_dimensions = plan.shape[1]

        # Calculate the number of rows needed for the subplots
        num_rows = (num_dimensions + 2) // 3  # +2 to account for remaining dimensions if not divisible by 3

        # Create subplots with 3 columns
        fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
        axs = axs.flatten()  # Flatten the axes for easy iteration

        # Plot each dimension of the plan on a separate subplot
        for i in range(num_dimensions):
            axs[i].plot(time, plan[:, i])
            axs[i].set_title(f'{var_name_full} dimension {i+1}')
            axs[i].set_xlabel('Time (s)')
            axs[i].set_ylabel(f'{var_name_full} values')
            axs[i].grid(True)

        # Turn off unused subplots if there are any
        for i in range(num_dimensions, len(axs)):
            fig.delaxes(axs[i])

        plt.tight_layout()


    
    def show_plots(self):
        plt.show()

    def print_timings(self):
        print()
        print_timings(self.timings)
        print_timings(self.solver.timings)

    def __del__(self):
        if self.velocity_goal: self.velocity_goal._stop_update_thread()
    
    def get_com_z_trajectory(self) -> np.ndarray:
        """
        FOR Norm Height
        """
        return np.array(self.com_z_traj)