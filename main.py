import os
import argparse
from typing import List
import pinocchio as pin
import mujoco as mj
import numpy as np
from mj_pin.abstract import VisualCallback, DataRecorder # type: ignore
from mj_pin.simulator import Simulator # type: ignore
from mj_pin.utils import get_robot_description   # type: ignore

from mpc_controller.mpc import LocomotionMPC

SIM_DT = 1.0e-3
VIEWER_DT = 1/30.

class ReferenceVisualCallback(VisualCallback):
    def __init__(self, mpc_controller, update_step = 1):
        super().__init__(update_step)
        self.mpc = mpc_controller
        self.radius = 0.01

    def add_visuals(self, mj_data):
        # Contact locations
        for i, foot_cnt in enumerate(self.mpc.solver.dyn.feet):
            cnt_pos = self.mpc.solver.params[foot_cnt.plane_point.name]
            cnt_pos_unique = np.unique(cnt_pos, axis=1).T
            for pos in cnt_pos_unique:
                if np.sum(pos) == 0.: continue
                self.add_sphere(pos, self.radius, self.colors.id(i))

        # Base reference
        BLACK = self.colors.name("black")
        base_ref = self.mpc.solver.cost_ref[self.mpc.solver.dyn.base_cost.name][:, 0]
        self.add_box(base_ref[:3], rot_euler=base_ref[3:6][::-1], size=[0.08, 0.04, 0.04], rgba=BLACK)
        
        # Base terminal reference
        base_ref = self.mpc.solver.cost_ref_terminal[self.mpc.solver.dyn.base_cost.name]
        self.add_box(base_ref[:3], rot_euler=base_ref[3:6][::-1], size=[0.08, 0.04, 0.04], rgba=BLACK)

class StateDataRecorder(DataRecorder):
    def __init__(
        self,
        record_dir: str = "",
        record_step: int = 1,
    ) -> None:
        """
        A simple data recorder that saves simulation data to a .npz file.
        """
        super().__init__(record_dir, record_step)
        self.data = {}
        self.reset()

    def reset(self) -> None:
        self.data = {"time": [], "q": [], "v": [], "ctrl": [],}

    def save(self) -> None:
        if not self.record_dir:
            self.record_dir = os.getcwd()
        os.makedirs(self.record_dir, exist_ok=True)

        timestamp = self.get_date_time_str()
        file_path = os.path.join(self.record_dir, f"simulation_data_{timestamp}.npz")

        try:
            # Uncomment to save data
            np.savez(file_path, **self.data)
            print(f"Data successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def _record(self, mj_data) -> None:
        """
        Record simulation data at the current simulation step.
        """
        # Record time and state
        self.data["time"].append(round(mj_data.time, 4))
        self.data["q"].append(mj_data.qpos.copy())
        self.data["v"].append(mj_data.qvel.copy())
        self.data["ctrl"].append(mj_data.ctrl.copy())

def run_traj_opt(args):
    # MPC Controller
    #robot_desc = get_robot_description(args.robot_name)
    # feet_frame_names = ["left_ankle_roll_link","right_ankle_roll_link"]
    feet_frame_names = ["left_foot_end_effector_link","right_foot_end_effector_link"]
    path_urdf_biped= "contact_tamp/rsc/g1_description/g1_biped_newee.urdf"
    xml_file = "contact_tamp/rsc/g1_description/g1_biped.xml"
    mj_model = mj.MjModel.from_xml_path(xml_file)
    joint_ref_biped = np.zeros(mj_model.nq)
    mpc = LocomotionMPC(
        path_urdf=path_urdf_biped,
        feet_frame_names = feet_frame_names,
        robot_name=args.robot_name,
        joint_ref = joint_ref_biped,
        sim_dt=SIM_DT,
        print_info=True,
        )
    mpc.set_command(args.v_des, 0.0)
    mpc.set_convergence_on_first_iter()

    q = np.array(mj_model.qpos0)
    v = np.zeros(mpc.pin_model.nv)
    q_plan, v_plan, _, _, dt_plan = mpc.optimize(q, v)
    
    q_plan_mj = np.array([mpc.solver.dyn.convert_to_mujoco(q_plan[i], v_plan[i])[0] for i in range(len(q_plan))])
    time_traj = np.concatenate(([0], np.cumsum(dt_plan)))

    sim = Simulator(xml_file, sim_dt=SIM_DT, viewer_dt=VIEWER_DT)
    sim.vs.set_high_quality()
    sim.visualize_trajectory(q_plan_mj, time_traj, record_video=args.record_video)

def run_mpc(args):
    # MPC Controller
    # robot_desc = get_robot_description(args.robot_name)
    # feet_frame_names = ["left_ankle_roll_link","right_ankle_roll_link"]
    feet_frame_names = ["left_foot_end_effector_link","right_foot_end_effector_link"]
    path_urdf_biped= "contact_tamp/rsc/g1_description/g1_biped_newee.urdf"
    xml_file = "contact_tamp/rsc/g1_description/g1_biped.xml"
    mj_model = mj.MjModel.from_xml_path(xml_file)
    joint_ref_biped = np.zeros(mj_model.nq)
    mpc = LocomotionMPC(
        path_urdf=path_urdf_biped,
        feet_frame_names = feet_frame_names,
        robot_name=args.robot_name,
        joint_ref = joint_ref_biped,
        interactive_goal=args.interactive,
        sim_dt=SIM_DT,
        print_info=False,
        solve_async=False,
        #TODO: Set async to False for testing
        )

    # q_init = np.array(mj_model.qpos0)
    # v_init = np.zeros(mj_model.nv)
    
    # q_sol, v_sol, a_sol, f_sol, dt_sol = mpc.optimize(q_init, v_init)
    
    if not args.interactive:
        mpc.set_command(args.v_des, 0.0)

    # Simulator with visual callback and state data recorder
    vis_feet_pos = ReferenceVisualCallback(mpc)
    data_recorder = StateDataRecorder(args.record_dir) if args.save_data else None

    sim = Simulator(xml_file, sim_dt=SIM_DT, viewer_dt=VIEWER_DT)
    sim.vs.track_obj = "base"
    sim.run(
        sim_time=args.sim_time,
        controller=mpc,
        visual_callback=vis_feet_pos,
        data_recorder=data_recorder
        )
    com_z_traj = mpc.get_com_z_trajectory()
    print("COM z trajectory:", com_z_traj)
    mpc.print_timings()
    mpc.plot_traj("q")
    mpc.plot_traj("f")
    mpc.plot_traj("tau")
    mpc.show_plots()

def run_open_loop(args):
    # MPC Controller
    # robot_desc = get_robot_description(args.robot_name)
    # feet_frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    feet_frame_names = ["left_foot_end_effector_link","right_foot_end_effector_link"]
    # feet_frame_names = ["left_ankle_roll_link","right_ankle_roll_link"]
    path_urdf_biped= "contact_tamp/rsc/g1_description/g1_biped_newee.urdf"
    xml_file = "contact_tamp/rsc/g1_description/g1_biped.xml"
    mj_model = mj.MjModel.from_xml_path(xml_file)
    joint_ref_biped = np.zeros(mj_model.nq)
    mpc = LocomotionMPC(
        path_urdf=path_urdf_biped, 
        feet_frame_names = feet_frame_names,
        robot_name=args.robot_name,
        joint_ref = joint_ref_biped,
        interactive_goal=False,
        sim_dt=SIM_DT,
        print_info=False,
        )
    mpc.set_command(args.v_des, 0.0)

    q = np.array(mj_model.qpos0)
    v = np.zeros(mpc.pin_model.nv)
    q_traj = mpc.open_loop(q, v, args.sim_time)
   
    mpc.print_timings()

    # sim = Simulator(robot_desc.xml_scene_path, sim_dt=SIM_DT)
    sim = Simulator(xml_file, sim_dt=SIM_DT)
    sim.vs.set_high_quality()
    sim.visualize_trajectory(q_traj, record_video=args.record_video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MPC simulations.")
    parser.add_argument('--mode', type=str, default="traj_opt", choices=['traj_opt', 'open_loop', 'close_loop'], help='Mode to run the simulation.')
    #parser.add_argument('--mode', type=str, default="open_loop", choices=['traj_opt', 'open_loop', 'close_loop'], help='Mode to run the simulation.')
    #parser.add_argument('--mode', type=str, default="close_loop", choices=['traj_opt', 'open_loop', 'close_loop'], help='Mode to run the simulation.')
    parser.add_argument('--sim_time', type=float, default=5, help='Simulation time.')
    parser.add_argument('--robot_name', type=str, default='g1', help='Name of the robot.')
    parser.add_argument('--record_dir', type=str, default='./data/', help='Directory to save recorded data.')
    parser.add_argument('--v_des', type=float, nargs=3, default=[0.0, 0.0, 0.0], help='Desired velocity.')
    parser.add_argument('--save_data', action='store_true', help='Flag to save data.')
    parser.add_argument('--interactive', action='store_true', help='Use keyboard to set the velocity goal (zqsd).')
    parser.add_argument('--record_video', action='store_true', help='Record a video of the viewer.')
    args = parser.parse_args()

    if args.mode == 'traj_opt':
        run_traj_opt(args)
    elif args.mode == 'open_loop':
        run_open_loop(args)
    elif args.mode == 'close_loop':
        run_mpc(args)

        
        
        
        
        