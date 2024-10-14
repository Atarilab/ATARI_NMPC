import numpy as np
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mj_pin_wrapper.simulator import Simulator
from mpc_controller.utils.solver import QuadrupedAcadosSolver

robot = MJPinQuadRobotWrapper(
    *RobotModelLoader.get_paths("go2", mesh_dir="assets"),
    rotor_inertia=0.5*0.250*(0.09/2)**2,
    gear_ratio=6.33,
    )

sim = Simulator(robot.mj)

solver = QuadrupedAcadosSolver(
    robot.pin,
    "trot"
)

v_des = np.array([0.2, 0., 0.])
solver.init(v_des=v_des)
q_traj, dt_traj = solver.solve()

sim.vis_trajectory(q_traj, dt_traj, loop=True, record_video=True)