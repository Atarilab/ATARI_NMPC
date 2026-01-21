import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir,'traj_opt_acados'))
sys.path.append(os.path.join(current_dir,'visualization'))

from contact import *
from model_utils import *
from inter_contact import *
from acados_helper import *
from visualization.vis_tools import vis_robot

def euler_from_quaternion(quat_):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_[0]
    y = quat_[1]
    z = quat_[2]
    w = quat_[3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = cs.atan2(t0, t1)

    # pitch out of range not considered
    t2 = +2.0 * (w * y - z * x)
    pitch_y = cs.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = cs.atan2(t3, t4)

    return cs.vcat([yaw_z, pitch_y, roll_x])  # in radians


class CentroidalKinoDynamics(Dynamics):
    def __init__(self, *args):
        super().__init__(*args)

    def setup(self, problem: ProblemFormulation):
        super().setup(problem)
        # problem.add_cost(self.get_base_cost())
        # problem.add_cost(self.get_contact_pos_cost())

    def get_base_cost(self):
        r = self.q[:3]  # position cost
        # quat = self.q[3:7]  # orientation cost
        # p = pin.XYZQUATToSE3(cs.vcat([[0.0] * 3, quat]))
        # euler = euler_from_quaternion(quat)
        euler = self.q[3:6]
        return cs.vcat([r, euler, self.v[:6]])

    def get_acc_cost(self):
        return self.a[6:]

    def get_contact_pos_cost(self):
        rc = []
        for c in self.contacts:
            rc.append(c.get_position())

        return cs.vcat(rc)


class ObjectDynamics(Dynamics):
    def __init__(self, *args):
        super().__init__(*args)


if __name__ == "__main__":
    # feet0 = ["FL_foot", "RL_foot", "RR_foot"]
    # feet1 = ["FR_foot", "RL_foot", "RR_foot"]
    #feet0 = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    feet0 = ["FL_foot","RR_foot"]
    # r_des = [
    #     np.array([0.1934, 0.142, 0.01320294]),
    #     np.array([0.1934, -0.142, 0.01320294]),
    #     np.array([-0.1934, 0.142, 0.01320294]),
    #     np.array([-0.1934, -0.142, 0.01320294]),
    # ]
    # feet1 = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    # feet2 = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    N = 50
    dyns = []
    phases = []
    mpf = SinglePhaseFormulation()

    # desired base pos
    xb_des = np.array([0.0, 0.0, 0.8] + [0.0] * 3 + [0.0] * 6)

    for i in [0]:
        feet = eval(f"feet{i}")
        problem = ProblemFormulation(dt_min=1e-2, dt_max=3e-2)
        dyn = CentroidalKinoDynamics(
            f"go2_{i}", *loadModel("rsc/go2_description/urdf/go2_description.urdf")
        )
        dyn.add_contacts(
            contactType=PointContact,
            frames=feet,
            # r_des=r_des,
            mu=0.7,
            z_des=0.013,
            p_gain=1000.0,
            v_gain=10.0,
        )
        dyn.setup(problem)
        # to add constraint
        # problem.add_eq_constr
        # problem.add_eq_constr_terminal
        # problem.add_ineq_constr
        # problem.add_ineq_constr_terminal

        # to add cost
        # problem.add_cost(casadi.SX)
        # problem.add_cost_terminal(casadi.SX)
        problem.add_cost(
            dyn.get_base_cost(),
            np.array([10.] * 6 + [0.1] * 6),
            name=dyn.name + "_base",
        )
        problem.add_cost(
            dyn.get_acc_cost(),
            np.array([1e-3] * 12),
            name=dyn.name + "_acc",
        )
        if i == 0:
            problem.add_cost_terminal(
                dyn.get_base_cost(),
                np.array([10.0] * 6 + [0.1] * 6),
                name=dyn.name + "_base_terminal",
            )
            # cost = dyn.get_contact_pos_cost()
            # problem.add_cost_terminal(cost, 10 * np.ones(12), name=dyn.name + "_fpos")

        phases.append(Phase(N, problem))
        mpf.set_phase(phases[-1])
        mpf.set_cost_ref(
            phase=phases[-1],
            name=dyn.name + "_base",
            y_ref=[xb_des] * N,
        )
        if i == 0:
            mpf.set_cost_terminal_ref(
                phase=phases[-1],
                name=dyn.name + "_base_terminal",
                y_ref=xb_des,
            )
        # np.array([0.1, 0.1, 0.] + [0.1, -0.1, 0.] + [-0.1, 0.1, 0.] + [-0.1, -0.1, 0.])
        dyns.append(dyn)

    mpf.setup(recompile=True)
    # mpf.setup(recompile=False)
    x0 = np.array(
        [0.0] * 6
        + [0.0, 0.0, 0.31, 0.0, 0.0, 0.0]
        + +2 * [0.0, 0.8, -1.6]
        + 2 * [0.0, 0.8, -1.6]
        + [0.0] * dyn.model.nv
    )
    [mpf.update_phase_cost(p) for p in phases]
    mpf.solve(x0=x0)
    mpf.get_solution(phase=phases[0])
    
    # Extract and print center of mass and joint positions
    q = phases[0].state["go2_0"]
    u = phases[0].input["go2_0"]
    
    print("Center of Mass Trajectory:")
    com_trajectory = []
    for i in range(N):
        com_pose = q[i][6:12]
        print(f"Time step {i}: {com_pose}")
        com_trajectory.append(com_pose)
    
    print("\nJoint Positions:")
    joint_trajectory = []
    for i in range(N):
        joint_position = q[i][12:24]
        print(f"Time step {i}: {joint_position}")
        joint_trajectory.append(joint_position)
    
    vis_robot(com_trajectory, joint_trajectory)
