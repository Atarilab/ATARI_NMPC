from typing import List
import pinocchio as pin
import numpy as np
from contact_tamp.traj_opt_acados.models.floating_base_dynamics import FloatingBaseDynamics
from contact_tamp.traj_opt_acados.models.point_contact import PointContact
from contact_tamp.traj_opt_acados.utils.model_utils import loadSymModel
from contact_tamp.traj_opt_acados.interface.acados_helper import ProblemFormulation, cs

class QuadrupedDynamics(FloatingBaseDynamics):
    MU_CONTACT = 0.7

    def __init__(self,
                 robot_name: str,
                 model_path: str,
                 feet_frame_names: List[str],
                 restrict_cnt: bool = False):
        
        model, data = loadSymModel(model_path)
        self.feet_frame_names = feet_frame_names
        self.feet_frame_id = [model.getFrameId(ee_name) for ee_name in self.feet_frame_names]

        super().__init__(robot_name, model, data)

        self.feet = [PointContact(
            dyn=self,
            frame=frame_name,
            mu=QuadrupedDynamics.MU_CONTACT,
            restriction=restrict_cnt) for frame_name in feet_frame_names]

        self.add_contacts(self.feet)
        self.base_cost = self.add_expr(name="base_cost", expr=self.get_base_cost())
        self.joint_cost = self.add_expr(name="joint_cost", expr=self.get_joint_cost())
        self.acc_cost = self.add_expr(name="acc_cost", expr=self.get_acc_cost())
        self.swing_cost = self.add_expr(name="sw_cost", expr=self.get_swing_foot_cost())

    def setup(self, problem: ProblemFormulation):
        for f in self.feet:
            f.setup(problem)
        super().setup(problem)
        problem.add_cost(self.base_cost, terminal=True)
        problem.add_cost(self.joint_cost, terminal=True)
        problem.add_cost(self.acc_cost)
        problem.add_cost(self.swing_cost, terminal=True)

    def get_hg(self):
        return self.h

    def get_base_cost(self):
        r = self.q[:3]  # position cost
        euler = self.q[3:6]
        return cs.vcat([r, euler, self.v[:6]])

    def get_joint_cost(self):
        return cs.vcat([self.q[6:], self.v[6:]])

    def get_acc_cost(self):
        return self.a[6:]

    def get_swing_foot_cost(self):
        z = cs.vcat([c.peak * c.get_position()[2] for c in self.feet])
        return z
    
    def get_torques(self,
                    model : pin.Model,
                    data : pin.Data,
                    q_plan : np.ndarray,
                    v_plan : np.ndarray,
                    a_plan : np.ndarray,
                    f_plan : np.ndarray,
                    ) -> np.ndarray:
        """
        Return torques for desired position, velocity, acceleration
        and external forces plan.

        Args:
            q_plan (np.ndarray): State position plan
            v_plan (np.ndarray): State velocity plan
            a_plan (np.ndarray): Acceleration plan
            f_plan (np.ndarray): Contact forces plan
        
        Return:
            torques:
        """
        # Inverse dynamics torques
        tau = pin.rnea(model, data, q_plan, v_plan, a_plan)[-12:]

        # Loop through each end-effector and accumulate external forces
        for frame_id, f_ee in zip(self.feet_frame_id, f_plan):
            J_ee = pin.computeFrameJacobian(model, data, q_plan, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, -12:]
            tau -= f_ee @ J_ee

        return tau