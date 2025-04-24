from typing import List, Tuple
import pinocchio as pin
import numpy as np
from contact_tamp.traj_opt_acados.models.floating_base_dynamics import FloatingBaseDynamics
from contact_tamp.traj_opt_acados.models.point_contact import PointContact
from contact_tamp.traj_opt_acados.utils.model_utils import toSymModel, loadModelImpl
from contact_tamp.traj_opt_acados.interface.acados_helper import ProblemFormulation, cs
from .transform import local_angular_to_euler_derivative, euler_derivative_to_local_angular

class QuadrupedDynamics(FloatingBaseDynamics):

    def __init__(self,
                 urdf_path,
                 feet_frame_names: List[str],
                 cnt_patch_restriction: bool = False,
                 mu_contact: float = 0.7,
                 torque_limit: bool = True,
                 ):
        # Load pinocchio model
        self.__raw_model = loadModelImpl(urdf_path)
        self.__raw_data = self.__raw_model.createData()
        self.nu = self.__raw_model.nv - 6
        # Load symbolic model
        model, data = toSymModel(self.__raw_model)
        super().__init__(model.name, model, data)

        # Init point contact
        self.feet_frame_id = [model.getFrameId(ee_name) for ee_name in feet_frame_names]
        self.feet = [PointContact(
            dyn=self,
            frame=frame_name,
            mu=mu_contact,
            patch_restriction=cnt_patch_restriction) for frame_name in feet_frame_names]

        self.add_contacts(self.feet)
        self.base_cost = self.add_expr(name="base_cost", expr=self.get_base_cost())
        self.joint_cost = self.add_expr(name="joint_cost", expr=self.get_joint_cost())
        self.acc_cost = self.add_expr(name="acc_cost", expr=self.get_acc_cost())
        self.torque_limit = torque_limit
    @property
    def pin_model(self):
        return self.__raw_model
    
    @property
    def pin_data(self):
        return self.__raw_data
    
    def update_pin(self, q : np.ndarray, v : np.ndarray):
        pin.framesForwardKinematics(self.pin_model, self.pin_data, q)
        pin.computeCentroidalMomentum(self.pin_model, self.pin_data, q, v)
    
    @staticmethod
    def convert_from_mujoco(q_mj : np.ndarray, v_mj : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Convert state from MuJoCo to Pinocchio model format
        q = np.zeros(len(q_mj) - 1)
        q[:3] = q_mj[:3]
        R_WB = pin.Quaternion(
                w=q_mj[3],
                x=q_mj[4],
                y=q_mj[5],
                z=q_mj[6]).toRotationMatrix()
        q[3:6] = pin.rpy.matrixToRpy(R_WB)[::-1]
        q[6:] = q_mj[7:]
        # Convert velocties from MuJoCo to Pinocchio model format
        # MuJoCo is v global and w local
        v = v_mj.copy()
        # w local to euler derivatives (z y x)
        # https://github.com/ANYbotics/kindr/blob/master/doc/cheatsheet/cheatsheet_latest.pdf
        v[3:6] = local_angular_to_euler_derivative(q[3:6], v[3:6])

        return q, v
    
    @staticmethod
    def convert_to_mujoco(q : np.ndarray, v : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Convert state from MuJoCo to Pinocchio model format
        q_mj = np.zeros(len(q) + 1)
        R_WB = pin.rpy.rpyToMatrix(q[3:6][::-1])
        quat = pin.Quaternion(R_WB)
        q_mj[:3] = q[:3]
        q_mj[4:7] = quat.coeffs()[:-1]
        q_mj[3] = quat.coeffs()[-1]
        q_mj[7:] = q[6:]
        # Convert velocties from MuJoCo to Pinocchio model format

        R_WB = pin.Quaternion(
                w=q_mj[3],
                x=q_mj[4],
                y=q_mj[5],
                z=q_mj[6]).toRotationMatrix()
        q[3:6] = pin.rpy.matrixToRpy(R_WB)[::-1]
        q[6:] = q_mj[7:]
        # Convert velocties from MuJoCo to Pinocchio model format
        # MuJoCo is v global and w local
        v_mj = v.copy()
        # w local to euler derivatives (z y x)
        # https://github.com/ANYbotics/kindr/blob/master/doc/cheatsheet/cheatsheet_latest.pdf
        v_mj[3:6] = euler_derivative_to_local_angular(q[3:6], v[3:6])

        return q_mj, v_mj
    
    def get_feet_position_w(self):
        feet_pos = np.array([
            self.__raw_data.oMf[frame_id].translation
            for frame_id in
            self.feet_frame_id])
        
        return feet_pos

    def setup(self, problem: ProblemFormulation):
        super().setup(problem)
        if self.torque_limit:
            self.add_torque_limit(problem)

        problem.add_cost(self.base_cost, terminal=True)
        problem.add_cost(self.joint_cost, terminal=True)
        problem.add_cost(self.acc_cost)

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

    def id_torques(self,
                   q_plan : np.ndarray,
                   v_plan : np.ndarray,
                   a_plan : np.ndarray,
                   f_plan : np.ndarray,
                   ) -> np.ndarray:
        """
        Return torques for desired position, velocity, acceleration
        and external forces plan.

        Args:
            q_plan (np.ndarray): State position plan [px, py, pz, y, p, r, joints]
            v_plan (np.ndarray): State velocity plan [vx, vy, vz, wz, wy, wx, joints]
            a_plan (np.ndarray): Acceleration plan   (idem)
            f_plan (np.ndarray): Contact forces plan
        
        Return:
            torques:
        """
        # Inverse dynamics torques
        tau = pin.rnea(self.pin_model, self.pin_data, q_plan, v_plan, a_plan)[-self.nu:]

        # Loop through each end-effector and accumulate external forces
        for frame_id, f_ee in zip(self.feet_frame_id, f_plan):
            J_ee = pin.computeFrameJacobian(self.pin_model, self.pin_data, q_plan, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, -self.nu:]
            tau -= f_ee @ J_ee

        return tau