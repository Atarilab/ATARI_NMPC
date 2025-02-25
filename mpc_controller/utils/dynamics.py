from typing import List, Tuple
import pinocchio as pin
import numpy as np
from contact_tamp.traj_opt_acados.models.floating_base_dynamics import FloatingBaseDynamics
from contact_tamp.traj_opt_acados.models.point_contact import PointContact, PlaneContact, PlaneContact_Wrench
from contact_tamp.traj_opt_acados.utils.model_utils import toSymModel, loadModelImpl
from contact_tamp.traj_opt_acados.interface.acados_helper import ProblemFormulation, cs
from .transform import local_angular_to_euler_derivative, euler_derivative_to_local_angular

class QuadrupedDynamics(FloatingBaseDynamics):

    def __init__(self,
                 urdf_path,
                 feet_frame_names: List[str],
                 cnt_patch_restriction: bool = False,
                 mu_contact: float = 0.7,
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
        self.swing_cost = self.add_expr(name="sw_cost", expr=self.get_swing_foot_cost())

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

class BipedDynamics (FloatingBaseDynamics):

    def __init__(self,
                 urdf_path,
                 feet_frame_names: List[str],
                 cnt_patch_restriction: bool = False,
                 mu_contact: float = 0.7,
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
        self.feet = [PlaneContact_Wrench(
            dyn=self,
            frame=frame_name,
            mu=mu_contact,
            patch_restriction=cnt_patch_restriction) for frame_name in feet_frame_names]

        self.add_contacts(self.feet)
        self.base_cost = self.add_expr(name="base_cost", expr=self.get_base_cost())
        self.joint_cost = self.add_expr(name="joint_cost", expr=self.get_joint_cost())
        self.acc_cost = self.add_expr(name="acc_cost", expr=self.get_acc_cost())
        self.swing_cost = self.add_expr(name="sw_cost", expr=self.get_swing_foot_cost())

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
    
    # def id_torques(self,
    #                q_plan : np.ndarray,
    #                v_plan : np.ndarray,
    #                a_plan : np.ndarray,
    #                w_plan : np.ndarray,
    #                ) -> np.ndarray:
    #     """
    #     Return torques for desired position, velocity, acceleration
    #     and external forces plan.

    #     Args:
    #         q_plan (np.ndarray): State position plan [px, py, pz, y, p, r, joints]
    #         v_plan (np.ndarray): State velocity plan [vx, vy, vz, wz, wy, wx, joints]
    #         a_plan (np.ndarray): Acceleration plan   (idem)
    #         f_plan (np.ndarray): Contact forces plan
        
    #     Return:
    #         torques:
    #     """
    #     tau = pin.rnea(self.pin_model, self.pin_data, q_plan, v_plan, a_plan)[-self.nu:]
    #     for frame_id, wrench_ee in zip(self.feet_frame_id, w_plan):
    #         J_ee=pin.computeFrameJacobian(self.pin_model, self.pin_data, q_plan, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    #         tau -= wrench_ee[:3] @ J_ee[:3, -self.nu:]
    #         tau -= wrench_ee[3:] @ J_ee[3:6, -self.nu:]
    def id_torques(self,
               q_plan: np.ndarray,
               v_plan: np.ndarray,
               a_plan: np.ndarray,
               w_plan: np.ndarray,
               ) -> np.ndarray:
        """
        Return torques for desired position, velocity, acceleration
        and external forces plan.
        """
        tau = pin.rnea(self.pin_model, self.pin_data, q_plan, v_plan, a_plan)[-self.nu:]
        print("[DEBUG] tau type:", type(tau), "shape:", tau.shape)
        for i, (frame_id, wrench_ee) in enumerate(zip(self.feet_frame_id, w_plan)):
            print(f"[DEBUG] wrench_ee[{i}] type: {type(wrench_ee)}, shape: {wrench_ee.shape}")
            J_ee = pin.computeFrameJacobian(self.pin_model, self.pin_data, q_plan, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            print(f"[DEBUG] J_ee[{i}] type: {type(J_ee)}, shape: {J_ee.shape}")
            tau -= wrench_ee[:3] @ J_ee[:3, -self.nu:]
            tau -= wrench_ee[3:] @ J_ee[3:6, -self.nu:]
        print("[DEBUG] Final tau:", tau)
        return tau


# class BipedDynamics(FloatingBaseDynamics):


#     def __init__(self,
#                  urdf_path,
#                  feet_frame_names: List[str],
#                  cnt_patch_restriction: bool = False,
#                  mu_contact: float = 0.7,
#                  ):
#         # Load pinocchio model
#         self.__raw_model = loadModelImpl(urdf_path)
#         self.__raw_data = self.__raw_model.createData()
#         self.nu = self.__raw_model.nv - 6
#         # Load symbolic model
#         model, data = toSymModel(self.__raw_model)
#         super().__init__(model.name, model, data)

#         # Init point contact
#         self.feet_frame_id = [model.getFrameId(ee_name) for ee_name in feet_frame_names]
#         self.feet = [PlaneContact(
#             dyn=self,
#             frame=frame_name,
#             mu=mu_contact,
#             patch_restriction=cnt_patch_restriction) for frame_name in feet_frame_names]

#         self.add_contacts(self.feet)
#         self.base_cost = self.add_expr(name="base_cost", expr=self.get_base_cost())
#         self.joint_cost = self.add_expr(name="joint_cost", expr=self.get_joint_cost())
#         self.acc_cost = self.add_expr(name="acc_cost", expr=self.get_acc_cost())
#         self.swing_cost = self.add_expr(name="sw_cost", expr=self.get_swing_foot_cost())
    
#     @property
#     def pin_model(self):
#         return self.__raw_model
    
#     @property
#     def pin_data(self):
#         return self.__raw_data
    
#     def update_pin(self, q: np.ndarray, v: np.ndarray):
#         # forward kinematics and Centroidal Momentum
#         pin.framesForwardKinematics(self.pin_model, self.pin_data, q)
#         pin.computeCentroidalMomentum(self.pin_model, self.pin_data, q, v)

#     @staticmethod
#     def convert_from_mujoco(q_mj : np.ndarray, v_mj : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Ref: QuadrupedDynamics MuJoCo -> Pinocchio
#         """
#         q = np.zeros(len(q_mj) - 1)
#         q[:3] = q_mj[:3]
#         R_WB = pin.Quaternion(
#                 w=q_mj[3],
#                 x=q_mj[4],
#                 y=q_mj[5],
#                 z=q_mj[6]).toRotationMatrix()
#         q[3:6] = pin.rpy.matrixToRpy(R_WB)[::-1]
#         q[6:] = q_mj[7:]

#         v = v_mj.copy()
#         v[3:6] = local_angular_to_euler_derivative(q[3:6], v[3:6])
#         return q, v

#     @staticmethod
#     def convert_to_mujoco(q: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Ref: QuadrupedDynamics Pinocchio -> MuJoCo 
#         """
#         q_mj = np.zeros(len(q) + 1)
#         R_WB = pin.rpy.rpyToMatrix(q[3:6][::-1])
#         quat = pin.Quaternion(R_WB)
#         q_mj[:3] = q[:3]
#         q_mj[4:7] = quat.coeffs()[:-1]
#         q_mj[3] = quat.coeffs()[-1]
#         q_mj[7:] = q[6:]

#         v_mj = v.copy()
#         v_mj[3:6] = euler_derivative_to_local_angular(q[3:6], v[3:6])
#         return q_mj, v_mj

#     def get_feet_position_w(self):
#         """
#         return the position of the feet in world frame
#         """
#         feet_pos = np.array([
#             self.__raw_data.oMf[f_id].translation
#             for f_id in self.feet_frame_id])
#         return feet_pos

#     def setup(self, problem: ProblemFormulation):
#         """
#         Add contact variables and constraints to the optimization problem,
#         """
#         for contact in self.feet:
#             contact.setup(problem)
#         super().setup(problem)


#         problem.add_cost(self.base_cost, terminal=True)
#         problem.add_cost(self.joint_cost, terminal=True)
#         problem.add_cost(self.acc_cost)
#         problem.add_cost(self.swing_cost, terminal=False) 
#         problem.add_cost(self.swing_cost, terminal=True)

#     def get_hg(self):
#         """
#         return FloatingBaseDynamics.h (centroidal dynamics)
#         """
#         return self.h

#     # ========== Cost ==========

#     def get_base_cost(self):
#         """
#         cost of base position, orientation and velocity
#         """
#         r = self.q[:3]  
#         euler = self.q[3:6]
#         return cs.vertcat(r, euler, self.v[:6])

#     def get_joint_cost(self):
#         """
#         cost of joint position and velocity
#         """
#         return cs.vertcat(self.q[6:], self.v[6:])

#     def get_acc_cost(self):
#         """
#         cost of joint acceleration
#         """
#         return self.a[6:]

#     def get_swing_foot_cost(self):
#         """
#         cost of swing foot height
#         """
#         z_list = [c.get_position()[2] for c in self.feet]
#         return cs.vertcat(*z_list)

#     # ========== IK ==========
#     def id_torques(
#         self,
#         q_plan: np.ndarray,
#         v_plan: np.ndarray,
#         a_plan: np.ndarray,
#         #wrenches_plan: np.ndarray,
#         forces_plan: np.ndarray,
#     ) -> np.ndarray:
#         """
#         Return joint torques for given q, v, a and external forces/torques
#         """
#         # calculate pure joint torques
#         tau = pin.rnea(self.pin_model, self.pin_data, q_plan, v_plan, a_plan)[-self.nu:]

#         # recorrect the joint torques with external forces/torques with the help of Jacobian
#         # For each foot, wrench[:3] = force, wrench[3:] = torque
#         # for frame_id, wrench_ee in zip(self.feet_frame_id, wrenches_plan):
#         #     f_ee = wrench_ee[:3]
#         #     tau_ee = wrench_ee[3:]

#         #     # Feet frame Jacobian
#         #     J_ee = pin.computeFrameJacobian(
#         #         self.pin_model, self.pin_data, q_plan,
#         #         frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
#         #     )

#         #     tau -= f_ee @ J_ee[:3, -self.nu:]
#         #     tau -= tau_ee @ J_ee[3:6, -self.nu:]
#         #     # Different from Quadruped, torque also need to be considered
        
#         for frame_id, force_ee in zip(self.feet_frame_id, forces_plan):
#         # Feet frame Jacobian
#             J_ee = pin.computeFrameJacobian(
#                 self.pin_model, self.pin_data, q_plan,
#                 frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
#             )

#             tau -= force_ee @ J_ee[:3, -self.nu:]

#         return tau