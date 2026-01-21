from ..interface.acados_helper import *
from ..utils.model_utils import *
from ..models.point_contact import *


class Quadruped(FloatingBaseDynamics):
    def __init__(self, model_path: str):
        self.__raw_model = loadModelImpl(model_path)
        self.__raw_data = self.__raw_model.createData()
        model, data = loadSymModel(model_path)
        super().__init__(model.name, model, data)
        feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        # note: 0.022 is the foot radius
        self.feet = [PointContact(dyn=self, frame=f, mu=0.7) for f in feet]
        # if you want to penalize contact location (horizontal) or restrict it
        # self.feet = [
        #     PointContact(
        #         dyn=self, frame=f, mu=0.7, penalize_loc=True, restriction=True
        #     )
        #     for f in feet
        # ]
        # then you need set the parameters
        # contact location penalty:
        # cost - [contact].pos_cost ref_dim: 3 (x, y, z) world frame
        # contact location restriction, center is [contact].plane_point
        # param - [contact].range_radius dim: 1
        # param - [contact].restrict (1 if restricted else 0)
        
        self.add_contacts(self.feet)
        self.base_cost = self.add_expr(name="base_cost", expr=self.get_base_cost())
        self.acc_cost = self.add_expr(name="acc_cost", expr=self.get_acc_cost())
        self.swing_cost = self.add_expr(name="sw_cost", expr=self.get_swing_foot_cost())

    def get_feet_position(self, q: np.ndarray) -> np.ndarray:
        pinocchio.forwardKinematics(self.__raw_model, self.__raw_data, q)
        pinocchio.updateFramePlacements(self.__raw_model, self.__raw_data)
        return np.array(
            [self.__raw_data.oMf[f.frame_id].translation for f in self.feet]
        ).T

    def setup(self, problem: ProblemFormulation):
        for f in self.feet:
            f.setup(problem)
        super().setup(problem)
        problem.add_cost(self.base_cost, terminal=True)
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
    
    def get_model(self):
        return self.__raw_model
    
    def get_data(self):
        return self.__raw_data
    
class Biped(FloatingBaseDynamics):
    def __init__(self, model_path: str):
        self.__raw_model = loadModelImpl(model_path)
        self.__raw_data = self.__raw_model.createData()
        model, data = loadSymModel(model_path)
        super().__init__(model.name, model, data)
        feet = ["left_ankle_pitch_link", "right_ankle_pitch_link"]
        # note: 0.022 is the foot radius
        self.feet = [PointContact(dyn=self, frame=f, mu=0.7) for f in feet]
        self.add_contacts(self.feet)
        self.base_cost = self.add_expr(name="base_cost", expr=self.get_base_cost())
        self.acc_cost = self.add_expr(name="acc_cost", expr=self.get_acc_cost())
        self.swing_cost = self.add_expr(name="sw_cost", expr=self.get_swing_foot_cost())

    def get_feet_position(self, q: np.ndarray) -> np.ndarray:
        pinocchio.forwardKinematics(self.__raw_model, self.__raw_data, q)
        pinocchio.updateFramePlacements(self.__raw_model, self.__raw_data)
        return np.array(
            [self.__raw_data.oMf[f.frame_id].translation for f in self.feet]
        ).T

    def setup(self, problem: ProblemFormulation):
        for f in self.feet:
            f.setup(problem)
        super().setup(problem)
        problem.add_cost(self.base_cost, terminal=True)
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
