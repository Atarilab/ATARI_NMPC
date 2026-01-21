from .floating_base_dynamics import *
from casadi import SX

class PointContact(AbstractModel):
    def __init__(
        self,
        dyn: FloatingBaseDynamics,
        frame: str,
        mu: float,
        patch_restriction: bool = False,
    ):
        super().__init__(name=f"{frame}_{dyn.name}")
        self.model = dyn.model
        self.data = dyn.data
        self.frame_name = frame
        self.q, self.v, self.a = dyn.q, dyn.v, dyn.a
        self.frame_id = self.model.getFrameId(frame)
        if self.frame_id >= len(self.model.frames):
            raise Exception(f"wrong frame name: {frame} for model: {self.model.name}")
        self.f = self.add_sym("f", 3)
        self.mu = mu
        self.p_gain = self.add_sym("kp")
        self.active = self.add_sym("active")
        self.restrict = self.add_sym("restrict")  # flag for restrictive contact region
        self.peak = self.add_sym("peak")

        # Plane parameters
        self.plane_point = self.add_sym("plane_point", 3)
        self.plane_normal = self.add_sym("plane_normal", 3)
        self.plane_rot = self.add_sym("plane_rot", 9)
        self.plane_border_size = 0.035

        # Constraints
        self.kin_constr = self.add_expr(name="c", expr=self.get_kin_expr())
        self.fric_cone = self.add_expr(name="f", **self.get_friction_cone())
        self.zero_force = self.add_expr(name="zf", expr=self.zero_force())
        self.f_reg = self.add_expr(name="f_reg", expr=copy.copy(self.f))
        self.foot_above_plane = self.add_expr(name="cnt_above", **self.get_contact_above_plane_constraint())
        
        # Cost
        self.swing_cost = self.add_expr(name="sw_cost", expr=self.get_swing_foot_cost())
        self.eeff_orientation_cost = self.add_expr(name="eeff_ori_cost", expr=self.get_contact_direction_along_normal_cost())
        
        # Contact has to be wihtin L1 dist from plane point
        # Add a cost on the contact position
        # (reference set to the next and last contact point) 
        self.__patch_restriction = patch_restriction
        if patch_restriction:
            self.size = self.add_sym("size_xy", 2)
            self.patch_cnt_constraint = self.add_expr(name="rg", **self.get_range_constr_lin())
            self.pos_cost = self.add_expr(name="rc", expr=self.get_position_at_contact())
            self.v_swing = self.add_expr(name="v_swing", expr=self.tangential_velocity_swing_cost())
            self.below_plane = self.add_expr(name="below_plane", expr=self.cost_below_plane())
            self.plane_border = self.add_expr(name="plane_border", expr=self.cost_on_plane_border())

    def setup(self, problem: ProblemFormulation):
        problem.add_input(self.f)
        problem.add_parameter(self.p_gain)
        problem.add_parameter(self.active)
        problem.add_parameter(self.peak)
        if self.__patch_restriction:
            problem.add_parameter(self.restrict)
            problem.add_parameter(self.size)
            problem.add_parameter(self.plane_rot)
            problem.add_ineq_constr(initial=False, **self.patch_cnt_constraint)
            problem.add_cost(self.pos_cost, terminal=True)
            problem.add_cost(self.v_swing, terminal=True)
            problem.add_cost(self.below_plane, terminal=True)
            problem.add_cost(self.plane_border, terminal=True)
        problem.add_parameter(self.plane_point)
        problem.add_parameter(self.plane_normal)
        problem.add_cost(self.f_reg, terminal=False)
        problem.add_cost(self.swing_cost, terminal=True)
        problem.add_cost(self.eeff_orientation_cost, terminal=True)

        problem.add_obs(self.r_c)
        problem.add_eq_constr(
            expr=self.kin_constr,
            initial=False,
            slack_idx=[0],
        )
        problem.add_ineq_constr(**self.fric_cone, terminal=False)
        problem.add_ineq_constr(**self.foot_above_plane, terminal=True)

    def has_contact_force(self):
        return True

    def has_contact_torque(self):
        return False

    def get_swing_foot_cost(self):
        # perpandicular distance to the plane
        d_normal = cs.dot((self.r_c - self.plane_point), self.plane_normal)
        return (1 - self.active) * d_normal
    
    def contact_force(self):
        return self.active * self.f

    def get_position(self):
        return self.r_c
    
    def get_position_at_contact(self):
        return self.r_c * self.restrict

    def get_friction_cone(self):
        # h = (self.mu * self.f[2]) ** 2 - cs.sumsqr(self.f[0:2])
        # Project force onto the plane
        f_normal = cs.dot(self.f, self.plane_normal)
        f_proj = self.f - f_normal * self.plane_normal
        h = cs.vcat([f_normal, (self.mu * f_normal) ** 2 - cs.sumsqr(f_proj)])
        return {
            "expr": self.active * h,
            "lb": [0.0] * 2,
            "ub": [1e11] * 2,
        }

    # -radius <= d_on_patch <= radius
    def get_range_constr_lin(self):
        # Translation in local patch frame
        # P_p_C = P_R_W @ P_p_C - P_R_W @ W_p_P
        P_p_C = cs.transpose(cs.reshape(self.plane_rot, (3, 3))) @ (self.r_c - self.plane_point)
        
        # coordinates on x, y (normal is direction z)
        h = cs.fabs(P_p_C[:2]) - self.size
        
        return {
            "expr": self.active * h,
            "lb": [-1e11, -1e11],
            "ub": [0., 0.],
        }
        
    def cost_on_plane_border(self):
        P_p_C = cs.transpose(cs.reshape(self.plane_rot, (3, 3))) @ (self.r_c - self.plane_point)
        stiffness = 20
        return self.active * (cs.tanh(- stiffness * (1 - (cs.fabs(P_p_C[:2]) / (self.size - self.plane_border_size)))) + 1)
    
    def cost_below_plane(self):
        P_p_C = cs.transpose(cs.reshape(self.plane_rot, (3, 3))) @ (self.r_c - self.plane_point)
        z = P_p_C[-1]
        beta = 1/1000000
        return (1 + np.sign(z)) * (beta / (beta + z**4)) + (1 - np.sign(z)) * ((5*z)**2 + 1)
        
    def tangential_velocity_swing_cost(self):
        norm_dot = cs.dot(self.v_c.linear, self.plane_normal)
        proj_vel = self.v_c.linear - norm_dot * self.plane_normal
        return (1 - self.active) * cs.fabs(proj_vel[:2]) * cs.fabs(norm_dot)

    def zero_force(self):
        return (1 - self.active) * self.f

    def get_kin_expr(self):
        pin.forwardKinematics(self.model, self.data, self.q, self.v)
        pin.updateFramePlacements(self.model, self.data)
        r_c = self.data.oMf[self.frame_id].translation
        self.r_c = self.add_expr("r_c", r_c)
        self.v_c = pin.getFrameVelocity(
            self.model, self.data, self.frame_id, pinocchio.LOCAL_WORLD_ALIGNED
        )

        # Calculate signed distance from contact point to plane
        signed_distance = cs.dot(r_c - self.plane_point, self.plane_normal)

        # constraint stabilization
        norm_vel = cs.dot(self.v_c.linear, self.plane_normal) * self.plane_normal
        proj_vel = self.v_c.linear - norm_vel
        kin_constr = cs.vcat([self.p_gain * signed_distance, proj_vel])

        return self.active * kin_constr

    def get_eeff_dot_prod_to_normal(self):
        pin.forwardKinematics(self.model, self.data, self.q, self.v)
        pin.updateFramePlacements(self.model, self.data)
        foot_up_direction = self.data.oMf[self.frame_id].rotation[:, -1] # z axis pointing up for go2
        dot = cs.dot(foot_up_direction, self.plane_normal)
        
        return dot
    
    def get_contact_above_plane_constraint(self):
        # dot positive => eeff oriented above the plane
        return {
            "expr": self.active * self.get_eeff_dot_prod_to_normal(),
            "lb": [0.],
            "ub": [1e11],
        }

    def get_contact_direction_along_normal_cost(self):
        # dot = 1 means normal and eeff are colinear
        return self.active * (1 - self.get_eeff_dot_prod_to_normal())


class PlaneContact(AbstractModel):
    def __init__(
        self,
        dyn: FloatingBaseDynamics,
        frame: str,
        mu: float,
        patch_restriction: bool = False,
    ):
        super().__init__(name=f"{frame}_{dyn.name}")
        self.model = dyn.model
        self.data = dyn.data
        self.frame_name = frame
        self.q, self.v, self.a = dyn.q, dyn.v, dyn.a
        self.frame_id = self.model.getFrameId(frame)
        if self.frame_id >= len(self.model.frames):
            raise Exception(f"wrong frame name: {frame} for model: {self.model.name}")
        self.f = self.add_sym("f", 3)
        self.mu = mu
        self.p_gain = self.add_sym("kp")
        self.active = self.add_sym("active")
        self.restrict = self.add_sym("restrict")  # flag for restrictive contact region
        self.impact = self.add_sym("impact")
        self.peak = self.add_sym("peak")

        # Plane parameters
        self.plane_point = self.add_sym("plane_point", 3)
        self.plane_normal = self.add_sym("plane_normal", 3)

        # constraints
        self.kin_constr = self.add_expr(name="c", expr=self.get_kin_expr())
        self.fric_cone = self.add_expr(name="f", **self.get_friction_cone())
        self.zero_force = self.add_expr(name="zf", expr=self.zero_force())
        self.f_reg = self.add_expr(name="f_reg", expr=copy.copy(self.f))
        

        # Contact has to be wihtin L1 dist from plane point
        # Add a cost on the contact position
        # (reference set to the next and last contact point) 
        self.__patch_restriction = patch_restriction
        if patch_restriction:
            self.range_radius = self.add_sym("rg_r")
            self.range = self.add_expr(name="rg", **self.get_range_constr_lin())
            self.pos_cost = self.add_expr(name="rc", expr=self.get_position_at_contact())
        else:
            self.range_radius = self.add_sym("rg_r")  
            # self.range = self.add_expr(name="rg", expr=SX.zeros(4, 1))   
            self.pos_cost = self.add_expr(name="rc", expr=SX.zeros(3, 1))
        # self.z_thresh = 0.05
        self.z_thresh = 0.02
        
        self.z_constr = self.add_expr(name="z_c", expr=self.get_z_constr())
    def setup(self, problem: ProblemFormulation):
        problem.add_input(self.f)
        problem.add_parameter(self.p_gain)
        problem.add_parameter(self.active)
        problem.add_parameter(self.impact)
        problem.add_parameter(self.peak)
        problem.add_parameter(self.restrict)
        problem.add_parameter(self.range_radius)
        if self.__patch_restriction:
            problem.add_ineq_constr(initial=False, **self.range)
        problem.add_cost(self.pos_cost, terminal=True)
        problem.add_parameter(self.plane_point)
        problem.add_parameter(self.plane_normal)
        problem.add_cost(self.f_reg, terminal=False)
        problem.add_obs(self.r_c)
        problem.add_eq_constr(
            expr=self.kin_constr,
            initial=True,
            slack_idx=[0],
        )
        # slack[0]: Relaxation corresponding to the vertical position error of the foot (p_gain × signed_distance).
        # slack[1], slack[2], slack[3]: Relaxation corresponding to the tangential velocity error of the foot in the plane (proj_vel).
        # slack[4], slack[5], slack[6]: Relaxation corresponding to the rotational error of the foot (log3(R_foot)).
        # slack[7], slack[8], slack[9]: Relaxation corresponding to the angular velocity error of the foot (v_c.angular).
        problem.add_ineq_constr(**self.fric_cone, terminal=False)
        #problem.add_eq_constr(expr=self.zero_force, terminal=False,initial=True)
        problem.add_ineq_constr(
            expr=self.z_constr,
            lb=[0.0],
            ub=[self.z_thresh],
            initial=True,terminal=True
        )

    def has_contact_force(self):
        return True

    def has_contact_torque(self):
        return False

    def contact_force(self):
        return self.active * self.f

    def get_position(self):
        return self.r_c
    
    def get_position_at_contact(self):
        return self.r_c * self.restrict

    def get_friction_cone(self):
        # h = (self.mu * self.f[2]) ** 2 - cs.sumsqr(self.f[0:2])
        # Project force onto the plane
        f_normal = cs.dot(self.f, self.plane_normal)
        f_proj = self.f - f_normal * self.plane_normal
        h = cs.vcat([f_normal, (self.mu * f_normal) ** 2 - cs.sumsqr(f_proj)])
        return {
            "expr": self.active * h,
            "lb": [0.0] * 2,
            "ub": [1e11] * 2,
        }

    # -radius <= r_c-range_center <= radius
    def get_range_constr_lin(self):
        d = self.r_c[:2] - self.plane_point[:2]
        h = cs.repmat(d, 2, 1)
        h[:2] -= self.range_radius
        h[2:] += self.range_radius
        return {
            "expr": self.restrict * self.active * h,
            "lb": [-1e11, -1e11, 0.0, 0.0],
            "ub": [0.0, 0.0, 1e11, 1e11],
        }

    def zero_force(self):
        return (1 - self.active) * self.f

    def get_kin_expr(self):
        pin.forwardKinematics(self.model, self.data, self.q, self.v)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.frame_id]
        r_c = oMf.translation
        self.r_c = self.add_expr("r_c", r_c)
        R_foot = oMf.rotation
        self.v_c = pin.getFrameVelocity(self.model, self.data, self.frame_id, pinocchio.LOCAL_WORLD_ALIGNED)
        signed_distance = cs.dot(r_c - self.plane_point, self.plane_normal)
        signed_distance=0#TODO 
        norm_vel = cs.dot(self.v_c.linear, self.plane_normal) * self.plane_normal
        proj_vel = self.v_c.linear - norm_vel
        kin_constr_lin = cs.vcat([self.p_gain * signed_distance, proj_vel])
        e_R = pin.log3(R_foot)
        kin_constr_ang = self.p_gain * e_R
        kin_constr_omega = self.v_c.angular 
        kin_constr = cs.vertcat(kin_constr_lin, kin_constr_ang, kin_constr_omega)
        return (1.0 - self.impact) * self.active * kin_constr
    def get_z_constr(self):
        # 单独提取eeframe的z坐标作为约束表达式，
        # 该约束将由setup中加入不等式约束，限定其在 [0, z_thresh] 内
        return self.r_c[2]


class PlaneContact_Wrench(AbstractModel):
    '''
    Extend the PlaneContact model to include the contact wrench (6 degrees wrench)
    '''
    def __init__(
        self,
        dyn: FloatingBaseDynamics,
        frame: str,
        mu: float,
        patch_restriction: bool = False,
    ):
        super().__init__(name=f"{frame}_{dyn.name}") 
        self.model = dyn.model
        self.data = dyn.data
        self.frame_name = frame
        self.q, self.v, self.a = dyn.q, dyn.v, dyn.a
        self.frame_id = self.model.getFrameId(frame)
        if self.frame_id >= len(self.model.frames):
            raise Exception(f"wrong frame name: {frame} for model: {self.model.name}")
        self.W = self.add_sym("w", 6)
        self.mu = mu
        self.p_gain = self.add_sym("kp")
        self.active = self.add_sym("active")
        self.restrict = self.add_sym("restrict")  # flag for restrictive contact region
        self.impact = self.add_sym("impact")
        self.peak = self.add_sym("peak")

        # Plane parameters
        self.plane_point = self.add_sym("plane_point", 3)
        self.plane_normal = self.add_sym("plane_normal", 3)

        # constraints
        self.kin_constr = self.add_expr(name="c", expr=self.get_kin_expr())
        self.fric_cone = self.add_expr(name="f", **self.get_friction_cone())
        self.zero_wrench = self.add_expr(name="zf", expr=self.zero_wrench())
        self.w_reg = self.add_expr(name="w_reg", expr=copy.copy(self.W))
        self.no_flip_expr=self.add_expr(name="no_flipping",**self.no_flipping())

        # Contact has to be wihtin L1 dist from plane point
        # Add a cost on the contact position
        # (reference set to the next and last contact point) 
        self.__patch_restriction = patch_restriction
        if patch_restriction:
            self.range_radius = self.add_sym("rg_r")
            self.range = self.add_expr(name="rg", **self.get_range_constr_lin())
            self.pos_cost = self.add_expr(name="rc", expr=self.get_position_at_contact())
        else:
            self.range_radius = self.add_sym("rg_r")  
            # self.range = self.add_expr(name="rg", expr=SX.zeros(4, 1))   
            self.pos_cost = self.add_expr(name="rc", expr=SX.zeros(3, 1))
        # self.z_thresh = 0.05
        self.z_thresh = 0.01
        
        self.z_constr = self.add_expr(name="z_c", expr=self.get_z_constr())

        
    def setup(self, problem: ProblemFormulation):
        problem.add_input(self.W)
        problem.add_parameter(self.p_gain)
        problem.add_parameter(self.active)
        problem.add_parameter(self.impact)
        problem.add_parameter(self.peak)
        problem.add_parameter(self.restrict)
        problem.add_parameter(self.range_radius)
        if self.__patch_restriction:
            problem.add_ineq_constr(initial=False, **self.range)
        problem.add_cost(self.pos_cost, terminal=True)
        problem.add_parameter(self.plane_point)
        problem.add_parameter(self.plane_normal)
        problem.add_cost(self.w_reg, terminal=False) #wrench regularization here
        problem.add_obs(self.r_c)
        problem.add_eq_constr(
            expr=self.kin_constr,
            initial=True,
            slack_idx=[0],
        )
        # slack[0]: Relaxation corresponding to the vertical position error of the foot (p_gain × signed_distance).
        # slack[1], slack[2], slack[3]: Relaxation corresponding to the tangential velocity error of the foot in the plane (proj_vel).
        # slack[4], slack[5], slack[6]: Relaxation corresponding to the rotational error of the foot (log3(R_foot)).
        # slack[7], slack[8], slack[9]: Relaxation corresponding to the angular velocity error of the foot (v_c.angular).
        problem.add_ineq_constr(**self.fric_cone, terminal=False)
        problem.add_eq_constr(expr=self.zero_wrench, terminal=False,initial=True)
        problem.add_ineq_constr(**self.no_flip_expr, terminal=False, initial=True)
        problem.add_ineq_constr(
            expr=self.z_constr,
            lb=[0.0],
            ub=[self.z_thresh],
            initial=True,terminal=True
        )
        
        
    def contact_force(self):
        return self.active * self.W[0:3]
    def contact_torque(self):

        return self.active * self.W[3:6]

    
    def has_contact_force(self):
        return True

    def has_contact_torque(self):
        return True

    def contact_wrench(self):
        return self.active * self.W

    def get_position(self):
        return self.r_c
    
    def get_position_at_contact(self):
        return self.r_c * self.restrict

    def get_friction_cone(self):
        # h = (self.mu * self.f[2]) ** 2 - cs.sumsqr(self.f[0:2])
        # Project force onto the plane
        # TODO： Need update?
        f_from_wrench=self.W[0:3]
        f_normal = cs.dot(f_from_wrench, self.plane_normal)
        f_proj = f_from_wrench - f_normal * self.plane_normal
        h = cs.vcat([f_normal, (self.mu * f_normal) ** 2 - cs.sumsqr(f_proj)])
        return {
            "expr": self.active * h,
            "lb": [0.0] * 2,
            "ub": [1e11] * 2,
        }

    # -radius <= r_c-range_center <= radius
    def get_range_constr_lin(self):
        pin.forwardKinematics(self.model, self.data, self.q, self.v)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.frame_id]
        r_c = oMf.translation
        self.r_c = self.add_expr("r_c", r_c)
        d = self.r_c[:2] - self.plane_point[:2]
        h = cs.repmat(d, 2, 1)
        h[:2] -= self.range_radius
        h[2:] += self.range_radius
        return {
            "expr": self.restrict * self.active * h,
            "lb": [-1e11, -1e11, 0.0, 0.0],
            "ub": [0.0, 0.0, 1e11, 1e11],
        }

    def zero_wrench(self):
        return (1 - self.active) * self.W

    def get_kin_expr(self):
        pin.forwardKinematics(self.model, self.data, self.q, self.v)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.frame_id]
        r_c = oMf.translation
        self.r_c = self.add_expr("r_c", r_c)
        R_foot = oMf.rotation
        self.v_c = pin.getFrameVelocity(self.model, self.data, self.frame_id, pinocchio.LOCAL_WORLD_ALIGNED)
        signed_distance = cs.dot(r_c - self.plane_point, self.plane_normal)
        signed_distance=0#TODO 
        norm_vel = cs.dot(self.v_c.linear, self.plane_normal) * self.plane_normal
        proj_vel = self.v_c.linear - norm_vel
        kin_constr_lin = cs.vcat([self.p_gain * signed_distance, proj_vel])
        e_R = pin.log3(R_foot)
        kin_constr_ang = self.p_gain * e_R
        kin_constr_omega = self.v_c.angular 
        kin_constr = cs.vertcat(kin_constr_lin, kin_constr_ang, kin_constr_omega)
        return (1.0 - self.impact) * self.active * kin_constr
    def get_z_constr(self):
        pin.forwardKinematics(self.model, self.data, self.q, self.v)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.frame_id]
        r_c = oMf.translation
        self.r_c = self.add_expr("r_c", r_c)
        return self.r_c[2]
    def no_flipping(self):
        """
        Define the no flipping constraint to ensure the foot does not tip over.
        The constraints are:
            |m_x| <= l_x_plus * f_z   for forward tipping,
            |m_x| <= l_x_minus * f_z  for backward tipping,
            |m_y| <= l_y * f_z,
            |m_z| <= μ_b * f_z,
            and ensure f_z > eps (a small positive number).
        Here, m_x, m_y, m_z are the moments (torques) about the eeff frame axes,
        and f_z is the normal force (assumed to be the third component of the force).
        """
        eps = 1e-3  # Small constant to enforce f_z > 0
        # Geometric parameters and frictional torque coefficient (can be tuned)
        l_x_plus = 0.14   # Forward lever arm limit along x-axis (in meters)
        l_x_minus = 0.06  # Backward lever arm limit along x-axis (in meters)
        l_y = 0.03        # Lever arm limit along y-axis (in meters)
        mu_b = 0.05       # Frictional torque coefficient for rotation about the vertical (z) axis
        
        # Extract force and moment from the 6D wrench
        f = self.W[0:3]   # f = [f_x, f_y, f_z]
        m = self.W[3:6]   # m = [m_x, m_y, m_z]
        f_z = f[2]
        
        # Construct the set of inequalities:
        # 1. m_x <= l_x_plus * f_z          --> m_x - l_x_plus * f_z <= 0
        # 2. m_x >= -l_x_minus * f_z         --> -m_x - l_x_minus * f_z <= 0
        # 3. m_y <= l_y * f_z                --> m_y - l_y * f_z <= 0
        # 4. m_y >= -l_y * f_z               --> -m_y - l_y * f_z <= 0
        # 5. m_z <= mu_b * f_z               --> m_z - mu_b * f_z <= 0
        # 6. m_z >= -mu_b * f_z              --> -m_z - mu_b * f_z <= 0
        # 7. f_z >= eps  (i.e., -f_z <= -eps)
        expr = cs.vertcat(
            m[0] - l_x_plus * f_z,
            -m[0] - l_x_minus * f_z,
            m[1] - l_y * f_z,
            -m[1] - l_y * f_z,
            m[2] - mu_b * f_z,
            -m[2] - mu_b * f_z,
            -f_z
        )
        
        # Set lower and upper bounds: the inequality is satisfied when expr <= 0,
        # and the last constraint enforces f_z >= eps (i.e., -f_z <= -eps).
        lb = [-1e11, -1e11, -1e11, -1e11, -1e11, -1e11, -1e11]
        ub = [0, 0, 0, 0, 0, 0, -eps]
        return {"expr": expr, "lb": lb, "ub": ub}
    # -------------------- End no_flipping --------------------
    