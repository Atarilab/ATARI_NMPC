from .point_contact import *


class Contact3D:
    def __init__(self, f, frame_id, r_c):
        self.f = f
        self.frame_id = frame_id
        self.r_c = r_c

    def has_contact_force(self):
        return True

    def has_contact_torque(self):
        return False

    def contact_force(self):
        return self.f

    def setup(self, problem):
        pass

    def get_position(self):
        return self.r_c


# contact constraint for the two dynamics
class PointContactMoving(AbstractModel):
    def __init__(
        self,
        dynA,
        dynB,
        p_gain,
        v_gain,
        mu,
        x_bound,
        y_bound,
        frame_name_A: str,
        frame_name_B: str,
    ):
        super().__init__(name=f"{frame_name_A}_{frame_name_B}")

        self.name = frame_name_A + "_" + frame_name_B
        self.f = cs.SX.sym(
            "fm_" + frame_name_A + "_" + frame_name_B, 3
        )  # local force expressed in B frame
        self.active = self.add_sym("active")
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.mu = mu
        # compute for dynA
        self.dynA = dynA
        self.frame_name_A = frame_name_A
        self.frame_id_A = dynA.model.getFrameId(frame_name_A)
        pin.forwardKinematics(dynA.model, dynA.data, dynA.q, dynA.v)
        pin.updateFramePlacement(dynA.model, dynA.data, self.frame_id_A)
        v_A = pin.getFrameVelocity(
            dynA.model, dynA.data, self.frame_id_A, pinocchio.LOCAL_WORLD_ALIGNED
        )
        r_A = dynA.data.oMf[self.frame_id_A].translation
        # compute for dynB
        self.dynB = dynB
        self.frame_name_B = frame_name_B
        self.frame_id_B = dynB.model.getFrameId(frame_name_B)
        pin.forwardKinematics(dynB.model, dynB.data, dynB.q, dynB.v)
        pin.updateFramePlacement(dynB.model, dynB.data, self.frame_id_B)
        v_B = pin.getFrameVelocity(
            dynB.model, dynB.data, self.frame_id_B, pinocchio.LOCAL_WORLD_ALIGNED
        )
        r_B = dynB.data.oMf[self.frame_id_B].translation
        R_B = dynB.data.oMf[self.frame_id_B].rotation
        # orthogonal constraint
        r_rel = R_B.T @ (r_A - r_B)
        z_error = r_rel[2]
        w_B_s = pin.skew(v_B.angular)
        v_rel = R_B.T @ (v_A - v_B).linear + (w_B_s @ R_B).T @ (r_A - r_B)
        # time derivative of the projected error
        v_rel_z = v_rel[2]
        # velocity level projected x and y
        v_rel_xy = v_rel[[0, 1]]
        a_A = pin.getFrameClassicalAcceleration(
            dynA.model, dynA.data, self.frame_id_A, pinocchio.LOCAL_WORLD_ALIGNED
        )
        a_B = pin.getFrameClassicalAcceleration(
            dynB.model, dynB.data, self.frame_id_B, pinocchio.LOCAL_WORLD_ALIGNED
        )
        # relative acceleration expressed in local frame
        dw_B_s = pin.skew(v_B.angular)
        a_rel = (
            R_B.T @ (a_A - a_B).linear
            + (w_B_s @ R_B).T @ (v_A - v_B).linear * 2
            + (dw_B_s @ R_B).T @ (r_A - r_B)
            + (w_B_s @ (w_B_s @ R_B)).T @ (r_A - r_B)
        )
        az_d = -v_gain * v_rel_z + -p_gain * z_error
        axy_d = -v_gain * v_rel_xy
        self.r_rel = r_rel
        # force in world frame
        self.f_world = R_B @ self.f
        self.r_c_A = r_A
        self.r_c_B = r_B

        # kinematics constraint
        kin_constr = self.active * (a_rel - cs.vcat([az_d, axy_d]))
        self.kin_constr = self.add_expr(name="c", expr=kin_constr)
        self.fric_cone = self.add_expr(name="f", **self.get_friction_cone())
        self.pos_bound = self.add_expr(name="p", **self.get_position_bound())

    def get_position_bound(self):
        h = cs.vcat([self.r_rel[2], self.r_rel[0], self.r_rel[1]])
        h_lb = -cs.vcat([0.0, self.x_bound, self.y_bound])
        h_ub = cs.vcat([100.0, self.x_bound, self.y_bound])
        return {
            "expr": self.active * h,
            "lb": [-0.0, -self.x_bound, -self.y_bound],
            "ub": [100.0, self.x_bound, self.y_bound],
        }

    def get_friction_cone(self):
        h = cs.vcat(
            [
                self.f[2],
                self.mu * self.f[2] - self.f[0],
                self.mu * self.f[2] + self.f[0],
                self.mu * self.f[2] - self.f[1],
                self.mu * self.f[2] + self.f[1],
            ]
        )
        return {
            "expr": self.active * h,
            "lb": [0] * 5,
            "ub": [1e10] * 5,
        }

    def setup(self, problem: ProblemFormulation):
        problem.add_input(self.f, self.name)
        problem.add_parameter(self.active)
        problem.add_eq_constr(expr=self.kin_constr)
        problem.add_ineq_constr(**self.fric_cone)
        problem.add_ineq_constr(**self.pos_bound)

    def add_to(self, dynA, dynB):
        c_p = Contact3D(self.f_world, self.frame_id_A, self.r_c_A)
        c_s = Contact3D(-self.f_world, self.frame_id_B, self.r_c_B)
        dynA.add_contact(c_p)
        dynB.add_contact(c_s)


class PointContactOnMovingSurface(AbstractModel):
    def __init__(
        self,
        dyn_p: FloatingBaseDynamics,  # dynamics with the point
        dyn_s: FloatingBaseDynamics,  # dynamics with the surface
        frame: tuple[str, str],
        point_radius: float,
        mu: float,
    ):
        super().__init__(name=f"{frame}_{dyn_p.name}_{frame}_{dyn_s.name}")
        self.dyn_p = dyn_p
        self.dyn_s = dyn_s
        self.frame_name_p, self.frame_name_s = frame
        self.frame_id_p = dyn_p.model.getFrameId(self.frame_name_p)
        if self.frame_id_p >= len(dyn_p.model.frames):
            raise Exception(
                f"wrong frame name: {self.frame_name_p} for model: {dyn_p.model.name}"
            )
        self.frame_id_s = dyn_s.model.getFrameId(self.frame_name_s)
        if self.frame_id_s >= len(dyn_s.model.frames):
            raise Exception(
                f"wrong frame name: {self.frame_name_s} for model: {dyn_s.model.name}"
            )
        self.f = self.add_sym("f", 3)
        self.mu = mu
        self.p_gain = self.add_sym("kp")
        self.active = self.add_sym("active")
        self.point_radius = point_radius

        # constraints
        self.kin_constr = self.add_expr(name="c", expr=self.get_kin_expr())
        self.fric_cone = self.add_expr(name="f", **self.get_friction_cone())
        self.f_reg = self.add_expr(name="f_reg", expr=copy.copy(self.f))
        # restriction on contact location in the surface local coordinate
        self.surf_lim = self.add_sym("surf_lim", 2)
        self.range = self.add_expr(name="rg", **self.get_range_constr())
        self.__add_to()

    def setup(self, problem: ProblemFormulation):
        problem.add_input(self.f)
        problem.add_parameter(self.p_gain)
        problem.add_parameter(self.active)
        problem.add_parameter(self.surf_lim)
        problem.add_ineq_constr(initial=False, **self.range)
        problem.add_cost(self.f_reg, terminal=False)
        problem.add_eq_constr(expr=self.kin_constr, initial=False)
        problem.add_ineq_constr(**self.fric_cone, terminal=False)

    def __add_to(self):
        A = Contact3D(self.active * self.f, self.frame_id_p, self.r_c_p)
        B = Contact3D(-self.active * self.f, self.frame_id_s, self.r_c_s)
        self.dyn_p.add_contact(A)
        self.dyn_s.add_contact(B)

    def get_friction_cone(self):
        f_normal = cs.dot(self.f, self.plane_normal)
        f_proj = self.f - f_normal * self.plane_normal
        h = cs.vcat([f_normal, (self.mu * f_normal) ** 2 - cs.sumsqr(f_proj)])
        return {
            "expr": self.active * h,
            "lb": [0.0] * 2,
            "ub": [1e11] * 2,
        }

    def get_range_constr(self):
        h = cs.vcat([self.surf_lim - self.proj_r[:2], self.proj_r[:2] + self.surf_lim])
        return {
            "expr": self.active * h,
            "lb": [0] * 4,
            "ub": [1e11] * 4,
        }

    def get_kin_expr(self):
        r_c = []
        v_c = []
        for frame, dyn in zip(
            [self.frame_id_p, self.frame_id_s],
            [self.dyn_p, self.dyn_s],
        ):
            pin.forwardKinematics(dyn.model, dyn.data, dyn.q, dyn.v)
            pin.updateFramePlacements(dyn.model, dyn.data)
            r_c.append(dyn.data.oMf[frame].translation)
            v_c.append(
                pin.getFrameVelocity(
                    dyn.model, dyn.data, frame, pinocchio.LOCAL_WORLD_ALIGNED
                )
            )
        self.r_c_p = self.add_expr("r_c_p", r_c[0])
        self.r_c_s = self.add_expr("r_c_s", r_c[1])
        # use the surface frame origin as plane point
        self.plane_point = self.r_c_s
        # z-axis of the surface
        self.plane_normal = self.dyn_s.data.oMf[self.frame_id_s].rotation[:, -1]
        # Calculate signed distance from contact point to plane
        r_world = r_c[0] - r_c[1]
        signed_distance = cs.dot(r_world, self.plane_normal) - self.point_radius
        # calculate projection contact point on the surface
        self.proj_r = self.dyn_s.data.oMf[self.frame_id_s].rotation.T @ r_world

        # constraint stabilization
        kin_constr = (v_c[0] - v_c[1]).linear + self.p_gain * signed_distance * self.plane_normal

        return self.active * kin_constr
