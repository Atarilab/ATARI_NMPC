from ..core.model import *
from ..core.pinocchio_fwd import *
from ..interface.problem_formuation import *
# import pinocchio.casadi as cpin
# import pinocchio as pin

class FloatingBaseDynamics(AbstractModel):
    def __init__(self, name: str, model: pin.Model, data: pin.Data):
        super().__init__(name)
        self.q = self.add_sym("q", model.nq)
        self.v = self.add_sym("v", model.nv)
        self.a = self.add_sym("a", model.nv)
        self.model = model
        self.data = data
        self.x = []
        self.u = self.a
        self.xnext = []
        self.h = self.add_sym("h", 6)  # centroidal momentum
        self.contacts = []

    def add_contact(self, contact):
        self.contacts.append(contact)

    def add_contacts(self, contacts: list):
        self.contacts.extend(contacts)

    def semi_implicit_euler(self, problem: ProblemFormulation):
        # centroidal dynamics
        dt = problem.dt
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        com = pin.centerOfMass(self.model, self.data, self.q)
        dh = cs.SX.zeros(6)
        for c in self.contacts:
            r = c.get_position()
            if c.has_contact_force():
                f = c.contact_force()
                dh += cs.vcat([f, cs.cross(r - com, f)])  # cs.DM.zeros(3)])#
            if c.has_contact_torque():
                dh[3:] += c.contact_torque()
        # impact duration 10ms
        self.weight = np.array(cs.DM(self.data.mass[0] * self.model.gravity981))
        dh[:3] += self.weight
        # semi-implicit euler integration
        vnext = self.v + self.a * dt
        qnext = pin.integrate(
            self.model,
            self.q,
            vnext * dt,
        )
        # this way the numerical conditioning can be better
        hnext = self.h + dh * dt
        problem.add_input(self.u)
        problem.add_state(self.h, hnext)
        problem.add_bound(expr=self.h, field="x", lb=[-1e11] * 6, ub=[1e11] * 6)
        problem.add_state(self.q, qnext)
        problem.add_bound(
            expr=self.q,
            field="x",
            lb=np.array(cs.DM(self.model.lowerPositionLimit)).flatten().tolist(),
            ub=np.array(cs.DM(self.model.upperPositionLimit)).flatten().tolist(),
        )
        problem.add_state(self.v, vnext)
        problem.add_bound(
            expr=self.v,
            field="x",
            lb=np.array(cs.DM(-self.model.velocityLimit)).flatten().tolist(),
            ub=np.array(cs.DM(self.model.velocityLimit)).flatten().tolist(),
        )

    # generate h - A(q)qdot=0
    def get_cen_momentum_constr(self):
        h = pin.computeCentroidalMomentum(self.model, self.data, self.q, self.v).np
        return self.add_expr(name="cmm", expr=h - self.h)

    def setup(self, problem: ProblemFormulation):
        self.cmm_constr = self.get_cen_momentum_constr()
        problem.add_eq_constr(expr=self.cmm_constr, initial=False)
        self.semi_implicit_euler(problem)
        for f in self.feet:
            f.setup(problem)

    def add_torque_limit(self, problem: ProblemFormulation):
    
        pin.forwardKinematics(self.model, self.data, self.q, self.v)
        pin.updateFramePlacements(self.model, self.data)
    
        M = pin.crba(self.model, self.data, self.q)
        non_lin = pin.nonLinearEffects(self.model, self.data, self.q, self.v)
        
        J_T_times_f = np.zeros(self.model.nv)
        for cnt in self.contacts:
            J = pin.computeFrameJacobian(self.model, self.data, self.q, 
                                        self.model.getFrameId(cnt.frame_name), 
                                        pinocchio.LOCAL_WORLD_ALIGNED)
            J_T_times_f += J[:3,:].T @ cnt.f

        tau = M @ self.a + non_lin - J_T_times_f

        tau_lim = self.add_expr(
            name="tau_lim",
            expr=tau[-self.nu:],
            lb=np.array(cs.DM(-self.model.effortLimit))[-self.nu:].flatten().tolist(),
            ub=np.array(cs.DM(self.model.effortLimit))[-self.nu:].flatten().tolist()
            )
        
        problem.add_ineq_constr(
            **tau_lim,
            terminal=False,
        )