from traj_opt_acados.models.quadruped import *
from traj_opt_acados.utils.model_utils import *
from traj_opt_acados.models.inter_contact import *
from visualization.vis_tools import *


class ObjectDynamics(FloatingBaseDynamics):
    def __init__(self, *args):
        super().__init__(*args)

    def get_contact_forces(self):
        f = cs.vcat([c.contact_force() for c in self.contacts])
        return f


if __name__ == "__main__":
    N = 100
    problem = ProblemFormulation(dt_min=1e-3, dt_max=2.5e-2)

    dyn_rob = Quadruped("rsc/go2_description/urdf/go2_description.urdf")
    dyn_box = ObjectDynamics(
        "box", *loadSymModel("rsc/go2_description/urdf/box_description.urdf")
    )

    box_frames = [f"p{i}" for i in [1, 2, 3, 4]]
    surf_frames = [f"face_{i}" for i in [1, 2]]

    box_contacts = [PointContact(dyn=dyn_box, frame=f, mu=0.7) for f in box_frames]
    inter_contact = [
        PointContactOnMovingSurface(
            dyn_p=dyn_rob,
            dyn_s=dyn_box,
            frame=("FL_foot", f),
            mu=0.7,
            point_radius=0.025,  # radius of foot
        )
        for f in surf_frames
    ]
    for s in inter_contact:
        s.setup(problem)

    for c in box_contacts:
        dyn_box.add_contact(c)
        c.setup(problem)

    dyn_box.setup(problem)
    dyn_rob.setup(problem)

    recompile = False
    use_cython = False

    solver = AcadosSolverHelper(problem=problem, N=N)
    solver.setup(recompile, use_cython)

    ########## configure the solver
    surf_idx = 0  # 0
    dh_ = 0.3
    dh = dh_ + 0.022
    data = solver.get_data_template()
    q0 = np.array([0.0, 0.0, dh, 0.0, 0.0, 0.0] + 4 * [0.0, 0.8, -1.6])
    data["x"][dyn_rob.q.name] = q0
    dt_nom = 2e-2
    data["u"]["dt"][0] = dt_nom
    data["yref"]["dt"][0] = dt_nom

    data["W"]["dt"][0] = 5e4
    data["W"][dyn_rob.base_cost.name] = np.array(
        [1] * 3 + [1e-2] * 3 + [1e1] * 3 + [1e1] * 3
    )
    data["W_e"][dyn_rob.base_cost.name] = np.array(
        [10] * 3 + [1e-2] * 3 + [1e1] * 3 + [1e1] * 3
    )
    # data["W_e"][dyn_rob.base_cost.name] = np.array(
    #     [1e3] * 3 + [1e-2] * 3 + [1e1] * 3 + [1e1] * 3
    # )
    data["W"][dyn_rob.acc_cost.name] = np.array([1e-3] * 12)
    data["W"][dyn_rob.swing_cost.name] = np.array([0] * len(dyn_rob.feet))
    for f in dyn_rob.feet:
        data["W"][f.f_reg.name] = np.array([1e-3, 1e-3, 1e-3])
    for c in box_contacts:
        data["W"][c.f_reg.name] = np.array([1e-3, 1e-3, 1e-3])
    # data["W"][inter_contact[surf_idx].f_reg.name] = np.array([10., 10., 10.])
    solver.set_cost_weight_constant(data["W"])
    solver.set_cost_weight_terminal(data["W_e"])

    x_d = 0.0
    data["yref"][dyn_rob.base_cost.name] = np.array(
        [x_d, 0.0, dh, 0.0, 0.0, 0.0] + [0.0] + [0.0] * 5
    )
    data["yref_e"][dyn_rob.base_cost.name] = np.array(
        [x_d, 0.0, dh, 0.0, 0.0, 0.0] + [0.0] + [0.0] * 5
    )
    solver.set_ref_constant(data["yref"])
    solver.set_ref_terminal(data["yref_e"])

    r_f = dyn_rob.get_feet_position(q0)
    for i, c in enumerate(dyn_rob.feet):
        data["p"][c.p_gain.name] = np.array([100.0])
        data["p"][c.active.name] = np.array([1])
        data["p"][c.plane_point.name] = np.array([0, 0, r_f[2, i]])
        data["p"][c.plane_normal.name] = np.array([0, 0, 1])

    frames = [f"p{i}" for i in [1, 2, 3, 4]]

    data["p"][dyn_box.impact_active.name][0] = 0.0
    for i, c in enumerate(box_contacts):
        data["p"][c.p_gain.name][0] = 100.0
        data["p"][c.impact.name][0] = 0.0
        data["p"][c.plane_point.name] = np.array([0, 0, 0])
        data["p"][c.plane_normal.name] = np.array([0, 0, 1])
        data["p"][c.active.name][0] = 1

    data["p"][inter_contact[surf_idx].surf_lim.name] = np.array([0.05, 0.05])
    data["p"][inter_contact[surf_idx].p_gain.name][0] = 100.0

    solver.set_parameters_constant(data["p"])

    solver.params[dyn_rob.feet[0].active.name][:, 20:] = np.array([0])
    solver.params[inter_contact[surf_idx].active.name][:, 30:] = np.array([1])

    ######################
    data["x"][dyn_box.q.name] = np.array([0.3, 0.0, 0.05, 0.0, 0.0, 0.0])
    solver.set_state_constant(data["x"])
    solver.set_initial_state(data["x"])
    solver.set_input_constant(data["u"])

    solver.update_cost_weights()
    solver.update_parameters()
    solver.solve(print_stats=True, print_time=True)
    solver.parse_sol()

    q = solver.states[dyn_rob.q.name]
    params = solver.params
    print("states: ", solver.states.keys())
    print("controls: ", solver.inputs.keys())
    print("params: ", params.keys())
    print("Length of q: ", solver.states["q_go2"].shape)
    print("Length of v: ", solver.states["v_go2"].shape)
    print("Length of dt: ", solver.inputs["dt"].shape)
    print("Length of f_FL: ", solver.inputs["f_FL_foot_go2"].shape)
    print("Length of f_FR: ", solver.inputs["f_FR_foot_go2"].shape)
    print("Length of f_RL: ", solver.inputs["f_RL_foot_go2"].shape)
    print("Length of f_RR: ", solver.inputs["f_RR_foot_go2"].shape)
    print("Length of acc", solver.inputs["a_go2"].shape)
    print("Length of active_FL", params["active_FL_foot_go2"].shape)
    print("Active FL", params["active_FL_foot_go2"])
    print("Forces FL", solver.inputs["f_FL_foot_go2"])
    print("Forces Inter", solver.inputs[inter_contact[surf_idx].f.name])
    print("Forces Box", solver.inputs[box_contacts[0].f.name])
    dt_ = solver.inputs["dt"].flatten()

    ee_frames = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    model_ = dyn_rob.get_model()
    data_ = dyn_rob.get_data()

    torques = np.zeros((len(q), model_.nv))

    qs = solver.states["q_go2"][:, 1:]
    s = solver.states[dyn_box.q.name]
    vs = solver.states["v_go2"][:, 1:]
    active_FL = params["active_FL_foot_go2"][:, 1:]

    accs = solver.inputs["a_go2"]
    forces = [
        solver.inputs["f_FL_foot_go2"],
        solver.inputs["f_FR_foot_go2"],
        solver.inputs["f_RL_foot_go2"],
        solver.inputs["f_RR_foot_go2"],
    ]

    for i in range(len(q)):
        # Set the current state
        pinocchio.forwardKinematics(model_, data_, qs[:, i])
        pinocchio.computeJointJacobians(model_, data_, qs[:, i])
        pinocchio.updateFramePlacements(model_, data_)

        # Compute the mass matrix
        M = pinocchio.crba(model_, data_, qs[:, i])

        # Compute the Coriolis and gravity terms
        b = pinocchio.nle(model_, data_, qs[:, i], vs[:, i])

        # Initialize the contact forces vector
        contact_forces = np.zeros((model_.nv,))

        # Loop through each end-effector and accumulate external forces
        for ee_name, f_ee in zip(ee_frames, forces):
            frame_id = model_.getFrameId(ee_name)
            J_ee = pinocchio.computeFrameJacobian(
                model_, data_, qs[:, i], frame_id, pin.ReferenceFrame.LOCAL
            )
            contact_forces += J_ee.T @ np.hstack((f_ee[:, i], np.zeros(3)))

        # Compute the torques using the inverse dynamics equation
        tau = M @ accs[:, i] - b - contact_forces
        torques[i, :] = tau

    box_com_trajectory = []
    for i in range(N):
        com_pose = []
        for j in range(6):
            com_pose.append(s[j][i + 1])
        box_com_trajectory.append(com_pose)
    timestep_trajectory = dt_.flatten().tolist()

    com_trajectory = []
    for i in range(N):
        com_pose = q[:6, i]
        com_trajectory.append(com_pose)

    joint_trajectory = []
    for i in range(N):
        joint_position = q[6:, i]
        joint_trajectory.append(joint_position)

    timestep_trajectory = dt_.flatten().tolist()
    print("Time ", timestep_trajectory)
    print("Box trajectory", s)
    vis_robot(com_trajectory, joint_trajectory, timestep_trajectory, box_com_trajectory)
