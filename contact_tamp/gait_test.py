from traj_opt_acados.models.quadruped import *
from traj_opt_acados.utils.model_utils import *
from visualization.vis_tools import vis_robot

def set_gait(gait_type):
    if gait_type == "crawl":
        N_trot = [25, 50, 75, 100, 125, 150, 175, 200]
        gait = [(0,), (1,), (2,), (3,),(0,), (1,), (2,), (3,),]
    elif gait_type == "trot":
        N_trot = [25, 50, 75, 100, 125, 150, 175, 200]
        gait = [(0, 3), (1, 2), (0, 3), (1, 2), (0, 3), (1, 2), (0, 3), (1, 2)]
    elif gait_type == "pace":
        N_trot = [25, 50, 75, 100, 125, 150, 175, 200]
        gait = [(0, 2), (1, 3),(0, 2), (1, 3),(0, 2), (1, 3),(0, 2), (1, 3),]
    elif gait_type == "bound":
        N_trot = [25, 50, 75, 100, 125, 150, 175, 200]
        gait = [(0, 1), (2, 3),(0, 1), (2, 3),(0, 1), (2, 3),(0, 1), (2, 3),]
    elif gait_type == "jump":
        N_trot = [25, 50, 75, 100, 125, 150, 175, 200]
        gait = [(), (0,1,2,3,), (), (0,1,2,3,), (), (0,1,2,3,), (), (0,1,2,3,)]
    elif gait_type == "hind_walk":
        N_trot = [25, 50, 75, 100, 125, 150, 175, 200]
        gait = [(0, 1, 2,), (0, 1, 3,), (0, 1, 2,), (0, 1, 3,), (0, 1, 2,), (0, 1, 3,), (0, 1, 2,), (0, 1,)]
    elif gait_type == "single_switch":
        N_trot = [30]
        gait = [()]
    else:
        raise ValueError("Invalid gait type. Choose from: crawl, trot, pace, jump.")
    
    return N_trot, gait

if __name__ == "__main__":
    dyn = Quadruped("rsc/go2_description/urdf/go2_description.urdf")
    problem = ProblemFormulation(dt_min=1e-2, dt_max=2.5e-2, enable_time_opt=True)
    dyn.setup(problem)
    N = 200

    recompile = True
    use_cython = False

    solver = AcadosSolverHelper(problem=problem, N=N)
    solver.setup(recompile, use_cython)

    ########## configure the solver
    dh_ = 0.3
    dh = dh_ + 0.022
    data = solver.get_data_template()
    q0 = np.array([0.0, 0.0, dh, 0.0, 0.0, 0.0] + 4 * [0.0, 0.8, -1.6])
    data["x"][dyn.q.name] = q0
    dt_nom = 2e-2
    data["u"]["dt"][0] = dt_nom
    data["yref"]["dt"][0] = dt_nom

    solver.set_state_constant(data["x"])
    solver.set_initial_state(data["x"])
    solver.set_input_constant(data["u"])

    data["W"]["dt"][0] = 0
    data["W"][dyn.base_cost.name] = np.array([1e1] * 3 + [1e-2] * 3 + [1e1] * 3 + [1e1] * 3)
    data["W_e"][dyn.base_cost.name] = np.array([1e-1] * 3 + [1e-2] * 3 + [1e1] * 3 + [1e1] * 3)
    data["W"][dyn.acc_cost.name] = np.array([1e-3] * 12)
    data["W"][dyn.swing_cost.name] = np.array([1e-1] * len(dyn.feet))
    for f in dyn.feet:
        data["W"][f.f_reg.name] = np.array([1e-2, 1e-2, 1e-3])
    solver.set_cost_weight_constant(data["W"])
    solver.set_cost_weight_terminal(data["W_e"])

    x_d = 0.0
    data["yref"][dyn.base_cost.name] = np.array([x_d, 0.0, dh, 0.0, 0.0, 0.0] + [0.5] + [0.0] * 5)
    data["yref_e"][dyn.base_cost.name] = np.array([x_d, 0.0, dh, 0.0, 0.0, 0.0] + [0.0] + [0.0] * 5)
    solver.set_ref_constant(data["yref"])
    solver.set_ref_terminal(data["yref_e"])

    r_f = dyn.get_feet_position(q0)
    for i, c in enumerate(dyn.feet):
        data["p"][c.p_gain.name] = np.array([1000.0])
        data["p"][c.active.name] = np.array([1])
        data["p"][c.plane_point.name] = np.array([0,0,r_f[2, i]])
        data["p"][c.plane_normal.name] = np.array([0,0,1])
    solver.set_parameters_constant(data["p"])

    print(solver.params[c.active.name])

    #### Gait setting: Choose gait type
    gait_type = input("Choose gait (crawl, trot, pace, bound, jump): ").lower()
    N_trot, gait = set_gait(gait_type)   

    for n, g in zip(N_trot, gait):
        nc = len(g)
        if nc > 0:
            fd = -dyn.weight / nc
            for i, c in enumerate(dyn.feet):
                if i in g:
                    air_start = n - 25 + 1
                    air_peak = n - 12
                    # contact_pos = contact_locations[i][n] 
                    solver.params[c.active.name][:, air_start : n] = 0
                    solver.params[c.peak.name][:, air_peak] = 1
            solver.params[dyn.impact_active.name][:, n] = 1

    solver.update_cost_weights()
    solver.update_parameters()
    solver.solve(print_stats=True, print_time=True)
    solver.parse_sol()

    q = solver.states[dyn.q.name]
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
    dt_ = solver.inputs["dt"].flatten()

    ee_frames = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    # model_, data_ = loadModel("rsc/go2_description/urdf/go2_description.urdf")
    model_ = dyn.get_model()
    data_ = dyn.get_data()

    torques = np.zeros((len(q), model_.nv))

    qs = solver.states["q_go2"][:,1:]
    vs = solver.states["v_go2"][:,1:]
    active_FL = params["active_FL_foot_go2"][:,1:]

    accs = solver.inputs["a_go2"]
    forces = [solver.inputs["f_FL_foot_go2"], 
              solver.inputs["f_FR_foot_go2"],
              solver.inputs["f_RL_foot_go2"],
              solver.inputs["f_RR_foot_go2"]]

    for i in range(len(q)):
        # Set the current state
        pinocchio.forwardKinematics(model_, data_, qs[:,i])
        pinocchio.computeJointJacobians(model_, data_, qs[:,i])
        pinocchio.updateFramePlacements(model_, data_)

        # Compute the mass matrix
        M = pinocchio.crba(model_, data_, qs[:,i])
        
        # Compute the Coriolis and gravity terms
        b = pinocchio.nle(model_, data_, qs[:,i], vs[:,i])
        
        # Initialize the contact forces vector
        contact_forces = np.zeros((model_.nv,))

        # Loop through each end-effector and accumulate external forces
        for ee_name, f_ee in zip(ee_frames, forces):
            frame_id = model_.getFrameId(ee_name)
            J_ee = pinocchio.computeFrameJacobian(model_, data_, qs[:,i], frame_id, pin.ReferenceFrame.LOCAL)
            contact_forces += J_ee.T @ np.hstack((f_ee[:,i], np.zeros(3)))
        
        # Compute the torques using the inverse dynamics equation
        tau = M @ accs[:,i] - b - contact_forces
        torques[i, :] = tau

    com_trajectory = []
    for i in range(N):
        com_pose = q[:6, i]
        com_trajectory.append(com_pose)

    joint_trajectory = []
    for i in range(N):
        joint_position = q[6:, i]
        joint_trajectory.append(joint_position)
    
    timestep_trajectory = dt_.flatten().tolist()
    vis_robot(com_trajectory, joint_trajectory, timestep_trajectory)
