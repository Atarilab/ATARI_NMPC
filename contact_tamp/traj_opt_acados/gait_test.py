from .models.quadruped import *
from visualization.vis_tools import vis_robot

if __name__ == "__main__":
    dyn = Quadruped("rsc/go2_description/urdf/go2_description.urdf")

    time_opt = True
    if time_opt:
        problem = ProblemFormulation(dt_min=1e-2, dt_max=2.5e-2, enable_time_opt=True)
    else:
        problem = ProblemFormulation(dt_nom=0.02)

    dyn.setup(problem)
    N = 65

    recompile = True
    # recompile = False
    # use_cython = True
    use_cython = False

    solver = AcadosSolverHelper(problem=problem, N=N)
    solver.setup(recompile, use_cython)

    ########## configure the solver
    # set initial states and guess
    # note: if the initial contact kin_constr residual
    # is close to solver reg_eps, SPEED mode would fail
    # But dt is a decision variable seems like the issue will be mitigated
    # dh_ = 0.2978
    # dh_ = 0.3
    dh_ = 0.3
    dh = dh_ + 0.022
    data = solver.get_data_template()
    q0 = np.array([0.0, 0.0, dh, 0.0, 0.0, 0.0] + 4 * [0.0, 0.8, -1.6])
    data["x"][dyn.q.name] = q0
    ######## set nominal dt
    dt_nom = 2e-2
    if time_opt:
        data["u"]["dt"][0] = dt_nom
        data["yref"]["dt"][0] = dt_nom

    ######## set states

    solver.set_state_constant(data["x"])
    solver.set_initial_state(data["x"])

    ######## set inputs
    solver.set_input_constant(data["u"])

    ######### set cost
    if time_opt:
        data["W"]["dt"][0] = 1e5
    # walking
    data["W"][dyn.base_cost.name] = np.array([1] * 3 + [1e2] * 3 + [100] * 3 + [10] * 3)
    # stance
    # data["W"][dyn.base_cost.name] = np.array([1e2] * 3 + [1e2] * 3 + [100] * 3 + [10] * 3)
    data["W_e"][dyn.base_cost.name] = np.array(
        [1e2] * 3 + [1e2] * 3 + [100] * 3 + [10] * 3
    )
    data["W"][dyn.acc_cost.name] = np.array([1e-3] * 12)
    data["W"][dyn.swing_cost.name] = np.array([1e5] * len(dyn.feet))
    for f in dyn.feet:
        data["W"][f.f_reg.name] = np.array([1e-2, 1e-2, 1e-3])  # 0.1
    solver.set_cost_weight_constant(data["W"])
    solver.set_cost_weight_terminal(data["W_e"])
    x_d = 0.0
    data["yref"][dyn.base_cost.name] = np.array(
        [x_d, 0.0, dh, 0.0, 0.0, 0.0] + [0.0] * 6
    )
    data["yref_e"][dyn.base_cost.name] = np.array(
        [x_d, 0.0, dh, 0.0, 0.0, 0.0] + [0.0] * 6
    )
    data["yref"][dyn.base_cost.name][0] = x_d
    data["yref"][dyn.swing_cost.name] = np.array([0.07] * len(dyn.feet))
    # for f in dyn.feet:
    #     data["yref"][f.f_reg.name] = -dyn.weight / 4
    solver.set_ref_constant(data["yref"])
    solver.set_ref_terminal(data["yref_e"])

    ######### set constraints and parameters
    r_f = dyn.get_feet_position(q0)
    for i, c in enumerate(dyn.feet):
        data["p"][c.plane_point.name][2] = r_f[2, i]
        data["p"][c.plane_normal.name] = np.array([0, 0, 1.0])
        data["p"][c.p_gain.name] = np.array([100.0])
        data["p"][c.active.name] = np.array([1])

    print(data["p"])
    # shortcut for setting params
    solver.set_parameters_constant(data["p"])
    #### gait setting
    N_trot = [15, 30, 45, 60]
    gait = [(0, 3), (1, 2)] * 2  # trot
    # N_trot = [30, 45]
    # gait = [(), (0, 1, 2, 3)] # jumping
    # N_trot = [15, 30, 45, 60]
    # gait = [(0,), (1,), (2,), (3,)] # crawling
    # N_trot = [10, 20]
    # gait = [(0, 3), (1, 2)]
    for n, g in zip(N_trot, gait):
        nc = len(g)
        if nc > 0:
            fd = -dyn.weight / nc
            for i, f in enumerate(dyn.feet):
                if i in g:
                    # swing foot
                    # n - 10 + 1: after impact immediately in air
                    # n - 10: before impact already in air
                    # n - 10 + 2: after impact at least dt in air
                    air_start = n - 15 + 1
                    air_peak = n - 7
                    solver.params[f.active.name][:, air_start:n] = 0
                    solver.params[f.peak.name][:, air_peak] = 1
                    # solver.params[f.impact.name][:, n] = 1
                    # solver.cost_weights[f.f_reg.name][2, n] = 1e-1
                # else:
                #     solver.cost_ref[f.f_reg.name][:, air_start: n] = fd
            solver.params[dyn.impact_active.name][:, n] = 1
    solver.update_cost_weights()
    solver.update_parameters()
    # print(solver.params)
    # exit(0)
    # solver.get_acados("p")
    solver.solve(print_stats=True, print_time=True)
    solver.parse_sol()
    # print(solver.raw["u"][:, 0])
    f0 = problem.obs_eval_func(dyn.feet[0].r_c.name)
    pprint(solver.eval_func(f0))
    # exit(0)
    np.set_printoptions(3)
    q = solver.states[dyn.q.name]
    v = solver.states[dyn.v.name]
    # print(q[:6, :])
    print("f")
    pprint(solver.inputs[dyn.feet[0].f.name])
    if time_opt:
        dt_ = solver.inputs["dt"].flatten()
    else:
        dt_ = np.array([dt_nom] * N)
    pprint("dt: ")
    pprint(dt_)
    # for f in dyn.feet:
    #     pprint(f.name)
    #     pprint(solver.inputs[f.f.name])

    # print("Center of Mass Trajectory:")
    com_trajectory = []
    for i in range(N):
        com_pose = q[:6, i]
        # print(f"Time step {i}: {com_pose}")
        com_trajectory.append(com_pose)

    # print("\nJoint Positions:")
    joint_trajectory = []
    for i in range(N):
        joint_position = q[6:, i]
        # print(f"Time step {i}: {joint_position}")
        joint_trajectory.append(joint_position)
    timestep_trajectory = dt_.flatten().tolist()
    vis_robot(com_trajectory, joint_trajectory, timestep_trajectory)
