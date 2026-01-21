from traj_opt_acados.models.quadruped import *
from visualization.vis_tools import vis_robot

def set_gait(gait_type):
    if gait_type == "stand":
        N_trot = [25, 50, 75, 100]
        gait = [(), (), (), (),]
    elif gait_type == "walk":
        N_trot = [25, 50, 75, 100]
        gait = [(0,), (1,), (0,), (1,),]
    elif gait_type == "jump":
        N_trot = [25, 50, 75, 100]
        gait = [(), (0,1), (), (0,1),]
    else:
        raise ValueError("Invalid gait type. Choose from: stand, walk, jump.")
    
    return N_trot, gait

if __name__ == "__main__":
    dyn = Biped("/home/michal/projects/tamp/tamp_warm_start/g1_description/g1_29dof.urdf")
    problem = ProblemFormulation(dt_min=1e-2, dt_max=2.5e-2)
    dyn.setup(problem)
    N = 200

    recompile = True
    use_cython = False

    solver = AcadosSolverHelper(problem=problem, N=N)
    solver.setup(recompile, use_cython)

    ########## configure the solver
    dh = 0.79
    data = solver.get_data_template()
    q0 = np.array([0.0, 0.0, dh, 0.0, 0.0, 0.0] + 29 * [0.0])
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
    data["W"][dyn.acc_cost.name] = np.array([1e-3] * 29)
    data["W"][dyn.swing_cost.name] = np.array([1e-1] * len(dyn.feet))
    for f in dyn.feet:
        data["W"][f.f_reg.name] = np.array([1e-2, 1e-2, 1e-3])
    solver.set_cost_weight_constant(data["W"])
    solver.set_cost_weight_terminal(data["W_e"])

    x_d = 0.0
    data["yref"][dyn.base_cost.name] = np.array([x_d, 0.0, 0.6, 0.0, 0.0, 0.0] + [0.0] + [0.0] * 5)
    data["yref_e"][dyn.base_cost.name] = np.array([x_d, 0.0, 0.6, 0.0, 0.0, 0.0] + [0.0] + [0.0] * 5)
    solver.set_ref_constant(data["yref"])
    solver.set_ref_terminal(data["yref_e"])

    r_f = dyn.get_feet_position(q0)
    for i, c in enumerate(dyn.feet):
        data["p"][c.zf.name][0] = r_f[2, i]
        data["p"][c.p_gain.name] = np.array([100.0])
        data["p"][c.active.name] = np.array([1])
    solver.set_parameters_constant(data["p"])

    #### Gait setting: Choose gait type
    gait_type = input("Choose gait (stand, walk, jump): ").lower()
    N_trot, gait = set_gait(gait_type) 

    for n, g in zip(N_trot, gait):
        nc = len(g)
        if nc > 0:
            fd = -dyn.weight / nc
            for i, c in enumerate(dyn.feet):
                if i in g:
                    air_start = n - 25 + 1
                    air_peak = n - 12
                    solver.params[c.active.name][:, air_start : n] = 0
                    solver.params[c.peak.name][:, air_peak] = 1
            solver.params[dyn.impact_active.name][:, n] = 1

    solver.update_cost_weights()
    solver.update_parameters()
    solver.solve(print_stats=True, print_time=True)
    solver.parse_sol()

    q = solver.states[dyn.q.name]
    dt_ = solver.inputs["dt"].flatten()

    com_trajectory = []
    for i in range(N):
        com_pose = q[:6, i]
        com_trajectory.append(com_pose)

    joint_trajectory = []
    for i in range(N):
        joint_position = q[6:, i]
        joint_trajectory.append(joint_position)
    
    timestep_trajectory = dt_.flatten().tolist()
    vis_robot(com_trajectory, joint_trajectory, timestep_trajectory, robot = "g1")