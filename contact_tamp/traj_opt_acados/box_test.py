from traj_opt_acados.interface.acados_helper import *
from traj_opt_acados.utils.model_utils import *
from traj_opt_acados.models.point_contact import *
from pinocchio.visualize import MeshcatVisualizer
import time


class ObjectDynamics(FloatingBaseDynamics):
    def __init__(self, *args):
        super().__init__(*args)

    def get_contact_forces(self):
        f = cs.vcat([c.contact_force() for c in self.contacts])
        return f


if __name__ == "__main__":
    N = 50
    problem = ProblemFormulation(dt_min=1e-3, dt_max=3e-2, enable_time_opt=True)
    ########## dynamics
    box = ObjectDynamics(
        "box", *loadSymModel("rsc/go2_description/urdf/box_description.urdf")
    )
    frames = [f"p{i}" for i in [1, 2, 3, 4]]
    contacts = [PointContact(dyn=box, frame=f, mu=0.7) for f in frames]
    for c in contacts:
        box.add_contact(c)
        c.setup(problem)

    box.setup(problem)

    res_eval = [problem.obs_eval_func(name=c.r_c.name) for c in contacts]
    ########### cost
    # base state cost
    # problem.add_cost(cost=box.get_contact_forces(), name="force_cost")
    # problem.add_cost(cost=box.q, name="base_vel_cost")  # useless
    problem.add_cost(cost=problem.dt, name="dt")
    # base state terminal cost
    # problem.add_cost_terminal(cost=box.q, name="base_vel_cost")  # useless

    solver = AcadosSolverHelper(problem=problem, N=N)
    use_cython = False
    solver.setup(recompile=True, use_cython=use_cython)
    # solver.setup(recompile=False, use_cython=use_cython)

    # set trajectory
    data = solver.get_data_template()
    data["x"][box.q.name] = np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0])
    solver.set_state_constant(data["x"])
    solver.set_initial_state(data["x"])
    data["u"]["dt"] = np.array([0.015])
    # for c in [contacts[0], contacts[2]]:
    #     data["u"][c.f.name] = [np.array([0., 0., 1.0])]
    solver.set_input_constant(data["u"])

    # set cost
    # data["W"]["base_vel_cost"] = np.array([1e-3] * 6)
    # data["W_e"]["base_vel_cost"] = np.array([1e-3] * 6)
    # data["W"]["f"] = np.array([0.001] * len(contacts) * 3)
    data["W"]["dt"] = np.array([100.0])
    solver.set_cost_weight_constant(data["W"])
    solver.set_cost_weight_terminal(data["W_e"])
    # data["yref"]["base_vel_cost"] = np.zeros(6)
    data["yref"]["dt"] = np.array([0.015])
    solver.set_ref_constant(data["yref"])
    # data["yref_e"]["base_vel_cost"] = np.zeros(6)
    # solver.set_ref_terminal(data["yref_e"])
    # set constraints
    data["p"][box.impact_active.name][0] = 0.0
    for c in contacts:
        data["p"][c.p_gain.name][0] = 100.0
        data["p"][c.active.name][0] = 1
        data["p"][c.impact.name][0] = 0.0
        data["p"][c.plane_point.name] = np.array([0,0,0])
        data["p"][c.plane_normal.name] = np.array([0,0,1])
    solver.set_parameters_constant(data["p"])

    # ###################################### falling test!
    solver.get_acados("p")
    N_impact = int(10)
    print(f"N_impact: {N_impact}")
    # set constraints
    for c in contacts[2:]:  # initially p3 p4 in air
        for i in range(0, N_impact):
            solver.params[c.active.name][:, i] = 0
    for c in contacts[2:]:  # initially p3 p4 in air
        solver.params[c.impact.name][:, N_impact] = 1.0

    solver.params[box.impact_active.name][:, N_impact] = 1.0
    # data["W"]["base_vel_cost"] = [np.array([50] * 6)]
    # solver.set_cost_weight(N_impact + 1, data["W"])
    solver.update_parameters()
    solver.get_acados("p")
    # # set initial state
    L = 0.05 * np.sqrt(2.0)
    angle = np.pi / 3 + 0.1
    data["x"][box.q.name] = np.array(
        [0.05 - L * np.cos(angle), 0.0, L * np.sin(angle), 0.0, angle - np.pi / 4, 0.0]
    )

    solver.set_state_constant(data["x"])
    solver.set_initial_state(data["x"])

    # for i in range(100):
    solver.solve(print_stats=True, print_time=True)
    # exit(0)
    solver.parse_sol()
    s = solver.states[box.q.name]
    ds = solver.states[box.v.name]
    for i in range(4):
        print(f"{i}:", solver.inputs[contacts[i].f.name][2, :])
    dt_ = solver.inputs["dt"].flatten()
    print(dt_[:N_impact])
    print(dt_[N_impact:])

    from meshcat.animation import Animation
    import meshcat.transformations as tf
    import sys

    robot = loadModelGeom("rsc/go2_description/urdf/box_description.urdf")
    viz = MeshcatVisualizer(*robot)
    try:
        viz.initViewer(open=True)
    except ImportError as err:
        print(err)
        sys.exit(0)
    viz.loadViewerModel()

    import time

    while True:
        viz.display(s[:, 0])
        # print("start: ")
        # input()
        for i, dt in enumerate(dt_[:N_impact]):
            viz.display(s[:, i])
            if i != N_impact:
                time.sleep(float(dt))
        # print("pre_impact: ")
        viz.display(s[:, N_impact])
        time.sleep(float(dt_[N_impact]))
        # print(ds[N_impact])
        # input()
        # print("post_impact: ")
        viz.display(s[:, N_impact + 1])
        # print(ds[N_impact + 1])
        time.sleep(float(dt_[N_impact + 1]))
        # input()
        for i, dt in enumerate(dt_[N_impact + 2 :]):
            # print("go on: ", i)
            viz.display(s[:, i + N_impact + 2])
            time.sleep(float(dt))
            # input()
        time.sleep(2)
