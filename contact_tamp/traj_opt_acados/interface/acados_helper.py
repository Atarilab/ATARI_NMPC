from acados_template import *
from .problem_formuation import *
import numpy as np
import os
import json
import time
from enum import Enum

os.environ["LD_LIBRARY_PATH"] += os.pathsep + "$ACADOS_SOURCE_DIR/lib"


def create_model(name: str, problem: ProblemFormulation) -> AcadosModel:
    model = AcadosModel()
    model.disc_dyn_expr = problem.get_all_expr("xn")
    for n in ["x", "u", "p"]:
        setattr(model, n, problem.get_all_expr(n))
    model.name = name
    problem.add_cost(name="reg", cost=cs.vcat([model.x, model.u]))
    problem.add_cost(name="reg", cost=model.x, path=False, terminal=True)
    return model


class HPIPM_MODE(Enum):
    speed = "SPEED"
    balance = "BALANCE"
    robust = "ROBUST"


class AcadosSolverHelper:

    def __init__(
        self,
        problem: ProblemFormulation,
        N: int,
        name="solver",
        reg_eps: float = 1e-6,
        reg_eps_e: float = 1e-5,
    ):
        self.name = name
        self.problem = problem
        self.N_max = 0
        self.N0 = 0  # idx of the first node
        self.reg_eps = reg_eps
        self.reg_eps_e = reg_eps_e
        self._acados_model = create_model(name=self.name + "_model", problem=problem)
        self.resize_N(N)
        self.set_flat_fields = ['x', 'u', 'z', 'pi', 'lam', 'sl', 'su', 'p']

    def resize_N(self, N):
        self.N = N
        if N > self.N_max:
            self.N_max = N
            # field: number of shooting nodes
            self.raw_fields = {
                "x": N + 1,
                "p": N + 1,  # todo check size
                "u": N,
                "yref": N,
                "W": N,
                "yref_e": 1,
                "W_e": 1,
            }
            self.__raw = {}  # raw data
            self.__data = (
                {}
            )  # stuctured data {field: {key: value}}, MUST be synchronized with acados!
            for f, N_ in self.raw_fields.items():
                self.__raw[f] = np.zeros((self.problem.get_dim(f), N_))
                d = {}
                for name in self.problem.get_names(f):
                    d[name] = np.zeros((self.problem.get_expr_dim(name, f), N_))
                self.__data[f] = d
            ########### bounds.
            # For now use for_all scheme for all bounds
            self.bound_fields = ["x", "u", "h", "h_e"]
            self.__bound_data = {"x": {}, "u": {}, "h": {}, "h_e": {}}
            for f in self.bound_fields:
                # d = {}
                for name in self.problem.get_names(f):
                    if f in ["x", "u"]:
                        lb, ub = self.problem.get_sym_bound(name, f)
                    else:
                        lb, ub = self.problem.get_constr_bound(name, f)
                    if lb.shape[0] != 0:
                        self.__bound_data[f][name] = {"l": lb, "u": ub}

    def get_data_template(
        self,
    ) -> dict[str, np.ndarray]:  # get an empty data struct for convenient mapping
        data = {}  # stuctured data
        for f in self.raw_fields.keys():
            d = {}
            for name in self.problem.get_names(f):
                d[name] = np.zeros((self.problem.get_expr_dim(name, f), 1))
            data[f] = d
        data["W"]["reg"].fill(self.reg_eps)
        data["W_e"]["reg"].fill(self.reg_eps_e)
        return data

    def eval_func(self, res_func: cs.Function):
        return np.array(res_func(np.array(self.__raw["x"]), np.array(self.__raw["p"])))

    def get_acados(self, field: str):  # todo W
        if field not in ["x", "u", "p"]:
            raise RuntimeError(f"field {field} doesn't has get method")
        self.__raw[field] = self.solver.get_flat(field).reshape(-1, self.raw_fields[field],  order="F")
        self.problem.parse_traj(
            traj=self.__raw[field], data=self.__data[field], field=field
        )

    def parse_sol(self):
        self.get_acados("x")
        self.get_acados("u")

    def __set_bounds(self, field: str):
        if field in ["x", "u"]:
            for b in ["lb" + field, "ub" + field]:
                for i in range(self.raw_fields[field]):
                    self.solver.set(i, b, self.__bound_data[b])
        else:
            for b in ["lh", "uh"]:
                for i in range(self.N) if field[-2] != "_e" else [self.N]:
                    self.solver.set(i, b, self.__bound_data[b])

    @typechecked
    def __set_acados(
        self, field: str, dst_idx: int = -1, for_all: bool = False, ext_data: dict = {}
    ):
        # set acados using structured data
        # forall: use one node to set for all nodes
        # single: set for only one node
        terminal = field[-2:] == "_e"
        single = dst_idx != -1
        if for_all:  # for_all == True, will use traj[0]
            dst_idx = 0
        if not single and not for_all:
            traj = self.__raw[field]  # stacked trajectories to be copied
        else:  # single node setting
            traj = self.__raw[field][:, dst_idx : dst_idx + 1]
        if ext_data:
            data = ext_data
        else:
            if for_all or single:  # for any of them, ext_data must be provided
                raise RuntimeError(
                    f"must provide external data for forall: {for_all}, single: {single}"
                )
            data = self.__data[field]
        # stack data
        self.problem.dump_traj(traj=traj, data=data, field=field)
        # update raw, keep sync with acados
        if for_all:
            self.__raw[field][:, :] = traj
        # update data, keep sync with acados
        if single:
            self.problem.parse_traj(
                traj=traj, data=self.__data[field], field=field, dst_idx=dst_idx
            )
        else:
            self.problem.parse_traj(
                traj=self.__raw[field], data=self.__data[field], field=field
            )
        W = field[0] == "W" or field[0] == "yref" # if set diagnal matrix
        setfunc = self.solver.cost_set if W else self.solver.set
        if single or for_all or terminal:
            trj = np.diag(traj[:, 0]) if W else traj[:, 0]
        if (not terminal) and (not single):  # set all nodes
            if not for_all and field in self.set_flat_fields:
                data_flatten = traj.reshape(-1, order="F")
                self.solver.set_flat(field, data_flatten)                
            else:
                for i in range(self.raw_fields[field]):
                    if for_all:
                        setfunc(i, field, trj)
                    else:
                        setfunc(i, field, np.diag(traj[:, i]) if W else traj[:, i])
        else:  # single node setting
            field_ = field[:-2] if terminal else field  # if terminal remove '_e'
            setfunc(self.N if terminal else dst_idx, field_, trj)

    ######## properties
    # to set the acados solver data, one can call to update_xxx after modifying corresponding data
    # or call to set_xxx with a data struct obtained from get_data_template
    # for terminal values, one can call to set_xxx by modifying xxx_terminal or providing data

    @property
    def raw(self):
        return self.__raw

    @property
    def states(self):
        return self.__data["x"]

    @property
    def inputs(self):
        return self.__data["u"]

    @property
    def params(self):
        return self.__data["p"]

    @property
    def cost_weights(self):
        return self.__data["W"]

    @property
    def cost_weights_terminal(self):
        return self.__data["W_e"]

    @property
    def cost_ref(self):
        return self.__data["yref"]

    @property
    def cost_ref_terminal(self):
        return self.__data["yref_e"]

    @property
    def constr_bound(self):
        return self.__bound_data["h"]

    @property
    def terminal_constr_bound(self):
        return self.__bound_data["h_e"]

    @property
    def state_bound(self):
        return self.__bound_data["x"]

    @property
    def input_bound(self):
        return self.__bound_data["u"]

    @typechecked
    def set_cost_weight_constant(self, data: dict[str, np.ndarray]):
        self.__set_acados(field="W", for_all=True, ext_data=data)

    @typechecked
    def set_cost_weight(self, idx: int, data: dict[str, np.ndarray]):
        self.__set_acados(field="W", dst_idx=idx, ext_data=data)

    @typechecked
    def set_ref_constant(self, data: dict[str, np.ndarray]):
        # constant reference yref(0...N-1) = y_des
        self.__set_acados(field="yref", for_all=True, ext_data=data)

    @typechecked
    def set_state_constant(self, data: dict[str, np.ndarray]):
        self.__set_acados(field="x", for_all=True, ext_data=data)

    @typechecked
    def warm_start_multipliers(self, start_node: int, n_warm_start: int, repeat_last: bool = False):
        """
        Warm start the dual variables in the solver from the past solution.

        Args:
            field (str): The name of the field to warm start, e.g., 'lam', 'u', 'pi'.
            start_node (int): The starting node of the previous solution to use for warm start.
            n_warm_start (int): The number of nodes to warm start.
            repeat_last (bool): If True, repeats the last warm-started value for remaining nodes.
        """
        field = "lam"
        # Retrieve the flattened data for the specified field
        raw_data = self.solver.get_flat(field)
        warm_start_data = raw_data[self.N0_lam: -self.N1_lam].reshape(-1, self.N-1, order="F")

        # Perform warm start for the first `n_warm_start` nodes
        warm_start_data[:, :n_warm_start-1] = warm_start_data[:, start_node:]

        # Optionally repeat the last warm-started value for the remaining nodes
        if repeat_last and n_warm_start < self.N:
            warm_start_data[:, n_warm_start:] = warm_start_data[:, n_warm_start]
        
        # Flatten the data back to its original form and set it in the solver
        raw_data[self.N0_lam: -self.N1_lam] = warm_start_data.reshape(-1, order="F")
        self.solver.set_flat(field, raw_data)

    @typechecked
    def set_input_constant(self, data: dict[str, np.ndarray]):
        self.__set_acados(field="u", for_all=True, ext_data=data)

    @typechecked
    def set_initial_state(self, data: dict[str, np.ndarray]):
        self.__set_acados(field="x", dst_idx=0, ext_data=data)
        self.solver.constraints_set(0, "lbx", self.__raw["x"][:, 0])
        self.solver.constraints_set(0, "ubx", self.__raw["x"][:, 0])

    @typechecked
    def set_cost_weight_terminal(self, data: dict[str, np.ndarray] = {}):
        # data can be empty so that it will use local __data['W_e']
        self.__set_acados(field="W_e", ext_data=data)

    @typechecked
    def set_ref_terminal(self, data: dict[str, np.ndarray] = {}):
        # data can be empty so that it will use local __data['yref_e']
        self.__set_acados(field="yref_e", ext_data=data)

    @typechecked
    def set_parameters_constant(self, data: dict[str, np.ndarray]):
        self.__set_acados(field="p", for_all=True, ext_data=data)

    def update_bounds(self, field: str):
        self.__set_bounds(field)

    def update_states(self):
        self.__set_acados(field="x")

    def update_inputs(self):
        self.__set_acados(field="u")

    def update_parameters(self):
        self.__set_acados(field="p")

    def update_cost_weights(self):
        self.__set_acados(field="W")

    def update_ref(self):
        self.__set_acados(field="yref")

    def update_ref_terminal(self):
        self.__set_acados(field="yref_e")

    def setup(
        self,
        recompile: bool = True,
        use_cython: bool = False,
        use_rti: bool = False,
        qp_max_iter: int = 20,
        hpipm_mode: HPIPM_MODE = HPIPM_MODE.balance,
    ):
        self.use_rti = use_rti
        ocp = AcadosOcp()
        problem = self.problem
        ocp.model = self._acados_model
        ocp.solver_options.N_horizon = self.N
        self.ns = [0, 0, 0]
        self.stages = ["_0", "", "_e"]  # initial, path, terminal
        for stage_i, sfx in enumerate(self.stages):
            # nonlinear constraints
            h = "h" + sfx
            if problem.get_dim(h) > 0:
                setattr(ocp.model, "con_h_expr" + sfx, problem.get_all_expr(h))
                lh, uh = problem.get_bound(h)
                setattr(ocp.constraints, "l" + h, lh)
                setattr(ocp.constraints, "u" + h, uh)
                idxs = problem.get_slack_idx(h)
                if idxs.size > 0:
                    setattr(ocp.constraints, "idxs" + h, idxs)
                    self.ns[stage_i] += idxs.size
            # cost, acados will copy automatically
            if sfx != "_0":
                setattr(ocp.cost, "cost_type" + sfx, "NONLINEAR_LS")
                ny = problem.get_dim("yref" + sfx)
                nW = problem.get_dim("W" + sfx)
                setattr(
                    ocp.model, "cost_y_expr" + sfx, problem.get_all_expr("yref" + sfx)
                )
                setattr(ocp.cost, "W" + sfx, np.diag(np.ones(nW)))
                setattr(ocp.cost, "yref" + sfx, np.zeros(ny))
        # primal variable bounds
        for v, sfx_ in {"x": self.stages, "u": [""]}.items():
            idx = problem.get_bound_idx(v)
            idxs = problem.get_slack_idx(v)
            for stage_i, sfx in enumerate(sfx_):
                if idx.size > 0:
                    lb, ub = problem.get_bound(v)
                    setattr(ocp.constraints, "idxb" + v + sfx, idx)
                    setattr(ocp.constraints, "lb" + v + sfx, lb)
                    setattr(ocp.constraints, "ub" + v + sfx, ub)
                if idxs.size > 0:  # slack, not tested
                    setattr(ocp.constraints, "idxsb" + v + sfx, idxs)
                    self.ns[stage_i] += idxs.size

        # set slack cost
        for n, s in zip(self.ns, self.stages):
            setattr(ocp.cost, "Zu" + s, np.ones(n) * 1e4)
            setattr(ocp.cost, "Zl" + s, np.ones(n) * 1e4)
            setattr(ocp.cost, "zu" + s, np.ones(n) * 0)
            setattr(ocp.cost, "zl" + s, np.ones(n) * 0)

        self.ocp = ocp
        self.ocp.parameter_values = np.zeros(self.problem.get_dim("p"))
        self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.ocp.solver_options.qp_solver_cond_N = self.N
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        if use_rti:
            self.ocp.solver_options.nlp_solver_type = "SQP_RTI"
        else:
            self.ocp.solver_options.nlp_solver_type = "SQP"
        # note: related to numerical conditioning
        self.ocp.solver_options.tf = 1.0
        self.ocp.solver_options.tol = 1e-4
        self.ocp.solver_options.integrator_type = "DISCRETE"
        self.ocp.solver_options.hpipm_mode = hpipm_mode.value
        self.ocp.solver_options.qp_solver_iter_max = qp_max_iter
        self.ocp.solver_options.nlp_solver_max_iter = 100

        json_file = f"{self.name}_" + "acados_ocp.json"
        verbose = False
        self.solver = AcadosOcpSolver(
            self.ocp,
            json_file=json_file,
            generate=recompile,
            build=recompile if not use_cython else False,
            verbose=verbose,
        )
        if use_cython:
            with open(json_file, "r") as f:
                acados_ocp_json = json.load(f)
            code_export_directory = acados_ocp_json["code_export_directory"]
            if recompile:
                self.solver.build(
                    code_export_directory, with_cython=True, verbose=verbose
                )
            self.solver = self.solver.create_cython_solver(json_file)
        # print("###################")
        # print(f"{self.name} setup!")
        # print("###################")
        # To warm start dual variables
        self.N0_lam = len(self.solver.get(0, "lam"))
        self.N1_lam = len(self.solver.get(self.N, "lam"))


    def solve(self, print_stats: bool = False, print_time: bool = False):
        try:
            if print_time:
                st = time.perf_counter()
            self.solver.solve()
            self.solver.get_cost()
            if print_time:
                ed = time.perf_counter()
            if print_stats:
                self.solver.print_statistics()
            if print_time:
                print("name\ttime(ms)")
                # python timer
                print("py_tot:\t%.2f" % ((ed - st) * 1000))
                t = [
                    ("qp_tot", "time_qp"),
                    ("qp_sol", "time_qp_solver_call"),
                    ("reg", "time_reg"),
                    ("c_tot", "time_tot"),
                ]
                for n, f in t:
                    print(f"{n}:\t{self.solver.get_stats(f)*1000:.2f}")
                n_qp = sum(self.solver.get_stats("qp_iter"))
                avg_qp = self.solver.get_stats("time_qp") * 1000 / n_qp
                print("avg_qp:\t%.2f" % avg_qp)
                print("avg_nd:\t%.3f" % (avg_qp / self.N))  # averge per node
            acados_status = self.solver.get_status()

            if acados_status > 0 and acados_status != 2:
                self.check_hessian_singular()
                raise RuntimeError(f"solver failure status: {acados_status}")

        # enum hpipm_status
        # 	{
        # 	SUCCESS, // found solution satisfying accuracy tolerance
        # 	MAX_ITER, // maximum iteration number reached
        # 	MIN_STEP, // minimum step length reached
        # 	NAN_SOL, // NaN in solution detected
        # 	INCONS_EQ, // unconsistent equality constraints
        # 	};
        except:
            raise RuntimeError("solver failure")

    @typechecked
    def set_max_iter(self, n_iter: int):
        if not self.use_rti:
            self.solver.options_set("max_iter", n_iter)

    @typechecked
    def set_qp_tol(self, tol: float):
        self.solver.options_set("qp_tol_stat", tol)
        self.solver.options_set("qp_tol_eq", tol)
        self.solver.options_set("qp_tol_ineq", tol)
        self.solver.options_set("qp_tol_comp", tol)

    @typechecked
    def set_nlp_tol(self, tol: float):
        self.solver.options_set("tol_eq", tol)
        self.solver.options_set("tol_stat", tol)
        self.solver.options_set("tol_ineq", tol)
        self.solver.options_set("tol_comp", tol)

    @typechecked
    def set_warm_start_nlp(self, warm_start: bool):
        if not self.use_rti:
            self.solver.options_set("warm_start_first_qp", warm_start)

    @typechecked
    def set_warm_start_inner_qp(self, warm_start: bool):
        if not self.use_rti:
            self.solver.options_set("qp_warm_start", warm_start)

    @typechecked
    def set_slack_cost(self, Z: float, z: float):
        """
        Z: [float] quadratic penalty
        z: [float] linear penalty
        """
        n_nodes = [[0], range(self.N), self.N]
        for i, (n, s) in enumerate(zip(self.ns, self.stages)):
            for i in n_nodes[i]:
                for z in ["zl", "zu", "Zl", "Zu"]:
                    self.solver.cost_set(i, z, np.ones(n) * eval(z[0]))

    def check_hessian_singular(self):
        H = [(self.solver.get_hessian_block(i)) for i in range(self.N + 1)]
        for i in range(self.N + 1):
            H, Q, R, S = self.solver.get_hessian_block(i)
            try:
                np.linalg.inv(H)
            except:
                print(f"{i}\n", np.diagonal(H))
                return
