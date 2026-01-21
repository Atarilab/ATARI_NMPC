from ..core.common import *


class ProblemFormulation:
    def __init__(
        self,
        dt_nom: float = 0.0,
        dt_min: float = 0.0,
        dt_max: float = 0.0,
        enable_time_opt: bool = False,
    ):
        self.dt = cs.SX.sym("dt")
        self.sym_fields: list[str] = ["x", "u"]
        # y is for observation such as contact location. Not involved in optimization
        self.param_fields: list[str] = ["p", "W", "W_e", "y"]
        self.constr_fields: list[str] = ["h_0", "h", "h_e"]
        self.cost_fields: list[str] = ["yref", "yref_e"]
        self.expr = {"xn": []}  # xn is x_next
        for d in ["lb", "ub", "ls", "us", "idx", "idxb", "idxs", "dim"]:
            setattr(self, d, {})
        for f in (
            self.sym_fields + self.param_fields + self.constr_fields + self.cost_fields
        ):
            self.expr[f] = []
            self.idx[f] = {}
            self.dim[f] = 0
        for f in self.sym_fields + self.constr_fields:
            self.lb[f] = []
            self.ub[f] = []
            self.ls[f] = []
            self.us[f] = []
            self.idxs[f] = []
        for f in self.sym_fields:
            self.idxb[f] = []
        if enable_time_opt:
            self.add_input(u=self.dt, name="dt")
            self.add_bound(name="dt", lb=[dt_min], ub=[dt_max], field="u", idx=[0])
            self.add_cost(name="dt", cost=self.dt)
        else:
            self.dt = dt_nom

    def get_names(self, field) -> list[str]:
        return self.idx[field].keys()

    def get_dim(self, field: str) -> int:
        return self.dim[field]

    def add_obs(self, y: cs.SX, name: str = ""):
        self.add_expr(s=y, field="y", name=name)

    def obs_eval_func(self, name: str):
        try:
            idx0, idx1 = self.idx["y"][name]
            return cs.Function(
                f"{name}_eval",
                [self.get_all_expr("x"), self.get_all_expr("p")],
                [self.get_all_expr("y")[idx0:idx1]],
            )
        except:
            raise RuntimeError(f"the obs {name} does not exist (?)")

    def constr_eval_func(self, name: str, terminal: bool = False):
        field = "h_e" if terminal else "h"
        idx0, idx1 = self.idx[field][name]
        return cs.Function(
            f"{name}_eval",
            [self.get_all_expr("x"), self.get_all_expr("p")],
            [self.get_all_expr(field)[idx0:idx1]],
        )

    @typechecked
    def add_expr(self, s: cs.SX, field: str, name: str):
        try:
            if name == "":
                name = s.name
        except:
            raise RuntimeError(f"cannot get expression name {s}")

        ns = self.dim[field]
        self.dim[field] = ns + s.shape[0]
        self.idx[field][name] = (ns, self.dim[field])
        self.expr[field].append(s)

    def get_all_expr(self, field: str) -> cs.SX:
        return cs.vcat(self.expr[field])

    def get_expr_dim(self, name: str, field: str) -> int:
        idx0, idx1 = self.idx[field][name]
        return idx1 - idx0

    @typechecked
    def add_bound(
        self,
        name: str = "",
        expr: cs.SX | None = None,
        lb: list[float] = [],
        ub: list[float] = [],
        field: str = "",
        idx: list[int] = [],
        slack_idx: list[int] = [],
    ):
        if expr != None:
            if name != "":
                if expr.name != name:
                    raise RuntimeError(
                        f"expression name {expr.name} and bound name {name} mismatch"
                    )
            else:
                name = expr.name

        if field in self.sym_fields:
            idx0, _ = self.idx[field][name]
            if idx:
                idx_ = [idx0 + i for i in idx]
            else:
                idx_ = [idx0 + i for i in range(self.get_expr_dim(name, field))]
            self.idxb[field].extend(idx_)
        if (field in self.sym_fields or field in self.constr_fields) and slack_idx:
            idx0, _ = self.idx[field][name]
            self.idxs[field].extend([idx0 + i for i in slack_idx])
        self.lb[field].extend(lb)
        self.ub[field].extend(ub)

    def get_bound(self, field: str) -> tuple[np.ndarray, np.ndarray]:
        """
        get bound of the field
        """
        return np.array(self.lb[field]), np.array(self.ub[field])

    def get_bound_idx(self, field: str) -> np.ndarray[np.int64]:
        """
        get bound indices of the field
        """
        return np.array(self.idxb[field], dtype=np.int64)

    def get_sym_bound_idx(self, name: str, field: str):
        idx0, idx1 = self.idx[field][name]
        b_idx = self.get_bound_idx(field)
        b_idx0 = np.searchsorted(b_idx, idx0)
        b_idx1 = np.searchsorted(b_idx, idx1)
        return b_idx0, b_idx1

    def get_sym_bound(self, name: str, field: str) -> tuple[np.ndarray, np.ndarray]:
        """
        get bound of symbolic variables by name
        """
        lb, ub = self.get_bound(field)
        b_idx0, b_idx1 = self.get_sym_bound_idx(name, field)
        return lb[b_idx0:b_idx1], ub[b_idx0:b_idx1]

    def get_constr_bound(self, name: str, field: str) -> tuple[np.ndarray, np.ndarray]:
        """
        get bound of constraints by name
        """
        lb, ub = self.get_bound(field)
        idx0, idx1 = self.idx[field][name]
        return lb[idx0:idx1], ub[idx0:idx1]

    def get_slack_idx(self, field: str) -> np.ndarray:
        return np.array(self.idxs[field], dtype=np.int64)

    def add_constr(
        self,
        name: str,
        expr: cs.SX,
        lb: list[float],
        ub: list[float],
        field: str,
        slack_idx: list[int] = [],
    ):
        self.add_expr(s=expr, name=name, field=field)
        self.add_bound(name=expr.name, lb=lb, ub=ub, field=field, slack_idx=slack_idx)

    @typechecked
    def parse_traj(
        self, traj: np.ndarray, data: dict[str, np.ndarray], field: str, dst_idx=-1
    ):
        for name, idx in self.idx[field].items():
            if dst_idx >= 0:
                data[name][:, dst_idx] = traj[idx[0] : idx[1], :].flatten()
            else:
                data[name] = traj[idx[0] : idx[1], :]

    @typechecked
    def dump_traj(self, traj: np.ndarray, data: dict[str, np.ndarray], field: str):
        for name, idx in self.idx[field].items():
            d = data[name]
            # for i, v in enumerate(traj):
            traj[idx[0] : idx[1], :] = d if len(d.shape) > 1 else d.reshape(-1, 1)

    def add_input(self, u: cs.SX, name: str = ""):
        self.add_expr(s=u, name=name, field="u")

    def add_state(self, x: cs.SX, xnext: cs.SX, name: str = ""):
        self.add_expr(s=x, name=name, field="x")
        self.expr["xn"].append(xnext)

    def add_parameter(self, p: cs.SX, name: str = ""):
        self.add_expr(s=p, name=name, field="p")

    def add_eq_constr(
        self,
        expr: cs.SX,
        name: str = "",
        initial: bool = True,
        path: bool = True,
        terminal: bool = True,
        slack_idx: list[int] = [],
    ):
        n = expr.shape[0]
        self.add_ineq_constr(
            expr, [0.0] * n, [0.0] * n, name, initial, path, terminal, slack_idx
        )

    def add_ineq_constr(
        self,
        expr: cs.SX,
        lb: list[float],
        ub: list[float],
        name: str = "",
        initial: bool = True,
        path: bool = True,
        terminal: bool = True,
        slack_idx: list[int] = [],
    ):
        for idx, stage in enumerate([initial, path, terminal]):
            if stage:
                self.add_constr(name, expr, lb, ub, self.constr_fields[idx], slack_idx)

    def add_cost(
        self, cost: cs.SX, path: bool = True, terminal: bool = False, name: str = ""
    ):
        # only nonlinear ls cost is considered
        def __add_cost__(terminal):
            self.add_expr(s=cost, name=name, field="yref" if not terminal else "yref_e")
            W_name = cost.name if name == "" else name
            W = cs.SX.sym(W_name, cost.shape[0])  # cost gain
            self.add_expr(s=W, name=W_name, field="W" if not terminal else "W_e")
        if path : __add_cost__(False)
        if terminal : __add_cost__(True)