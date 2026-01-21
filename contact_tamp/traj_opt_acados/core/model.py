from .common import *


# this class is to add prefix and names for expressions
class AbstractModel:
    def __init__(self, name: str):
        self.name = name
        self.__expr: dict[str, cs.SX] = {}

    # @staticmethod
    # def set_hash(name: str, expr: cs.SX) -> None:
    #     # todo: casadi hash bug

    def add_sym(self, name: str, nrow: int = 1, ncol: int = 1):
        name = f"{name}_{self.name}"
        expr = cs.SX.sym(name, nrow, ncol)
        expr.name = name
        self.__expr[name] = expr
        return expr

    @typechecked
    def add_expr(
        self,
        name: str,
        expr: cs.SX,
        lb: list[float] = [],
        ub: list[float] = [],
    ):
        name = f"{name}_{self.name}"
        expr.name = name
        self.__expr[name] = expr
        if lb and ub:
            return {"expr": expr, "lb": lb, "ub": ub}
        else:
            return expr

    def expr(self, name: str):
        return self.__expr[f"{name}_{self.name}"]
