import math
from typing import Callable, List, NamedTuple

import numpy as np
import z3

from pyz3_utils import BinarySearch


class Metric(NamedTuple):
    z3ExprRef: z3.ExprRef
    lo: float
    hi: float
    eps: float
    maximize: bool = True

    def name(self) -> str:
        ret = self.z3ExprRef.decl().name()
        assert isinstance(ret, str)
        return ret


def fix_metrics(solver: z3.Solver, metric_list: List[Metric]):
    for metric in metric_list:
        if(metric.maximize):
            solver.add(metric.z3ExprRef == metric.lo)
        else:
            solver.add(metric.z3ExprRef == metric.hi)


def optimize_var(s: z3.Solver, variable: z3.ExprRef, lo, hi, eps, maximize=True):
    """
    WLOG, assume we are maximizing, Find the maximum output value of input
    variable in the range [lo, hi] (with accuracy of eps), such that the formula
    checked by solver s is unsatisfiable.

    To minimize set maximize = False.

    If we want optimum value such that s is satisfiable, just reverse polarity of maximize.
    """

    # Assert that input is function application with zero
    assert len(variable.children()) == 0
    assert variable.num_args() == 0

    """
    The binary search process assumes lo, hi map to 1 1 1... 2 2 2... 3 3 3...
    So if we want maximum value for unsat, lo/1 must be registered unsat.
    Otherwise, if lo is sat then maximum value is less than lo - eps.
    """
    sat_value_1 = 'unsat'
    sat_value_3 = 'sat'

    if(not maximize):
        sat_value_1 = 'sat'
        sat_value_3 = 'unsat'

    print(f"Optimizing {variable.decl().name()}.")

    binary_search = BinarySearch(lo, hi, eps)
    while True:
        pt = binary_search.next_pt()
        if(pt is None):
            break

        print(f"Optimizing {variable.decl().name()}. Trying value: {pt}")
        s.push()
        s.add(variable == pt)
        sat = s.check()
        s.pop()

        if(str(sat) == sat_value_1):
            binary_search.register_pt(pt, 1)
        elif str(sat) == "unknown":
            binary_search.register_pt(pt, 2)
        else:
            assert str(sat) == sat_value_3, f"Unknown value: {str(sat)}"
            binary_search.register_pt(pt, 3)

    optimal_bounds = binary_search.get_bounds()
    if(maximize):
        optimal_value = math.floor(optimal_bounds[0]/eps) * eps
    else:
        optimal_value = math.ceil(optimal_bounds[-1]/eps) * eps

    return optimal_value


def optimize_var_nopushpop(
    s: z3.Solver, variable: z3.ExprRef, lo, hi, eps, maximize=True):
    # This is same as the non fast, it just does not use push/pop
    # This allows solver to make use of better preprocessing techniques.
    """
    WLOG, assume we are maximizing, Find the maximum output value of input
    variable in the range [lo, hi] (with accuracy of eps), such that the formula
    checked by solver s is unsatisfiable.

    To minimize set maximize = False.

    If we want optimum value such that s is satisfiable, just reverse polarity of maximize.
    """

    # Assert that input is function application with zero
    assert len(variable.children()) == 0
    assert variable.num_args() == 0

    """
    The binary search process assumes lo, hi map to 1 1 1... 2 2 2... 3 3 3...
    So if we want maximum value for unsat, lo/1 must be registered unsat.
    Otherwise, if lo is sat then maximum value is less than lo - eps.
    """
    sat_value_1 = 'unsat'
    sat_value_3 = 'sat'

    if(not maximize):
        sat_value_1 = 'sat'
        sat_value_3 = 'unsat'

    print("Optimizing {}.".format(variable.decl().name()))

    def create_verifier():
        verifier = z3.Solver()
        for assertion in s.assertions():
            verifier.add(assertion)
        return verifier

    binary_search = BinarySearch(lo, hi, eps)
    while True:
        pt = binary_search.next_pt()
        if(pt is None):
            break

        print("Optimizing {}. Trying value: {}".format(variable.decl().name(), pt))
        verifier = create_verifier()
        verifier.add(variable == pt)
        sat = verifier.check()

        if(str(sat) == sat_value_1):
            binary_search.register_pt(pt, 1)
        elif str(sat) == "unknown":
            binary_search.register_pt(pt, 2)
        else:
            assert str(sat) == sat_value_3, f"Unknown value: {str(sat)}"
            binary_search.register_pt(pt, 3)

    optimal_bounds = binary_search.get_bounds()
    if(maximize):
        optimal_value = math.floor(optimal_bounds[0]/eps) * eps
    else:
        optimal_value = math.ceil(optimal_bounds[-1]/eps) * eps

    return optimal_value


def get_raw_value(expr: z3.ExprRef):
    try:
        if(isinstance(expr, z3.RatNumRef)):
            return expr.as_fraction()
        elif(isinstance(expr, z3.BoolRef)):
            return bool(expr)
        elif(isinstance(expr, z3.ArithRef)):
            return np.nan
        else:
            raise NotImplementedError
    except z3.z3types.Z3Exception as e:
        return np.nan


def unroll_assertions(expression: z3.ExprRef) -> List[z3.ExprRef]:
    """
    If the input expression is And of multiple expressions,
    then this returns a list of the constituent expressions.
    This is done recursively untill the constituent expressions
    use a different z3 operation at the AST root.
    """
    ret = []
    if(z3.is_and(expression)):
        for constituent in expression.children():
            ret.extend(unroll_assertions(constituent))
    else:
        ret.append(expression)
    return ret


def custom_get_unsat_core(solver: z3.Solver):
    dummy = z3.Solver()
    dummy.set(unsat_core=True)

    assertions = []
    for assertion in solver.assertions():
        for expr in unroll_assertions(assertion):
            assertions.append(expr)

    non_track_assertions = assertions

    for assertion in non_track_assertions:
        dummy.add(assertion)

    track_assertions = []
    i = 0
    for assertion in track_assertions:
        i += 1
        dummy.assert_and_track(assertion, str(assertion) + f"  :{i}")

    import ipdb; ipdb.set_trace()

    assert(str(dummy.check()) == "unsat")
    unsat_core = dummy.unsat_core()
    import ipdb; ipdb.set_trace()
    return unsat_core


def get_unsat_core(solver: z3.Solver):
    dummy = z3.Solver()
    dummy.set(unsat_core=True)

    assertions = []
    for assertion in solver.assertions():
        for expr in unroll_assertions(assertion):
            assertions.append(expr)

    i = 0
    for assertion in assertions:
        i += 1
        dummy.assert_and_track(assertion, str(assertion) + f"  :{i}")

    assert(str(dummy.check()) == "unsat")
    unsat_core = dummy.unsat_core()
    return unsat_core


def qe_simplify(g: z3.BoolRef) -> z3.BoolRef:
    tactic = z3.Then(
        # 'fm',

        # 'qe2',
        # 'qe',
        'solve-eqs',
        'propagate-values2',
        'propagate-ineqs',

        # 'demodulator',
        # 'dom-simplify',

        'purify-arith',
        'simplify',
        'unit-subsume-simplify',

        'solver-subsumption',

        # 'aig',
        # 'sat-preprocess',
        # 'fm',
        # 'ctx-solver-simplify',
        # 'ctx-simplify,
        # 'add-bounds',
    )
    ret = tactic(g)
    return ret.as_expr()


def try_except(function: Callable):
    try:
        return function()
    except Exception:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
        assert False


def try_except_wrapper(function):
    def func_to_return(*args, **kwargs):
        def func_to_try():
            return function(*args, **kwargs)
        return try_except(func_to_try)
    return func_to_return


def z3_min(a: z3.ArithRef, b: z3.ArithRef):
    ret = z3.If(a < b, a, b)
    assert isinstance(ret, z3.ArithRef)
    return ret


def z3_max(a: z3.ArithRef, b: z3.ArithRef):
    ret = z3.If(a > b, a, b)
    assert isinstance(ret, z3.ArithRef)
    return ret


def z3_min_list(args: List[z3.ArithRef]):
    ret = args[0]
    for arg in args[1:]:
        ret = z3_min(ret, arg)
    return ret


def z3_max_list(args: List[z3.ArithRef]):
    ret = args[0]
    for arg in args[1:]:
        ret = z3_max(ret, arg)
    return


def z3_abs(x: z3.ArithRef) -> z3.ArithRef:
    ret = z3.If(x >= 0, x, -x)
    assert isinstance(ret, z3.ArithRef)
    return ret
