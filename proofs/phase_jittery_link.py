# %%
import time
from typing import List

import sympy as sp
import z3
from IPython.display import display
from util import (
    Metric,
    fix_metrics,
    optimize_var,
    optimize_var_nopushpop,
    z3_abs,
    z3_max,
    z3_max_list,
    z3_min,
)

sp.init_printing(use_unicode=True)
N_G = 2
PRECISION = 4

"""
--------------------------------------------------------------------------------
Assumptions:
--------------------------------------------------------------------------------
0. Dumbbell topology, with network bandwidth C mbps that may only vary slowly
   over time.

1. Propagation delay same for all flows = R seconds. R includes transmission
   delay etc., think of it as min RTT.

2. Jitter bound same for all flows = D seconds.

3. Number of flows = N_G.

4. Slots and rounds synchronized. All measurements occur before all updates.

- See paper for detailed list of assumptions and design.

--------------------------------------------------------------------------------
Conventions:
--------------------------------------------------------------------------------

1. hat for estimates, non-hats for reference state.

--------------------------------------------------------------------------------
Design:
--------------------------------------------------------------------------------

Parameters set by us: theta, alpha, delta_H, delta_L, gamma, gamma_R, deltaR

1. Contract:
>>> target_delay = theta * N_C_i_hat

2. Measurement:
>>> rate_i_hat = max of ACK rates over packet-timed RTT.
The max helps filter out decrease in ACK rate due to other flows' probes.
>>> excess = gamma * rate_i_hat * N_T_i_hat * deltaR
This excess creates gamma * deltaR delay when rate_i_hat = fair share and N_T_i_hat = N_G.
>>> probing_cwnd = cwnd + excess
>>> C_i_hat = excess / excess_delay_created_by_excess
>>> excess_delay_created_by_excess = excess / C (with an error of plusminus deltaR).

3. Update:
>>> N_C_i_hat = C_i_hat / rate_i_hat
>>> target_factor = N_C_i_hat / N_T_i_hat
>>> target_factor = min(target_factor, delta_H)
>>> target_factor = max(target_factor, delta_L)
>>> target_cwnd = target_factor * cwnd
>>> next_cwnd = (1-alpha) * cwnd + alpha * target_cwnd
"""


class ParameterChoices:
    alpha = 1                      # 0 < alpha <= 1. cwnd averaging: next_cwnd = (1-alpha) * cwnd + alpha * target_cwnd.
    delta_H = 1.3                  # delta_H > 1. Upper clamp: N_C_i_hat <= N_T_i_hat * delta_H.
    delta_L = 0.8                  # delta_L < 1. Lower clamp: N_C_i_hat >= N_T_i_hat * delta_L.
    jitter_scale = 1.
    gamma = 4 * jitter_scale       # gamma1 > 1. probing multiplier: probe = gamma1 * rate_i_hat * N_T_hat * deltaR.
    deltaR_by_theta = 0.5 / jitter_scale
    gammaR_ub = 1 + 1/jitter_scale


class Z3Variables():
    # Network parameters
    R = z3.Real('R')
    gamma_R = z3.Real('gamma_R')   # Multiplicative error in RTT (affects error in rate estimate compared to reference execution)

    # Design parameters
    theta = z3.Real('theta')
    alpha = z3.Real('alpha')
    delta_H = z3.Real('delta_H')
    delta_L = z3.Real('delta_L')
    gamma = z3.Real('gamma')
    deltaR_by_theta = z3.Real('deltaR_by_theta')

    # State in reference execution
    N_C = [z3.Real(f"N_C_{k}") for k in range(N_G)]
    N_T = z3.Real('N_T')

    # Derived state and choices
    scaling_k = [z3.Real(f"scaling_{k}") for k in range(N_G)]
    # This is the value of N_C_hat_k/N_T_hat_k used in the cwnd update.
    # Z3 picks this based on the error bounds we compute.
    # FRCC works for all possible observations of "scaling" that are within the modeled error bounds.
    scaling_k_clamp = [z3.Real(f"scaling_{k}_clamp") for k in range(N_G)]
    # This represents the clamped value of the scaling factor.

    def __init__(self):
        self.state_domain = [self.N_C[k] > 1 for k in range(N_G)] + [self.N_T > 0]
        self.network_domain = [
            self.R > 0,
            self.gamma_R >= 1,
        ]
        self.parameter_choices = [
            self.theta > 0,
            self.alpha == ParameterChoices.alpha,
            self.delta_H == ParameterChoices.delta_H,
            self.delta_L == ParameterChoices.delta_L,
            self.gamma == ParameterChoices.gamma,
            self.deltaR_by_theta == ParameterChoices.deltaR_by_theta,
        ]
        self.parameter_domain = [
            self.theta > 0,
            self.alpha > 0,
            self.alpha <= 1,
            self.delta_H > 1,
            self.delta_L < 1,
            self.gamma > 1,
        ]

        assert N_G == 2
        init_constraints = [self.N_C[1] * (self.N_C[0] - 1) == self.N_C[0]]

        N_T_largest = self.N_T + self.deltaR_by_theta
        self.N_T_largest = N_T_largest

        error_bounds = []
        for k in range(N_G):
            lb1 = self.scaling_k[k] * N_T_largest >= 1  # type: ignore
            lb2 = (
                self.scaling_k[k]
                * (self.gamma * self.gamma_R * N_T_largest + self.N_C[k])
            ) >= (
                self.gamma * self.N_C[k]
            )  # type: ignore
            ub = z3.Implies(
                self.gamma * self.N_T - self.N_C[k] > 0,
                (self.scaling_k[k] * (self.gamma * self.N_T - self.N_C[k]))
                <= (self.gamma * self.N_C[k]),  # type: ignore
            )
            clamps = self.scaling_k_clamp[k] == z3_max(
                z3_min(self.scaling_k[k], self.delta_H), self.delta_L
            )
            error_bounds += [lb1, lb2, ub, clamps]

        # This is the main cwnd update that Z3 takes forever to reason about.
        error_bounds_stable = []
        for k in range(N_G):
            lb = self.scaling_k[k] * (self.R + self.theta * N_T_largest) >= (
                self.R + self.theta * 1
            )
            lb1 = (
                self.scaling_k[k]
                * (self.R + self.theta * self.N_T)
                * (self.gamma * self.gamma_R * self.N_T + self.N_C[k])
            ) >= (  # type: ignore
                self.R * (self.gamma * self.gamma_R * self.N_T + self.N_C[k])
                + self.theta * (self.N_C[k] * self.gamma * self.N_T)
            )  # type: ignore
            lb2 = (
                self.scaling_k[k]
                * (self.R + self.theta * N_T_largest)
                * (self.gamma * self.gamma_R * N_T_largest + self.N_C[k])
            ) >= (  # type: ignore
                self.R * (self.gamma * self.gamma_R * N_T_largest + self.N_C[k])
                + self.theta * (self.N_C[k] * self.gamma * N_T_largest)
            )  # type: ignore
            ub = z3.Implies(
                self.gamma * self.N_T - self.N_C[k] > 0,
                (
                    self.scaling_k[k]
                    * (self.gamma * self.N_T - self.N_C[k])
                    * (self.R + self.theta * self.N_T)
                )
                <= (
                    self.R * (self.gamma * self.N_T - self.N_C[k])
                    + self.theta * (self.N_C[k] * self.gamma * self.N_T)
                ),  # type: ignore
            )
            clamps = self.scaling_k_clamp[k] == z3_max(
                z3_min(self.scaling_k[k], self.delta_H), self.delta_L
            )
            error_bounds_stable += [lb, z3.Or(lb1, lb2), ub, clamps]

        error_bounds = error_bounds
        self.update_model = init_constraints + error_bounds


class SpVariables():
    C, R = sp.symbols('C R', domain=sp.S.Reals, positive=True)
    deltaR = sp.Symbol('\\DeltaR', domain=sp.S.Reals, positive=True)

    theta = sp.symbols('theta', domain=sp.S.Reals, positive=True)
    alpha = sp.symbols('alpha', domain=sp.S.Reals, positive=True)
    alpha = 1
    delta_H, delta_L = sp.symbols('delta_H delta_L', domain=sp.S.Reals, positive=True)
    gamma, gamma_R = sp.symbols('gamma gamma_R', domain=sp.S.Reals, positive=True)

    N_C = [sp.symbols(f"N_{{C_{i}}}", domain=sp.S.Reals, positive=True) for i in range(N_G)]
    N_C_hat_k = [sp.symbols(f"\\widehat{{N_{{C_{i}}}}}", domain=sp.S.Reals, positive=True) for i in range(N_G)]
    N_T_hat_k = [sp.symbols(f"\\widehat{{N_{{T_{i}}}}}", domain=sp.S.Reals, positive=True) for i in range(N_G)]
    N_T = sp.symbols('N_T', domain=sp.S.Reals, positive=True)
    scaling_k = [sp.symbols(f"scaling_{i}", domain=sp.S.Reals, positive=True) for i in range(N_G)]

    def __init__(self):
        self.N_last = 1 / (1 - sum([1/self.N_C[i] for i in range(N_G-1)]))
        self.next_N_C, self.next_N_T = round_update(self, self.N_C, self.N_T, self.N_C_hat_k, self.N_T_hat_k)


def round_update(sp: SpVariables, N_C, N_T, N_C_hat_k, N_T_hat_k):
    # Compute next state given current state, and estimates.
    RTT = N_T * sp.theta + sp.R
    cwnd_k = [sp.C * RTT / N_C[i] for i in range(N_G)]

    next_cwnd_k = []
    for i in range(N_G):
        # scaling = (sp.R + sp.theta * N_C_hat_k[i])/(sp.R + sp.theta * N_T_hat_k[i])
        # scaling = N_C_hat_k[i]/N_T_hat_k[i]
        scaling = sp.scaling_k[i]
        next_cwnd_k.append(cwnd_k[i] * (1-sp.alpha + sp.alpha * scaling))

    next_RTT = sum(next_cwnd_k) / sp.C
    next_N_C = [sp.C * next_RTT / next_cwnd_k[i] for i in range(N_G)]
    next_N_T = (next_RTT - sp.R) / sp.theta

    return next_N_C, next_N_T


def print_state(m: z3.ModelRef, zv: Z3Variables, sv: SpVariables):
    for k in range(N_G):
        print(f"N_C_{k} = ", m.eval(zv.N_C[k]).as_decimal(PRECISION))  # type: ignore
    print(f"N_T = ", m.eval(zv.N_T).as_decimal(PRECISION))  # type: ignore

    z3_next_N_C = [convert_to_z3(sv.next_N_C[k], sv, zv) for k in range(N_G)]
    for k in range(N_G):
        print(f"next_N_{k} = ", m.eval(z3_next_N_C[k]).as_decimal(PRECISION))  # type: ignore
    z3_next_N_T = convert_to_z3(sv.next_N_T, sv, zv)
    print(f"next_N_T = ", m.eval(z3_next_N_T).as_decimal(PRECISION))  # type: ignore

    print("Checksum before: ", m.eval(z3.Sum([1/zv.N_C[k] for k in range(N_G)])))
    print("Checksum after: ", m.eval(z3.Sum([1/z3_next_N_C[k] for k in range(N_G)])))

    for k in range(N_G):
        print(f"scaling_{k} = ", m.eval(zv.scaling_k[k]).as_decimal(PRECISION))  # type: ignore
        print(f"scaling_{k}_clamp = ", m.eval(zv.scaling_k_clamp[k]).as_decimal(PRECISION))  # type: ignore

        print(f"lb1_{k} = ", m.eval(1 / zv.N_T_largest).as_decimal(PRECISION))  # type: ignore
        print(f"lb2_{k} = ", m.eval(zv.gamma * zv.N_C[k] / (zv.gamma * zv.gamma_R * zv.N_T_largest + zv.N_C[k])).as_decimal(PRECISION))  # type: ignore
        print(f"ub_{k} = ", m.eval(zv.gamma * zv.N_C[k] / (zv.gamma * zv.N_T - zv.N_C[k])).as_decimal(PRECISION))  # type: ignore


def convert_to_z3(expr, sv: SpVariables, zv: Z3Variables):
    # We use sympy to simplify the state update equations and then feed those
    # to Z3 to reason about possible trajectories.

    # Ideally none of the simplified expressions should mention C, so we don't
    # substitute it.

    # The eval at the end of the function uses zv.
    zv = zv

    ret = (
        sp.simplify(expr).subs(sv.R, "zv_R")
        .subs(sv.deltaR, "zv_deltaR")
        .subs(sv.theta, "zv_theta")
        # .subs(sv.alpha, "zv_alpha") # alpha = 1
        .subs(sv.delta_H, "zv_delta_H")
        .subs(sv.delta_L, "zv_delta_L")
        .subs(sv.gamma, "zv_gamma")
        .subs(sv.gamma_R, "zv_gamma_R")
        .subs(sv.N_T, "zv_N_T")
    )
    ret = sp.simplify(ret)
    str_ret = str(ret)
    for k in range(N_G):
        # str_ret = str_ret.replace(str(sv.N_T_hat_k[k]), f"zv_N_T_hat_k[{k}]")
        # str_ret = str_ret.replace(str(sv.N_C_hat_k[k]), f"zv_N_C_hat_k[{k}]")
        str_ret = str_ret.replace(str(sv.N_C[k]), f"zv_N_C[{k}]")
        str_ret = str_ret.replace(str(sv.scaling_k[k]), f"zv_scaling_k_clamp[{k}]")
    str_ret = str_ret.replace("zv_", "zv.")

    return eval(str_ret)


def setup_solver(zv: Z3Variables):
    s = z3.Solver()
    s.add(z3.And(*zv.state_domain, *zv.network_domain, *zv.parameter_domain))
    s.add(z3.And(zv.parameter_choices))
    s.add(z3.And(zv.update_model))
    return s


def check_lemma(
    s: z3.Solver,
    lemma: z3.BoolRef,
    assumptions: z3.BoolRef,
    zv: Z3Variables,
    sv: SpVariables,
):
    print("Finding cex to lemma")
    s.add(assumptions)
    s.add(z3.Not(lemma))

    start = time.time()
    ret = s.check()
    print(f"{ret} in {time.time() - start:.2f} seconds")

    if ret == z3.sat:
        m = s.model()
        print_state(m, zv, sv)
        try:
            print("lemma = ", m.eval(lemma))
        except AttributeError:
            pass
        print("\nModel:")
        print(m)
        return m

    return None


def print_expressions():
    sv = SpVariables()
    zv = Z3Variables()

    print("State update equations computed using sympy. The variable scaling is based on estimates and is noisy.")
    for k in range(N_G):
        display(f"next_N_C_{k} = ", sp.simplify(sv.next_N_C[k]))
    display(f"next_N_T = ", sp.simplify(sv.next_N_T))

    print("Converting to Z3:")
    for k in range(N_G):
        print(f"next_N_C_{k} = ", convert_to_z3(sv.next_N_C[k], sv, zv))
    print(f"next_N_T = ", convert_to_z3(sv.next_N_T, sv, zv))


# %%
class Lemmas:
    # TODO: can convert them into SteadyStateVariables object from ccmatic
    # https://github.com/108anup/ccmatic/blob/main/ccmatic/verifier/__init__.py#L103

    # We found these bounds using binary search for the tightest values at
    # which the corresponding lemmas still hold.
    _bounds = {
        "speed": 1.01,
        "nt_lb": 0.15,  # lemma_steady2
        "nt_ub": 3.3,  # lemma_steady2
        "nci_ub": 6.6,  # lemma_steady1
        "nt_lb_bottom": 0.45,  # lemma_bottom
        "nt_ub_top": 2.66,  # lemma_top
        "nt_lb_band": 1.8,
        "nt_ub_band": 2.2,
    }

    def __init__(self, symbolic_bounds=False):
        sv = SpVariables()
        zv = Z3Variables()

        # TODO: remove redundancy in code here.
        z3_next_nt = convert_to_z3(sv.next_N_T, sv, zv)
        final_nt_num = convert_to_z3(sp.simplify(sv.next_N_T).as_numer_denom()[0], sv, zv)
        final_nt_den = convert_to_z3(sp.simplify(sv.next_N_T).as_numer_denom()[1], sv, zv)

        z3_next_nc0 = convert_to_z3(sv.next_N_C[0], sv, zv)
        final_nci_num = convert_to_z3(sp.simplify(sv.next_N_C[0]).as_numer_denom()[0], sv, zv)
        final_nci_den = convert_to_z3(sp.simplify(sv.next_N_C[0]).as_numer_denom()[1], sv, zv)

        z3_next_nc1 = convert_to_z3(sv.next_N_C[1], sv, zv)
        final_nci_num_other = convert_to_z3(sp.simplify(sv.next_N_C[1]).as_numer_denom()[0], sv, zv)
        final_nci_den_other = convert_to_z3(sp.simplify(sv.next_N_C[1]).as_numer_denom()[1], sv, zv)

        init_nci = zv.N_C[0]
        init_nci_other = zv.N_C[1]
        init_nt = zv.N_T

        speed = Lemmas._bounds["speed"]
        nt_lb = Lemmas._bounds["nt_lb"]
        nt_ub = Lemmas._bounds["nt_ub"]
        nt_lb_bottom = Lemmas._bounds["nt_lb_bottom"]
        nt_ub_top = Lemmas._bounds["nt_ub_top"]
        nt_lb_band = Lemmas._bounds["nt_lb_band"]
        nt_ub_band = Lemmas._bounds["nt_ub_band"]
        nci_ub = Lemmas._bounds["nci_ub"]
        nci_lb = nci_ub/(nci_ub - 1)

        if symbolic_bounds:
            speed = z3.Real('speed')
            nt_lb = z3.Real('nt_lb')
            nt_ub = z3.Real('nt_ub')
            nci_ub = z3.Real('nci_ub')
            nci_lb = nci_ub/(nci_ub - 1)

            EPS = 5 * 1e-2
            metrics = [
                Metric(speed, Lemmas._bounds["speed"], 2, EPS),
                Metric(nt_lb, Lemmas._bounds["nt_lb"], N_G, EPS),
                Metric(nt_ub, N_G, Lemmas._bounds["nt_ub"], EPS, maximize=False),
                Metric(nci_ub, N_G, Lemmas._bounds["nci_ub"], EPS, maximize=False),
            ]
            self.metrics = metrics

        # We actually do not need any assumption about ni here.
        lemma_top = z3.Implies(
            init_nt > nt_ub_top,
            z3.Or(
                final_nt_num <= final_nt_den * init_nt / speed,
                final_nt_num <= final_nt_den * nt_ub_top,
            ),  # nr decreases or converges
            final_nt_num >= final_nt_den * nt_lb,
        )
        assert isinstance(lemma_top, z3.BoolRef)

        lemma_bottom = z3.Implies(
            init_nt < nt_lb_bottom,
            z3.Or(
                final_nt_num > final_nt_den * init_nt * speed,
                final_nt_num >= final_nt_den * nt_lb_bottom,
            ),  # nr decreases or converges
            final_nt_num <= final_nt_den * nt_ub,
        )
        assert isinstance(lemma_bottom, z3.BoolRef)

        lemma_steady1 = z3.Implies(
            z3.And(
                # init_nt <= nt_ub,
                # init_nt >= nt_lb,
                init_nci >= nci_lb,
                init_nci <= nci_ub,
            ),
            z3.And(
                final_nci_num <= final_nci_den * nci_ub,
                final_nci_num >= final_nci_den * nci_lb,
            )
        )
        lemma_steady2 = z3.Implies(
            z3.And(
                init_nt <= nt_ub,
                init_nt >= nt_lb,
                # init_nci >= nci_lb,
                # init_nci <= nci_ub,
            ),
            z3.And(
                final_nt_num <= nt_ub * final_nt_den,
                final_nt_num >= nt_lb * final_nt_den,
            )
        )
        assert isinstance(lemma_steady1, z3.BoolRef)
        assert isinstance(lemma_steady2, z3.BoolRef)

        init_nci_converged = z3.And(
            init_nci <= nci_ub,
            init_nci >= nci_lb,
        )
        init_nt_converged = z3.And(
            init_nt <= nt_ub,
            init_nt >= nt_lb,
        )
        final_nci_converged = z3.And(
            final_nci_num <= final_nci_den * nci_ub,
            final_nci_num >= final_nci_den * nci_lb,
        )
        final_nt_converged = z3.And(
            final_nt_num <= final_nt_den * nt_ub,
            final_nt_num >= final_nt_den * nt_lb,
        )

        bad_nci_improves = z3.Or(
            z3.And(
                init_nci > nci_ub,
                final_nci_num >= final_nci_den * nci_lb,
                z3.Or(
                    final_nci_num <= final_nci_den * init_nci / speed,
                    final_nci_num <= final_nci_den * nci_ub,
                ),
            ),
            z3.And(
                init_nci_other > nci_ub,
                final_nci_num_other >= final_nci_den_other * nci_lb,
                z3.Or(
                    final_nci_num_other <= final_nci_den_other * init_nci_other / speed,
                    final_nci_num_other <= final_nci_den_other * nci_ub,
                ),
            ),
        )

        bad_nt_improves = z3.Or(
            z3.And(
                init_nt > nt_ub_band,
                final_nt_num >= final_nt_den * nt_lb_band,
                z3.Or(
                    final_nt_num <= final_nt_den * init_nt / speed,
                    final_nt_num <= final_nt_den * nt_ub_band,
                ),
            ),
            z3.And(
                init_nt < nt_lb_band,
                final_nt_num <= final_nt_den * nt_ub_band,
                z3.Or(
                    final_nt_num >= final_nt_den * init_nt * speed,
                    final_nt_num >= final_nt_den * nt_lb_band,
                ),
            ),
        )

        nci_does_not_degrade = z3.And(
            z3.Implies(
                init_nci < nci_lb,
                final_nci_num >= final_nci_den * init_nci,
            ),
            z3.Implies(
                init_nci > nci_ub,
                final_nci_num <= final_nci_den * init_nci,
            )
        )

        nt_does_not_degrade = z3.And(
            z3.Implies(
                init_nt < nt_lb,
                final_nt_num >= final_nt_den * init_nt,
            ),
            z3.Implies(
                init_nt > nt_ub,
                final_nt_num <= final_nt_den * init_nt,
            )
        )

        self.nt_does_not_degrade = nt_does_not_degrade
        self.nci_does_not_degrade = nci_does_not_degrade
        self.bad_nt_improves = bad_nt_improves
        self.bad_nci_improves = bad_nci_improves
        self.final_nt_converged = final_nt_converged
        self.final_nci_converged = final_nci_converged
        self.init_nt_converged = init_nt_converged
        self.init_nci_converged = init_nci_converged

        lemma_movement = z3.Implies(
            z3.Not(z3.And(init_nci_converged, init_nt_converged)),
            z3.Or(
                z3.And(final_nci_converged, final_nt_converged),
                z3.And(
                    z3.Or(bad_nci_improves, bad_nt_improves),
                    nci_does_not_degrade,
                    nt_does_not_degrade,
                ),
            ),
        )
        assert isinstance(lemma_movement, z3.BoolRef)

        assumptions = z3.And(
            True,
            zv.theta * N_G > zv.R,
            zv.theta == 1,
            zv.gamma_R <= ParameterChoices.gammaR_ub,
        )
        assert isinstance(assumptions, z3.BoolRef)

        self.lemma_top = lemma_top
        self.lemma_bottom = lemma_bottom
        self.lemma_steady1 = lemma_steady1
        self.lemma_steady2 = lemma_steady2
        self.lemma_movement = lemma_movement
        self.assumptions = assumptions

        self.zv = zv
        self.sv = sv
        self.final_nt_num = final_nt_num
        self.final_nt_den = final_nt_den
        self.final_nci_num = final_nci_num
        self.final_nci_den = final_nci_den

    def check_lemmas(self):
        zv, sv = self.zv, self.sv
        final_nt_num, final_nt_den = self.final_nt_num, self.final_nt_den
        final_nci_num, final_nci_den = self.final_nci_num, self.final_nci_den

        def this_check_lemma(lemma, assumptions, self=self, expect_sat=False):
            s = setup_solver(zv)
            m = check_lemma(s, lemma, assumptions, zv, sv)
            if not expect_sat and m is not None:
                print("z3_nt_num = ", m.eval(final_nt_num).as_decimal(PRECISION))  # type: ignore
                print("z3_nt_den = ", m.eval(final_nt_den).as_decimal(PRECISION))  # type: ignore
                print("z3_nt = ", m.eval(final_nt_num / final_nt_den).as_decimal(PRECISION))  # type: ignore
                print("z3_nci_num = ", m.eval(final_nci_num).as_decimal(PRECISION))  # type: ignore
                print("z3_nci_den = ", m.eval(final_nci_den).as_decimal(PRECISION))   # type: ignore
                print("z3_nci = ", m.eval(final_nci_num / final_nci_den).as_decimal(PRECISION))  # type: ignore

                import ipdb; ipdb.set_trace()
            # True if lemma is true, False otherwise
            return m is None

        print("-"*80)
        print("Checking if model is feasible. This should be SAT.")
        ret = this_check_lemma(False, z3.And(self.assumptions), expect_sat=True)
        assert ret is False

        print("-"*80)
        print("Checking if there are counteexamples to lemmas. These should be UNSAT.")
        print("Checking top (Lemma B.7 part 1)")
        this_check_lemma(self.lemma_top, self.assumptions)
        print("Checking bottom (Lemma B.7 part 2)")
        this_check_lemma(self.lemma_bottom, self.assumptions)
        print("Checking steady state (Lemma B.6 part 1)")
        this_check_lemma(self.lemma_steady1, self.assumptions)
        print("Checking steady state (Lemma B.6 part 2)")
        this_check_lemma(self.lemma_steady2, self.assumptions)
        print("Checking movement (Lemma B.8)")
        this_check_lemma(self.lemma_movement, self.assumptions)


    def binary_search_performance_bounds(self):
        # TODO: Can we use binary search style process to build analytical
        # relation between design param choices and objectives?

        # TODO: Use binary search to find best parameters given objective
        # bounds.

        def optimize_metric(metric: Metric, s: z3.Solver):
            others = self.metrics.copy()
            others.remove(metric)
            fix_metrics(s, others)

            print("Optimizing ", metric.name())
            ret = optimize_var_nopushpop(
                s, metric.z3ExprRef, metric.lo, metric.hi, metric.eps, metric.maximize
            )
            print(f"Optimized {metric.name()} to {ret}")

        # Optimize the metrics one by one
        metric_dict = {metric.name(): metric for metric in self.metrics}

        s = setup_solver(self.zv)
        s.add(self.assumptions)
        s.add(z3.Not(self.lemma_steady2))
        # s.add(z3.Not(self.lemma_top))
        optimize_metric(metric_dict["nt_ub"], s)

        s = setup_solver(self.zv)
        s.add(self.assumptions)
        s.add(z3.Not(self.lemma_steady2))
        # s.add(z3.Not(self.lemma_bottom))
        optimize_metric(metric_dict["nt_lb"], s)

        s = setup_solver(self.zv)
        s.add(self.assumptions)
        s.add(z3.Not(self.lemma_steady1))
        optimize_metric(metric_dict["nci_ub"], s)

        s = setup_solver(self.zv)
        s.add(self.assumptions)
        s.add(z3.Not(self.lemma_movement))
        optimize_metric(metric_dict["speed"], s)


# %%
if __name__ == "__main__":
    print_expressions()

    lemmas = Lemmas()
    lemmas.check_lemmas()

    # Uncomment these to run binary search to find tight performance bounds.
    # lemmas = Lemmas(symbolic_bounds=True)
    # lemmas.binary_search_performance_bounds()
