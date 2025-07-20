import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from fluid_parking_lot import binary_search
from IPython.display import display
from plot_config_light import get_fig_size_paper, get_fig_size_ppt, get_style

sp.init_printing(use_unicode=True)

outdir = "outputs/"
ppt = True
ppt = False
style = get_style(use_markers=False, paper=True, use_tex=False)  # paper
get_fig_size = get_fig_size_paper
ext = "pdf"
if ppt:
    style = get_style(use_markers=False, paper=False, use_tex=False)  # ppt
    get_fig_size = get_fig_size_ppt
    ext = "svg"


class DifferentRTTModel:
    def __init__(self, probe0: bool):
        # Define symbols (see parking lot file for meanings and conventions of symbols)
        N_G = 2
        C = sp.symbols("C", domain=sp.S.Reals, positive=True)
        D = sp.symbols("D", domain=sp.S.Reals, positive=True)
        f0 = sp.symbols("f_0", domain=sp.S.Reals, positive=True)
        rtts = [sp.Symbol(f"R_{i}", domain=sp.S.Reals, positive=True) for i in range(N_G)]
        excess_fraction = sp.symbols("f_X", domain=sp.S.Reals, positive=True)
        excess_delay = sp.symbols("d_H", domain=sp.S.Reals, positive=True)

        # Compute Rates and Excess
        rate0 = C * f0
        rate1 = C - rate0
        rates = [rate0, rate1]

        # Compute cwnds and new RTTs
        cwnds = [rate * rtt for rate, rtt in zip(rates, rtts)]
        cwnds_after = cwnds.copy()
        if probe0:
            excess = rate0 * D * excess_fraction
            cwnds_after[0] = cwnds[0] + excess
            probes_frac = f0
        else:
            excess = rate1 * D * excess_fraction
            cwnds_after[1] = cwnds[1] + excess
            probes_frac = 1 - f0
        rtts_after = [rtt + excess_delay for rtt in rtts]

        # Solve for Excess Delay
        balance_after = -C + sum(cwnd / rtt for cwnd, rtt in zip(cwnds_after, rtts_after))
        display("Balance equation:", balance_after)

        ret = sp.solve(balance_after, excess_delay)
        excess_delay_solution = sp.simplify(ret[1])
        # excess_delay_solution = sp.factor(excess_delay_solution, deep=True)
        C_est = excess / excess_delay_solution
        ratio_C = sp.simplify(C_est / C)
        # ratio_C = sp.factor(ratio_C, deep=True)
        fc_est = C_est / (C * probes_frac)

        display("Excess delay created by probe:", excess_delay_solution)
        display("Bias in bandwidth estimate:", ratio_C)
        display("Bias in flow count estimate:", fc_est)

        self.N_G = N_G
        self.C = C
        self.D = D
        self.f0 = f0
        self.rtts = rtts
        self.excess_fraction = excess_fraction
        self.excess_delay = excess_delay
        self.excess_delay_solution = excess_delay_solution
        self.C_est = C_est
        self.ratio_C = ratio_C
        self.fc_est = fc_est
        self.probes_frac = probes_frac


def evaluate(m: DifferentRTTModel, v_D: float, v_f0: float, v_rtts: List[float], v_excess_fraction: float):
    """Evaluates excess delay and ratio_C for given parameter values."""
    D = m.D
    f0 = m.f0
    rtts = m.rtts
    excess_fraction = m.excess_fraction
    excess_delay_solution = m.excess_delay_solution
    ratio_C = m.ratio_C

    subs = {
        D: v_D,
        f0: v_f0,
        rtts[0]: v_rtts[0],
        rtts[1]: v_rtts[1],
        excess_fraction: v_excess_fraction,
    }
    # print(subs)
    # print(excess_delay_solution)
    v_excess_delay_solution = sp.simplify(excess_delay_solution.subs(subs))
    v_ratio_C = sp.simplify(ratio_C.subs(subs))
    v_fc_est = sp.simplify(m.fc_est.subs(subs))
    return (
        sp.factor(v_excess_delay_solution, deep=True),
        sp.factor(v_ratio_C, deep=True),
        sp.factor(v_fc_est, deep=True),
    )


def get_fixed_point(
    eqn: sp.Expr,
    m: DifferentRTTModel,
    v_D: float,
    v_rtts: List[float],
    v_excess_fraction: float,
    v_C: float = 1,
):
    subs = eqn.subs(
        {
            m.D: v_D,
            m.rtts[0]: v_rtts[0],
            m.rtts[1]: v_rtts[1],
            m.excess_fraction: v_excess_fraction,
            m.C: v_C,
        }
    )
    # display(sp.simplify(subs))
    initial_guess = 0.5
    # ret = sp.solve(subs, m.f0)
    ret = sp.nsolve(subs, m.f0, initial_guess)
    return ret


# %%
# Symobolic analysis
print("Consequences when flow 0 probes")
m0 = DifferentRTTModel(probe0=True)
print("Consequences when flow 1 probes")
m1 = DifferentRTTModel(probe0=False)
# display(evaluate(m0, m0.D, m0.f0, m0.rtts, m0.excess_fraction))
# display(evaluate(m1, m1.D, m1.f0, m1.rtts, m1.excess_fraction))

# %%
# Numeric analysis
v_D = 200  # ms
v_rtts = [200, 2000]
v_excess_fraction = 10
v_f0 = 0.5
v_C = 1

# display(evaluate(m0, v_D, v_f0, v_rtts, v_excess_fraction))  # flow 0 probes
# display(evaluate(m1, v_D, v_f0, v_rtts, v_excess_fraction))  # flow 1 probes

# %%
print("-"*80)
print("Symbolic values for flow count estimates as a function of initial RTTs.")
fc_est0 = evaluate(m0, m0.D, m0.f0, m0.rtts, m0.excess_fraction)[2]
display("N_C_0 = ", fc_est0)
fc_est1 = evaluate(m1, m1.D, m1.f0, m1.rtts, m1.excess_fraction)[2]
display("N_C_1 = ", fc_est1)

# When do the flow count ests ratio equal the target flow count ratio (=1)
fixed_point_eqn = fc_est0 - fc_est1
display(sp.simplify(fixed_point_eqn))
# This has many solutions, sp only gives us a few imaginary roots.
# sp.solve(fixed_point_eqn, m0.f0)
get_fixed_point(fixed_point_eqn, m0, v_D, v_rtts, v_excess_fraction, v_C)
# get_fixed_point(fixed_point_eqn, m0, v_D, [200, 400], v_excess_fraction, v_C)

# %%
# # Taking limit D -> 0 gives vacuous results.
# fixed_point_ratio_eqn = fc_est0/fc_est1
# display(sp.simplify(fixed_point_ratio_eqn))
# ret = sp.limit(fixed_point_ratio_eqn, m0.D, 0)
# display(ret)
# # sp.solve(fixed_point_ratio_eqn - 1, m0.f0)

# display(sp.limit(fc_est0, m0.D, 0))
# display(sp.factor(fc_est0, deep=True))
# display(sp.expand((m0.D * m0.f0 * m0.excess_fraction - m0.rtts[0] * (1-m0.f0) - m0.rtts[1] * m0.f0)**2))

# %%
# Exploration
# display(
#     sp.simplify(
#         fixed_point_ratio_eqn.subs(
#             {m0.D: 200, m0.rtts[0]: 2000, m1.rtts[1]: 200, m0.excess_fraction: 1e-5}
#         )
#     )
# )
get_fixed_point(fixed_point_eqn, m0, v_D, [200, 400], 1e-10, v_C)
get_fixed_point(fixed_point_eqn, m0, v_D, [200, 600], 1e-10, v_C)
# worse case seems to be 1/(rtt_ratio + 1), when the ratio is small.

# %%
# Exploration for simulation

# v_D = 10  # ms
# v_rtts = [10, 100]
# v_excess_fraction = 2 ** -3
# v_excess_fraction = 4
# v_C = 10
#
# v_f0 = get_fixed_point(fixed_point_eqn, m0, v_D, v_rtts, v_excess_fraction, v_C)
# print(v_f0)
# m0.ratio_C.subs({
#     m0.D: v_D,
#     m0.rtts[0]: v_rtts[0],
#     m0.rtts[1]: v_rtts[1],
#     m0.excess_fraction: v_excess_fraction,
#     m0.f0: v_f0,
# })
# m1.ratio_C.subs({
#     m1.D: v_D,
#     m1.rtts[0]: v_rtts[0],
#     m1.rtts[1]: v_rtts[1],
#     m1.excess_fraction: v_excess_fraction,
#     m1.f0: v_f0,
# })

# %%
# Plotting fixed point for different parameter choices
fixed_point_subs = fixed_point_eqn.subs(
    {
        m0.D: v_D,
        m0.rtts[0]: v_rtts[0],
        m0.C: v_C,
    }
)

print("-"*80)
print("Data points for the fixed points with different RTprops")

plt.style.use(style)
figsize = get_fig_size(0.49, 0.49)
fig, ax = plt.subplots(figsize=figsize)

for this_v_excess_fraction_exp in range(-7, 9):
    this_v_excess_fraction = 2 ** this_v_excess_fraction_exp
    records = []
    for this_rtt_ratio_exp in range(1, 9):
        this_rtt_ratio = 2 ** this_rtt_ratio_exp
        try:
            # fp = get_fixed_point(fixed_point_eqn, m0, v_D, v_rtts, this_v_excess_fraction, this_v_hops, v_C)

            # Using binary search because sp many times finds the imaginary root which we do not want.
            this_fixed_point_subs = fixed_point_subs.subs(
                {m0.excess_fraction: this_v_excess_fraction, m0.rtts[1]: this_rtt_ratio * v_rtts[0]}
            )
            fp = binary_search(
                this_fixed_point_subs,
                m0.f0,
                1e-10,
                0.6,
                1e-10,
            )
            fc_est = evaluate(m0, v_D, fp, [v_rtts[0], v_rtts[0]*this_rtt_ratio], this_v_excess_fraction)[2]
            records.append({"rtt_ratio": this_rtt_ratio, "f0": fp, "tput_ratio": (1-fp) / fp, "fc_est": fc_est})
        except ValueError:
            continue
    df = pd.DataFrame(records)

    display(df)
    ax.plot(df["rtt_ratio"], df["tput_ratio"], label=f"_$\\gamma={this_v_excess_fraction}$")

rtt_ratios = np.array([2**x for x in range(1, 9)])
ax.plot(rtt_ratios, rtt_ratios, label="RTT ratio", color='black', ls='dashed')
ax.plot(rtt_ratios, rtt_ratios*0 + 1, label="1", color='black', ls='solid')
ax.legend()
ax.set_xlabel("RTT ratio")
ax.set_ylabel("Tput ratio (long/short)")
ax.yaxis.set_label_coords(-0.2, 0.3)
# ax.set_ylim(bottom=1)
# ax.grid(True)
# ax.minorticks_on()
# ax.set_xscle('log', base=2)
# ax.set_yscale('log')
fig.set_layout_engine('tight', pad=0.03)
os.makedirs(outdir, exist_ok=True)
fig.savefig(f"{outdir}/different_rtt_fixed_point.pdf")
plt.close(fig)
