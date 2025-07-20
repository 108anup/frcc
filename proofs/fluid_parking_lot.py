import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
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


class ParkingLotModel:
    def __init__(self, probe0: bool):
        """
        Convention, flow 0 sees delays from all the hops
        """

        # Define Symbols
        N_G = 2
        C = sp.symbols("C", domain=sp.S.Reals, positive=True)
        D = sp.symbols("D", domain=sp.S.Reals, positive=True) # Probe excess = gamma * D * rate * N_T
        f0 = sp.symbols("f_0", domain=sp.S.Reals, positive=True) # Current fraction of link consumed by flow 0
        rtts = [sp.Symbol(f"R_{i}", domain=sp.S.Reals, positive=True) for i in range(N_G)]
        excess_fraction = sp.symbols("f_X", domain=sp.S.Reals, positive=True)
        # "N_T * gamma" from paper.
        # probe excess = N_T * gamma * D * rate
        #              = excess_fraction * D * rate
        hop_excess_delay = sp.symbols("d_X", domain=sp.S.Reals, positive=True)
        hops = sp.symbols("hops", domain=sp.S.Naturals, positive=True)
        # rtts[1] = rtts[0]

        # Compute Rates and Excess
        rate0 = C * f0
        rate1 = C - rate0
        rates = [rate0, rate1]

        # Compute cwnds and new RTTs
        cwnds = [rate * rtt for rate, rtt in zip(rates, rtts)]
        cwnds_after = cwnds.copy()
        if probe0:
            excess = rate0 * D * hops * excess_fraction
            # Flow 0 sees delay from all hops, so its N_T and excess_fraction is hops times larger.
            cwnds_after[0] = cwnds[0] + excess
            excess_delay = hops * hop_excess_delay
            probes_frac = f0
        else:
            excess = rate1 * D * excess_fraction
            cwnds_after[1] = cwnds[1] + excess
            excess_delay = hop_excess_delay
            probes_frac = 1 - f0
        rtts_after = [rtt + hop_excess_delay for rtt in rtts]
        rtts_after[0] += (hops-1) * hop_excess_delay

        # Solve for excess delay
        balance_after = -C + sum(cwnd / rtt for cwnd, rtt in zip(cwnds_after, rtts_after))
        display("Balance equation:", balance_after)

        ret = sp.solve(balance_after, hop_excess_delay)
        hop_excess_delay_solution = sp.simplify(ret[1])
        excess_delay_solution = excess_delay.subs(hop_excess_delay, hop_excess_delay_solution)
        # excess_delay_solution = sp.factor(excess_delay_solution, deep=True)
        C_est = excess / excess_delay_solution
        ratio_C = sp.simplify(C_est / C)
        fc_est = C_est / (C * probes_frac)
        # ratio_C = sp.factor(ratio_C, deep=True)

        display("Excess delay created by probe:", excess_delay_solution)
        display("Bias in bandwidth estimate:", ratio_C)
        display("Bias in flow count estimate:", fc_est)

        self.N_G = N_G
        self.C = C
        self.D = D
        self.f0 = f0
        self.rtts = rtts
        self.excess_fraction = excess_fraction
        self.hop_excess_delay = hop_excess_delay
        self.hops = hops
        self.excess_delay = excess_delay
        self.excess_delay_solution = excess_delay_solution
        self.hop_excess_delay_solution = hop_excess_delay_solution
        self.C_est = C_est
        self.ratio_C = ratio_C
        self.fc_est = fc_est
        self.probes_frac = probes_frac


def evaluate(m: ParkingLotModel, v_D: float, v_f0: float, v_rtts: List[float], v_excess_fraction: float, v_hops: int):
    """Evaluates excess delay and ratio_C for given parameter values."""
    D = m.D
    f0 = m.f0
    rtts = m.rtts
    excess_fraction = m.excess_fraction
    excess_delay_solution = m.excess_delay_solution
    ratio_C = m.ratio_C
    hops = m.hops

    subs = {
        D: v_D,
        f0: v_f0,
        rtts[0]: v_rtts[0],
        rtts[1]: v_rtts[1],
        excess_fraction: v_excess_fraction,
        hops: v_hops,
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


def binary_search(eqn: sp.Expr, x: sp.Symbol, lo: float, hi: float, eps: float):
    """Binary search to find the root of the equation."""
    hi_negative = eqn.subs(x, hi) < 0
    hi_value = eqn.subs(x, hi)
    lo_value = eqn.subs(x, lo)
    # print(lo_value, hi_value)
    assert hi_value * lo_value < 0
    while hi - lo > eps:
        mid = (lo + hi) / 2
        value = eqn.subs(x, mid)
        # print(mid, value)
        if (not hi_negative or value < 0) and (hi_negative or value > 0):
            hi = mid
        else:
            lo = mid
    return lo


def get_fixed_point(
    eqn: sp.Expr,
    m: ParkingLotModel,
    v_D: float,
    v_rtts: List[float],
    v_excess_fraction: float,
    v_hops: int,
    v_C: float = 1,
):
    subs = eqn.subs(
        {
            m.D: v_D,
            m.rtts[0]: v_rtts[0],
            m.rtts[1]: v_rtts[1],
            m.excess_fraction: v_excess_fraction,
            m.hops: v_hops,
            m.C: v_C,
        }
    )
    # display(sp.simplify(subs))
    initial_guess = 1./(v_hops+1)
    # ret = sp.solve(subs, m.f0)
    ret = sp.nsolve(subs, m.f0, initial_guess)
    return ret


@matplotlib.rc_context(rc=style)
def main():
    # import ipdb; ipdb.set_trace()
    # Symobolic analysis
    print("Consequences when flow 0 (sees delays from all hops) probes")
    m0 = ParkingLotModel(probe0=True)
    print("")
    print("Consequences when flow 1 (sees delays from 1 hop only) probes")
    m1 = ParkingLotModel(probe0=False)
    # display(evaluate(m0, m0.D, m0.f0, [m0.rtts[0], m0.rtts[0]], m0.excess_fraction, m0.hops))
    # display(evaluate(m1, m1.D, m1.f0, [m1.rtts[0], m1.rtts[0]], m1.excess_fraction, m1.hops))

    # Numeric analysis. Initial conditions.
    v_D = 200  # ms
    v_rtts = [200, 200]
    v_excess_fraction = 10
    v_hops = 3
    v_f0 = 0.25
    v_C = 1

    # display(evaluate(m0, v_D, v_f0, v_rtts, v_excess_fraction, v_hops))  # flow 0 probes
    # display(evaluate(m1, v_D, v_f0, v_rtts, v_excess_fraction, v_hops))  # flow 1 probes
    (_, ratio_C_0, fc_est0) = evaluate(m0, m0.D, m0.f0, [m0.rtts[0], m0.rtts[0]], m0.excess_fraction, m0.hops)
    (_, ratio_C_1, fc_est1) = evaluate(m1, m1.D, m1.f0, [m1.rtts[0], m1.rtts[0]], m1.excess_fraction, m1.hops)

    # When do the flow count ests ratio equal the target flow count ratio (=hops)
    fixed_point_eqn = fc_est0/m0.hops - fc_est1
    print("-"*80)
    display("Fixed point equation. 0 = ", sp.simplify(fixed_point_eqn))

    # # This has many solutions, sp only gives us a few imaginary roots, so we numerically compute real root.
    # # sp.solve(fixed_point_eqn, m0.f0)
    # display("Solving fixed point eqn numerically for the initial conditions.")
    # get_fixed_point(fixed_point_eqn, m0, v_D, v_rtts, v_excess_fraction, v_hops, v_C)

    # Taking limit D -> 0 gives vacuous results. The hope was that this will
    # show what happens when probes are very small sompares to RTTs.
    # # Mathematically, this analysis just means that when D = 0, convergence
    # # only happens when h = 1. We want to make D tends to 0 in the solution to
    # # f0.
    # fixed_point_ratio_eqn = fc_est0/m0.hops/fc_est1 - 1
    # ret = sp.limit(fixed_point_ratio_eqn, m0.D, 0)
    # display(ret)  # The worst case ratio is hops^2
    # # den = sp.simplify(ret.as_numer_denom()[1]/m0.rtts[0]/m0.hops)
    # # display(den)
    # # f0_solution = sp.solve(den, m0.f0)
    # # display(f0_solution)
    # display(sp.limit(fc_est0/fc_est1, m0.D, 0))
    # display(sp.limit(ratio_C_0/ratio_C_1, m0.D, 0))
    # display(sp.limit(fc_est0, m0.D, 0))

    fixed_point_subs = fixed_point_eqn.subs(
        {
            m0.D: v_D,
            m0.rtts[0]: v_rtts[0],
            m0.rtts[1]: v_rtts[1],
            m0.C: v_C,
        }
    )

    print("-"*80)
    print("Data points for the fixed points on parking lot")

    figsize = get_fig_size(0.49, 0.49)
    fig, ax = plt.subplots(figsize=figsize)

    for this_v_excess_fraction_exp in range(-7, 9):
        this_v_excess_fraction = 2 ** this_v_excess_fraction_exp
        records = []
        for this_v_hops in range(1, 9):
            # print(this_v_excess_fraction, this_v_hops)
            try:
                # fp = get_fixed_point(fixed_point_eqn, m0, v_D, v_rtts, this_v_excess_fraction, this_v_hops, v_C)

                # Using binary search because sp many times finds the imaginary root which we do not want.
                this_fixed_point_subs = fixed_point_subs.subs(
                    {m0.excess_fraction: this_v_excess_fraction, m0.hops: this_v_hops}
                )
                fp = binary_search(
                    this_fixed_point_subs,
                    m0.f0,
                    1e-10,
                    1 / (this_v_hops + 1) + 1e-10,
                    1e-10,
                )
                fc_est = evaluate(m0, v_D, fp, v_rtts, this_v_excess_fraction, this_v_hops)[2]
                records.append({"hops": this_v_hops, "f0": fp, "tput_ratio": (1-fp) / fp, "fc_est": fc_est})
            except ValueError:
                continue
        df = pd.DataFrame(records)

        display(df)
        ax.plot(df["hops"], df["tput_ratio"], label=f"_D/R={this_v_excess_fraction}")

    hops = np.array(list(range(1, 9)))
    ax.plot(hops, hops**2, label="$hops^2$", color='black', ls='dashed')
    ax.plot(hops, hops, label="hops", color='black', ls='solid')
    ax.legend()
    ax.set_xlabel("Hops")
    ax.set_ylabel("Tput ratio (short/long)")
    ax.yaxis.set_label_coords(-0.2, 0.3)
    fig.set_layout_engine('tight', pad=0.03)
    ax.grid(True)
    ax.minorticks_on()
    # ax.set_yscale('log')
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f"{outdir}/fluid_parking_lot_fixed_point.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
