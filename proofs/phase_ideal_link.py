# %%
import os
from typing import Callable, Literal, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from analytical_ideal_link import (
    N_C,
    N_T,
    R,
    alpha,
    full_overlap,
    partial_overlap,
    partial_overlap_other,
    sequential,
    synchronous,
    theta,
)
from matplotlib.axes import Axes
from plot_config_light import colors, get_fig_size_paper, get_fig_size_ppt, get_style

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


# %%
# Phase plots for FRCC on ideal link.

class Params:
    alpha = 1
    theta = 0.2  # 10 ms
    R = 1  # 50 ms
    # In the analysis, these are unitless quantities. Our default setup uses
    # R=50ms and theta=10ms, so we use a ratio of 5:1 in setting these.


@mpl.rc_context(rc=style)
def plot_phase_portrait(
    p: Params,
    next_n_c_func: Callable,
    next_n_t_func: Callable,
    type: Literal["quiver", "stream"],
    opath: str,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
):
    # For quiver, the magnitude of movement overshadows when N_T_1 is close to 0,
    # and we only get vertical up lines. So we restrict the range of N_T_1.
    # Ideally, we should change the scale of the arrows. E.g., show relative
    # change in N_T_1 and N_C_1. Naively doing this does not work, so sticking with stream plot.

    n_points = 50
    n_c_1 = np.linspace(1+1e-12, 4, n_points)
    n_t = np.linspace(0.5, 4, n_points)
    N_C_1, N_T_1 = np.meshgrid(n_c_1, n_t)

    N_C_1_next = next_n_c_func(p, N_C_1, N_T_1)
    N_T_1_next = next_n_t_func(p, N_C_1, N_T_1)

    N_C_1_dot = (N_C_1_next - N_C_1)
    N_T_1_dot = (N_T_1_next - N_T_1)

    fixed_points = np.array(
        [
            [2.0, 2.0],
        ]
    )

    fig = None
    if ax is None:
        figsize = get_fig_size(0.6, 0.6, False)
        fig, ax = plt.subplots(figsize=figsize)
    assert isinstance(ax, Axes)

    # See curved quiver and manual streamplot
    # https://stackoverflow.com/questions/51843313/flow-visualisation-in-python-using-curved-path-following-vectors/51990140#51990140
    if type == "quiver":
        ax.quiver(N_C_1, N_T_1, N_C_1_dot, N_T_1_dot)
    elif type == "stream":
        ax.streamplot(
            N_C_1,
            N_T_1,
            N_C_1_dot,
            N_T_1_dot,
            arrowstyle="->",
            linewidth=style["lines.linewidth"] * 0.75,
            density=1.2,
            arrowsize=style["lines.linewidth"] * 0.75,
        )
    else:
        raise ValueError("Invalid type")

    ax.scatter(
        *fixed_points.T,
        marker="o",
        color=colors[2],
        s=style["lines.markersize"] * 10 / 3,
        linewidth=style["lines.linewidth"],
    )
    ax.set_ylabel("$N_T$")
    ax.set_xlabel("$N_{C_i}$")
    if title is not None:
        ax.set_title(title)

    if fig is not None:
        fig.set_layout_engine('tight', pad=0.03)
        fig.savefig(opath)
        plt.close(fig)

# %%
@mpl.rc_context(rc=style)
def main():
    os.makedirs(outdir, exist_ok=True)
    p = Params()
    title = {
        'synchronous': '(a) Synchronous',
        'sequential': '(b) Sequential',
        'full_overlap': '(c) Full overlap',
        'partial_overlap': '(d) Partial overlap (1)',
        'partial_overlap_other': '(e) Partial overlap (2)',
    }
    ts = {
        'synchronous': synchronous,
        'sequential': sequential,
        'full_overlap': full_overlap,
        'partial_overlap': partial_overlap,
        'partial_overlap_other': partial_overlap_other,
    }

    figsize1 = get_fig_size(0.6, 0.6, False)
    figsize2 = get_fig_size(1, 1, True)
    figsize = (figsize2[0], figsize1[1])
    fig, axes = plt.subplots(1, 5, figsize=figsize, sharex=True, sharey=True)
    i = 0

    for k, v in ts.items():
        next_N_k, next_N_T_1 = v()

        def substitute_all(p: Params, expr):
            if isinstance(alpha, sp.Symbol):
                expr = expr.subs(alpha, p.alpha)
            expr = expr.subs(theta, p.theta)
            expr = expr.subs(R, p.R)
            expr = expr.subs(N_C[0], "this_N_C_1")
            expr = expr.subs(N_T, "this_N_T_1")
            return expr

        def N_C_1_next(p: Params, this_N_C_1, this_N_T_1):
            expr = substitute_all(p, next_N_k[0])
            # import ipdb; ipdb.set_trace()
            return eval(str(expr))

        def N_T_1_next(p: Params, this_N_C_1, this_N_T_1):
            expr = substitute_all(p, next_N_T_1)
            # print(expr)
            return eval(str(expr))


        # Individual plots
        plot_phase_portrait(
            p, N_C_1_next, N_T_1_next, "stream", f"{outdir}/ideal_phase_stream_{k}.{ext}", title=title[k]
        )

        # Combined plot
        plot_phase_portrait(
            p, N_C_1_next, N_T_1_next, "stream", f"{outdir}/ideal_phase_stream_{k}.{ext}", axes[i], title=title[k]
        )

        axes[i].label_outer()
        i += 1

    fig.set_layout_engine('tight', pad=0.03)
    fig.savefig(f"{outdir}/ideal_phase_portrait_all.{ext}")


if __name__ == "__main__":
    main()
