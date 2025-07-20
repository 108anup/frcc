# FRCC analysis and proofs

1. Ideal link, N flows, equal RTprop, no probe collisions.
`analytical_ideal_link.py` derives and prints the state transition equations used in Appendix B.3.
2. Jittery link, **2 flows**, equal RTprop, no probe collisions.
`phase_jittery_link.py` uses Z3 to prove the lemmas describing the state trajectories.
3. Ideal link, **2 flows**, equal RTprop, **with collisions**.
`phase_ideal_link.py` derives state update equations for different types of probe collisions and uses this to plot the phase portrait (Figure 14).
4. Fluid model, different RTprops and multiple bottlenecks.
`fluid_different_rtt.py` and `fluid_parking_lot.py` derive, print and plot the steady-state fixed-point of FRCC (Figure 18).

Just run these files as: `python phase_ideal_link.py`. These create any output figures in the `outputs/` directory.
