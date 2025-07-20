# FRCC analysis and proofs

1. Ideal link, N flows, equal RTprop, no probe collisions.
`analytical_ideal_link.py` derives the state transition equations.
2. Jittery link, **2 flows**, equal RTprop, no probe collisions.
`phase_jittery_link.py` uses Z3 to prove the lemmas describing the state trajectories.
3. Ideal link, **2 flows**, equal RTprop, **with collisions**.
`phase_ideal_link.py` plots the phase portrait in Figure 14.
4. Fluid model, different RTprops and multiple bottlenecks.
`fluid_different_rtt.py` and `fluid_parking_lot.py` derive the steady-state fixed point of FRCC.
