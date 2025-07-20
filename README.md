# FRCC

FRCC is a congestion control algorithm designed to *provably* guarantee bounds
on fairness on networks with jitter.

This repository accompanies the paper: "Towards Fair and Robust Congestion
Control (to appear in NSDI 26)", and includes FRCC's kernel implementation,
benchmarking, and proofs.

# Getting started
Ensure that you cloned all the submodules.
```bash
git submodule update --init --recursive
```

## Dependencies

```bash
conda create -yn frcc python=3
conda activate frcc
conda install numpy matplotlib pandas sympy
pip install z3-solver  # For verifying the proofs
```

## Compiling and installing FRCC's kernel module
```bash
cd frcc_kernel
make
sudo insmod tcp_frcc.ko
```

## Setting up test bench
1. Refer to `experiments/cc_bench/setup.sh` for installing mahimahi, iperf3, etc. used for running experiments.
2. Refer to `experiments/cc_bench/boot.sh` for setting up kernel parameters (e.g., TCP buffers). This needs to be run after every boot.

## Hello world experiment
Run FRCC on a simple dumbbell topology to check if all the dependencies are correctly installed.
```bash
cd experiments/cc_bench
python sweep.py -t debug -o ../data/logs/frcc-nsdi26/  # Run simple 60s experiment
python parse_pcap.py -i ../data/logs/frcc-nsdi26/debug  # Plot throughput and rtt
# Output logs will be in experiments/data/logs/frcc-nsdi26/debug
# Output plots will be in experiments/data/figs/frcc-nsdi26/debug
```

# Reproducing results

## Empirical experiments

## Proofs
`proofs/` contains code to generate the state update equations, verify the lemmas, and plot the phase portraits of FRCC.

1. Ideal link, N flows, equal RTprop, no probe collisions.
`proofs/analytical_ideal_link.py` derives and prints the state transition equations used in Appendix B.3.
2. Jittery link, **2 flows**, equal RTprop, no probe collisions.
`proofs/phase_jittery_link.py` uses Z3 to prove the lemmas describing the state trajectories.
3. Ideal link, **2 flows**, equal RTprop, **with collisions**.
`proofs/phase_ideal_link.py` derives state update equations for different types of probe collisions and uses this to plot the phase portrait (Figure 14).
4. Fluid model, different RTprops and multiple bottlenecks.
`proofs/fluid_different_rtt.py` and `proofs/fluid_parking_lot.py` derive, print and plot the steady-state fixed-point of FRCC (Figure 18).

Run these files as:
```bash
cd proofs
python phase_ideal_link.py
```

These will output figures in the `proofs/outputs/` directory.
