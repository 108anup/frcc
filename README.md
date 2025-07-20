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

Note, following instructions do not produce Figure 22 (parking lot experiment) and runs for BBRv3, as we have separate VMs for these experiments.

1. Push button run of the entire suite (covers all figures). We recommend running a single experiment first (see below) to ensure everything is working fine.

```bash
cd experiments/cc_bench

python sweep.py -t sweeps -o ../data/logs/frcc-nsdi26 -p
# -p runs 20 experiments in parallel.
# You can edit `sweep.py` based on the number of cores available.
# We use 20 cores when machine has 32 physical cores to ensure limited contention.

./plot_all.sh
# This will parse all the logs and copy all the relevant figures to `experiments/data/figs/frcc-nsdi26/evaluation`.
# See plot.sh for mapping between the pdf files and figures in paper.
```

2. Experiments for individual figures:

```bash
cd experiments/cc_bench

# Ideal link sweeps (Figure 19)

## Sweep flow count
python sweep.py -t sweep_flows -o ../data/logs/frcc-nsdi26/sweep_flows -p
python parse_pcap.py -i ../data/logs/frcc-nsdi26/n_flows --agg n_flows
## Output figures: `experiments/data/figs/frcc-nsdi26/sweep_flows/{jfi, rtt}.pdf`

# Sweep bandwidth
python sweep.py -t sweep_bw -o ../data/logs/frcc-nsdi26/sweep_bw -p
python parse_pcap.py -i ../data/logs/frcc-nsdi26/sweep_bw --agg bw_mbps
## Output figures: `experiments/data/figs/frcc-nsdi26/sweep_bw/{jfi, rtt}.pdf`

# Sweep RTprop
python sweep.py -t sweep_rtprop -o ../data/logs/frcc-nsdi26/sweep_rtprop -p
python parse_pcap.py -i ../data/logs/frcc-nsdi26/sweep_rtprop --agg rtprop_ms
## Output figures: `experiments/data/figs/frcc-nsdi26/sweep_rtprop/{jfi, rtt}.pdf`

# Sweeps with jitter. Figure 20 (left and right respectively)
python sweep.py -t different_rtt_sweep_bw -o ../data/logs/frcc-nsdi26/different_rtt_sweep_bw -p
python parse_pcap.py -i ../data/logs/frcc-nsdi26/different_rtt_sweep_bw --agg bw_mbps
## Output figure (left): `experiments/data/figs/frcc-nsdi26/different_rtt_sweep_bw/xput_ratio.pdf`

python sweep.py -t different_rtt -o ../data/logs/frcc-nsdi26/different_rtt -p
python parse_pcap.py -i ../data/logs/frcc-nsdi26/different_rtt --agg rtprop_ratio
## Output figure (right): `experiments/data/figs/frcc-nsdi26/different_rtt/xput_ratio.pdf`

# Sweeps with jitter. Figure 21 (left and right respectively)
python sweep.py -t sweep_jitter_bw -o ../data/logs/frcc-nsdi26/sweep_jitter_bw -p
python parse_pcap.py -i ../data/logs/frcc-nsdi26/sweep_jitter_bw --agg bw_mbps
## Output figure (left): `experiments/data/figs/frcc-nsdi26/sweep_jitter_bw/xput_ratio.pdf`

python sweep.py -t sweep_jitter -o ../data/logs/frcc-nsdi26/sweep_jitter -p
python parse_pcap.py -i ../data/logs/frcc-nsdi26/sweep_jitter --agg jitter_ms
## Output figure (right): `experiments/data/figs/frcc-nsdi26/sweep_jitter/xput_ratio.pdf`

# Figure 1, 2, 26, 27
## The previous sweeps produce the logs for these figures.
## See or run plot.sh to aggregate and copy the figures to
## `experiments/data/figs/frcc-nsdi26/evaluation/timeseries`.
```

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
