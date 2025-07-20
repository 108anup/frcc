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

### Compiling and running
```bash
conda create -yn frcc python=3
conda activate frcc
conda install numpy matplotlib pandas sympy
pip install z3-solver  # For verifying the proofs
```

### Development
```bash
sudo apt install bear  # For generating compile_commands.json for clangd
```

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

