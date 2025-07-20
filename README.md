# FRCC

FRCC is a congestion control algorithm designed to *provably* guarantee bounds
on fairness on networks with jitter.

This repository accompanies the paper: "Towards Fair and Robust Congestion
Control (to appear in NSDI 26)", and includes FRCC's kernel implementation,
benchmarking, and proofs.

# Getting started

## Dependencies
```bash
# Compiling and running
conda create -yn frcc python=3 numpy matplotlib pandas sympy
pip install z3-solver  # For verifying the proofs

# Development
sudo apt install bear  # For generating compile_commands.json for clangd
```

## Hello world experiment

# Reproducing results

## Empirical experiments

## Proofs

