# FRCC

FRCC is a congestion control algorithm designed to *provably* guarantee bounds
on fairness on networks with jitter.

This repository accompanies the paper: "Towards Fair and Robust Congestion
Control (to appear in NSDI 26)", and includes FRCC's kernel implementation,
benchmarking, and proofs.

This artifact is split across 3 repositories:
1. https://github.com/108anup/frcc/tree/main. This includes the other two
   repositories as submodules and has proofs about FRCC's performance.
2. https://github.com/108anup/frcc_kernel/tree/main. This has the code for
   FRCC's kernel module.
3. https://github.com/108anup/cc_bench/tree/main. This has scripts for running
   and plotting experiments.

## Structure (most relevant files/directories)

```text
.
├── frcc_kernel # (kernel module)
│   ├── Makefile
│   ├── set_frcc_params.py # (changing parameters and features at runtime)
│   ├── tcp_frcc.c
├── experiments
│   ├── ccas # (implementations of other CCAs, e.g., Copa)
│   │   └── genericCC
│   ├── cc_bench # (benchmarking framework)
│   │   ├── sweep.py # (main entry point)
│   │   └── parse_pcap.py
│   │       # (parse and plot pcap traces logged by tcpdump for the experiment runs)
│   ├── data # (organization of logs and figures)
│   │   ├── figs
│   │   └── logs
│   └── mahimahi-traces # (traces for emulation)
├── proofs # (proofs and analysis of FRCC's performance properties)
│   ├── analytical_ideal_link.py
│   ├── fluid_different_rtt.py
│   ├── fluid_parking_lot.py
│   ├── phase_ideal_link.py
│   └── phase_jittery_link.py
└── README.md
```

## Getting started

Ensure that you cloned all the submodules. After cloning, you can either follow
instructions here or refer to `setup.sh`.

```bash
sudo apt update
sudo apt install -y build-essential git openssh-server

mkdir -p $HOME/Projects
cd $HOME/Projects
git clone https://github.com/108anup/frcc.git
git submodule update --init --recursive
```

### Dependencies

We manage dependencies using conda.

```bash
conda create -yn frcc python=3
conda activate frcc
conda install -y numpy matplotlib pandas sympy pip ipython
conda install ipdb -c conda-forge
pip install z3-solver  # For verifying the proofs
```

### Compiling and installing FRCC's kernel module

```bash
cd frcc_kernel
make -j
sudo insmod tcp_frcc.ko
```

### Setting up test bench

1. Refer to `experiments/cc_bench/setup.sh` for installing mahimahi, iperf3,
   etc. used for running experiments. Note, you may want to deactivate conda
environment, as the environment's protobuf may interfere with mahimahi and
genericCC protobuf. Reactivate after running the script.
2. Refer to `experiments/cc_bench/boot.sh` for setting up kernel parameters
   (e.g., TCP buffers). This needs to be run after every boot.

## Hello world experiment

If you find yourself killing an experiment in between (e.g., Ctrl+C on a script
run), run `experiments/cc_bench/clean_processes.sh` to kill all the test bench
processes.

1. Run FRCC on a simple dumbbell topology to check if all the dependencies are
   correctly installed. Runs for roughly a minute.

    ```bash
    cd experiments/cc_bench
    # Run simple 60s experiment
    python sweep.py -t debug -o ../data/logs/frcc-nsdi26/
    # Plot throughput and rtt
    python parse_pcap.py -i ../data/logs/frcc-nsdi26/debug
    # Output logs will be in experiments/data/logs/frcc-nsdi26/debug
    # Output plots will be in experiments/data/figs/frcc-nsdi26/debug
    ```

2. Check if all the congestion control algorithms (CCAs) are working fine. Runs
   5 CCAs for 1 min each (total 5 mins).

    ```bash
    cd experiments/cc_bench
    python sweep.py -t debug_all -o ../data/logs/frcc-nsdi26/
    python parse_pcap.py -i ../data/logs/frcc-nsdi26/debug_all
    # Output logs will be in experiments/data/logs/frcc-nsdi26/debug_all
    # Output plots will be in experiments/data/figs/frcc-nsdi26/debug_all
    ```

3. Check if parallel experiments run fine. Same as previous but executes up to
   N_PROCESSES=10 (in `experiments/cc_bench/sweep.py`) runs in parallel by
default. Edit `sweep.py` to change the number of parallel runs based on the
number of physical cores. For reference, we set 10 cores here when our machine
has 16 physical cores to ensure limited contention.

    ```bash
    python sweep.py -t debug_all -o ../data/logs/frcc-nsdi26/ -p
    python parse_pcap.py -i ../data/logs/frcc-nsdi26/debug_all
    ```

## Reproducing results

### Empirical experiments

Note, following instructions do not produce Figure 22 (parking lot experiment)
and runs for BBRv3, as we have separate VMs for these experiments.

1. Push button run of the entire suite (covers all figures). We recommend
   running a single experiment first (see below) to ensure everything is
working fine. We typically leave this running overnight.

    ```bash
    cd experiments/cc_bench

    python sweep.py -t sweeps -o ../data/logs/frcc-nsdi26 -p
    # -p executes runs in parallel.
    # You can edit `sweep.py` to change the number of parallel runs based on
    # the number of cores available. For reference, we use 10 cores when machine
    # has 16 physical cores to ensure limited contention.

    ./plot_all.sh
    # This will parse all the logs and copy all the relevant figures to
    # `experiments/data/figs/frcc-nsdi26/evaluation`. See plot.sh for mapping
    # between the pdf files and figures in paper.
    ```

2. Experiments for individual figures:

    Note, a single run of a congestion control algorithm (CCA) on a single
    scenario (e.g., choice of link capacity, number of flows, etc.) takes 5
    mins. An experiment like sweep_flows runs 5 CCAs and varies flow count from
    1 to 8, for a total of 40 runs. When executing 10 runs in parallel, this
    experiment would take about 20 mins.

    ```bash
    cd experiments/cc_bench

    # Ideal link sweeps (Figure 19)

    ## Sweep flow count
    python sweep.py -t sweep_flows -o ../data/logs/frcc-nsdi26/ -p
    python parse_pcap.py -i ../data/logs/frcc-nsdi26/n_flows --agg n_flows
    ## Output figures: `experiments/data/figs/frcc-nsdi26/sweep_flows/{jfi, rtt}.pdf`

    # Sweep bandwidth
    python sweep.py -t sweep_bw -o ../data/logs/frcc-nsdi26/ -p
    python parse_pcap.py -i ../data/logs/frcc-nsdi26/sweep_bw --agg bw_mbps
    ## Output figures: `experiments/data/figs/frcc-nsdi26/sweep_bw/{jfi, rtt}.pdf`

    # Sweep RTprop
    python sweep.py -t sweep_rtprop -o ../data/logs/frcc-nsdi26/ -p
    python parse_pcap.py -i ../data/logs/frcc-nsdi26/sweep_rtprop --agg rtprop_ms
    ## Output figures: `experiments/data/figs/frcc-nsdi26/sweep_rtprop/{jfi, rtt}.pdf`

    # Sweeps with jitter. Figure 20 (left and right respectively)
    python sweep.py -t different_rtt_sweep_bw -o ../data/logs/frcc-nsdi26/ -p
    python parse_pcap.py -i ../data/logs/frcc-nsdi26/different_rtt_sweep_bw --agg bw_mbps
    ## Output figure (left): `experiments/data/figs/frcc-nsdi26/different_rtt_sweep_bw/xput_ratio.pdf`

    python sweep.py -t different_rtt -o ../data/logs/frcc-nsdi26/ -p
    python parse_pcap.py -i ../data/logs/frcc-nsdi26/different_rtt --agg rtprop_ratio
    ## Output figure (right): `experiments/data/figs/frcc-nsdi26/different_rtt/xput_ratio.pdf`

    # Sweeps with jitter. Figure 21 (left and right respectively)
    python sweep.py -t sweep_jitter_bw -o ../data/logs/frcc-nsdi26/ -p
    python parse_pcap.py -i ../data/logs/frcc-nsdi26/sweep_jitter_bw --agg bw_mbps
    ## Output figure (left): `experiments/data/figs/frcc-nsdi26/sweep_jitter_bw/xput_ratio.pdf`

    python sweep.py -t sweep_jitter -o ../data/logs/frcc-nsdi26/ -p
    python parse_pcap.py -i ../data/logs/frcc-nsdi26/sweep_jitter --agg jitter_ms
    ## Output figure (right): `experiments/data/figs/frcc-nsdi26/sweep_jitter/xput_ratio.pdf`

    # Figure 1, 2, 26, 27
    ## The previous sweeps produce the logs for these figures.
    ## See or run plot.sh to aggregate and copy the figures to
    ## `experiments/data/figs/frcc-nsdi26/evaluation/timeseries`.

    # Convergence (Figure 23)
    ## This runs for 12 mins per CCA.
    python sweep.py -t staggered -o ../data/logs/frcc-nsdi26/ -p
    python parse_pcap.py -i ../data/logs/frcc-nsdi26/staggered
    ## Output figure:
    `experiments/data/figs/frcc-nsdi26/staggered/bw_ppms[8]-ow_delay_ms[25]-n_flows[8]/bw_ppms[8]-ow_delay_ms[25]-n_flows[8]-buf_size_bdp[100]-cca[frcc]/tcpdump_throughput.pdf
    ```

### Proofs

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
python analytical_ideal_link.py
python phase_ideal_link.py
python phase_jittery_link.py
python fluid_parking_lot.py
python fluid_different_rtt.py
```

These will output figures in the `proofs/outputs/` directory.
