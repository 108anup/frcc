#!/bin/sh

set -euo pipefail

{
  SCRIPT=$(realpath "$0")
  REPO=$(dirname "$SCRIPT")
  FRCC_KERNEL="$REPO/frcc_kernel/"
  EXPERIMENTS=$(realpath "$REPO/experiments/")
  BENCH="$EXPERIMENTS/cc_bench"

  # Install conda
  cur_dir=$(pwd)
  mkdir -p $HOME/opt
  cd $HOME/opt
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  chmod u+x ./Miniconda3-latest-Linux-x86_64.sh
  ./Miniconda3-latest-Linux-x86_64.sh -b -u
  cd $cur_dir

  eval "$($HOME/miniconda3/bin/conda shell.zsh hook)"
  conda init --all
  conda tos accept

  # Install dependencies
  conda create -yn frcc python=3
  conda activate frcc
  conda install -y numpy matplotlib pandas sympy pip
  pip install z3-solver  # For verifying the proofs

  # Install FRCC kernel module
  cd $FRCC_KERNEL
  make -j
  sudo insmod tcp_frcc.ko

  # Setup test bench
  # conda's protobuf installation can interfere with
  # genericCC and mahimahi required protobuf
  cd $BENCH
  conda deactivate
  ./setup.sh
  ./boot.sh
  conda activate frcc
  cd $REPO

exit 0
}
