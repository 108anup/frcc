#!/bin/sh

set -euo pipefail

{
  # Install conda
  OLD_PWD=$(pwd)
  cd /tmp
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  chmod u+x ./Miniconda3-latest-Linux-x86_64.sh
  ./Miniconda3-latest-Linux-x86_64.sh -b -u
  rm ./Miniconda3-latest-Linux-x86_64.sh
  cd $OLD_PWD

  eval "$($HOME/miniconda3/bin/conda shell.zsh hook)"
  conda init --all
  conda tos accept

  # Install dependencies
  conda create -yn frcc python=3
  conda activate frcc
  conda install -y numpy matplotlib pandas sympy pip
  pip install z3-solver  # For verifying the proofs









exit 0
}
