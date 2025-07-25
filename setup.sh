#!/bin/sh

set -euo pipefail

{
  sudo apt update
  sudo apt install -y openssh-server git build-essential

  mkdir -p $HOME/Projects/
  cd $HOME/Projects/
  git clone https://github.com/108anup/frcc.git
  git submodule update --init --recursive






exit 0
}
