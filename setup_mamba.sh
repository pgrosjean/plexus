#!/bin/bash
# This script uses CUDA 12.1. You can swap with CUDA 11.8.
mamba create --name plexus \
    python=3.10 \
    pytorch-cuda=12.1 \
    pytorch-lightning=2.4.0 \
    pytorch=2.3.0 cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
mamba activate plexus

pip install -e . -v