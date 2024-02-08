#!/bin/bash
# Script for installation of environment

# Create environment
conda create --name env_multigml python=3.8
# Activate environment
conda activate env_multigml
# Install requirements
pip install -r requirements.txt
# Install the source repository
python3 -m pip install -e .
