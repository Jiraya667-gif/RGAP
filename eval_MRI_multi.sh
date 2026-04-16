#!/bin/bash
set -euo pipefail

python main.py \
  --type 2d \
  --config fastmri_knee_320_complex.yml \
  --dataset_path "/home/pc/sda1/zjc-codes/python_project/datasets/Brain_data/T2" \
  --deg MRI-multi \
  --acc_factor 8 \
  --device cuda \
  -i ./results
