#!/bin/bash

python3 main_dist.py \
  --batch_size 1024 \
  --output_dir ./test \
  --workers 16
  "$@"
