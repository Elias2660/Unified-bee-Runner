#!/bin/bash

echo "Submitting slurm job. Remember to edit the Unifier_Run.sh (especially if you want to change input/output) if you want to make changes."

sbatch \
  -n 8 \
  -c 10 -G 1 --mem=600G \
  --time=28800 \
  -o "/dev/null" \
  Unified-bee-Runner/Unifier_Run.sh

echo "Slurm Submitted"
