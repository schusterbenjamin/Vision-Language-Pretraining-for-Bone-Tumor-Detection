#!/bin/bash

# Description: This script submits a SLURM job to run a sweep of training experiments and tails the output log in real-time.

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <sweep> [number_of_runs]"
  exit 1
fi

SWEEP=$1
COUNT=$2

# Construct the --export string
EXPORTS="ALL,SWEEP=$SWEEP"

# Append COUNT only if it's provided
if [ -n "$COUNT" ]; then
  EXPORTS+=",COUNT=$COUNT"
fi

# Submit the job and capture the output
sbatch_output=$(sbatch --export="$EXPORTS" slurm/train_sweep.sh)

# Extract the Job ID from sbatch output (e.g., "Submitted batch job 123456")
jobid=$(echo "$sbatch_output" | awk '{print $4}')

echo "Job submitted with Job ID: $jobid"

# Show the current queue
echo "Current SLURM queue:"
squeue -u "$USER"

# Wait a moment to ensure the output file is generated (optional)
sleep 2

# Tail the output log
log_file="/mnt/nfs/homedirs/benjamins/slurm-output/slurm-${jobid}.out"
echo "Tailing log file: $log_file"

# Check if the log file exists yet
if [ -f "$log_file" ]; then
  tail -f "$log_file"
else
  echo "Log file not yet available. Waiting..."
  while [ ! -f "$log_file" ]; do
    sleep 2
  done
  tail -f "$log_file"
fi