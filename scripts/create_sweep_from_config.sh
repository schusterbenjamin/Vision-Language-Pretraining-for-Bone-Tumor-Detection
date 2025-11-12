#!/bin/bash

# Description: This script creates a Weights & Biases sweep from a given configuration file
# and outputs the sweep ID along with instructions on how to run the sweep using the provided scripts.

# Check for correct usage
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <wandb_project> <sweep_config.yaml>"
  exit 1
fi

WANDB_ENTITY="benjamin-schuster"
WANDB_PROJECT="$1"
SWEEP_CONFIG="$2"

# Run the sweep and capture output
sweep_output=$(wandb sweep --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" "$SWEEP_CONFIG" 2>&1)
echo "Sweep output: $sweep_output"

# Extract the sweep ID from the output
sweep_id=$(echo "$sweep_output" | sed -n 's/.*Creating sweep with ID: //p')

sweep=$WANDB_ENTITY/$WANDB_PROJECT/$sweep_id

# Print the full sweep path
echo "Sweep created: ${sweep}"

echo ""
echo "You can now run the sweep with the following command:"
echo "sbatch --export=ALL,SWEEP=${sweep},COUNT=<COUNT> slurm/train_sweep.sh"
echo "Replace <COUNT> with the number of runs you want or leave it empty to run until stopped."
echo ""
echo "OR: You can also use the following command to submit the sweep and follow the logs:"
echo "./slurm/sweep_submit_and_follow.sh ${sweep} <COUNT>"
echo "Replace <COUNT> with the number of runs you want or leave it empty to run until stopped."
