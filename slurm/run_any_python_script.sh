#!/bin/bash
#SBATCH --job-name=benjamin_train       # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --gres=gpu:1                    # Number of GPU per node
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --time=12:00:00                 # Time limit hrs:min:sec
#SBATCH --output="/mnt/nfs/homedirs/%u/slurm-output/slurm-%j.out"  # Stdout and stderr log

# Description: This script runs any specified Python script. It should be submitted via SLURM.

home_dir="/mnt/nfs/homedirs/$USER"
cd ${SLURM_SUBMIT_DIR} || exit 1
echo "Starting job ${SLURM_JOBID}"
echo "SLURM assigned me these nodes:"
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

# Activate your conda environment
MY_CONDA_ENV="project"
source "$home_dir/miniconda3/bin/activate" "$MY_CONDA_ENV"
echo "Environment activated: $MY_CONDA_ENV"

# Path to the Python executable in that environment
python_path="$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python"

# Check for the Python script argument
if [ -z "$1" ]; then
    echo "[ERROR] No Python script specified. Usage: sbatch run_any_python_script.sh <path_to_script.py> [additional args]"
    exit 1
else
    SCRIPT_PATH="$1"
    shift  # shift past the script name, so "$@" can hold additional arguments
    echo "[INFO] Running Python script: $SCRIPT_PATH"
fi

# Finally, run the specified Python script (with any additional arguments passed)
"$python_path" "$SCRIPT_PATH" "$@"