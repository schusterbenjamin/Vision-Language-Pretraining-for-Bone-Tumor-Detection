#!/bin/bash
#SBATCH --job-name=benjamin_train       # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1     # Number of tasks per node
#SBATCH --gres=gpu:1            # Number of GPU per node
#SBATCH --cpus-per-task=1       # Number of CPU cores per task
#SBATCH --time=192:00:00         # Time limit hrs:min:sec
#SBATCH --output="/mnt/nfs/homedirs/%u/slurm-output/slurm-%j.out"       # Standard output and error log

# Description: This script runs a training experiment using a specified experiment configuration. It should be submitted via SLURM.
home_dir="/mnt/nfs/homedirs/$USER"
cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

# Activate your conda environment
MY_CONDA_ENV="project"
source $home_dir/miniconda3/bin/activate $MY_CONDA_ENV     
echo Environment activated

# Run the Python script
python_path=$home_dir/miniconda3/envs/$MY_CONDA_ENV/bin/python

# if experiment starts with config/experiment, remove that prefix
# That way I can simply use the path to the experiment file
if [[ "$EXPERIMENT" == configs/experiment/* ]]; then
  EXPERIMENT=${EXPERIMENT#configs/experiment/}
fi

if [ -z "$EXPERIMENT" ]; then
    echo "[ERROR] No EXPERIMENT specified. Please specify one!"
    exit -1
else
    echo "[INFO] Using EXPERIMENT: $EXPERIMENT"
fi
$python_path src/train.py experiment=$EXPERIMENT