#!/bin/bash
# Submission script for yosef partition
# Check job status with: squeue --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"
#SBATCH --job-name=Ivans_Dist_Getter
#SBATCH --array=0-399
#SBATCH --time=100-00:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
# SBATCH --nodelist=s0  # Do not use, let slurm auto-figure it out. More likely that I will mess up.
#SBATCH --mem-per-cpu=4096 # 4GB
#SBATCH --cpus-per-task=1

# SBATCH --comment= # I don't care
#
#SBATCH --open-mode=append
#SBATCH --output=logs/post_sim/post_sim.job%A.stdout
#SBATCH --error=logs/post_sim/post_sim.job%A.stderr


source ~/.bashrc
conda activate cass
cd /home/eecs/ivalexander13/datadir/in-the-distance/

NEW_ARRAY_ID=$(($SLURM_ARRAY_TASK_ID + 2000))

echo "Running with: $NEW_ARRAY_ID"

python post_sim_analytics.py -t $NEW_ARRAY_ID --cached
