#!/bin/bash
#SBATCH --job-name=Ivans_Benchmarking__Reconstruction
#SBATCH --array=0-999
#SBATCH --time=100-00:00:00 # days-hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4096 # 4GB
#SBATCH --cpus-per-task=1
#SBATCH --open-mode=append
#SBATCH --output=logs/1-reconstruct_new/job%A.stdout
#SBATCH --error=logs/1-reconstruct_new/job%A.stderr

# Submit with: sbatch 1-reconstruct_new.sbatch.sh <ARRAY_OFFSET> && watch -d squeue

source ~/.bashrc
conda activate cass

if [ $# -eq 0 ]; 
then
    ARRAY_OFFSET=0
else
    ARRAY_OFFSET=$1
fi

NEW_ARRAY_ID=$(($SLURM_ARRAY_TASK_ID + $ARRAY_OFFSET))

echo "Running t: $NEW_ARRAY_ID  |  on $(date)"

python 1-reconstruct_new.py -t $NEW_ARRAY_ID --cached
