#!/bin/bash
#SBATCH --job-name=Ivans_Benchmarking__Score
#SBATCH --time=100-00:00:00 # days-hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4096 # 4GB
#SBATCH --cpus-per-task=20
#SBATCH --open-mode=append
#SBATCH --output=logs/2-score_all_trees/job%A.stdout
#SBATCH --error=logs/2-score_all_trees/job%A.stderr

source ~/.bashrc
conda activate cass

# Params
numcells=2000
alg=nj_iwhd

python 2-score_all_trees.py $numcells ./data/scores/$alg --focus_algorithm $alg --threads 20 --verbose 

