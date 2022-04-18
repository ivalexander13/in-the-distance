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

numcells=2000

python 2-score_all_trees.py $numcells ./data/scores/nj_iwhd_oracle --focus_algorithm nj_iwhd_oracle --threads 20 --verbose 

