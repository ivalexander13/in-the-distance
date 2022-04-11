#!/bin/bash

# *** "#PBS" lines must come before any non-blank, non-comment lines ***### Set the job name
### Job's Name
#PBS -N Cassiopeia_StressTest
### Redirect stdout and stderr by first telling Torque to redirect do /dev/null and then redirecting yourself via exec. This is the way the IT recommends.
#PBS -e localhost:/dev/null
#PBS -o localhost:/dev/null
### Set the queue to which to send
#PBS -q yosef3
### Limit the resources used
#PBS -l nodes=1:ppn=20
### Change the walltime and cpu time limit from their default (the default is currently an hour)
#PBS -l walltime=2000:00:00
#PBS -l cput=20000:00:00
### Move all your environment variables to the job
#PBS -V

### Change to the directory where the job was submitted
### Alternately, use the -d option and set $PBS_O_INITDIR

workdir=$PBS_O_WORKDIR
run=$PBS_ARRAYID

source ~/.bashrc
conda activate cass
cd $workdir

algorithms=("nj")

NCELLS="2000"
SCORE_SCRIPT="/home/eecs/ivalexander13/datadir/ivan-cassiopeia-benchmarking/solver_benchmarking_whole_regime/2-score_all_trees.py"
SCORE_BASELINE_SCRIPT="/data/yosef2/users/richardz/projects/CassiopeiaV2-Reproducibility/scripts/score_baseline_trees_no_missing.py"
algorithm=${algorithms[$run]};
output_stub="data/scores/${NCELLS}_${algorithm}"

exec 2> $workdir/_log.stderr_$algorithm > $workdir/_log.stdout_$algorithm

cd $workdir

if [ $algorithm == "baseline" ]; then
    output_stub="${output_stub}_no_missing"
    cmd="python ${SCORE_BASELINE_SCRIPT} ${NCELLS} ${output_stub} --verbose --threads 20";
    echo ${cmd}
else
    cmd="python ${SCORE_SCRIPT} ${NCELLS} ${output_stub} --verbose --threads 20 --focus_algorithm ${algorithm}";
    echo ${cmd}
fi
${cmd}