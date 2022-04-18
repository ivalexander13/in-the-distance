#!/bin/bash

# *** "#PBS" lines must come before any non-blank, non-comment lines ***### Set the job name
### Job's Name
#PBS -N Cassiopeia_StressTest_Ivan
### Redirect stdout and stderr by first telling Torque to redirect do /dev/null and then redirecting yourself via exec. This is the way the IT recommends.
#PBS -e /data/yosef2/users/ivalexander13/ivan-cassiopeia-benchmarking/solver_benchmarking_whole_regime/stderr.txt
#PBS -o /data/yosef2/users/ivalexander13/ivan-cassiopeia-benchmarking/solver_benchmarking_whole_regime/stdout.txt
### Set the queue to which to send
#PBS -q yosef3
### Limit the resources used
#PBS -l nodes=1:ppn=1
### Change the walltime and cpu time limit from their default (the default is currently an hour)
#PBS -l walltime=2000:00:00
#PBS -l cput=20000:00:00
### Move all your environment variables to the job
#PBS -V
# PBS -t0-2099

### Change to the directory where the job was submitted
### Alternately, use the -d option and set $PBS_O_INITDIR

workdir=$PBS_O_WORKDIR
t=$PBS_ARRAYID
# exec 2> $workdir/log4/_log.stderr_$t > $workdir/log4/_log.stdout_$t

source ~/.bashrc
conda activate cass
cd $workdir

python 1-reconstruct_new.py -t $t --cached