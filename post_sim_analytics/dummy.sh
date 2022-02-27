#!/bin/bash

# *** "#PBS" lines must come before any non-blank, non-comment lines ***### Set the job name
### Job's Name
#PBS -N Cassiopeia_Distance_Getting
### Redirect stdout and stderr by first telling Torque to redirect do /dev/null and then redirecting yourself via exec. This is the way the IT recommends.
#PBS -e localhost:/dev/null
#PBS -o localhost:/dev/null
### Set the queue to which to send
#PBS -q yosef3
### Limit the resources used
#PBS -l nodes=1:ppn=1
### Change the walltime and cpu time limit from their default (the default is currently an hour)
#PBS -l walltime=2000:00:00
#PBS -l cput=20000:00:00
### Move all your environment variables to the job
#PBS -V

### Change to the directory where the job was submitted
### Alternately, use the -d option and set $PBS_O_INITDIR

workdir=$PBS_O_WORKDIR
t=$PBS_ARRAYID
exec 2> $workdir/log/_log.stderr_$t > $workdir/log/_log.stdout_$t

echo $t
             
# cmd="python post_sim_analytics.py $t"

# ${cmd}
