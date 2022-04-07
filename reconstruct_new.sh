#!/bin/bash

# *** "#PBS" lines must come before any non-blank, non-comment lines ***### Set the job name
### Job's Name
#PBS -N Cassiopeia_StressTest_Ivan
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
exec 2> $workdir/log4/_log.stderr_$t > $workdir/log4/_log.stdout_$t

source ~/.bashrc
conda activate cass
cd $workdir

topology_type=exponential_plus_c
numcells=2000cells
algs=(nj_iwhd)

numtrees=50
fits=(no_fit high_fit)
priors_types=(no_priors)
sims=(char10 char20 char40 char60 char90 char150 mut10 mut30 mut70 mut90 drop0 stoch_high_drop10 stoch_high_drop30 stoch_high_drop40 stoch_high_drop50 stoch_high_drop60 states5 states10 states25 states50 states500)

# Some modular arithmetic used to "flatten" previous arrays
tree_num=$(($t%$numtrees))
sim=${sims[$(($t/$numtrees%${#sims[@]}))]}
priors_type=${priors_types[$(($t/$(($numtrees*${#sims[@]}))%${#priors_types[@]}))]}
fit=${fits[$(($t/$(($numtrees*${#sims[@]}*${#priors_types[@]}))%${#fits[@]}))]}

path="/data/yosef2/users/ivalexander13/in-the-distance"
BENCHMARK_SCRIPT="${path}/reconstruct_new.py"
SIM_DIR="/data/yosef2/users/richardz/projects/CassiopeiaV2-Reproducibility/trees/$topology_type/$numcells/$fit/$sim"
                
for alg in ${algs[@]};
do
    SAVE_DIR="${path}/whole_regime_benchmark_recons/$topology_type/$numcells/$priors_type/$fit/$sim/$alg"

    if [ ! -d ${SAVE_DIR} ]; then
        mkdir -p ${SAVE_DIR};
    fi

    # Give information about the run
    echo Working directory is $workdir
    cd $workdir
    NPROCS=`wc -l < $PBS_NODEFILE`
    echo This job has allocated $NPROCS cpus
    echo Algorithm Used: $alg, Fitness Regime: $fit, Parameter Regime: $sim, Tree Number: $tree_num

    # Get the paths for the ground truth and reconstructed trees
    tree="${SIM_DIR}/tree${tree_num}.pkl"
    output="${SAVE_DIR}/recon${tree_num}"
    echo Reconstructing tree from $tree
    echo Saving tree to $output

    # If priors are to be used, get the right priors and set the argument
    if [ $priors_type == "no_priors" ]; then
        use_priors=""
        priors_path=""
        echo No priors used
    else
        use_priors="--use_priors True"
        regime=${sim//[0-9]/}
        if [ $regime == "states" ]; then
            priors_path="--priors_path priors/$sim.pkl"
        else
            priors_path="--priors_path priors/states100.pkl"
        fi
        echo Drawing priors from $priors_path
    fi

    # Create logfiles for algorithms using ILP
    if [[ "$alg" == *"ilp"* ]]; then
        logpath="alg_logs/$topology_type/$numcells/$priors_type/$fit/$sim/$alg"
        if [ ! -d ${logpath} ]; then
            mkdir -p ${logpath};
        fi
        logfile="--logfile $logpath/$tree_num.log"
    else
        logfile=""
    fi

    cmd="python ${BENCHMARK_SCRIPT} $tree $output $alg $use_priors $priors_path $logfile"
    ${cmd}
done
