# Ivan's Cassiopeia Benchmarking Suite
Benchmarking scripts built primarily to test our inverse weighted hamming distance approach to Neighbor Joining.

## Single-Regime Solver Benchmarking 
Located in `solver_benchmarking_single/`, the notebook runs low-throughput benchmarks of a single solver on a single set of trees. Included is a `BenchmarkModule` class that neatly manages the various input/output files, while allowing heavy user modification of specific elements (solvers, character matrices, custom distance functions, etc) through subclassing.


## Whole-Regime Solver Benchmarking
Located in the folder `solver_benchmarking_whole_regime/`, the scripts allow the reconstruction, scoring, and plotting of multple algorithms (solvers), stressors, and parameters simultaneously. Here are the following usages:

### 1. **Reconstructing Trees**
Given ground truth trees with specified fitness regimes, cell counts, stressors, and priors, use a custom solver to reconstruct it only from its character matrix. The output will be one file per tree that contains the topology in its newick format.

In order to set up this section, do these changes in `1-reconstruct_new.py`:
1. Ensure the directories for ground-truth trees and new reconstructed trees are correct.
2. Under "GT Tree Params", fill in the conditions to run the solver through, while ensuring each condition is accompanied by a corresponding set of ground-truth trees.
3. Under "Recon Tree Params", fill in the priors type(s) to use and algorithm(s) to run.
4. The script can be run with `python 1-reconstruct_new.py -t <t>` where `t` is the array ID to use. This array ID will be the determiner of which combination of conditions and tree number to feed into the solver.

And apply these changes to `1-reconstruct_new.sbatch.sh`:
1. At line 15, make sure the conda environment name is correct.
2. At line 28, determine whether or not to have caching enabled. If so, then existing files will not be overwritten.

To run the script, do the following:
```bash
sbatch 1-reconstruct_new.sbatch.sh <ARRAY_OFFSET>
```
Where `ARRAY_OFFSET` is the number to add to `ARRAY_ID`, since the server limits `ARRAY_ID`s only to 999. Therefore to run, for example, 2100 trees, you would run the command three times, with `ARRAY_OFFSET`s of 0, 1000, and 2000 (and caching enabled).

### 2. **Scoring Trees**





