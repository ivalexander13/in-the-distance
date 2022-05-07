# Ivan's Cassiopeia Benchmarking Suite
Benchmarking scripts built primarily to test our inverse weighted hamming distance approach to Neighbor Joining.

# Whole-Regime Solver Benchmarking (Cache-cade Version)
To generate the plot, run the following commands:
```python
from src.plot_stressor_regimes import plot_stressor_regimes
plot_stressor_regimes()
```
Alternatively, this code is written in `plot.ipynb`.

### Notes for editing:
- In order to change parameters, set them as arguments for `plot_stressor_regimes()`.
- The implementation is currently written to solve each tree on runtime, but there is an option to use pre-calculated scores (RF and triplets correct) to replicate the plot previously made by deprecated code. To do this, open `src/plot_stressor_regimes.py` and under the `plot_stressor_regimes` function, do the following changes:
    1.  Comment out the section named "Uncomment to Run Cascade"
    2.  Uncomment the section named "Uncomment to Use Cached Scores"
- The implementation is currently written to use pre-simulated ground-truth trees on Richard's account. If you want to generate your own trees, open `src/benchmark.py` and override the `get_gt_tree` function (and enable caching).
- To add a new solver, do the following changes:
    - Open `src/benchmark.py` and under `get_solver_by_name()`, add the solver name and instance to the elif cascade.
    - When calling `plot_stressor_regimes()`, add the solver name to the `solver_names` list and its corresponding color to `solver_plot_params`
<br><br>

# Single-Regime Solver Benchmarking 
Located in `solver_benchmarking_single/`, the `Solver Benchmarking.ipynb` notebook runs low-throughput benchmarks of a single solver on a single set of trees. Included is a `BenchmarkModule` class that neatly manages the various input/output files, while allowing heavy user modification of specific elements (solvers, character matrices, custom distance functions, etc) through subclassing.

Note: This code is not written with the caching decorator, but it already has a caching mechanism built in.
<br><br>

# Whole-Regime Solver Benchmarking (Deprecated: Pipeline Version)
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
N/A
<br><br>

# Tree Generation
This section is for generating trees and doing distance analysis. It does not use the caching decorator yet. The pipeline involves four steps:
1. `simulate_topologies.ipynb`: Simulate tree topologies given a set of tree parameters.
2. `simulate_trees`: From each topology, create a set of trees given a different set of tree parameters. 
    - `2-simulate_trees.sbatch.sh`: Run this script to use SLURM to parallelize the tree generation.
    - `2-simulate_trees.solo.py`: Contains the function to generate one tree under a single set of tree parameters, given an array id `-t`. This is used in the sbatch script above.
    - `2-simulate_trees.all.py` (Deprecated): A runnable script to generate all the trees without parallelization.
3. `compute_distance`: A parallelization setup to calculate the true distance and the weighted hamming distance for every pair of leaves in every single tree in the specified dataset.
    - `3-compute_distances.sbatch.sh`: Run this script to use SLURM to parallelize the distance calculation.
    - `3-compute_distances.py`: Contains the function to compute the distance of one tree, given an array id `-t`.
    - `3-compute_distances.zen.sh` (Deprecated): An alternative parallelization script using zen instead of sbatch.
4. `distance_analysis.ipynb`: Analyze the distance data and generate plots.

The dataset parameters are collected in `config.json` and shared across the scripts.




