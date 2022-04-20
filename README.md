# Ivan's Cassiopeia Benchmarking Suite
Benchmarking scripts built primarily to test our inverse weighted hamming distance approach to Neighbor Joining.

### Single-Regime Solver Benchmarking 
Located in `solver_benchmarking_single/`, the notebook runs low-throughput benchmarks of a single solver on a single set of trees. Included is a `BenchmarkModule` class that neatly manages the various input/output files, while allowing heavy user modification of specific elements (solvers, character matrices, custom distance functions, etc) through subclassing.


### Whole-Regime Solver Benchmarking
Refers to the folder `solver_benchmarking_whole_regime/` 


