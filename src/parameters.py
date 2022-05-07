from collections import namedtuple

TreeParameters = namedtuple(
    "TreeParameters",
    [
        "topology_type",
        "n_cells",
        "fitness",
        "random_seed",
    ],
)

LineageTracingParameters = namedtuple(
    "LineageTracingParameters",
    [
        "numchars",
        "numstates",
        "drop_total",
        "mut_prop",
        "random_seed",
    ],
)

SolverParameters = namedtuple(
    "SolverParameters",
    ["solver_name", "collapse_mutationless_edges", "priors_type"],
)
