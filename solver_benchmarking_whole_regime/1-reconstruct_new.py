import argparse
from pathlib import Path
import pickle as pic
import sys

sys.path.append('../')
from nj_iwhd import InverseNJSolver, InverseNJSolverOracle

import cassiopeia.solver as solver
from cassiopeia.data.CassiopeiaTree import CassiopeiaTree


##########
# PARAMS #
##########

# IO
gt_tree_stub = "/data/yosef2/users/richardz/projects/CassiopeiaV2-Reproducibility/trees/{topology_type}/{numcells}/{fit}/{sim}/tree{numtree}.pkl"
save_recon_stub = "/data/yosef2/users/ivalexander13/ivan-cassiopeia-benchmarking/solver_benchmarking_whole_regime/data/{topology_type}/{numcells}/{priors_type}/{fit}/{sim}/{alg}/recon{numtree}"
priors_path = ""

# GT Tree Params
topology_type = "exponential_plus_c"
numcells = "2000cells"
fits = ['no_fit', 'high_fit']
sims = ["char10", "char20", "char40", "char60", "char90", "char150", "mut10", "mut30", "mut70", "mut90", "drop0", "stoch_high_drop10", "stoch_high_drop30", "stoch_high_drop40", "stoch_high_drop50", "stoch_high_drop60", "states5", "states10", "states25", "states50", "states500"]
numtrees = range(50)

# Recon Tree Params
priors_types = ["no_priors"]
alg = 'nj_iwhd_oracle'

####################
# Logistical Stuff #
####################

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-t", type=int, help="job number", default=1)
parser.add_argument("--cached", action='store_true', help="load from cache", default=False)
args = parser.parse_args()
t = args.t
cached = args.cached
og_t = t

# Array Parsing
t, numtree_idx = divmod(t, len(numtrees))
t, sim_idx = divmod(t, len(sims))
t, priors_type_idx = divmod(t, len(priors_types))
t, fit_idx = divmod(t, len(fits))

numtree = numtrees[numtree_idx]
sim = sims[sim_idx]
priors_type = priors_types[priors_type_idx]
fit = fits[fit_idx]

# IO Filling
gt_tree_path = gt_tree_stub.format(
    topology_type=topology_type,
    numcells=numcells,
    fit=fit,
    sim=sim,
    numtree=numtree
)
save_recon_path = save_recon_stub.format(
    topology_type=topology_type,
    numcells=numcells,
    priors_type=priors_type,
    fit=fit,
    sim=sim,
    alg=alg,
    numtree=numtree
)

# Mkdir if not found
Path(gt_tree_path).parent.mkdir(parents=True, exist_ok=True)
Path(save_recon_path).parent.mkdir(parents=True, exist_ok=True)

#################
# Main Function #
#################
CUSTOM_SOLVE = False

# Check cached
if Path(save_recon_path).exists() and cached:
    print(f"Done for t={og_t}. Cached: True. --> {save_recon_path}")
    sys.exit(1)

# Load the ground_truth character matrix
tree = pic.load(open(gt_tree_path, 'rb'))
cm = tree.character_matrix
cm = cm.replace(-2, -1)

tree_solver = solver.NeighborJoiningSolver()

# Initialize the reconstructed tree, with or without priors
if priors_type is not 'no_priors':
    priors = pic.load(open(priors_path, "rb"))
    priors_per_character = {}
    for i in range(cm.shape[1]):
        priors_per_character[i] = priors
    recon_tree = CassiopeiaTree(character_matrix = cm, missing_state_indicator = -1, priors=priors_per_character)
else:
    recon_tree = CassiopeiaTree(character_matrix = cm, missing_state_indicator = -1)

# Initialize the solver
if alg == "greedy":
    tree_solver = solver.VanillaGreedySolver()
elif alg == "maxcut_greedy":
    tree_solver = solver.MaxCutGreedySolver()
elif alg == "maxcut":
    tree_solver = solver.MaxCutSolver()
elif alg == "nj":
    tree_solver = solver.NeighborJoiningSolver(add_root = True)
elif alg == "upgma":
    tree_solver = solver.UPGMASolver()
elif alg == "ilp":
    if priors_type is not 'no_priors':
        tree_solver = solver.ILPSolver(maximum_potential_graph_lca_distance = 20, weighted = True)
    else:
        tree_solver = solver.ILPSolver(maximum_potential_graph_lca_distance = 20, weighted = False)
elif alg == "greedy_over_maxcut":
    greedy_solver = solver.VanillaGreedySolver()
    maxcut_solver = solver.MaxCutSolver()
    tree_solver = solver.HybridSolver(greedy_solver, maxcut_solver, lca_cutoff=20, threads=10)
elif alg == "greedy_over_ilp":
    greedy_solver = solver.VanillaGreedySolver()
    if priors_type is not 'no_priors':
        ilp_solver = solver.ILPSolver(maximum_potential_graph_lca_distance = 20, weighted = True)
    else:
        ilp_solver = solver.ILPSolver(maximum_potential_graph_lca_distance = 20, weighted = False)
    tree_solver = solver.HybridSolver(greedy_solver, ilp_solver, lca_cutoff=20, threads=10)
elif alg == "maxcut_greedy_over_ilp":
    greedy_solver = solver.MaxCutGreedySolver()
    if priors_type is not 'no_priors':
        ilp_solver = solver.ILPSolver(maximum_potential_graph_lca_distance = 20, weighted = True)
    else:
        ilp_solver = solver.ILPSolver(maximum_potential_graph_lca_distance = 20, weighted = False)
    tree_solver = solver.HybridSolver(greedy_solver, ilp_solver, lca_cutoff=20, threads=10)
elif alg == "spectral":
    tree_solver = solver.SpectralSolver()
elif alg == "spectral_greedy":
    tree_solver = solver.SpectralGreedySolver()
elif alg == "nj_iwhd":
    tree_solver = InverseNJSolver(add_root=True)
elif alg == "nj_iwhd_oracle":
    tree_solver = InverseNJSolverOracle(
        add_root=True,
        gt_tree_path=gt_tree_path,
        )
    CUSTOM_SOLVE = True
    tree_solver.solve(
        recon_tree, 
        collapse_mutationless_edges = True,
        )

# Solve the reconstructed tree, using the provided logfile
if not CUSTOM_SOLVE:
    tree_solver.solve(recon_tree, collapse_mutationless_edges = True)

# Save the tree newick
with open(save_recon_path, "w+") as f:
    f.write(recon_tree.get_newick())

print(f"Done for t={og_t}. Cached: False. --> {save_recon_path}")