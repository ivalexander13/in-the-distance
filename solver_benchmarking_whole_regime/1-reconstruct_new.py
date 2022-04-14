import argparse
import pickle as pic
import sys
sys.path.append('../')
from nj_iwhd import InverseNJSolver, InverseNJSolverOracle

import cassiopeia.solver as solver
from cassiopeia.data.CassiopeiaTree import CassiopeiaTree


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("tree_path", type=str, help="character matrix path")
    parser.add_argument("out_path", type=str, help="output path for reconstructed tree")
    parser.add_argument("alg", type=str, help="name of algorithm to use for reconstruction")
    parser.add_argument("--use_priors", type=bool, default=False, help="whether or not to use priors in reconstruction")
    parser.add_argument("--priors_path", type=str, default="", help="path specifying the location of priors")
    parser.add_argument("--logfile", type=str, default="", help="log file for solver")

    args = parser.parse_args()
    tree_path = args.tree_path
    out_path = args.out_path
    alg = args.alg
    use_priors = args.use_priors
    priors_path = args.priors_path
    logfile = args.logfile
    CUSTOM_SOLVE = False

    # Load the ground_truth character matrix
    tree = pic.load(open(tree_path, 'rb'))
    cm = tree.character_matrix
    cm = cm.replace(-2, -1)

    tree_solver = solver.NeighborJoiningSolver()

    # Initialize the reconstructed tree, with or without priors
    if use_priors:
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
        if use_priors:
            tree_solver = solver.ILPSolver(maximum_potential_graph_lca_distance = 20, weighted = True)
        else:
            tree_solver = solver.ILPSolver(maximum_potential_graph_lca_distance = 20, weighted = False)
    elif alg == "greedy_over_maxcut":
        greedy_solver = solver.VanillaGreedySolver()
        maxcut_solver = solver.MaxCutSolver()
        tree_solver = solver.HybridSolver(greedy_solver, maxcut_solver, lca_cutoff=20, threads=10)
    elif alg == "greedy_over_ilp":
        greedy_solver = solver.VanillaGreedySolver()
        if use_priors:
            ilp_solver = solver.ILPSolver(maximum_potential_graph_lca_distance = 20, weighted = True)
        else:
            ilp_solver = solver.ILPSolver(maximum_potential_graph_lca_distance = 20, weighted = False)
        tree_solver = solver.HybridSolver(greedy_solver, ilp_solver, lca_cutoff=20, threads=10)
    elif alg == "maxcut_greedy_over_ilp":
        greedy_solver = solver.MaxCutGreedySolver()
        if use_priors:
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
            gt_tree_path=tree_path,
            )
        CUSTOM_SOLVE = True
        tree_solver.solve(
            recon_tree, 
            logfile = logfile, 
            collapse_mutationless_edges = True,
            )

    
    # Solve the reconstructed tree, using the provided logfile
    if not CUSTOM_SOLVE:
        tree_solver.solve(recon_tree, logfile = logfile, collapse_mutationless_edges = True)

    # Save the tree newick
    with open(out_path, "w") as f:
        f.write(recon_tree.get_newick())

if __name__=="__main__":
    main()