import pickle as pic
from typing import Tuple

import numpy as np

from nj_iwhd import InverseNJSolver
from src.caching import cached, set_cache_dir, set_use_hash
from src.parameters import *

import cassiopeia as cas
import cassiopeia.solver as solver
from cassiopeia.data.CassiopeiaTree import CassiopeiaTree

# Caching settings
set_cache_dir("./_cache/")
set_use_hash(True)


def get_stressor_by_params(
    lt_parameters
) -> Tuple[str, str]:
    numstates__default = 100
    mut_prop__default = 0.5
    drop_total__default = 0.2

    stressor, param = "char", lt_parameters.numchars
    if lt_parameters.numstates != numstates__default:
        stressor, param = "states", lt_parameters.numstates
    elif lt_parameters.mut_prop != mut_prop__default:
        stressor, param = "mut", int(lt_parameters.mut_prop * 100)
    elif lt_parameters.drop_total != drop_total__default:
        if lt_parameters.drop_total == 0:
            stressor, param = "drop", 0
        else:
            stressor, param = "stoch_high_drop", int(lt_parameters.drop_total * 100)

    return stressor, str(param)


# cached
def get_gt_tree(
    tree_parameters,
    lt_parameters
) -> CassiopeiaTree:
    gt_tree_stub = "/data/yosef2/users/richardz/projects/CassiopeiaV2-Reproducibility/trees/{topology_type}/{numcells}cells/{fit}/{stressor}/tree{numtree}.pkl"

    stressor, param = get_stressor_by_params(lt_parameters)

    gt_tree_path = gt_tree_stub.format(
        topology_type=tree_parameters.topology_type,
        numcells=tree_parameters.n_cells,
        fit=tree_parameters.fitness,
        stressor=stressor+param,
        numtree=tree_parameters.random_seed,
    )

    gt_tree: CassiopeiaTree = pic.load(open(gt_tree_path, 'rb'))

    cm = gt_tree.character_matrix
    if cm is not None:
        cm = cm.replace(-2, -1)
        gt_tree.character_matrix = cm

    return gt_tree

def get_solver_by_name(
    solver_name,
    priors_type='no_priors'
) -> solver.CassiopeiaSolver:  # type: ignore
    # Initialize the solver
    if solver_name == "greedy":
        tree_solver = solver.VanillaGreedySolver()
    elif solver_name == "maxcut_greedy":
        tree_solver = solver.MaxCutGreedySolver()
    elif solver_name == "maxcut":
        tree_solver = solver.MaxCutSolver()
    elif solver_name == "nj":
        tree_solver = solver.NeighborJoiningSolver(add_root = True)
    elif solver_name == "upgma":
        tree_solver = solver.UPGMASolver()
    elif solver_name == "ilp":
        if priors_type is not 'no_priors':
            tree_solver = solver.ILPSolver(maximum_potential_graph_lca_distance = 20, weighted = True)
        else:
            tree_solver = solver.ILPSolver(maximum_potential_graph_lca_distance = 20, weighted = False)
    elif solver_name == "greedy_over_maxcut":
        greedy_solver = solver.VanillaGreedySolver()
        maxcut_solver = solver.MaxCutSolver()
        tree_solver = solver.HybridSolver(greedy_solver, maxcut_solver, lca_cutoff=20, threads=10)
    elif solver_name == "greedy_over_ilp":
        greedy_solver = solver.VanillaGreedySolver()
        if priors_type is not 'no_priors':
            ilp_solver = solver.ILPSolver(maximum_potential_graph_lca_distance = 20, weighted = True)
        else:
            ilp_solver = solver.ILPSolver(maximum_potential_graph_lca_distance = 20, weighted = False)
        tree_solver = solver.HybridSolver(greedy_solver, ilp_solver, lca_cutoff=20, threads=10)
    elif solver_name == "maxcut_greedy_over_ilp":
        greedy_solver = solver.MaxCutGreedySolver()
        if priors_type is not 'no_priors':
            ilp_solver = solver.ILPSolver(maximum_potential_graph_lca_distance = 20, weighted = True)
        else:
            ilp_solver = solver.ILPSolver(maximum_potential_graph_lca_distance = 20, weighted = False)
        tree_solver = solver.HybridSolver(greedy_solver, ilp_solver, lca_cutoff=20, threads=10)
    elif solver_name == "spectral":
        tree_solver = solver.SpectralSolver()
    elif solver_name == "spectral_greedy":
        tree_solver = solver.SpectralGreedySolver()
    elif solver_name == "nj_iwhd":
        tree_solver = InverseNJSolver(add_root=True)
    elif solver_name == "nj_iwhd_oracle":
        tree_solver = InverseNJSolver(add_root=True)
    else:
        raise NameError('Solver not found.')

    return tree_solver

@cached()
def run_solver(
    tree_parameters,
    lt_parameters,
    solver_parameters
) -> CassiopeiaTree:
    solver = get_solver_by_name(
        solver_parameters.solver_name, 
        priors_type=solver_parameters.priors_type
        )

    gt_tree = get_gt_tree(
        tree_parameters,
        lt_parameters
    )

    if solver_parameters.solver_name == 'nj_iwhd_oracle':
        # Setting default oracle params
        solver.set_numstates(lt_parameters.numstates) 
        solver.set_mut_prop(lt_parameters.mut_prop)
        solver.set_numstates(lt_parameters.numstates)
        solver.set_total_time(gt_tree.get_time(gt_tree.leaves[0]))

    recon_tree = CassiopeiaTree(character_matrix = gt_tree.character_matrix, missing_state_indicator = -1) #todo: implement priors
    solver.solve(recon_tree, collapse_mutationless_edges=solver_parameters.collapse_mutationless_edges)
    
    return recon_tree

@cached()
def get_score(
    metric_name,
    tree_parameters,
    lt_parameters,
    solver_parameters,
) -> float:
    # get gt tree
    gt_tree = get_gt_tree(
        tree_parameters,
        lt_parameters
    )

    # get solved tree
    recon_tree = run_solver(
        tree_parameters,
        lt_parameters,
        solver_parameters
    )

    if metric_name == 'rf':
        rf, rf_max = cas.critique.compare.robinson_foulds(gt_tree, recon_tree)
        return rf / rf_max
    elif metric_name == 'triplets':
        triplets = cas.critique.compare.triplets_correct(gt_tree, recon_tree, number_of_trials=500)
        return np.mean(list(triplets[0].values()))
    else:
        raise NameError('Metric not found.')