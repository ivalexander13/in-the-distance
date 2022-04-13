import pandas as pd
import pickle as pic
import numpy as np

# import matplotlib.pyplot as plt
import json
import os
import argparse
import cassiopeia as cas

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("-t", type=int, help="job number", default=1)
parser.add_argument("--cached", action='store_true', help="load from cache", default=False)
args = parser.parse_args()
t = args.t
cached = args.cached
og_t = t

# Load config
with open("config.json") as f:
    config = json.load(f)

# IO Setup
out_folder = "./post_sim_analytics/"
in_tree_raw = config["tree_dir"]

# Helper Functions
is_cached = False
def get_dists(tree, dists_file):
    if cached and os.path.isfile(dists_file):
        try:
            dists = pic.load(open(dists_file, "rb"))
            if np.isnan(dists.tolist()).all():  # type: ignore
                pass
            else:
                global is_cached
                is_cached = True
                return dists
        except:
            pass

    dists = pd.DataFrame(columns=tree.leaves, index=tree.leaves)
    for leaf in tree.leaves:
        dists[leaf] = tree.get_distances(
            leaf, leaves_only=True
        ).values()  # assume sorted

    melted = melt_triu(dists)
    pic.dump(melted, open(dists_file, "wb+"))

    return melted


def get_dissim_whd(tree, in_tree_path, numtree):
    dissim_file = in_tree_path + "dissim_whd" + str(numtree) + ".pkl"

    if cached and os.path.isfile(dists_file):
        try:
            return pic.load(open(dissim_file, "rb"))
        except:
            pass

    # Get computed dissimilarity matrix.
    tree.compute_dissimilarity_map(
        dissimilarity_function=cas.solver.dissimilarity_functions.weighted_hamming_distance  # type: ignore
    )
    dissim_raw = tree.get_dissimilarity_map()
    dissim = melt_triu(dissim_raw)
    pic.dump(dissim, open(dissim_file, "wb+"))

    return dissim


def melt_triu(dataf):
    return dataf.values[np.triu_indices_from(dataf, 1)]


# mathy math
t, numtree = divmod(t, 50)
t, numstates_idx = divmod(t, len(config["numstates"]))
t, numcassetes_idx = divmod(t, len(config["numcassettes"]))
t, mutrate_idx = divmod(t, len(config["mutation_proportions"]))

# Main Loop
in_tree_path = in_tree_raw.format(
    mutation_proportions=config["mutation_proportions"][mutrate_idx],
    numcassettes=config["numcassettes"][numcassetes_idx],
    numstates=config["numstates"][numstates_idx],
    numcells=config['numcells']
)

# get tree
tree_file = in_tree_path + "tree" + str(numtree) + ".pkl"
dists_file = in_tree_path + "dists" + str(numtree) + ".pkl"

if not os.path.isfile(tree_file):
    print(f'Tree not found: {tree_file}')
    exit()

tree = pic.load(open(tree_file, "rb"))

# get cm
tree.character_matrix = tree.character_matrix.replace(-2, -1)

# get dists
dists_melt = get_dists(tree, dists_file)

# get dissims
dissim_melt = get_dissim_whd(tree, in_tree_path, numtree)
print("Done for t=" + str(og_t) + f". Cached: {is_cached}. --> {tree_file}")


