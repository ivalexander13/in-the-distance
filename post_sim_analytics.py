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
parser.add_argument('-t', type=int, help='job number', default=1)
args = parser.parse_args()
t = args.t
og_t = t

# Load config
with open('config.json') as f:
    config = json.load(f)

# IO Setup
out_folder = "./post_sim_analytics/"
in_tree_raw = config['dir_structure']

# Helper Functions
def get_dists(tree, dists_file):
    if os.path.isfile(dists_file):
        try: 
            return pic.load(open(dists_file, 'rb'))
        except:
            pass

    dists = pd.DataFrame(columns=tree.leaves, index=tree.leaves)
    for leaf in tree.leaves:
        dists[leaf] = tree.get_distances(leaf, leaves_only=True).values() # assume sorted

    melted = melt_triu(dists)
    pic.dump(melted, open(dists_file, 'wb+'))

    return melted

def melt_triu(dataf):
    return np.fliplr(dataf.values)[np.triu_indices_from(dataf)]

# mathy math
t, mutrate_idx = divmod(t, len(config['mutation_proportions']))
t, numcassetes_idx = divmod(t, len(config['numcassettes']))
t, numstates_idx = divmod(t, len(config['numstates']))
t, numtree = divmod(t, 50)

# Main Loop
in_tree_path = in_tree_raw.format(
    mutation_proportions=config['mutation_proportions'][mutrate_idx],
    numcassettes=config['numcassettes'][numcassetes_idx],
    numstates=config['numstates'][numstates_idx]
)

# get tree
tree_file = in_tree_path + "tree"+ str(numtree) + ".pkl"
dists_file = in_tree_path + "dists"+ str(numtree) + ".pkl"
dissim_file = in_tree_path + "dissim_whd"+ str(numtree) + ".pkl"

if not os.path.isfile(tree_file):
    exit()

tree = pic.load(open(tree_file, "rb"))

# get cm
tree.character_matrix = tree.character_matrix.replace(-2, -1)

# get dists
dists_melt = get_dists(tree, dists_file)

# get dissims
# if os.path.isfile(dissim_file):
#     try: 
#         dissim = pic.load(open(dissim_file, 'rb'))
#     except:
#         pass
# else:
#     # Get computed dissimilarity matrix.
#     tree.compute_dissimilarity_map(dissimilarity_function=cas.solver.dissimilarity_functions.weighted_hamming_distance)
#     dissim_raw = tree.get_dissimilarity_map()
#     dissim = melt_triu(dissim_raw)
#     pic.dump(dissim, open(dissim_file, 'wb+'))


print('Done for t='+str(og_t))
