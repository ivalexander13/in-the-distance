# %% [markdown]
# # Simulating Character Matrices on Existing Topologies

# %%
# Imports
import numpy as np
import networkx as nx
import pickle as pic
import os
from scipy import interpolate
from IPython.display import Image
from tqdm.auto import tqdm
import json

# import time

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.simulator.TreeSimulator import TreeSimulatorError
from cassiopeia.simulator.BirthDeathFitnessSimulator import (
    BirthDeathFitnessSimulator,
)
from cassiopeia.simulator.Cas9LineageTracingDataSimulator import (
    Cas9LineageTracingDataSimulator,
)
from cassiopeia.simulator.UniformLeafSubsampler import UniformLeafSubsampler

from cassiopeia.plotting import itol_utilities

# %% [markdown]
# ## Function Definition and Call

# %%
def populate_trees(
    num_trees,
    folders,
    num_cells,
    fitness_regimes,
    number_of_cassettes,
    size_of_cassette,
    mutation_proportion,
    state_priors,
    total_dropout_proportion,
    stochastic_proportion,
    tree_regime,
    seed_string,
    seed_path,
    seen_seeds=[],
):
    pbar0 = tqdm(folders, leave=False)
    for folder in pbar0:
        pbar0.set_description(f"Topology: {folder}")

        pbar1 = tqdm(num_cells, leave=False)
        for num_cell in pbar1:
            pbar1.set_description(f"Num Cells: {num_cell}")

            pbar2 = tqdm(fitness_regimes, leave=False)
            for fitness in pbar2:
                pbar2.set_description(f"Fitness Regime: {fitness}")

                top_folder = (
                    "/data/yosef2/users/richardz/projects/CassiopeiaV2-Reproducibility/topologies/"
                    + folder
                    + "/"
                    + str(num_cell)
                    + "cells/"
                    + fitness
                    + "/"
                )
                path_elements = [
                    f"mutrate{mutation_proportion}", 
                    tree_regime,
                    f"states{len(state_priors)}", 
                    folder,
                    f"cells{num_cell}",
                    fitness,
                ]
                tree_folder = (
                    "/data/yosef2/users/ivalexander13/simulation_data/trees/"
                )

                for i in path_elements:
                    tree_folder += i + "/"
                    if os.path.exists(tree_folder) == False:
                        os.mkdir(tree_folder)
                seeds = np.random.choice(
                    range(10000000), num_trees, replace=False
                )

                pbar3 = tqdm(range(num_trees), leave=False)
                for num in pbar3:
                    pbar3.set_description(f"Tree #: {num}")

                    seed = seeds[num]
                    while seed in seen_seeds:
                        seed += 1
                    top_path = top_folder + "topology" + str(num) + ".pkl"
                    topology = pic.load(open(top_path, "rb"))

                    total_time = -1
                    for node in topology.nodes:
                        if topology.is_leaf(node):
                            total_time = topology.get_time(node)
                            break

                    lt_sim = Cas9LineageTracingDataSimulator(
                        number_of_cassettes=number_of_cassettes,
                        size_of_cassette=size_of_cassette,
                        mutation_rate=np.log(1 - mutation_proportion)
                        / (-1 * total_time),
                        state_priors=state_priors,
                        heritable_silencing_rate=np.log(
                            (total_dropout_proportion - 1)
                            / (stochastic_proportion - 1)
                        )
                        / (-1 * total_time),
                        stochastic_silencing_rate=stochastic_proportion,
                        heritable_missing_data_state=-2,
                        stochastic_missing_data_state=-1,
                        random_seed=seed,
                    )
                    seen_seeds.append(seed)
                    f = open(seed_path, "a")
                    f.write(
                        seed_string
                        + "\t"
                        + fitness
                        + "\t"
                        + str(num)
                        + "\t"
                        + str(seed)
                        + "\n"
                    )
                    f.close()
                    lt_sim.overlay_data(topology)

                    # out dir
                    pic.dump(
                        topology,
                        open(tree_folder + "tree" + str(num) + ".pkl", "wb"),
                    )


# %%
## Loading Config File
with open('config.json', 'r') as f:
    config = json.load(f)

## Hyperparams to Vary
num_trees = 50
folders = config.get(
    'folders', 
    ["exponential_plus_c"])
num_cells = config.get(
    'num_cells', 
    [2000])
fitness_regimes = config.get(
    'fitness_regimes', 
    ["no_fit"])
size_of_cassette = 1
# number_of_cassettes = 40
# mutation_proportion = 0.5
# numstates = 10

total_dropout_proportion = 0
stochastic_proportion = 0
seed_path = "/data/yosef2/users/ivalexander13/simulation_data/seeds_021622.txt"

# External Loops
mutation_proportions = config.get(
    'mutation_proportions', 
    [0.1, 0.5, 1, 2, 5])
numcassettes = config.get(
    'numcassettes', 
    [10, 20, 40, 100])
numstateses = config.get(
    'numstates', 
    [1, 10, 50, 100])

pbar1 = tqdm(mutation_proportions)
for mutation_proportion in pbar1:
    pbar1.set_description(f"Mutation Rate: {mutation_proportion}")

    pbar2 = tqdm(numcassettes)
    for numchars in pbar2:
        pbar2.set_description(f"Num Chars: {numchars}")

        pbar3 = tqdm(numstateses)
        for numstates in pbar3:
            pbar3.set_description(f"Num States: {numstates}")

            state_priors = dict(enumerate([1 / numstates] * numstates, 1))
            populate_trees(
                num_trees,
                folders,
                num_cells,
                fitness_regimes,
                numchars,
                size_of_cassette,
                mutation_proportion,
                state_priors,
                total_dropout_proportion,
                stochastic_proportion,
                tree_regime="char" + str(numchars),
                seed_string="char\t" + str(numchars),
                seed_path=seed_path,
                seen_seeds=[],
            )

# %%
