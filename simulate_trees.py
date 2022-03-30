# %% [markdown]
# # Simulating Character Matrices on Existing Topologies

# %%
# Imports
from pathlib import Path
import numpy as np
import pickle as pic
from tqdm import trange
import json

from cassiopeia.simulator.Cas9LineageTracingDataSimulator import (
    Cas9LineageTracingDataSimulator,
)

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
    pbar = trange(len(folders) * len(fitness_regimes) * num_trees, leave=False)
    for folder in folders:
        for fitness in fitness_regimes:
            top_folder = config["top_dir"].format(
                topology=folder, fitness_regime=fitness, numcells=num_cells
            )

            tree_folder = config["tree_dir"].format(
                mutation_proportions=mutation_proportion,
                numcassettes=number_of_cassettes,
                numstates=len(state_priors),
                numcells=config['numcells']
            )

            Path(tree_folder).mkdir(parents=True, exist_ok=True)

            seeds = np.random.choice(range(10000000), num_trees, replace=False)

            for num in range(num_trees):
                # Update pbar
                pbar.update(1)
                pbar.set_description(
                    f"Doing [ {folder} > {fitness} > tree{num} ]"
                )

                tree_file = tree_folder + f"tree{num}.pkl"

                # Get seed
                seed = seeds[num]
                while seed in seen_seeds:
                    seed += 1

                # Load topology
                top_path = top_folder + f"topology{num}.pkl"
                topology = pic.load(open(top_path, "rb"))

                # Get actual time
                total_time = -1
                for node in topology.nodes:
                    if topology.is_leaf(node):
                        total_time = topology.get_time(node)
                        break

                # Main function call
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

                # Seed writing
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

                # Tree output writing
                pic.dump(
                    topology,
                    open(tree_file, "wb"),
                )


# %%
## Loading Config File
with open("config.json", "r") as f:
    config = json.load(f)

## Hyperparams to Vary
num_trees = 50
folders = config.get("folders", ["exponential_plus_c"])
num_cells = config.get("numcells", [400])
fitness_regimes = config.get("fitness_regimes", ["no_fit"])
size_of_cassette = 1

total_dropout_proportion = 0
stochastic_proportion = 0
seed_path = "/data/yosef2/users/ivalexander13/simulation_data/seeds/030922.sim_trees.txt"

# External Loops
mutation_proportions = config.get("mutation_proportions", [0.5])
numcassettes = config.get("numcassettes", [40])
numstateses = config.get("numstates", [1, 10, 50, 100])

pbar = trange(len(mutation_proportions) * len(numcassettes) * len(numstateses))
for mutation_proportion in mutation_proportions:
    for numchars in numcassettes:
        for numstates in numstateses:
            pbar.update(1)
            pbar.set_description(
                f"Doing [ mut{mutation_proportion} > chars{numchars} > states{numstates} ]"
            )

            state_priors = dict(enumerate([1 / numstates] * numstates, 1))
            populate_trees(
                num_trees=num_trees,
                folders=folders,
                num_cells=num_cells,
                fitness_regimes=fitness_regimes,
                number_of_cassettes=numchars,
                size_of_cassette=size_of_cassette,
                mutation_proportion=mutation_proportion,
                state_priors=state_priors,
                total_dropout_proportion=total_dropout_proportion,
                stochastic_proportion=stochastic_proportion,
                tree_regime="char" + str(numchars),
                seed_string="char\t" + str(numchars),
                seed_path=seed_path,
                seen_seeds=[],
            )

# %%
