import pickle as pic
import numpy as np
import json
import argparse
from pathlib import Path

from cassiopeia.simulator.Cas9LineageTracingDataSimulator import (
    Cas9LineageTracingDataSimulator,
)


# argparse
parser = argparse.ArgumentParser()
parser.add_argument("-t", type=int, help="job number", default=1)
parser.add_argument("--cached", type=bool, help="load from cache", default=False)
args = parser.parse_args()
t = args.t
cached = args.cached
og_t = t

# Load config
with open("config.json") as f:
    config = json.load(f)

# mathy math
t, numstates_idx = divmod(t, len(config["numstates"]))
t, numcassetes_idx = divmod(t, len(config["numcassettes"]))
t, mutrate_idx = divmod(t, len(config["mutation_proportions"]))

mutation_proportion=config["mutation_proportions"][mutrate_idx]
number_of_cassettes=config["numcassettes"][numcassetes_idx]
numstates=config["numstates"][numstates_idx]


# Params
num_trees = config['num_trees']
folders = config.get("folders", ["exponential_plus_c"])
num_cells = config.get("numcells", [400])
fitness_regimes = config.get("fitness_regimes", ["no_fit"])
size_of_cassette = 1

total_dropout_proportion = 0
stochastic_proportion = 0

seen_seeds = []
seed_path = "/data/yosef2/users/ivalexander13/simulation_data/seeds/030922.sim_trees.txt"

# Param processing
state_priors = dict(enumerate([1 / numstates] * numstates, 1))
seeds = np.random.choice(range(10000000), num_trees, replace=False)

# IO
tree_folder = config["tree_dir"].format(
            mutation_proportions=mutation_proportion,
            numcassettes=number_of_cassettes,
            numstates=len(state_priors),
            numcells=config['numcells']
        )
Path(tree_folder).mkdir(parents=True, exist_ok=True)

# Main call 
for folder in folders:
    for fitness in fitness_regimes:
        top_folder = config["top_dir"].format(
            topology=folder, fitness_regime=fitness, numcells=num_cells
        )

        for num in range(num_trees):
            tree_file = tree_folder + f"tree{num}.pkl"

            # cached if cached
            if cached and Path(tree_file).is_file():
                print(f'Loaded cache for t={t}')
                exit()

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
                mutation_rate=np.log(1 - mutation_proportion)  # type: ignore
                / (-1 * total_time),
                state_priors=state_priors,
                heritable_silencing_rate=np.log(  # type: ignore
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
                "char\t" + str(number_of_cassettes)
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

print(f'Done for t={og_t} -> {tree_folder}')