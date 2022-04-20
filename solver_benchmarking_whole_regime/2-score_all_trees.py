import os
from typing import List, Tuple
import argparse
from collections import namedtuple
from joblib import delayed
import pickle as pic
from tqdm import tqdm

import ngs_tools
import pandas as pd
import cassiopeia as cas

RECONSTRUCTION_DIR = "./data/exponential_plus_c"
# RECONSTRUCTION_DIR = "/data/yosef2/users/richardz/projects/CassiopeiaV2-Reproducibility/reconstructed/exponential_plus_c"
GROUND_TRUTH_DIR = "/data/yosef2/users/richardz/projects/CassiopeiaV2-Reproducibility/trees/exponential_plus_c"
RECONSTRUCTION = namedtuple(
    "Reconstruction",
    [
        "number_of_cells",
        "priors_used",
        "fitness",
        "stressor",
        "param",
        "algorithm",
        "replicate",
        "filename",
    ],
)
NUMBER_OF_TRIPLETS = 1000
MINIMUM_NUMBER_OF_TRIPLETS_AT_DEPTH = 50


def get_stressor_param_from_directory_name(dir_name: str) -> Tuple[str, str]:
    """Gets stressor paramater number.

    Stressors appear as directories with an alphanumeric name - the first set of
    characters correspond to the stressor and the numbers correspond to the
    parameter. For example "char10" would indicate that this directory stores
    results from benchmarks with 10 characters. This function separates the
    stressor name and parameter.

    Args:
        filename: Stressor directory name.

    Returns:
        Stressor name and parameter.
    """
    param = ""
    stressor_name = ""
    for character in dir_name:
        if character.isdigit():
            param += character
        else:
            stressor_name += character

    return stressor_name, param


def get_replicate_from_file(filename: str) -> str:
    """Gets replicate number.

    The reconstructions are currently saved as a file of the form "reconX" where
    X is the replicate. This function will extract X from the file name.

    Args:
        filename: Reconstruction file name.

    Returns:
        Replicate number.
    """
    replicate_number = filename.split("recon")[1]
    return replicate_number


def gather_all_files(
    number_of_cells: str,
    include_priors: bool = False,
    focus_algorithm: str = None,  # type: ignore
) -> List[RECONSTRUCTION]:
    """Gathers all files into a nested dictionary for batch processing.

    Crawls through results directory and gathers all the reconstruction
    filepaths into a nested dictionary for batch processing. The keys of the
    dictionary correspond to the parameters used to generate the reconstruction
    or the parameters of the ground truth tree.

    Args:
        number_of_cells: Number of cells for the ground truth topology.
            Currently this is limited to either 400 or 2000.
        include_priors: Score trees that were reconstructed with priors.
        focus_algorithm: Algorithm to focus on.

    Returns:
        A nested dictionary of all the file paths.
    """
    reconstructions_to_score = []

    priors_directories = ["no_priors"]

    if include_priors:
        priors_directories.append("priors")

    fitnesses = ["no_fit", "high_fit"]
    number_of_cells_directory = f"{number_of_cells}cells"

    for fitness in fitnesses:

        # iterate through priors
        for _priors in priors_directories:

            # get all stressors
            for stressor in os.listdir(
                os.path.join(
                    RECONSTRUCTION_DIR,
                    number_of_cells_directory,
                    _priors,
                    fitness,
                )
            ):
                (
                    stressor_name,
                    stressor_param,
                ) = get_stressor_param_from_directory_name(stressor)

                # get all algorithms
                for algorithm in os.listdir(
                    os.path.join(
                        RECONSTRUCTION_DIR,
                        number_of_cells_directory,
                        _priors,
                        fitness,
                        stressor,
                    )
                ):

                    if (
                        focus_algorithm is not None
                        and algorithm != focus_algorithm
                    ):
                        continue

                    # get all replicates
                    for replicate in os.listdir(
                        os.path.join(
                            RECONSTRUCTION_DIR,
                            number_of_cells_directory,
                            _priors,
                            fitness,
                            stressor,
                            algorithm,
                        )
                    ):

                        replicate_number = get_replicate_from_file(replicate)

                        filename = os.path.join(
                            RECONSTRUCTION_DIR,
                            number_of_cells_directory,
                            _priors,
                            fitness,
                            stressor,
                            algorithm,
                            replicate,
                        )
                        new_reconstruction = RECONSTRUCTION(
                            number_of_cells,
                            _priors,
                            fitness,
                            stressor_name,
                            stressor_param,
                            algorithm,
                            replicate_number,
                            filename,
                        )

                        reconstructions_to_score.append(new_reconstruction)

    return reconstructions_to_score


def split_into_batches(
    reconstruction_list: List[RECONSTRUCTION], number_of_threads: int
) -> List[List[RECONSTRUCTION]]:
    """Splits tasks into batches.

    Using the specified number of threads, create approximately evenly sized
    batches of reconstructions to score.

    Args:
        reconstruction_list: List of reconstructions to score.
        number_of_threads: Number of threads to utilize.

    Returns:
        A list of batches of reconstructions.
    """

    k, m = divmod(len(reconstruction_list), number_of_threads)
    batches = [
        reconstruction_list[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(number_of_threads)
    ]
    return batches


def score_batch(
    batch_of_reconstructions: List[RECONSTRUCTION],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scores a batch of reconstructions.

    Scores each reconstruction's Triplets Correct and Robinson Foulds. Will
    return a dataframe for each of these values.

    Args:
        batch_of_reconstructions: A batch of reconstruction tuples.

    Returns:
        Two dataframes. One corresponding to triplets correct results, one
            corresponding to robinson-foulds results.
    """

    triplets = pd.DataFrame(
        columns=[
            "NumberOfCells",
            "Priors",
            "Fitness",
            "Stressor",
            "Parameter",
            "Algorithm",
            "Replicate",
            "Depth",
            "TripletsCorrect",
        ]
    )
    RF = pd.DataFrame(
        columns=[
            "NumberOfCells",
            "Priors",
            "Fitness",
            "Stressor",
            "Parameter",
            "Algorithm",
            "Replicate",
            "UnNormalizedRobinsonFoulds",
            "MaxRobinsonFoulds",
            "NormalizedRobinsonFoulds",
        ]
    )

    for reconstruction in tqdm(batch_of_reconstructions):

        reconstruction_file_path = reconstruction.filename
        number_of_cells = reconstruction.number_of_cells
        priors_used = reconstruction.priors_used
        fitness_level = reconstruction.fitness
        stressor = reconstruction.stressor
        stressor_param = reconstruction.param
        algorithm_used = reconstruction.algorithm
        replicate_number = reconstruction.replicate

        ground_truth_filepath = os.path.join(
            GROUND_TRUTH_DIR,
            f"{number_of_cells}cells",
            fitness_level,
            f"{stressor+stressor_param}",
            f"tree{replicate_number}.pkl",
        )

        reconstructed_tree = cas.data.CassiopeiaTree(
            tree=reconstruction_file_path
        )
        ground_truth_tree = pic.load(open(ground_truth_filepath, "rb"))

        # score triplets, rf for reconstructed_tree
        triplet_correct = cas.critique.triplets_correct(
            ground_truth_tree,
            reconstructed_tree,
            number_of_trials=NUMBER_OF_TRIPLETS,
            min_triplets_at_depth=MINIMUM_NUMBER_OF_TRIPLETS_AT_DEPTH,
        )[0]
        rf, rf_max = cas.critique.robinson_foulds(
            ground_truth_tree, reconstructed_tree
        )

        RF = RF.append(
            pd.Series(
                [
                    number_of_cells,
                    priors_used,
                    fitness_level,
                    stressor,
                    stressor_param,
                    algorithm_used,
                    replicate_number,
                    rf,
                    rf_max,
                    rf / rf_max,
                ],
                index=RF.columns,
            ),
            ignore_index=True,
        )

        for depth in triplet_correct:
            triplets = triplets.append(
                pd.Series(
                    [
                        number_of_cells,
                        priors_used,
                        fitness_level,
                        stressor,
                        stressor_param,
                        algorithm_used,
                        replicate_number,
                        depth,
                        triplet_correct[depth],
                    ],
                    index=triplets.columns,
                ),
                ignore_index=True,
            )

    return triplets, RF


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "number_of_cells",
        type=str,
        help="The size of the ground truth dataset.",
    )
    parser.add_argument(
        "output_stub",
        type=str,
        help="Output file name stub. Will add `triplets` and `rf` to stub.",
    )
    parser.add_argument("--score_priors", action="store_true")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="The number of threads to utilize.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable extra verbosity."
    )
    parser.add_argument(
        "--debug_n_entries",
        type=int,
        default=None,
        help="Number of entries to test for debugging.",
    )
    parser.add_argument(
        "--focus_algorithm",
        type=str,
        default=None,
        help="Algorithm to focus on.",
    )
    args = parser.parse_args()

    num_cells = args.number_of_cells
    output_stub = args.output_stub
    score_priors = args.score_priors
    threads = args.threads
    verbose = args.verbose
    debug_entries = args.debug_n_entries
    focus_algorithm = args.focus_algorithm

    triplets_output = f"{output_stub}.triplets_correct.tsv"
    rf_output = f"{output_stub}.robinson_foulds.tsv"

    if os.path.exists(triplets_output) or os.path.exists(rf_output):
        raise Exception(
            "Either the triplets correct or RF file already exists."
            " Either move the file, or specify a different output"
            " stub."
        )

    all_reconstructions = gather_all_files(
        num_cells, include_priors=score_priors, focus_algorithm=focus_algorithm
    )

    if debug_entries:
        print(f"Testing on {debug_entries} entries.")
        all_reconstructions = all_reconstructions[:debug_entries]

    file_batches = split_into_batches(all_reconstructions, threads)

    if verbose:
        print(
            f"Split {len(all_reconstructions)} into {len(file_batches)} batches of length {len(file_batches[0])} each."
        )

    if threads > 0:
        all_triplets_correct_dfs = []
        all_rf_dfs = []
        for triplets_correct, rf in ngs_tools.utils.ParallelWithProgress(
            n_jobs=threads,
            total=len(file_batches),
            desc="Scoring reconstructions in batches",
        )(delayed(score_batch)(batch) for batch in file_batches):
            # triplets_correct, rf = score_batch(batch)
            all_triplets_correct_dfs.append(triplets_correct)
            all_rf_dfs.append(rf)

        triplets_correct_df = pd.concat(all_triplets_correct_dfs)
        robinson_foulds_df = pd.concat(all_rf_dfs)

    else:
        triplets_correct_df, robinson_foulds_df = score_batch(
            all_reconstructions
        )

    triplets_correct_df.to_csv(triplets_output, sep="\t", index=False)
    robinson_foulds_df.to_csv(rf_output, sep="\t", index=False)


if __name__ == "__main__":
    main()
