from pathlib import Path
import cassiopeia as cas
import pandas as pd
import pickle as pic
from tqdm import trange
import os
from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.solver.CassiopeiaSolver import CassiopeiaSolver


class BenchmarkModule:
    """A class to handle input trees and output benchmarking data, as well as 
    modular methods to implement custom steps for solving trees.
    """
    def __init__(
        self,
        test_name: str,
        solver: CassiopeiaSolver = cas.solver.NeighborJoiningSolver(add_root = True, dissimilarity_function=cas.solver.dissimilarity_functions.weighted_hamming_distance),  # type: ignore
        gt_trees_dir: str = "/data/yosef2/users/richardz/projects/CassiopeiaV2-Reproducibility/trees/exponential_plus_c/400cells/no_fit/char40/",
        numtrees: int = 50,
        out_basefolder: str = "./benchmarking/",
        ):
        """Init
        Args:
            test_name (str): the unique name of the instance, to be used as output filenames. Examples include "nj_iwhd" and "snj_yaffe".
            solver (CassiopeiaSolver, optional): Defaults to cas.solver.NeighborJoiningSolver(add_root = True, dissimilarity_function=cas.solver.dissimilarity_functions.weighted_hamming_distance).
            gt_trees_dir (str, optional): The directory of a single simulation condition, containing at least <numtree> trees, each with a character matrix. This directory must have files called tree<numtree>.pkl directly in them. Defaults to "/data/yosef2/users/richardz/projects/CassiopeiaV2-Reproducibility/trees/exponential_plus_c/400cells/no_fit/char40/".
            numtrees (int, optional): The number of trees ('replicates') per simulation condition. Defaults to 50.
            out_basefolder (str, optional): The primary output parent folder to keep our reconstructed trees and metric dataframes. Defaults to "./benchmarking/".
        """
        self.test_name = test_name
        self.solver = solver
        self.gt_trees_dir = gt_trees_dir
        self.numtrees = numtrees
        self.out_basefolder = out_basefolder

    def get_gt_tree(self, i: int) -> CassiopeiaTree:
        """Get a ground truth tree from the ground truth tree directory.

        Args:
            i (int): the tree number to get.

        Returns:
            CassiopeiaTree: the ground-truth tree.
        """
        gt_tree_file = os.path.join(self.gt_trees_dir, f"tree{i}.pkl")
        gt_tree = pic.load(open(gt_tree_file, "rb"))

        return gt_tree

    def get_recon_tree(self, i: int) -> CassiopeiaTree:
        """Get a reconstructed tree from the output directory.

        Args:
            i (int): the tree number to get.

        Returns:
            CassiopeiaTree: the reconstructed tree.
        """
        recon_file = os.path.join(self.out_basefolder, self.test_name, f"recon{i}")
        recon_tree = cas.data.CassiopeiaTree(
                tree=recon_file
            )

        return recon_tree

    def run_solver(self, i: int, cm: pd.DataFrame, collapse_mutationless_edges: bool) -> str:
        """Run the solver on a single tree.

        Args:
            i (int): the tree number to solve.
            cm (pd.DataFrame): the character matrix of the tree.
            collapse_mutationless_edges (bool): whether to collapse mutationless edges.

        Returns:
            str: the newick string of the solved tree.
        """
        # Initialize output recon tree
        recon_tree = cas.data.CassiopeiaTree(
            character_matrix=cm, 
            missing_state_indicator = -1
            )
        
        # Instantiate Solver
        self.solver.solve(recon_tree, collapse_mutationless_edges = collapse_mutationless_edges)
        
        return recon_tree.get_newick()

    def get_cm(self, i: int) -> pd.DataFrame:
        """Get the character matrix of a tree.

        Args:
            i (int): the tree number to get.

        Returns:
            pd.DataFrame: the character matrix of the tree.
        """
        cm_file = os.path.join(self.gt_trees_dir, f"cm{i}.txt")
        cm = pd.read_table(cm_file, index_col = 0)
        cm = cm.replace(-2, -1)  # type: ignore

        return cm

    def reconstruct(
        self,
        overwrite: bool=False,
        collapse_mutationless_edges: bool=True
        ) -> None:
        """Reconstruct trees from the ground truth trees and run the solver on them.

        Args:
            overwrite (bool, optional): Whether to overwrite existing reconstructed tree files, if exists. Defaults to False.
            collapse_mutationless_edges (bool, optional): Whether to collapse mutationless edges. Defaults to True.
        """

        pbar = trange(self.numtrees)
        for i in pbar:
            pbar.set_description(f"Reconstructing tree {i}")

            # Output File
            recon_outfile = Path(os.path.join(self.out_basefolder, self.test_name, f"recon{i}"))
            recon_outfile.parent.mkdir(parents=True, exist_ok=True)

            if not overwrite and recon_outfile.exists():
                pbar.set_description(f"Skipping reconstruction {i}")
                continue

            # Get CM
            cm = self.get_cm(i)

            # Instantiate Solver
            recon_newick = self.run_solver(i, cm, collapse_mutationless_edges)
            
            # Save
            with open(recon_outfile, "w+") as f:
                f.write(recon_newick)
                f.close()

    def evaluate(self, overwrite: bool=False) -> None:
        """Evaluate the reconstruction.

        Args:
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
        """
        # Output Files
        rf_out = Path(os.path.join(self.out_basefolder, f"{self.test_name}.rf.csv"))
        triplets_out = Path(os.path.join(self.out_basefolder, f"{self.test_name}.triplets.csv"))

        # Check overwrites
        if not overwrite and rf_out.exists() and triplets_out.exists():
            return

        # Init datframes
        triplets_df = pd.DataFrame(
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
        RF_df = pd.DataFrame(
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

        # Main Loop
        pbar = trange(self.numtrees)
        for i in pbar:
            pbar.set_description(f"Evaluating tree {i}")

            # GT Tree
            gt_tree = self.get_gt_tree(i)

            # Recon Tree
            recon_tree = self.get_recon_tree(i)

            # Triplets
            triplet_correct = cas.critique.triplets_correct(
                gt_tree,
                recon_tree,
                number_of_trials=1000,
                min_triplets_at_depth=50,
            )[0]

            for depth in triplet_correct:
                triplets_df = triplets_df.append(
                    pd.Series(
                        [
                            400,
                            "no_priors",
                            "no_fit",
                            "char",
                            40,
                            "SNJ",
                            i,
                            depth,
                            triplet_correct[depth],
                        ],
                        index=triplets_df.columns,
                    ),
                    ignore_index=True,
                )

            # RF
            rf, rf_max = cas.critique.robinson_foulds(
                gt_tree, recon_tree
            )

            RF_df = RF_df.append(
                pd.Series(
                    [
                        400,
                        "no_priors",
                        "no_fit",
                        "char",
                        40,
                        "SNJ",
                        i,
                        rf,
                        rf_max,
                        rf / rf_max,
                    ],
                    index=RF_df.columns,
                ),
                ignore_index=True,
            )

        # Save
        triplets_df.to_csv(triplets_out)
        RF_df.to_csv(rf_out)

        return 



