import pickle
from typing import Optional, Tuple
from cassiopeia.data.CassiopeiaTree import CassiopeiaTree # type: ignore
from cassiopeia import solver # type: ignore
import pandas as pd
import numpy as np
from scipy import interpolate

# Copied from score_all_trees.py
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

class IWHD:
    def __init__(
        self, 
        state_distribution,
        mut_prop,
        total_time=1): 

        self.total_time = total_time
        self.q = np.sum(np.array([*state_distribution])**2)

        # Getting mutation rate
        self.mut_rate = np.log(1 - mut_prop)/(-1 * self.total_time) #type: ignore

    def _ewhd_given_h(self, mut_rate, collision_rate, height, time):
        t = time
        r = mut_rate
        q = collision_rate
        h = height
    
        return (2 * (1 - np.exp(-h * r)) ** 2 * (1 - q) + 2 * (1 - np.exp(-h * r)) * (np.exp(-h * r))) * (np.exp(r * (h - t)))   # type: ignore

    def _inverse(self, f, y, lower, upper, error_tolerance, depth):
        x = (upper + lower) / 2.0
        if abs(f(x) - y) < error_tolerance or depth >= 10:
            return x
        elif f(x) < y:
            return self._inverse(f, y, x, upper, error_tolerance, depth+1)
        else:
            return self._inverse(f, y, lower, x, error_tolerance, depth+1)

    def _iwhd(self, mut_rate, collision_rate, whd, time, error_tolerance):
        f = lambda x: self._ewhd_given_h(mut_rate, collision_rate, x, time)
        return self._inverse(f, whd, 0, time,  error_tolerance, 0)

    def __call__(
        self,
        s1,
        s2,
        missing_state_indicator=-1,
        weights=None,
    ) -> float:
        
        # Weighted Hamming Distance
        whd = solver.dissimilarity.weighted_hamming_distance(s1, s2, missing_state_indicator, weights)
                
        return 2 * self._iwhd(
            mut_rate=self.mut_rate,     
            collision_rate=self.q, 
            whd=whd, 
            time=self.total_time, 
            error_tolerance=0.001
        )

class InverseNJSolver(solver.NeighborJoiningSolver):
    def get_dissimilarity_map(
        self, 
        cassiopeia_tree: CassiopeiaTree,
        layer: Optional[str] = None
    ) -> pd.DataFrame: 

        # Estimating parameters
        cm = cassiopeia_tree.character_matrix
        cm = cm.replace(-2, -1)
        mut_prop = np.count_nonzero(cm.replace(-1, 0)) / np.count_nonzero(cm+1)

        numstates = cm.max().max()
        state_distribution = dict(enumerate([1 / numstates] * numstates, 1))

        # Set up the iwhd dissimilarity function
        self.dissimilarity_function = IWHD(
            state_distribution=state_distribution,
            mut_prop=mut_prop
            )

        # Get the dissimilarity map
        self.setup_dissimilarity_map(cassiopeia_tree, layer)
        dissimilarity_map = cassiopeia_tree.get_dissimilarity_map()


        return dissimilarity_map
        
class InverseNJSolverOracle(solver.NeighborJoiningSolver):
    def __init__(
        self,
        dissimilarity_function = solver.dissimilarity.weighted_hamming_distance,
        add_root: bool = False,
        prior_transformation: str = "negative_log",
        gt_tree_path = None
    ):

        super().__init__(
            dissimilarity_function = dissimilarity_function,
            add_root = add_root,
            prior_transformation = prior_transformation
            )

        # Default params. If none are provided, use estimations.
        # Obtained from https://docs.google.com/document/d/1lMg90cV8k55hgRPdWqitPgMlrOGq_XBt3h5i-O9amoQ/edit
        self.numstates = 100
        self.state_distribution = None
        # self.state_distribution = dict(enumerate([1 / self.numstates] * self.numstates, 1))
        self.mut_prop = 0.5
        self.total_time = None

        if gt_tree_path is not None:
            stressor_name, stressor_value = get_stressor_param_from_directory_name(gt_tree_path)

            if stressor_name == "states":
                self.numstates = int(stressor_value)
            elif stressor_name == "mut":
                self.mut_prop = float(stressor_value) / 100

            # Time Param
            tree = pickle.load(open(gt_tree_path, 'rb'))
            self.total_time = tree.get_time(tree.leaves[0])

    def _spline_qdist(self, numstates):
        state_priors = pickle.load(open("/data/yosef2/users/richardz/projects/notebooks/full_indel_dist.pkl", "rb"))
        tck = interpolate.splrep(list(state_priors.keys()), list(state_priors.values()))
        bins = []
        for i in range(numstates + 1):
            increment = len(state_priors)//numstates
            bins.append(increment * i)

        vals = []
        for i in range(len(bins) - 1):
            vals.append(np.mean(interpolate.splev(list(range(bins[i], bins[i + 1])), tck)))

        total = sum(vals)
        down_sampled_vals = [i/total for i in vals]
        print(f'Using spling qdist with {numstates} states.')

        return down_sampled_vals

    def get_dissimilarity_map(
        self, 
        cassiopeia_tree: CassiopeiaTree,
        layer: Optional[str] = None
    ) -> pd.DataFrame: 

        # Estimating parameters
        cm = cassiopeia_tree.character_matrix
        cm = cm.replace(-2, -1)

        # Getting Oracle Params if Available
        if self.numstates is None:
            numstates = cm.max().max()
        else:
            numstates = self.numstates

        if self.state_distribution is None:
            state_distribution = self._spline_qdist(numstates)
        else:
            state_distribution = self.state_distribution

        if self.mut_prop is None:
            mut_prop = np.count_nonzero(cm.replace(-1, 0)) / np.count_nonzero(cm+1)
        else:
            mut_prop = self.mut_prop

        if self.total_time is None:
            total_time = 1
        else:
            total_time = self.total_time

        # Set up the iwhd dissimilarity function
        print(f'mut{mut_prop}, states{numstates}, time{total_time}')
        self.dissimilarity_function = IWHD(
            state_distribution=state_distribution,
            mut_prop=mut_prop,
            total_time=total_time
            )

        # Get the dissimilarity map
        self.setup_dissimilarity_map(cassiopeia_tree, layer)
        dissimilarity_map = cassiopeia_tree.get_dissimilarity_map()


        return dissimilarity_map