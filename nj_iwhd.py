import pickle
from typing import List, Optional
from cassiopeia.data.CassiopeiaTree import CassiopeiaTree 
from cassiopeia import solver
import pandas as pd
import numpy as np
from scipy import interpolate


class IWHD:
    def __init__(
        self, 
        state_distribution: List[float],
        mut_prop: float,
        total_time: float=1): 

        self.total_time = total_time
        self.q = np.sum(np.array([*state_distribution])**2)
        self.mut_rate = np.log(1 - mut_prop)/(-1 * self.total_time) #type: ignore

    def _ewhd_given_h(
        self, 
        mut_rate: float, 
        collision_rate: float, 
        height: float, 
        time: float):
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
    """A Neighbor Joining solver that uses the inverse weighted hamming distance
    as the dissimilarity function.
    """
    def _spline_qdist(self, numstates: int) -> List:
        """Get an empirical state distribution.

        Args:
            numstates (int): = number of states in the character matrix.

        Returns:
            List: A list of floats representing the state distribution.
        """
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

        return down_sampled_vals

    def set_numstates(self, numstates: int) -> None:
        """Set the number of states for the inverse weighted hamming distance function.

        Args:
            numstates (int): = number of states 
        """
        self.numstates: int = numstates
        
    def set_mut_prop(self, mut_prop: float) -> None:
        """Set the mutation proportion for the inverse weighted hamming distance function.

        Args:
            mut_prop (float): = mutation proportion
        """
        self.mut_prop: float = mut_prop

    def set_total_time(self, total_time: float) -> None:
        """Set the total time of a tree for the inverse weighted hamming distance function.

        Args:
            total_time (float): = total time of a tree
        """
        self.total_time: float = total_time

    def set_state_distribution(self, state_distribution: List) -> None:
        """Set the state distribution for the inverse weighted hamming distance function.

        Args:
            state_distribution (List): = state distribution
        """
        self.state_distribution: List = state_distribution

    def get_dissimilarity_map(
        self, 
        cassiopeia_tree: CassiopeiaTree,
        layer: Optional[str] = None
        ) -> pd.DataFrame: 
        """Get the dissimilarity map for the given tree. This method overrides the same method of the super class.
        """
        # Estimating parameters
        cm = cassiopeia_tree.character_matrix
        cm = cm.replace(-2, -1)

        # Parameter Estimation
        if hasattr(self, "numstates"):
            numstates = self.numstates
            del self.numstates

        else:
            numstates = cm.max().max()

        if hasattr(self, "mut_prop"):
            mut_prop = self.mut_prop
            del self.mut_prop
        else:
            mut_prop = np.count_nonzero(cm.replace(-1, 0)) / np.count_nonzero(cm+1)

        if hasattr(self, "total_time"):
            total_time = self.total_time
            del self.total_time
        else:
            total_time = 1

        if hasattr(self, "state_distribution"):
            state_distribution = self.state_distribution
            del self.state_distribution
        else:
            try:
                state_distribution = self._spline_qdist(numstates)
            except:
                state_distribution = [1 / numstates] * numstates

        # Set up the iwhd dissimilarity function
        self.dissimilarity_function = IWHD(
            state_distribution=state_distribution,
            mut_prop=mut_prop,
            total_time=total_time
            )

        # Get the dissimilarity map
        self.setup_dissimilarity_map(cassiopeia_tree, layer)
        dissimilarity_map = cassiopeia_tree.get_dissimilarity_map()

        return dissimilarity_map
        