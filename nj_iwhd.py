from typing import Optional
from cassiopeia.data.CassiopeiaTree import CassiopeiaTree # type: ignore
from cassiopeia.solver.dissimilarity_functions import weighted_hamming_distance # type: ignore
from cassiopeia.solver import NeighborJoiningSolver # type: ignore
import pandas as pd
import numpy as np

class IWHD:
    # Copy this class to dissimilarity_functions.py
    def __init__(
        self, 
        numstates,
        mut_prop,
        total_time=1): 

        self.total_time = total_time

        # Getting q
        state_distribution = dict(enumerate([1 / numstates] * numstates, 1))
        self.q = np.sum(np.array([*state_distribution.values()])**2)

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
        whd = weighted_hamming_distance(s1, s2, missing_state_indicator, weights)
                
        return 2 * self._iwhd(
            mut_rate=self.mut_rate,     
            collision_rate=self.q, 
            whd=whd, 
            time=self.total_time, 
            error_tolerance=0.001
        )

class InverseNJSolver(NeighborJoiningSolver):
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

        # Set up the iwhd dissimilarity function
        self.dissimilarity_function = IWHD(
            numstates=numstates,
            mut_prop=mut_prop
            )

        # Get the dissimilarity map
        self.setup_dissimilarity_map(cassiopeia_tree, layer)
        dissimilarity_map = cassiopeia_tree.get_dissimilarity_map()


        return dissimilarity_map