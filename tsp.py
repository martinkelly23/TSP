from itertools import permutations
from typing import Any, List, Optional, Tuple

import numpy as np


class TravelingSalesmanProblemSolver():
    
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
    
    def solve_brute_force(self) -> Tuple[Optional[List], Any]:
        """Solve TSP to optimality with a brute force approach
        Parameters
        ----------
        distance_matrix
            Distance matrix of shape (n x n) with the (i, j) entry indicating the
            distance from node i to j. It does not need to be symmetric
        Returns
        -------
        A permutation of nodes from 0 to n that produces the least total
        distance
        The total distance the optimal permutation produces
        Notes
        ----
        The algorithm checks all permutations and returns the one with smallest
        distance. In principle, the total number of possibilities would be n! for
        n nodes. However, we can fix node 0 and permutate only the remaining,
        reducing the possibilities to (n - 1)!.
        """

        # Exclude 0 from the range since it is fixed as starting point
        points = range(1, self.distance_matrix.shape[0])
        best_distance = np.inf
        best_permutation = None

        for partial_permutation in permutations(points):
            # Add the starting node before evaluating it
            permutation = [0] + list(partial_permutation)
            distance = self._compute_permutation_distance(permutation)

            if distance < best_distance:
                best_distance = distance
                best_permutation = permutation

        return best_permutation, best_distance
    
    
    def solve_brute_force_fixed_start_and_end(self) -> Tuple[Optional[List], Any]:
        last_point = (self.distance_matrix.shape[0] - 1)
        # Exclude first and last points since are the start and end
        points = range(1, last_point)
        best_distance = np.inf
        best_permutation = None

        for partial_permutation in permutations(points):
            # Add the starting and ending nodes before evaluating it
            permutation = [0] + list(partial_permutation) + [last_point]
            print("Computing permutation: ", permutation)
            distance = self._compute_permutation_distance(permutation)

            if distance < best_distance:
                best_distance = distance
                best_permutation = permutation

        return best_permutation, best_distance
    
    
    def _compute_permutation_distance(self, permutation: List[int]) -> float:
        """Compute the total route distance of a given permutation
        Parameters
        ----------
        distance_matrix
            Distance matrix of shape (n x n) with the (i, j) entry indicating the
            distance from node i to j. It does not need to be symmetric
        permutation
            A list with nodes from 0 to n - 1 in any order
        Returns
        -------
        Total distance of the path given in ``permutation`` for the provided
        ``distance_matrix``
        Notes
        -----
        Suppose the permutation [0, 1, 2, 3], with four nodes. The total distance
        of this path will be from 0 to 1, 1 to 2, 2 to 3, and 3 back to 0. This
        can be fetched from a distance matrix using:
            distance_matrix[ind1, ind2], where
            ind1 = [0, 1, 2, 3]  # the FROM nodes
            ind2 = [1, 2, 3, 0]  # the TO nodes
        This can easily be generalized to any permutation by using ind1 as the
        given permutation, and moving the first node to the end to generate ind2.
        """
        ind1 = permutation
        ind2 = permutation[1:] + permutation[:1]
        return self.distance_matrix[ind1, ind2].sum()