"""
Method 1: Nearest Neighbor Construction with 2-opt Local Search
"""
import numpy as np
from typing import List, Tuple
import time


class NearestNeighbor2Opt:
    """Nearest Neighbor heuristic with 2-opt improvement."""

    def __init__(self, distance_matrix: np.ndarray, max_passes: int = 100):
        """
        Args:
            distance_matrix: n x n distance matrix (0-indexed)
            max_passes: Maximum number of 2-opt improvement passes
        """
        self.dist = distance_matrix
        self.n = len(distance_matrix)
        self.max_passes = max_passes

    def nearest_neighbor(self, start: int = 0) -> List[int]:
        """Construct tour using Nearest Neighbor heuristic.

        Args:
            start: Starting city index (0-based)

        Returns:
            Tour as list of city indices (0-based)
        """
        tour = [start]
        unvisited = set(range(self.n))
        unvisited.remove(start)

        current = start
        while unvisited:
            # Find nearest unvisited city
            nearest = min(unvisited, key=lambda j: self.dist[current][j])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        return tour

    def tour_length(self, tour: List[int]) -> int:
        """Calculate tour length."""
        length = 0
        n = len(tour)
        for k in range(n):
            length += self.dist[tour[k]][tour[(k + 1) % n]]
        return length

    def two_opt_swap(self, tour: List[int], i: int, j: int) -> List[int]:
        """Perform 2-opt swap: reverse tour[i+1:j+1].

        Args:
            tour: Current tour
            i, j: Indices for 2-opt swap (i < j)

        Returns:
            New tour after swap
        """
        new_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
        return new_tour

    def two_opt(self, tour: List[int], first_improvement: bool = True) -> Tuple[List[int], int]:
        """Apply 2-opt local search.

        Args:
            tour: Initial tour
            first_improvement: If True, accept first improving move; else best improvement

        Returns:
            (improved_tour, number_of_passes)
        """
        improved = True
        passes = 0

        while improved and passes < self.max_passes:
            improved = False
            passes += 1

            for i in range(self.n - 1):
                for j in range(i + 2, self.n):
                    # Calculate improvement delta
                    # Current edges: (tour[i], tour[i+1]) and (tour[j], tour[j+1])
                    # New edges: (tour[i], tour[j]) and (tour[i+1], tour[j+1])
                    i_next = (i + 1) % self.n
                    j_next = (j + 1) % self.n

                    current_cost = (self.dist[tour[i]][tour[i_next]] +
                                    self.dist[tour[j]][tour[j_next]])
                    new_cost = (self.dist[tour[i]][tour[j]] +
                                self.dist[tour[i_next]][tour[j_next]])

                    delta = current_cost - new_cost

                    if delta > 0:  # Improvement found
                        tour = self.two_opt_swap(tour, i, j)
                        improved = True
                        if first_improvement:
                            break

                if improved and first_improvement:
                    break

        return tour, passes

    def solve(self, start_city: int = 0, use_2opt: bool = True) -> dict:
        """Solve TSP using NN + 2-opt.

        Args:
            start_city: Starting city (0-based)
            use_2opt: Whether to apply 2-opt improvement

        Returns:
            dict with solution details
        """
        start_time = time.time()

        # Nearest Neighbor construction
        nn_tour = self.nearest_neighbor(start_city)
        nn_length = self.tour_length(nn_tour)
        construction_time = time.time() - start_time

        # 2-opt improvement
        if use_2opt:
            opt_start = time.time()
            final_tour, passes = self.two_opt(nn_tour)
            final_length = self.tour_length(final_tour)
            improvement_time = time.time() - opt_start
        else:
            final_tour = nn_tour
            final_length = nn_length
            improvement_time = 0
            passes = 0

        total_time = time.time() - start_time

        # Convert to 1-based indexing for output
        tour_1based = [city + 1 for city in final_tour]

        return {
            'tour': tour_1based,
            'length': final_length,
            'nn_length': nn_length,
            'improvement': nn_length - final_length,
            'improvement_pct': 100 * (nn_length - final_length) / nn_length if nn_length > 0 else 0,
            'passes': passes,
            'construction_time': construction_time,
            'improvement_time': improvement_time,
            'total_time': total_time,
            'start_city': start_city + 1  # 1-based
        }

    def multi_start(self, num_starts: int = None) -> dict:
        """Run NN+2-opt from multiple start cities.

        Args:
            num_starts: Number of different start cities (default: all cities)

        Returns:
            dict with best solution
        """
        if num_starts is None:
            num_starts = self.n

        best_solution = None
        best_length = float('inf')
        all_solutions = []

        for start in range(min(num_starts, self.n)):
            solution = self.solve(start_city=start, use_2opt=True)
            all_solutions.append(solution)

            if solution['length'] < best_length:
                best_length = solution['length']
                best_solution = solution

        return {
            'best_solution': best_solution,
            'all_solutions': all_solutions,
            'num_starts': num_starts
        }
