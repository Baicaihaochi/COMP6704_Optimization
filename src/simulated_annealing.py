"""
Method 4: Simulated Annealing with 2-opt Neighborhood
"""
import numpy as np
from typing import List, Tuple
import random
import math
import time


class SimulatedAnnealing:
    """Simulated Annealing for TSP using 2-opt moves."""

    def __init__(self,
                 distance_matrix: np.ndarray,
                 initial_temp: float = None,
                 cooling_rate: float = 0.99,
                 moves_per_temp: int = None,
                 min_temp: float = 1e-3,
                 max_no_improve: int = 20,
                 initial_acceptance: float = 0.5):
        """
        Args:
            distance_matrix: n x n distance matrix (0-indexed)
            initial_temp: Initial temperature (auto-computed if None)
            cooling_rate: Cooling rate alpha (T_new = alpha * T_old), default 0.99
            moves_per_temp: Number of moves per temperature level (default: 100*n)
            min_temp: Minimum temperature threshold
            max_no_improve: Stop if no improvement for N temperature levels
            initial_acceptance: Target probability of accepting uphill moves when
                estimating the starting temperature
        """
        self.dist = distance_matrix
        self.n = len(distance_matrix)
        self.cooling_rate = cooling_rate
        self.moves_per_temp = moves_per_temp if moves_per_temp else 100 * self.n
        self.min_temp = min_temp
        self.max_no_improve = max_no_improve
        self.initial_temp = initial_temp
        self.initial_acceptance = max(0.05, min(0.95, initial_acceptance))

        # Statistics tracking
        self.history = {
            'temperature': [],
            'best_length': [],
            'current_length': [],
            'acceptance_rate': [],
            'iteration': []
        }

    def tour_length(self, tour: List[int]) -> int:
        """Calculate tour length."""
        length = 0
        n = len(tour)
        for k in range(n):
            length += self.dist[tour[k]][tour[(k + 1) % n]]
        return length

    def generate_initial_tour(self, method: str = 'nn') -> List[int]:
        """Generate initial tour.

        Args:
            method: 'random' or 'nn' (nearest neighbor)

        Returns:
            Initial tour (0-indexed)
        """
        if method == 'random':
            tour = list(range(self.n))
            random.shuffle(tour)
            return tour
        elif method == 'nn':
            # Simple nearest neighbor from city 0
            tour = [0]
            unvisited = set(range(1, self.n))
            current = 0

            while unvisited:
                nearest = min(unvisited, key=lambda j: self.dist[current][j])
                tour.append(nearest)
                unvisited.remove(nearest)
                current = nearest

            return tour

    def two_opt_neighbor(self, tour: List[int]) -> Tuple[List[int], int]:
        """Generate random 2-opt neighbor and compute delta.

        Args:
            tour: Current tour

        Returns:
            (new_tour, delta) where delta = old_length - new_length
        """
        n = len(tour)
        i, j = sorted(random.sample(range(n), 2))

        # Ensure i < j and j is not n-1 followed by i=0 (no-op case)
        if j == i + 1:
            j = (j + 1) % n
            if j <= i:
                i, j = j, i

        # Calculate delta without actually modifying tour
        i_next = (i + 1) % n
        j_next = (j + 1) % n

        old_cost = self.dist[tour[i]][tour[i_next]] + self.dist[tour[j]][tour[j_next]]
        new_cost = self.dist[tour[i]][tour[j]] + self.dist[tour[i_next]][tour[j_next]]

        delta = old_cost - new_cost

        # Create new tour if needed
        new_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]

        return new_tour, delta

    def _two_opt_delta(self, tour: List[int], i: int, j: int) -> int:
        """Compute 2-opt delta for fixed breakpoints."""
        n = len(tour)
        i_next = (i + 1) % n
        j_next = (j + 1) % n
        old_cost = self.dist[tour[i]][tour[i_next]] + self.dist[tour[j]][tour[j_next]]
        new_cost = self.dist[tour[i]][tour[j]] + self.dist[tour[i_next]][tour[j_next]]
        return old_cost - new_cost

    def _local_two_opt_improvement(self, tour: List[int], max_passes: int = None) -> Tuple[List[int], int]:
        """Apply first-improvement 2-opt passes to get a stronger starting point."""
        current_tour = tour
        current_length = self.tour_length(current_tour)
        if max_passes is None:
            max_passes = max(5, min(20, self.n // 2))
        passes = 0
        improved = True

        while improved and passes < max_passes:
            improved = False
            passes += 1

            for i in range(self.n - 1):
                for j in range(i + 2, self.n):
                    if i == 0 and j == self.n - 1:
                        continue

                    delta = self._two_opt_delta(current_tour, i, j)
                    if delta > 0:
                        current_tour = current_tour[:i+1] + current_tour[i+1:j+1][::-1] + current_tour[j+1:]
                        current_length -= delta
                        improved = True
                        break

                if improved:
                    break

        return current_tour, current_length

    def compute_initial_temperature(self, initial_tour: List[int], samples: int = 500) -> float:
        """Compute initial temperature from sample of positive cost increases.

        Args:
            initial_tour: Starting tour
            samples: Number of samples to collect

        Returns:
            Initial temperature
        """
        positive_deltas = []
        all_deltas = []

        # Sample more moves to get better statistics
        for _ in range(samples):
            _, delta = self.two_opt_neighbor(initial_tour)
            abs_delta = abs(delta)
            if abs_delta == 0:
                continue
            all_deltas.append(abs_delta)
            if delta < 0:  # Cost increase
                positive_deltas.append(-delta)

        if positive_deltas:
            uphill = np.array(positive_deltas, dtype=float)
            median_uphill = float(np.median(uphill))
            upper_uphill = float(np.percentile(uphill, 75))
            delta_ref = 0.5 * (median_uphill + upper_uphill)
        elif all_deltas:
            delta_ref = float(np.median(all_deltas))
        else:
            return 200.0  # Fallback for degenerate cases

        # P(accept) = exp(-delta/T) => T = -delta / ln(P)
        T0 = delta_ref / max(1e-8, -math.log(self.initial_acceptance))

        # Clamp to avoid extremely hot or cold starts
        min_T0 = max(50.0, 0.5 * delta_ref)
        max_T0 = max(min_T0, 5.0 * delta_ref)
        return max(min_T0, min(max_T0, T0))

    def anneal(self) -> dict:
        """Run simulated annealing."""
        start_time = time.time()

        # Initialize tour - use NN + bounded 2-opt passes for a solid baseline
        current_tour = self.generate_initial_tour(method='nn')
        current_tour, current_length = self._local_two_opt_improvement(current_tour)

        best_tour = current_tour.copy()
        best_length = current_length

        # Set initial temperature
        if self.initial_temp is None:
            T = self.compute_initial_temperature(current_tour)
        else:
            T = self.initial_temp

        # SA loop
        iterations = 0
        temp_levels = 0
        no_improve_count = 0

        while T > self.min_temp and no_improve_count < self.max_no_improve:
            temp_levels += 1
            accepted_moves = 0
            improved_in_level = False

            for _ in range(self.moves_per_temp):
                iterations += 1

                # Generate neighbor
                new_tour, delta = self.two_opt_neighbor(current_tour)
                new_length = current_length - delta

                # Acceptance criterion
                if delta > 0:  # Improvement
                    accept = True
                else:  # Worse solution
                    probability = math.exp(delta / T)
                    accept = random.random() < probability

                if accept:
                    current_tour = new_tour
                    current_length = new_length
                    accepted_moves += 1

                    # Update best
                    if current_length < best_length:
                        best_tour = current_tour.copy()
                        best_length = current_length
                        improved_in_level = True

            # Record statistics
            acceptance_rate = accepted_moves / self.moves_per_temp
            self.history['temperature'].append(T)
            self.history['best_length'].append(best_length)
            self.history['current_length'].append(current_length)
            self.history['acceptance_rate'].append(acceptance_rate)
            self.history['iteration'].append(iterations)

            # Check improvement
            if improved_in_level:
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Cool down
            T *= self.cooling_rate

        # Final local refinement to make sure we return a 2-opt local optimum
        refinement_passes = max(2, min(10, self.n // 5))
        best_tour, best_length = self._local_two_opt_improvement(best_tour, max_passes=refinement_passes)

        total_time = time.time() - start_time

        # Convert to 1-based
        tour_1based = [city + 1 for city in best_tour]

        return {
            'tour': tour_1based,
            'length': best_length,
            'iterations': iterations,
            'temperature_levels': temp_levels,
            'total_time': total_time,
            'history': self.history,
            'method': 'SimulatedAnnealing'
        }

    def solve(self, num_runs: int = 1) -> dict:
        """Run SA multiple times and return best solution.

        Args:
            num_runs: Number of independent runs

        Returns:
            dict with best solution and statistics
        """
        if num_runs == 1:
            return self.anneal()

        best_solution = None
        best_length = float('inf')
        all_solutions = []

        for run in range(num_runs):
            # Reset history
            self.history = {
                'temperature': [],
                'best_length': [],
                'current_length': [],
                'acceptance_rate': [],
                'iteration': []
            }

            solution = self.anneal()
            all_solutions.append(solution)

            if solution['length'] < best_length:
                best_length = solution['length']
                best_solution = solution

        # Compute statistics across runs
        all_lengths = [sol['length'] for sol in all_solutions]

        return {
            'best_solution': best_solution,
            'all_solutions': all_solutions,
            'mean_length': np.mean(all_lengths),
            'std_length': np.std(all_lengths),
            'min_length': np.min(all_lengths),
            'max_length': np.max(all_lengths),
            'num_runs': num_runs
        }
