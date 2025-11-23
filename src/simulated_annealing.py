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
                 cooling_rate: float = 0.995,
                 moves_per_temp: int = None,
                 min_temp: float = 1e-3,
                 max_no_improve: int = 15):
        """
        Args:
            distance_matrix: n x n distance matrix (0-indexed)
            initial_temp: Initial temperature (auto-computed if None)
            cooling_rate: Cooling rate alpha (T_new = alpha * T_old), default 0.995
            moves_per_temp: Number of moves per temperature level (default: 50*n)
            min_temp: Minimum temperature threshold
            max_no_improve: Stop if no improvement for N temperature levels
        """
        self.dist = distance_matrix
        self.n = len(distance_matrix)
        self.cooling_rate = cooling_rate
        self.moves_per_temp = moves_per_temp if moves_per_temp else 50 * self.n
        self.min_temp = min_temp
        self.max_no_improve = max_no_improve
        self.initial_temp = initial_temp

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
            all_deltas.append(abs(delta))
            if delta < 0:  # Cost increase
                positive_deltas.append(-delta)

        if not positive_deltas or len(positive_deltas) < 10:
            # If starting from good solution (like NN), use average absolute delta
            if all_deltas:
                avg_delta = np.mean(all_deltas)
                # Set T0 high enough to accept 80% of average moves initially
                # P = exp(-delta/T) = 0.8 => T = -delta / ln(0.8)
                T0 = avg_delta / (-math.log(0.8))
                return max(T0, 100.0)  # At least 100
            return 100.0  # Fallback

        # Set T0 so that 70% of median positive delta is accepted initially
        # P = exp(-delta/T) = 0.7 => T = -delta / ln(0.7)
        median_delta = np.median(positive_deltas)
        T0 = median_delta / (-math.log(0.7))

        # Ensure reasonable range
        return max(T0, 50.0)

    def anneal(self) -> dict:
        """Run simulated annealing."""
        start_time = time.time()

        # Initialize tour
        current_tour = self.generate_initial_tour(method='nn')
        current_length = self.tour_length(current_tour)

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
