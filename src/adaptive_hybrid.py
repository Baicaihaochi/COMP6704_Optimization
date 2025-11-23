"""
Method 5: Adaptive Multi-Strategy Hybrid (AMSH) - Novel Approach

This method combines multiple strategies in an adaptive framework:
1. Population-based search with diversity maintenance
2. Adaptive operator selection based on performance
3. Multiple neighborhood structures (2-opt, 3-opt, Or-opt)
4. Periodic intensification and diversification phases
5. Solution pool management with quality-diversity balance

Key Innovation: Unlike traditional hybrid methods that use fixed operator
probabilities, AMSH dynamically adjusts the probability of applying each
operator based on their recent success rates, similar to adaptive large
neighborhood search (ALNS) but for metaheuristic operators.
"""
import numpy as np
from typing import List, Tuple, Dict
import random
import time
from collections import deque


class AdaptiveMultiStrategyHybrid:
    """Novel adaptive hybrid TSP solver combining multiple strategies."""

    def __init__(self,
                 distance_matrix: np.ndarray,
                 pool_size: int = 10,
                 iterations: int = 5000,
                 intensification_freq: int = 100,
                 diversification_freq: int = 500,
                 learning_rate: float = 0.1,
                 min_operator_prob: float = 0.05):
        """
        Args:
            distance_matrix: n x n distance matrix (0-indexed)
            pool_size: Size of solution pool
            iterations: Total number of iterations
            intensification_freq: Frequency of intensification phase
            diversification_freq: Frequency of diversification phase
            learning_rate: Learning rate for operator weight adaptation
            min_operator_prob: Minimum probability for each operator
        """
        self.dist = distance_matrix
        self.n = len(distance_matrix)
        self.pool_size = pool_size
        self.iterations = iterations
        self.intensification_freq = intensification_freq
        self.diversification_freq = diversification_freq
        self.learning_rate = learning_rate
        self.min_operator_prob = min_operator_prob

        # Operator weights (dynamically adjusted)
        self.operators = {
            '2opt': {'weight': 1.0, 'successes': 0, 'attempts': 0},
            '3opt': {'weight': 1.0, 'successes': 0, 'attempts': 0},
            'or_opt': {'weight': 1.0, 'successes': 0, 'attempts': 0},
            'swap': {'weight': 1.0, 'successes': 0, 'attempts': 0},
            'insert': {'weight': 1.0, 'successes': 0, 'attempts': 0}
        }

        # History tracking
        self.history = {
            'best_length': [],
            'pool_diversity': [],
            'operator_probs': {op: [] for op in self.operators},
            'iteration': []
        }

    def tour_length(self, tour: List[int]) -> int:
        """Calculate tour length."""
        length = 0
        n = len(tour)
        for k in range(n):
            length += self.dist[tour[k]][tour[(k + 1) % n]]
        return length

    def tour_diversity(self, tour1: List[int], tour2: List[int]) -> float:
        """Compute diversity between two tours (edge-based)."""
        edges1 = set()
        edges2 = set()

        for k in range(len(tour1)):
            i, j = tour1[k], tour1[(k + 1) % len(tour1)]
            edges1.add((min(i, j), max(i, j)))

        for k in range(len(tour2)):
            i, j = tour2[k], tour2[(k + 1) % len(tour2)]
            edges2.add((min(i, j), max(i, j)))

        # Jaccard distance
        intersection = len(edges1 & edges2)
        union = len(edges1 | edges2)
        return 1.0 - (intersection / union if union > 0 else 0)

    def initialize_pool(self) -> List[Tuple[List[int], int]]:
        """Initialize solution pool with diverse tours."""
        pool = []

        for start in range(min(self.pool_size * 2, self.n)):
            # Nearest neighbor from different starts
            tour = self.nearest_neighbor(start)
            # Quick 2-opt
            tour = self.apply_2opt(tour, max_iterations=10)
            length = self.tour_length(tour)
            pool.append((tour, length))

        # Sort by quality
        pool.sort(key=lambda x: x[1])

        # Select diverse subset
        final_pool = [pool[0]]
        for tour, length in pool[1:]:
            if len(final_pool) >= self.pool_size:
                break
            # Check diversity
            min_diversity = min(self.tour_diversity(tour, existing[0]) for existing in final_pool)
            if min_diversity > 0.1:  # Sufficient diversity threshold
                final_pool.append((tour, length))

        # Fill remaining slots with random tours
        while len(final_pool) < self.pool_size:
            tour = list(range(self.n))
            random.shuffle(tour)
            tour = self.apply_2opt(tour, max_iterations=5)
            length = self.tour_length(tour)
            final_pool.append((tour, length))

        return final_pool

    def nearest_neighbor(self, start: int) -> List[int]:
        """Nearest neighbor construction."""
        tour = [start]
        unvisited = set(range(self.n))
        unvisited.remove(start)
        current = start

        while unvisited:
            nearest = min(unvisited, key=lambda j: self.dist[current][j])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        return tour

    def apply_2opt(self, tour: List[int], max_iterations: int = 50) -> List[int]:
        """Apply 2-opt local search."""
        improved = True
        iterations = 0

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            for i in range(self.n - 1):
                for j in range(i + 2, self.n):
                    i_next = (i + 1) % self.n
                    j_next = (j + 1) % self.n

                    delta = (self.dist[tour[i]][tour[i_next]] + self.dist[tour[j]][tour[j_next]]) - \
                            (self.dist[tour[i]][tour[j]] + self.dist[tour[i_next]][tour[j_next]])

                    if delta > 0:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
                        break
                if improved:
                    break

        return tour

    def apply_3opt(self, tour: List[int]) -> Tuple[List[int], int]:
        """Apply single random 3-opt move."""
        n = len(tour)
        i, j, k = sorted(random.sample(range(n), 3))

        # Calculate delta for one 3-opt case (there are multiple)
        # Current: i->i+1, j->j+1, k->k+1
        # New: i->j, i+1->k, j+1->k+1
        i_next = (i + 1) % n
        j_next = (j + 1) % n
        k_next = (k + 1) % n

        old_cost = (self.dist[tour[i]][tour[i_next]] +
                    self.dist[tour[j]][tour[j_next]] +
                    self.dist[tour[k]][tour[k_next]])

        # One of the 3-opt reconnections
        new_tour = tour[:i+1] + tour[j+1:k+1] + tour[i+1:j+1] + tour[k+1:]
        new_cost = self.tour_length(new_tour)

        delta = self.tour_length(tour) - new_cost

        return new_tour, delta

    def apply_or_opt(self, tour: List[int]) -> Tuple[List[int], int]:
        """Or-opt: relocate a sequence of 1-3 cities."""
        n = len(tour)
        seq_len = random.randint(1, 3)

        i = random.randint(0, n - seq_len)
        j = random.randint(0, n - 1)

        if i <= j < i + seq_len:
            return tour, 0  # Invalid move

        new_tour = tour.copy()
        sequence = new_tour[i:i+seq_len]
        del new_tour[i:i+seq_len]

        insert_pos = j if j < i else j - seq_len
        new_tour[insert_pos:insert_pos] = sequence

        delta = self.tour_length(tour) - self.tour_length(new_tour)
        return new_tour, delta

    def apply_swap(self, tour: List[int]) -> Tuple[List[int], int]:
        """Swap two random cities."""
        new_tour = tour.copy()
        i, j = random.sample(range(len(tour)), 2)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        delta = self.tour_length(tour) - self.tour_length(new_tour)
        return new_tour, delta

    def apply_insert(self, tour: List[int]) -> Tuple[List[int], int]:
        """Remove and reinsert a city at best position."""
        new_tour = tour.copy()
        i = random.randint(0, len(tour) - 1)
        city = new_tour.pop(i)

        # Find best insertion position
        best_pos = 0
        best_cost = float('inf')

        for pos in range(len(new_tour)):
            test_tour = new_tour[:pos] + [city] + new_tour[pos:]
            cost = self.tour_length(test_tour)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos

        new_tour.insert(best_pos, city)
        delta = self.tour_length(tour) - self.tour_length(new_tour)
        return new_tour, delta

    def select_operator(self) -> str:
        """Select operator based on adaptive weights."""
        total_weight = sum(op['weight'] for op in self.operators.values())
        probs = {name: max(self.min_operator_prob, op['weight'] / total_weight)
                 for name, op in self.operators.items()}

        # Normalize
        total_prob = sum(probs.values())
        probs = {name: p / total_prob for name, p in probs.items()}

        return random.choices(list(probs.keys()), weights=list(probs.values()))[0]

    def update_operator_weights(self):
        """Update operator weights based on success rates."""
        for name, op in self.operators.items():
            if op['attempts'] > 0:
                success_rate = op['successes'] / op['attempts']
                # Reward successful operators
                op['weight'] = op['weight'] * (1 + self.learning_rate * success_rate)
                # Reset counters
                op['successes'] = 0
                op['attempts'] = 0

    def solve(self) -> dict:
        """Run adaptive multi-strategy hybrid algorithm."""
        start_time = time.time()

        # Initialize solution pool
        pool = self.initialize_pool()
        best_tour, best_length = min(pool, key=lambda x: x[1])

        iteration = 0
        while iteration < self.iterations:
            iteration += 1

            # Select solution from pool (biased towards better solutions)
            weights = [1.0 / (1 + i) for i in range(len(pool))]
            current_tour, current_length = random.choices(pool, weights=weights)[0]

            # Select and apply operator
            operator = self.select_operator()
            self.operators[operator]['attempts'] += 1

            if operator == '2opt':
                new_tour = self.apply_2opt(current_tour.copy(), max_iterations=5)
                delta = current_length - self.tour_length(new_tour)
            elif operator == '3opt':
                new_tour, delta = self.apply_3opt(current_tour)
            elif operator == 'or_opt':
                new_tour, delta = self.apply_or_opt(current_tour)
            elif operator == 'swap':
                new_tour, delta = self.apply_swap(current_tour)
            elif operator == 'insert':
                new_tour, delta = self.apply_insert(current_tour)

            new_length = self.tour_length(new_tour)

            # Update best
            if new_length < best_length:
                best_tour = new_tour.copy()
                best_length = new_length
                self.operators[operator]['successes'] += 1

            # Update pool
            if new_length < pool[-1][1] or self.tour_diversity(new_tour, pool[0][0]) > 0.15:
                pool.append((new_tour, new_length))
                pool.sort(key=lambda x: x[1])
                pool = pool[:self.pool_size]

            # Intensification phase
            if iteration % self.intensification_freq == 0:
                best_in_pool = pool[0][0].copy()
                improved_tour = self.apply_2opt(best_in_pool, max_iterations=100)
                improved_length = self.tour_length(improved_tour)
                if improved_length < best_length:
                    best_tour = improved_tour
                    best_length = improved_length
                pool[0] = (improved_tour, improved_length)

            # Diversification phase
            if iteration % self.diversification_freq == 0:
                # Replace worst solutions with random ones
                for i in range(len(pool) // 3, len(pool)):
                    random_tour = list(range(self.n))
                    random.shuffle(random_tour)
                    random_tour = self.apply_2opt(random_tour, max_iterations=10)
                    pool[i] = (random_tour, self.tour_length(random_tour))
                pool.sort(key=lambda x: x[1])

            # Update operator weights periodically
            if iteration % 100 == 0:
                self.update_operator_weights()

            # Record history
            if iteration % 10 == 0:
                pool_diversity = np.mean([
                    self.tour_diversity(pool[i][0], pool[j][0])
                    for i in range(len(pool))
                    for j in range(i+1, len(pool))
                ])
                self.history['best_length'].append(best_length)
                self.history['pool_diversity'].append(pool_diversity)
                self.history['iteration'].append(iteration)

                total_weight = sum(op['weight'] for op in self.operators.values())
                for name in self.operators:
                    prob = self.operators[name]['weight'] / total_weight
                    self.history['operator_probs'][name].append(prob)

        total_time = time.time() - start_time

        # Convert to 1-based
        tour_1based = [city + 1 for city in best_tour]

        return {
            'tour': tour_1based,
            'length': best_length,
            'iterations': iteration,
            'total_time': total_time,
            'history': self.history,
            'final_operator_weights': {name: op['weight'] for name, op in self.operators.items()},
            'method': 'AdaptiveMultiStrategyHybrid'
        }
