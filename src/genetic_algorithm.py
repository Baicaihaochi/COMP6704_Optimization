"""
Method 3: Genetic Algorithm with Permutation Encoding
"""
import numpy as np
from typing import List, Tuple, Callable
import random
import time


class GeneticAlgorithm:
    """Genetic Algorithm for TSP with permutation encoding."""

    def __init__(self,
                 distance_matrix: np.ndarray,
                 population_size: int = 100,
                 generations: int = 500,
                 tournament_size: int = 3,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.2,
                 elitism: int = 2,
                 memetic_2opt: bool = True,
                 memetic_fraction: float = 0.1,
                 memetic_frequency: int = 10,
                 early_stop: int = 50):
        """
        Args:
            distance_matrix: n x n distance matrix (0-indexed)
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            tournament_size: Tournament size for selection
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism: Number of best individuals to preserve
            memetic_2opt: Whether to apply 2-opt to top individuals
            memetic_fraction: Fraction of population to apply 2-opt
            memetic_frequency: Apply 2-opt every N generations
            early_stop: Stop if no improvement for N generations
        """
        self.dist = distance_matrix
        self.n = len(distance_matrix)
        self.pop_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.memetic_2opt = memetic_2opt
        self.memetic_fraction = memetic_fraction
        self.memetic_frequency = memetic_frequency
        self.early_stop = early_stop

        # Statistics tracking
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_length': [],
            'generation': []
        }

    def tour_length(self, tour: List[int]) -> int:
        """Calculate tour length."""
        length = 0
        n = len(tour)
        for k in range(n):
            length += self.dist[tour[k]][tour[(k + 1) % n]]
        return length

    def fitness(self, tour: List[int]) -> float:
        """Calculate fitness (inverse of tour length)."""
        return 1.0 / self.tour_length(tour)

    def initialize_population(self) -> List[List[int]]:
        """Create initial population with random permutations."""
        population = []
        base_tour = list(range(self.n))

        for _ in range(self.pop_size):
            tour = base_tour.copy()
            random.shuffle(tour)
            population.append(tour)

        return population

    def tournament_selection(self, population: List[List[int]], fitnesses: List[float]) -> List[int]:
        """Select individual using tournament selection."""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
        return population[winner_idx]

    def order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order Crossover (OX) operator.

        Copies a segment from parent1, fills remaining from parent2 in order.
        """
        size = len(parent1)

        # Select random segment
        start, end = sorted(random.sample(range(size), 2))

        # Create child with segment from parent1
        child = [-1] * size
        child[start:end] = parent1[start:end]

        # Fill remaining positions from parent2
        child_pos = end
        for city in parent2[end:] + parent2[:end]:
            if city not in child:
                child[child_pos % size] = city
                child_pos += 1

        return child

    def inversion_mutation(self, tour: List[int]) -> List[int]:
        """Inversion mutation: reverse a random subsequence."""
        size = len(tour)
        i, j = sorted(random.sample(range(size), 2))
        mutated = tour.copy()
        mutated[i:j+1] = reversed(mutated[i:j+1])
        return mutated

    def swap_mutation(self, tour: List[int]) -> List[int]:
        """Swap mutation: swap two random cities."""
        mutated = tour.copy()
        i, j = random.sample(range(len(tour)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    def two_opt_local_search(self, tour: List[int], max_iterations: int = 50) -> List[int]:
        """Apply quick 2-opt local search (limited iterations)."""
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
                        # Perform swap
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
                        break
                if improved:
                    break

        return tour

    def evolve(self) -> dict:
        """Run genetic algorithm evolution."""
        start_time = time.time()

        # Initialize population
        population = self.initialize_population()

        best_ever_tour = None
        best_ever_length = float('inf')
        generations_without_improvement = 0

        for gen in range(self.generations):
            # Evaluate fitness
            fitnesses = [self.fitness(ind) for ind in population]
            lengths = [self.tour_length(ind) for ind in population]

            # Track best
            best_idx = lengths.index(min(lengths))
            best_length = lengths[best_idx]
            best_tour = population[best_idx]

            if best_length < best_ever_length:
                best_ever_length = best_length
                best_ever_tour = best_tour.copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Record statistics
            self.history['generation'].append(gen)
            self.history['best_length'].append(best_ever_length)
            self.history['best_fitness'].append(max(fitnesses))
            self.history['avg_fitness'].append(np.mean(fitnesses))

            # Early stopping
            if generations_without_improvement >= self.early_stop:
                break

            # Sort population by fitness
            sorted_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
            sorted_population = [population[i] for i in sorted_indices]

            # Elitism: preserve best individuals
            new_population = sorted_population[:self.elitism]

            # Generate offspring
            while len(new_population) < self.pop_size:
                # Selection
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)

                # Crossover
                if random.random() < self.crossover_rate:
                    child = self.order_crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                # Mutation
                if random.random() < self.mutation_rate:
                    if random.random() < 0.5:
                        child = self.inversion_mutation(child)
                    else:
                        child = self.swap_mutation(child)

                new_population.append(child)

            population = new_population

            # Memetic: apply 2-opt to top individuals
            if self.memetic_2opt and (gen + 1) % self.memetic_frequency == 0:
                num_to_improve = max(1, int(self.memetic_fraction * self.pop_size))
                for i in range(num_to_improve):
                    population[i] = self.two_opt_local_search(population[i])

        total_time = time.time() - start_time

        # Convert to 1-based
        tour_1based = [city + 1 for city in best_ever_tour]

        return {
            'tour': tour_1based,
            'length': best_ever_length,
            'generations': gen + 1,
            'total_time': total_time,
            'history': self.history,
            'method': 'GeneticAlgorithm'
        }

    def solve(self, num_runs: int = 1) -> dict:
        """Run GA multiple times and return best solution.

        Args:
            num_runs: Number of independent runs

        Returns:
            dict with best solution and statistics
        """
        if num_runs == 1:
            return self.evolve()

        best_solution = None
        best_length = float('inf')
        all_solutions = []

        for run in range(num_runs):
            # Reset history
            self.history = {
                'best_fitness': [],
                'avg_fitness': [],
                'best_length': [],
                'generation': []
            }

            solution = self.evolve()
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
