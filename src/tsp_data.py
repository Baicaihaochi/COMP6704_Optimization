"""
TSP Data Loading and Distance Calculation with TSPLIB Rounding Convention
"""
import numpy as np
import math
from typing import Dict, Tuple, List


class TSPData:
    """Handle TSPLIB format TSP data with rounded Euclidean distances."""

    def __init__(self, filepath: str):
        """Load TSP data from TSPLIB format file.

        Args:
            filepath: Path to .tsp file
        """
        self.filepath = filepath
        self.name = ""
        self.dimension = 0
        self.coordinates: Dict[int, Tuple[float, float]] = {}
        self.distance_matrix: np.ndarray = None

        self._load_data()
        self._compute_distance_matrix()

    def _load_data(self):
        """Parse TSPLIB format file."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        reading_coords = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Remove inline comments (after '#')
            if '#' in line:
                line = line.split('#')[0].strip()

            if not line:
                continue

            if line.startswith('NAME'):
                self.name = line.split(':')[1].strip()
            elif line.startswith('DIMENSION'):
                # Extract just the number, ignore any trailing text
                dim_part = line.split(':')[1].strip().split()[0]
                self.dimension = int(dim_part)
            elif line == 'NODE_COORD_SECTION':
                reading_coords = True
            elif line == 'EOF':
                break
            elif reading_coords:
                parts = line.split()
                if len(parts) >= 3:
                    city_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    self.coordinates[city_id] = (x, y)

    def _compute_distance_matrix(self):
        """Compute distance matrix with TSPLIB rounding: floor(euclidean + 0.5)."""
        n = self.dimension
        self.distance_matrix = np.zeros((n, n), dtype=int)

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i == j:
                    self.distance_matrix[i-1][j-1] = 0
                else:
                    xi, yi = self.coordinates[i]
                    xj, yj = self.coordinates[j]
                    euclidean = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                    # TSPLIB rounding: floor(d + 0.5) = round with ties to +inf
                    rounded = math.floor(euclidean + 0.5)
                    self.distance_matrix[i-1][j-1] = rounded

    def get_distance(self, i: int, j: int) -> int:
        """Get distance between cities i and j (1-indexed).

        Args:
            i, j: City indices (1-based)

        Returns:
            Integer distance
        """
        return self.distance_matrix[i-1][j-1]

    def calculate_tour_length(self, tour: List[int]) -> int:
        """Calculate total length of a tour.

        Args:
            tour: List of city indices (1-based), does not need to repeat first city

        Returns:
            Total tour length
        """
        length = 0
        n = len(tour)
        for k in range(n):
            i = tour[k]
            j = tour[(k + 1) % n]
            length += self.get_distance(i, j)
        return length

    def verify_known_optimum(self) -> Tuple[int, bool]:
        """Verify the known optimal tour for Berlin52.

        Returns:
            (tour_length, is_optimal)
        """
        # Known optimal tour from berlin52.tsp comments
        optimal_tour = [
            1, 49, 32, 45, 19, 41, 8, 9, 10, 43, 33, 51, 11, 52, 14, 13, 47,
            26, 27, 28, 12, 25, 4, 6, 15, 5, 24, 48, 38, 37, 40, 39, 36, 35,
            34, 44, 46, 16, 29, 50, 20, 23, 30, 2, 7, 42, 21, 17, 3, 18, 31, 22
        ]

        length = self.calculate_tour_length(optimal_tour)
        is_optimal = (length == 7542)

        return length, is_optimal

    def __repr__(self):
        return f"TSPData(name={self.name}, dimension={self.dimension})"
