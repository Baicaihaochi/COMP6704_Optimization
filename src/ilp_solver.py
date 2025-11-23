"""
Method 2: Integer Linear Programming with Lazy Subtour Elimination Constraints
Uses Gurobi optimizer with callback-based lazy constraint separation
"""
import numpy as np
from typing import List, Tuple, Set
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("Warning: Gurobi not available. ILP solver will not work.")


class ILPLazySEC:
    """TSP solver using ILP with lazy subtour elimination constraints."""

    def __init__(self, distance_matrix: np.ndarray, time_limit: int = 900):
        """
        Args:
            distance_matrix: n x n distance matrix (0-indexed)
            time_limit: Time limit in seconds (default: 15 minutes)
        """
        if not GUROBI_AVAILABLE:
            raise RuntimeError("Gurobi is not installed. Install with: pip install gurobipy")

        self.dist = distance_matrix
        self.n = len(distance_matrix)
        self.time_limit = time_limit
        self.model = None
        self.x_vars = {}
        self.callback_count = 0

    def _build_model(self):
        """Build ILP model with degree constraints only (SEC added lazily)."""
        self.model = gp.Model("TSP_LazySEC")
        self.model.Params.OutputFlag = 1  # Show progress
        self.model.Params.TimeLimit = self.time_limit
        self.model.Params.LazyConstraints = 1  # Enable lazy constraints

        # Decision variables: x[i,j] = 1 if edge (i,j) is in tour
        # Only create variables for i < j (undirected)
        self.x_vars = {}
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.x_vars[i, j] = self.model.addVar(
                    vtype=GRB.BINARY,
                    name=f'x_{i}_{j}',
                    obj=self.dist[i][j]
                )

        # Objective: minimize total distance
        self.model.ModelSense = GRB.MINIMIZE

        # Degree constraints: each city has exactly 2 incident edges
        for i in range(self.n):
            edges = []
            for j in range(self.n):
                if i < j:
                    edges.append(self.x_vars[i, j])
                elif i > j:
                    edges.append(self.x_vars[j, i])
            self.model.addConstr(gp.quicksum(edges) == 2, name=f'degree_{i}')

        self.model.update()

    def _find_subtours(self, x_val: dict) -> List[List[int]]:
        """Find connected components (subtours) in current solution.

        Args:
            x_val: Dictionary mapping (i,j) to edge value (0 or 1)

        Returns:
            List of subtours, each subtour is a list of city indices
        """
        # Build adjacency list
        adj = {i: [] for i in range(self.n)}
        for (i, j), val in x_val.items():
            if val > 0.5:  # Edge is selected
                adj[i].append(j)
                adj[j].append(i)

        # Find connected components using DFS
        visited = set()
        subtours = []

        for start in range(self.n):
            if start in visited:
                continue

            # DFS to find component
            component = []
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)

            if component:
                subtours.append(sorted(component))

        return subtours

    def _subtour_callback(self, model, where):
        """Callback function to add subtour elimination constraints."""
        if where == GRB.Callback.MIPSOL:
            self.callback_count += 1

            # Get current solution
            x_val = {}
            for (i, j), var in self.x_vars.items():
                x_val[i, j] = model.cbGetSolution(var)

            # Find subtours
            subtours = self._find_subtours(x_val)

            # If more than one subtour, add lazy constraints
            if len(subtours) > 1:
                for subtour in subtours:
                    if len(subtour) < self.n:  # Not the complete tour
                        # Add cutset constraint: sum of edges crossing cut >= 2
                        S = set(subtour)
                        edges_crossing = []
                        for i in S:
                            for j in range(self.n):
                                if j not in S:
                                    if i < j:
                                        edges_crossing.append(self.x_vars[i, j])
                                    else:
                                        edges_crossing.append(self.x_vars[j, i])

                        model.cbLazy(gp.quicksum(edges_crossing) >= 2)

    def solve(self, warm_start_tour: List[int] = None) -> dict:
        """Solve TSP using ILP with lazy SEC.

        Args:
            warm_start_tour: Optional initial tour (1-based) for warm start

        Returns:
            dict with solution details
        """
        start_time = time.time()
        self.callback_count = 0

        # Build model
        self._build_model()

        # Warm start if provided
        if warm_start_tour is not None:
            tour_0based = [c - 1 for c in warm_start_tour]
            for k in range(len(tour_0based)):
                i = tour_0based[k]
                j = tour_0based[(k + 1) % len(tour_0based)]
                if i > j:
                    i, j = j, i
                if (i, j) in self.x_vars:
                    self.x_vars[i, j].Start = 1

        # Solve with callback
        self.model.optimize(self._subtour_callback)

        solve_time = time.time() - start_time

        # Extract solution
        if self.model.SolCount > 0:
            # Get selected edges
            selected_edges = []
            for (i, j), var in self.x_vars.items():
                if var.X > 0.5:
                    selected_edges.append((i, j))

            # Construct tour from edges
            tour = self._edges_to_tour(selected_edges)
            tour_1based = [city + 1 for city in tour]

            obj_value = int(round(self.model.ObjVal))
            gap = self.model.MIPGap
            node_count = int(self.model.NodeCount)
            bound = self.model.ObjBound

            status = "optimal" if self.model.Status == GRB.OPTIMAL else "time_limit"

        else:
            tour_1based = []
            obj_value = None
            gap = None
            node_count = 0
            bound = None
            status = "no_solution"

        return {
            'tour': tour_1based,
            'length': obj_value,
            'gap': gap,
            'mip_gap_pct': 100 * gap if gap is not None else None,
            'lower_bound': bound,
            'node_count': node_count,
            'callback_count': self.callback_count,
            'solve_time': solve_time,
            'status': status,
            'method': 'ILP_LazySEC'
        }

    def _edges_to_tour(self, edges: List[Tuple[int, int]]) -> List[int]:
        """Convert edge list to tour sequence.

        Args:
            edges: List of (i, j) edges

        Returns:
            Tour as list of city indices
        """
        # Build adjacency
        adj = {i: [] for i in range(self.n)}
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        # Start from city 0 and follow the path
        tour = [0]
        current = 0
        prev = -1

        while len(tour) < self.n:
            for neighbor in adj[current]:
                if neighbor != prev:
                    tour.append(neighbor)
                    prev = current
                    current = neighbor
                    break

        return tour


class ILPMTZ:
    """TSP solver using Miller-Tucker-Zemlin (MTZ) formulation."""

    def __init__(self, distance_matrix: np.ndarray, time_limit: int = 900):
        """
        Args:
            distance_matrix: n x n distance matrix (0-indexed)
            time_limit: Time limit in seconds
        """
        if not GUROBI_AVAILABLE:
            raise RuntimeError("Gurobi is not installed.")

        self.dist = distance_matrix
        self.n = len(distance_matrix)
        self.time_limit = time_limit
        self.model = None

    def solve(self, warm_start_tour: List[int] = None) -> dict:
        """Solve TSP using MTZ formulation.

        Args:
            warm_start_tour: Optional initial tour (1-based)

        Returns:
            dict with solution details
        """
        start_time = time.time()

        # Build model
        self.model = gp.Model("TSP_MTZ")
        self.model.Params.OutputFlag = 1
        self.model.Params.TimeLimit = self.time_limit

        # Decision variables: y[i,j] = 1 if arc i->j is used
        y = {}
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    y[i, j] = self.model.addVar(vtype=GRB.BINARY, name=f'y_{i}_{j}')

        # Ordering variables: u[i] represents position in tour
        u = {}
        u[0] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=1, name='u_0')
        for i in range(1, self.n):
            u[i] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=2, ub=self.n, name=f'u_{i}')

        # Objective: minimize total distance
        obj = gp.quicksum(self.dist[i][j] * y[i, j] for i in range(self.n) for j in range(self.n) if i != j)
        self.model.setObjective(obj, GRB.MINIMIZE)

        # Constraints: each city has exactly one outgoing and one incoming arc
        for i in range(self.n):
            self.model.addConstr(gp.quicksum(y[i, j] for j in range(self.n) if j != i) == 1, name=f'out_{i}')
            self.model.addConstr(gp.quicksum(y[j, i] for j in range(self.n) if j != i) == 1, name=f'in_{i}')

        # MTZ subtour elimination constraints
        for i in range(1, self.n):
            for j in range(1, self.n):
                if i != j:
                    self.model.addConstr(
                        u[i] - u[j] + self.n * y[i, j] <= self.n - 1,
                        name=f'mtz_{i}_{j}'
                    )

        self.model.update()

        # Solve
        self.model.optimize()
        solve_time = time.time() - start_time

        # Extract solution
        if self.model.SolCount > 0:
            # Reconstruct tour
            tour = [0]
            current = 0
            for _ in range(self.n - 1):
                for j in range(self.n):
                    if j != current and y[current, j].X > 0.5:
                        tour.append(j)
                        current = j
                        break

            tour_1based = [city + 1 for city in tour]
            obj_value = int(round(self.model.ObjVal))
            gap = self.model.MIPGap
            status = "optimal" if self.model.Status == GRB.OPTIMAL else "time_limit"
        else:
            tour_1based = []
            obj_value = None
            gap = None
            status = "no_solution"

        return {
            'tour': tour_1based,
            'length': obj_value,
            'gap': gap,
            'mip_gap_pct': 100 * gap if gap is not None else None,
            'solve_time': solve_time,
            'status': status,
            'method': 'ILP_MTZ'
        }
