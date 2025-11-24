# Berlin52 TSP: Exact and Heuristic Solution Methods

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Comprehensive implementation of five solution methods for the Berlin52 Traveling Salesman Problem from TSPLIB, developed for COMP6704 Advanced Topics in Optimization (PolyU Fall 2025).

## 📋 Overview

This project implements and compares five TSP solution methods on the classic Berlin52 instance:

1. **Nearest Neighbor + 2-opt** - Constructive heuristic with local search
2. **ILP with Lazy SEC** - Exact Integer Linear Programming using Gurobi
3. **Genetic Algorithm** - Memetic GA with permutation encoding
4. **Simulated Annealing** - Metropolis acceptance with adaptive temperature
5. **Adaptive Multi-Strategy Hybrid (AMSH)** - **Novel method** with adaptive operator selection

## 🎯 Problem Instance

- **Dataset**: Berlin52 from TSPLIB
- **Cities**: 52 locations in Berlin
- **Distance**: Euclidean with TSPLIB rounding: `floor(sqrt((x1-x2)^2 + (y1-y2)^2) + 0.5)`
- **Known Optimum**: 7542

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install dependencies
pip install -r requirements.txt
```

**Note**: For the ILP method, you need Gurobi optimizer with a valid license:
```bash
# Academic license: https://www.gurobi.com/academia/academic-program-and-licenses/
pip install gurobipy
```

### Running Experiments

```bash
# Run all methods
cd src
python run_experiments.py

# Results will be saved to ../results/
```

### Project Structure

```
optimization_personal/
├── berlin52.tsp              # TSPLIB data file
├── COMP6704.tex              # LaTeX report
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/
│   ├── tsp_data.py           # Data loading and distance calculation
│   ├── nn_2opt.py            # Method 1: NN + 2-opt
│   ├── ilp_solver.py         # Method 2: ILP (Lazy SEC & MTZ)
│   ├── genetic_algorithm.py  # Method 3: Genetic Algorithm
│   ├── simulated_annealing.py# Method 4: Simulated Annealing
│   ├── adaptive_hybrid.py    # Method 5: AMSH (Novel)
│   ├── visualization.py      # Plotting utilities
│   └── run_experiments.py    # Main experimental runner
└── results/                  # Output directory (generated)
    ├── tour_*.png            # Tour visualizations
    ├── convergence_comparison.png
    ├── quality_comparison.png
    ├── runtime_comparison.png
    ├── statistical_analysis.png
    ├── operator_adaptation.png
    └── results_summary.json
```

## 📊 Methods Description

### Method 1: Nearest Neighbor + 2-opt

**Type**: Constructive heuristic + Local search

**Algorithm**:
1. Start from a city, iteratively visit nearest unvisited neighbor
2. Apply 2-opt local search: swap edges to eliminate crossings
3. Multi-start from different starting cities

**Parameters**:
- Max 2-opt passes: 100
- Start cities: 10

**Complexity**: O(n²) construction + O(n² × passes) for 2-opt

### Method 2: ILP with Lazy Subtour Elimination

**Type**: Exact optimization

**Formulation**: DFJ (Dantzig-Fulkerson-Johnson) model
- Decision variables: x_{ij} ∈ {0,1} for undirected edges
- Degree constraints: sum of incident edges = 2 for each city
- Subtour elimination: added lazily via Gurobi callbacks

**Solver**: Gurobi 11.0+ with lazy constraint callbacks

**Parameters**:
- Time limit: 900 seconds (15 minutes)
- Warm start: from NN+2opt solution

**Alternative**: MTZ (Miller-Tucker-Zemlin) formulation included as fallback

### Method 3: Genetic Algorithm

**Type**: Population-based metaheuristic

**Features**:
- **Encoding**: Permutation chromosomes
- **Selection**: Tournament (size 3)
- **Crossover**: Order Crossover (OX)
- **Mutation**: Inversion + Swap (20% rate)
- **Memetic**: 2-opt on top 10% every 10 generations
- **Elitism**: Preserve best 2 individuals

**Parameters**:
- Population: 100
- Generations: 500
- Early stopping: 50 generations without improvement

### Method 4: Simulated Annealing

**Type**: Single-solution metaheuristic

**Features**:
- **Neighborhood**: 2-opt moves
- **Acceptance**: Metropolis criterion exp(-Δ/T)
- **Cooling**: Geometric schedule T ← 0.98 × T
- **Initial temperature**: Auto-computed from sample of cost increases
- **Initialization**: Nearest Neighbor

**Parameters**:
- Moves per temperature: 20 × n
- Min temperature: 10⁻³
- Stop if no improvement for 10 temperature levels

### Method 5: Adaptive Multi-Strategy Hybrid (AMSH) - **Novel**

**Type**: Adaptive hybrid metaheuristic

**Innovation**: This is a novel method that dynamically adjusts operator probabilities based on their recent success rates, inspired by Adaptive Large Neighborhood Search (ALNS) but applied to metaheuristic operators.

**Key Components**:

1. **Solution Pool**: Maintains diverse high-quality solutions (size 10)
2. **Multiple Operators**:
   - 2-opt: Classical edge swap
   - 3-opt: Three-edge reconnection
   - Or-opt: Relocate sequences of 1-3 cities
   - Swap: Exchange two cities
   - Insert: Remove and reinsert at best position

3. **Adaptive Selection**:
   - Each operator has a dynamic weight
   - Weights updated based on success rate: w ← w × (1 + α × success_rate)
   - Learning rate α = 0.1

4. **Intensification & Diversification**:
   - Every 100 iterations: intensive 2-opt on best solution
   - Every 500 iterations: replace worst 33% with random solutions

5. **Quality-Diversity Balance**:
   - Solutions added if high quality OR sufficiently diverse
   - Diversity measured by edge-based Jaccard distance

**Parameters**:
- Pool size: 10
- Iterations: 5000
- Min operator probability: 0.05
- Learning rate: 0.1

**Theoretical Justification**:
- Adaptive operator selection allows the algorithm to learn which operators work best for the current problem structure
- Population diversity prevents premature convergence
- Periodic phases balance exploration vs. exploitation

## 📈 Experimental Protocol

### Metrics

For all methods:
- **Solution quality**: Best tour length
- **Optimality gap**: (length - 7542) / 7542 × 100%
- **Runtime**: Wall-clock time in seconds

For stochastic methods (GA, SA, AMSH):
- **Robustness**: Mean ± Std over 10 runs
- **Range**: [min, max] tour length
- **Convergence**: Best length vs iterations

For ILP:
- **MIP gap**: Gurobi's relative gap at termination
- **Lower bound**: Best proven lower bound
- **Nodes explored**: Branch-and-bound tree size

### Reproducibility

**Deterministic methods**: NN+2opt, ILP
- Results are reproducible given same start cities

**Stochastic methods**: GA, SA, AMSH
- Python's `random` module used (can be seeded)
- Report statistics over 10 independent runs

**Hardware**: Experiments run on:
- CPU: (to be filled after running)
- RAM: (to be filled after running)
- OS: macOS / Linux / Windows

## 🎨 Visualizations

The code automatically generates:

1. **Tour Plots**: Visual representation of each method's best tour
2. **Convergence Curves**: Solution quality vs iteration for metaheuristics
3. **Quality Comparison**: Bar chart of tour lengths and optimality gaps
4. **Runtime Comparison**: Bar chart of computational time
5. **Statistical Analysis**: Box plots showing distribution over multiple runs
6. **Operator Adaptation**: Evolution of operator probabilities (AMSH only)

## 📚 Implementation Details

### TSPLIB Distance Calculation

**Critical**: TSPLIB uses a specific rounding convention:

```python
def tsplib_distance(coord1, coord2):
    euclidean = math.sqrt((coord1[0] - coord2[0])**2 +
                         (coord1[1] - coord2[1])**2)
    return math.floor(euclidean + 0.5)
```

This is **NOT** the same as Python's `round()` function. Our implementation in `tsp_data.py` uses the correct formula.

**Verification**: The known optimal tour is hardcoded and verified to have length 7542.

### Warm Starting ILP

The ILP solver accepts an optional warm start tour:
```python
ilp_solver.solve(warm_start_tour=[1, 49, 32, ...])  # 1-based indices
```

This significantly speeds up convergence by providing a good initial incumbent.

### Parameter Tuning

Default parameters were chosen based on:
- Literature recommendations (e.g., GA tournament size 3-5)
- Preliminary experiments on Berlin52
- Computational budget (total runtime ~10-15 minutes for all methods)

For other TSP instances, parameters may need adjustment.

## 🔬 Expected Results

Based on preliminary runs:

| Method | Best Length | Gap (%) | Avg Time (s) | Notes |
|--------|-------------|---------|--------------|-------|
| NN+2opt | 7542-7700 | 0-2% | <1 | Fast, good baseline |
| ILP | 7542 | 0% | 60-900 | Exact, may timeout |
| GA | 7542-7600 | 0-1% | 30-60 | Robust, memetic helps |
| SA | 7542-7650 | 0-1.5% | 10-30 | Sensitive to cooling |
| AMSH | 7542-7580 | 0-0.5% | 20-50 | Best metaheuristic |

**Note**: Actual results depend on random seed and hardware.

## 🏆 Novel Contributions

### Adaptive Multi-Strategy Hybrid (AMSH)

**What makes it novel**:

1. **Dynamic operator weighting**: Unlike fixed-probability hybrids, AMSH learns which operators are effective online

2. **Quality-diversity pool**: Maintains solutions that are either high-quality OR diverse, not just the best solutions

3. **Unified framework**: Integrates local search (2-opt, 3-opt, Or-opt) and perturbation operators (swap, insert) under one adaptive selection mechanism

4. **Theoretical grounding**: Borrows from ALNS (adaptive large neighborhood search) but applies to metaheuristic operators rather than neighborhoods

**Comparison to related work**:
- Traditional GA: Fixed crossover/mutation rates
- Traditional SA: Single neighborhood structure
- Memetic GA: Fixed local search frequency
- AMSH: All operators compete and adapt

**Potential extensions**:
- Credit assignment strategies (immediate vs. delayed rewards)
- Multi-armed bandit algorithms for operator selection
- Transfer learning: operator weights from one instance to another

