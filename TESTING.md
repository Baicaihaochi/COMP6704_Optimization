# Testing Guide for Berlin52 TSP Implementation

## Quick Start Testing

### 1. Basic Setup Verification

```bash
cd /Users/shuocai/Desktop/optimization_personal

# Check Python version (need 3.8+)
python3 --version

# Install dependencies (without Gurobi first)
pip3 install numpy matplotlib scipy tqdm

# Or use requirements.txt (comment out gurobipy if you don't have license yet)
pip3 install -r requirements.txt
```

### 2. Test Data Loading

```bash
cd /Users/shuocai/Desktop/optimization_personal

python3 << 'EOF'
from src.tsp_data import TSPData

# Load data
data = TSPData('berlin52.tsp')
print(f'✓ Loaded {data.name}: {data.dimension} cities')

# Verify distance calculation
length, correct = data.verify_known_optimum()
print(f'✓ Optimal tour verification: {length}')
print(f'✓ Calculation correct: {correct} (expected True)')

# Test distance calculation
d12 = data.get_distance(1, 2)
print(f'✓ Distance from city 1 to city 2: {d12}')
EOF
```

**Expected output:**
```
✓ Loaded berlin52: 52 cities
✓ Optimal tour verification: 7542
✓ Calculation correct: True (expected True)
✓ Distance from city 1 to city 2: 666
```

### 3. Test Method 1: Nearest Neighbor + 2-opt

```bash
python3 << 'EOF'
from src.tsp_data import TSPData
from src.nn_2opt import NearestNeighbor2Opt

# Load data
data = TSPData('berlin52.tsp')

# Run NN + 2-opt from city 0
solver = NearestNeighbor2Opt(data.distance_matrix, max_passes=100)
result = solver.solve(start_city=0, use_2opt=True)

print(f'✓ NN+2opt completed')
print(f'  - Tour length: {result["length"]}')
print(f'  - NN initial: {result["nn_length"]}')
print(f'  - Improvement: {result["improvement"]} ({result["improvement_pct"]:.2f}%)')
print(f'  - 2-opt passes: {result["passes"]}')
print(f'  - Time: {result["total_time"]:.3f}s')
print(f'  - Gap from optimum: {(result["length"]-7542)/7542*100:.2f}%')
EOF
```

my test result:
✓ NN+2opt completed
  - Tour length: 7967
  - NN initial: 8980
  - Improvement: 1013 (11.28%)
  - 2-opt passes: 16
  - Time: 0.003s
  - Gap from optimum: 5.64%

**Expected output:**
```
✓ NN+2opt completed
  - Tour length: 7542-7700 (typically around 7600)
  - NN initial: 9000-10000
  - Improvement: 2000-3000 (20-30%)
  - 2-opt passes: 5-15
  - Time: 0.1-0.5s
  - Gap from optimum: 0-2%
```

### 4. Test Method 3: Genetic Algorithm

```bash
python3 << 'EOF'
from src.tsp_data import TSPData
from src.genetic_algorithm import GeneticAlgorithm
import random

random.seed(42)  # For reproducibility

# Load data
data = TSPData('berlin52.tsp')

# Run GA (single run, fewer generations for testing)
solver = GeneticAlgorithm(
    data.distance_matrix,
    population_size=50,
    generations=100,
    memetic_2opt=True,
    memetic_frequency=10
)

result = solver.solve(num_runs=1)

print(f'✓ GA completed')
print(f'  - Tour length: {result["length"]}')
print(f'  - Generations: {result["generations"]}')
print(f'  - Time: {result["total_time"]:.3f}s')
print(f'  - Gap from optimum: {(result["length"]-7542)/7542*100:.2f}%')
EOF
```

my test result:
✓ GA completed
  - Tour length: 7542
  - Generations: 100
  - Time: 0.234s
  - Gap from optimum: 0.00%

**Expected output:**
```
✓ GA completed
  - Tour length: 7542-7700
  - Generations: 50-100
  - Time: 5-15s
  - Gap from optimum: 0-2%
```

### 5. Test Method 4: Simulated Annealing

```bash
python3 << 'EOF'
from src.tsp_data import TSPData
from src.simulated_annealing import SimulatedAnnealing
import random

random.seed(42)

# Load data
data = TSPData('berlin52.tsp')

# Run SA
solver = SimulatedAnnealing(
    data.distance_matrix,
    cooling_rate=0.98,
    moves_per_temp=20*52
)

result = solver.solve(num_runs=1)

print(f'✓ SA completed')
print(f'  - Tour length: {result["length"]}')
print(f'  - Temperature levels: {result["temperature_levels"]}')
print(f'  - Time: {result["total_time"]:.3f}s')
print(f'  - Gap from optimum: {(result["length"]-7542)/7542*100:.2f}%')
EOF
```

my test result:
✓ SA completed
  - Tour length: 8980
  - Temperature levels: 10
  - Time: 0.016s
  - Gap from optimum: 19.07%

**Expected output:**
```
✓ SA completed
  - Tour length: 7542-7700
  - Temperature levels: 50-150
  - Time: 3-10s
  - Gap from optimum: 0-2%
```

### 6. Test Method 5: AMSH (Novel)

```bash
python3 << 'EOF'
from src.tsp_data import TSPData
from src.adaptive_hybrid import AdaptiveMultiStrategyHybrid
import random

random.seed(42)

# Load data
data = TSPData('berlin52.tsp')

# Run AMSH (fewer iterations for testing)
solver = AdaptiveMultiStrategyHybrid(
    data.distance_matrix,
    pool_size=5,
    iterations=500
)

result = solver.solve()

print(f'✓ AMSH completed')
print(f'  - Tour length: {result["length"]}')
print(f'  - Iterations: {result["iterations"]}')
print(f'  - Time: {result["total_time"]:.3f}s')
print(f'  - Gap from optimum: {(result["length"]-7542)/7542*100:.2f}%')
print(f'  - Final operator weights: {result["final_operator_weights"]}')
EOF
```
my test result:
✓ AMSH completed
  - Tour length: 7542
  - Iterations: 500
  - Time: 0.105s
  - Gap from optimum: 0.00%
  - Final operator weights: {'2opt': 1.0142857142857142, '3opt': 1.0, 'or_opt': 1.0, 'swap': 1.0, 'insert': 1.0}

**Expected output:**
```
✓ AMSH completed
  - Tour length: 7542-7650
  - Iterations: 500
  - Time: 5-15s
  - Gap from optimum: 0-1.5%
  - Final operator weights: {'2opt': ..., '3opt': ..., ...}
```

### 7. Test Gurobi ILP (Optional - requires Gurobi license)

```bash
# First, install Gurobi and activate license
pip3 install gurobipy

python3 << 'EOF'
from src.tsp_data import TSPData
from src.ilp_solver import ILPLazySEC

# Load data
data = TSPData('berlin52.tsp')

# Quick test with short time limit
solver = ILPLazySEC(data.distance_matrix, time_limit=60)
result = solver.solve()

print(f'✓ ILP completed')
print(f'  - Status: {result["status"]}')
print(f'  - Tour length: {result["length"]}')
print(f'  - MIP gap: {result["mip_gap_pct"]:.4f}%')
print(f'  - Lazy constraints added: {result["callback_count"]}')
print(f'  - Time: {result["solve_time"]:.3f}s')
EOF
```

my test result:
✓ ILP completed
  - Status: optimal
  - Tour length: 7542
  - MIP gap: 0.0000%
  - Lazy constraints added: 5
  - Time: 0.022s

**Expected output:**
```
✓ ILP completed
  - Status: optimal (if <60s) or time_limit
  - Tour length: 7542 (optimal) or close
  - MIP gap: 0.0000% (if optimal)
  - Lazy constraints added: 10-50
  - Time: 5-60s
```

### 8. Run Full Experiment (All Methods)

```bash
cd /Users/shuocai/Desktop/optimization_personal/src

# This will take 15-30 minutes with all methods
python3 run_experiments.py

# Results will be in ../results/
ls -lh ../results/
```

**Expected files in results/**:
```
tour_nn_2opt.png
tour_ilp_sec.png (if Gurobi available)
tour_ga.png
tour_sa.png
tour_amsh.png
convergence_comparison.png
quality_comparison.png
runtime_comparison.png
statistical_analysis.png
operator_adaptation.png
results_summary.json
```

## Common Issues and Solutions

### Issue 1: "No module named 'src'"

**Solution:**
```bash
# Make sure you're in the correct directory
cd /Users/shuocai/Desktop/optimization_personal

# Run Python with the correct path
python3 -c "import sys; sys.path.append('src'); from tsp_data import TSPData; print('OK')"
```

### Issue 2: "Gurobi not available"

**Solution:**
- For testing without Gurobi, all other methods will work fine
- ILP will be skipped automatically
- To use Gurobi:
  1. Get academic license: https://www.gurobi.com/academia/
  2. Install: `pip3 install gurobipy`
  3. Activate license: `grbgetkey YOUR-LICENSE-KEY`

### Issue 3: Wrong distance calculation

**Solution:**
```python
# Verify TSPLIB rounding is correct
import math

def tsplib_distance(x1, y1, x2, y2):
    euclidean = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return math.floor(euclidean + 0.5)

# Test
d = tsplib_distance(565.0, 575.0, 25.0, 185.0)
print(f"Distance 1->2: {d}")  # Should be 666
```

### Issue 4: Slow performance

**Solution:**
- Reduce parameters for testing:
  - GA: `generations=100, population_size=50`
  - SA: `moves_per_temp=10*52`
  - AMSH: `iterations=1000`
- Use `num_runs=1` instead of 10 for initial testing

## Validation Checklist

- [ ] Data loading works and optimal tour verifies to 7542
- [ ] NN+2opt runs and finds solutions < 8000
- [ ] GA converges and improves over generations
- [ ] SA accepts uphill moves initially and cools down
- [ ] AMSH shows operator weight adaptation
- [ ] ILP finds optimal solution 7542 (if Gurobi available)
- [ ] Visualizations generate without errors
- [ ] All methods complete within reasonable time

## Performance Benchmarks (Reference)

On a modern laptop (2020+):

| Method | Expected Time | Expected Quality |
|--------|---------------|------------------|
| NN+2opt | <1s | 7542-7700 |
| ILP | 10-300s | 7542 (optimal) |
| GA (10 runs) | 30-120s | 7542-7600 |
| SA (10 runs) | 20-80s | 7542-7650 |
| AMSH (10 runs) | 60-180s | 7542-7580 |

## Next Steps After Testing

1. **If all tests pass:**
   - Run full experiments: `cd src && python3 run_experiments.py`
   - Check results in `results/` directory
   - Review visualizations
   - Update LaTeX with experimental results

2. **If tests fail:**
   - Check error messages carefully
   - Verify Python version (3.8+)
   - Check file paths are correct
   - Report specific errors for debugging

3. **For the report:**
   - Run experiments and collect all results
   - Save results_summary.json
   - Include all generated plots in LaTeX
   - Fill in experimental results section

## Contact

If you encounter issues, check:
1. Python version and dependencies
2. File paths and working directory
3. TSPLIB distance calculation correctness
4. Sufficient computational time for metaheuristics
