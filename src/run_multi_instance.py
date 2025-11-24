"""
Multi-instance TSP experiments to analyze scalability
Tests algorithms on instances of different sizes: 26, 52, 107, 208 cities
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tsp_data import TSPData
from nn_2opt import NearestNeighbor2Opt
from genetic_algorithm import GeneticAlgorithm
from simulated_annealing import SimulatedAnnealing
from adaptive_hybrid import AdaptiveMultiStrategyHybrid
import json
import time
import numpy as np
import matplotlib.pyplot as plt

# Try to import ILP solver
try:
    from ilp_solver import ILPLazySEC
    GUROBI_AVAILABLE = True
except:
    GUROBI_AVAILABLE = False


# TSPLIB instances metadata
INSTANCES = {
    'eil51': {'n': 51, 'optimum': 426, 'file': 'eil51.tsp'},
    'berlin52': {'n': 52, 'optimum': 7542, 'file': 'berlin52.tsp'},
    'st70': {'n': 70, 'optimum': 675, 'file': 'st70.tsp'},
    'pr107': {'n': 107, 'optimum': 44303, 'file': 'pr107.tsp'},
    'ch130': {'n': 130, 'optimum': 6110, 'file': 'ch130.tsp'},
    'a280': {'n': 280, 'optimum': 2579, 'file': 'a280.tsp'},
}


def test_instance(instance_name: str, data_path: str, optimum: int, n: int, output_dir: str):
    """Test all methods on a single TSP instance."""

    print("\n" + "="*80)
    print(f"Testing {instance_name}: n={n}, optimum={optimum}")
    print("="*80)

    # Load data
    tsp_data = TSPData(data_path)
    dist_matrix = tsp_data.distance_matrix

    results = {}

    # ========================================================================
    # Method 1: NN + 2-opt (always run, very fast)
    # ========================================================================
    print(f"\n[1/5] NN+2opt (multi-start from {min(10, n)} cities)...")
    nn_solver = NearestNeighbor2Opt(dist_matrix, max_passes=100)
    nn_result = nn_solver.multi_start(num_starts=min(10, n))
    nn_best = nn_result['best_solution']

    print(f"  ✓ Length: {nn_best['length']}, Gap: {(nn_best['length']-optimum)/optimum*100:.2f}%, Time: {nn_best['total_time']:.3f}s")
    results['NN+2opt'] = nn_best

    # ========================================================================
    # Method 2: ILP (only for small instances)
    # ========================================================================
    # Note: Gurobi free license limits variables to 2000
    # TSP ILP has n² variables, so even n=45 (2025 vars) exceeds limit
    # For size-limited license, skip ILP for all instances
    if GUROBI_AVAILABLE and n <= 44:  # Only instances with n² < 2000
        print(f"\n[2/5] ILP with Lazy SEC (time limit: {300 if n < 70 else 900}s)...")
        try:
            time_limit = 300 if n < 70 else 900
            ilp_solver = ILPLazySEC(dist_matrix, time_limit=time_limit)
            ilp_result = ilp_solver.solve(warm_start_tour=nn_best['tour'])

            print(f"  ✓ Length: {ilp_result['length']}, Gap: {ilp_result['mip_gap_pct']:.4f}%, Time: {ilp_result['solve_time']:.3f}s, Status: {ilp_result['status']}")
            results['ILP'] = ilp_result
        except Exception as e:
            print(f"  ✗ ILP failed: {e}")
    else:
        if not GUROBI_AVAILABLE:
            reason = "Gurobi not available"
        elif n > 44:
            reason = f"n={n}, n²={n*n} variables > 2000 (license limit)"
        else:
            reason = f"n={n} too large"
        print(f"\n[2/5] ILP skipped ({reason})")

    # ========================================================================
    # Method 3: Genetic Algorithm (scale parameters with n)
    # ========================================================================
    print(f"\n[3/5] Genetic Algorithm (3 runs)...")
    ga_pop = min(100, max(50, n))
    ga_gen = min(500, max(200, 10*n))

    ga_solver = GeneticAlgorithm(
        dist_matrix,
        population_size=ga_pop,
        generations=ga_gen,
        memetic_2opt=True,
        memetic_frequency=10,
        early_stop=50
    )

    ga_results = ga_solver.solve(num_runs=3)
    ga_best = ga_results['best_solution']

    print(f"  ✓ Best: {ga_best['length']}, Mean: {ga_results['mean_length']:.1f}, Gap: {(ga_best['length']-optimum)/optimum*100:.2f}%, Time: {ga_best['total_time']:.3f}s")
    results['GA'] = ga_best
    results['GA_stats'] = ga_results

    # ========================================================================
    # Method 4: Simulated Annealing (scale parameters with n)
    # ========================================================================
    print(f"\n[4/5] Simulated Annealing (3 runs)...")
    sa_moves = min(30*n, 6000)  # Cap iterations to keep runtime manageable
    sa_cooling = 0.98 if n <= 120 else 0.993
    sa_patience = max(200, int(1.5 * n))

    sa_solver = SimulatedAnnealing(
        dist_matrix,
        initial_temp=None,
        cooling_rate=sa_cooling,
        moves_per_temp=sa_moves,
        min_temp=1e-2,
        max_no_improve=sa_patience
    )

    sa_results = sa_solver.solve(num_runs=3)
    sa_best = sa_results['best_solution']

    print(f"  ✓ Best: {sa_best['length']}, Mean: {sa_results['mean_length']:.1f}, Gap: {(sa_best['length']-optimum)/optimum*100:.2f}%, Time: {sa_best['total_time']:.3f}s")
    results['SA'] = sa_best
    results['SA_stats'] = sa_results

    # ========================================================================
    # Method 5: AMSH (scale parameters with n)
    # ========================================================================
    print(f"\n[5/5] AMSH (3 runs)...")
    amsh_pool = min(10, max(5, n//10))
    amsh_iters = min(4000, max(1500, 20*n))

    amsh_all_results = []
    amsh_best_overall = None
    amsh_best_length = float('inf')

    for run in range(3):
        amsh_solver = AdaptiveMultiStrategyHybrid(
            dist_matrix,
            pool_size=amsh_pool,
            iterations=amsh_iters,
            intensification_freq=max(60, n//2),
            diversification_freq=max(300, n),
            learning_rate=0.15
        )

        amsh_result = amsh_solver.solve()
        amsh_all_results.append(amsh_result)

        if amsh_result['length'] < amsh_best_length:
            amsh_best_length = amsh_result['length']
            amsh_best_overall = amsh_result

    amsh_lengths = [r['length'] for r in amsh_all_results]
    amsh_stats = {
        'best_solution': amsh_best_overall,
        'mean_length': np.mean(amsh_lengths),
        'min_length': np.min(amsh_lengths)
    }

    print(f"  ✓ Best: {amsh_best_overall['length']}, Mean: {amsh_stats['mean_length']:.1f}, Gap: {(amsh_best_overall['length']-optimum)/optimum*100:.2f}%, Time: {amsh_best_overall['total_time']:.3f}s")
    results['AMSH'] = amsh_best_overall
    results['AMSH_stats'] = amsh_stats

    return results


def run_multi_instance_experiments(data_dir: str = '..', output_dir: str = '../results_multi'):
    """Run experiments on multiple TSP instances."""

    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    # Test each available instance
    for instance_name, metadata in INSTANCES.items():
        data_path = os.path.join(data_dir, metadata['file'])

        if not os.path.exists(data_path):
            print(f"\n⚠️  Skipping {instance_name}: file not found at {data_path}")
            continue

        try:
            results = test_instance(
                instance_name,
                data_path,
                metadata['optimum'],
                metadata['n'],
                output_dir
            )
            all_results[instance_name] = {
                'n': metadata['n'],
                'optimum': metadata['optimum'],
                'results': results
            }
        except Exception as e:
            print(f"\n✗ Error processing {instance_name}: {e}")
            import traceback
            traceback.print_exc()

    # ========================================================================
    # Generate Summary Report
    # ========================================================================
    print("\n" + "="*80)
    print("SCALABILITY ANALYSIS")
    print("="*80)

    print(f"\n{'Instance':<12} {'n':<6} {'Opt':<8} {'Method':<10} {'Best':<10} {'Gap(%)':<8} {'Time(s)':<10}")
    print("-"*80)

    for instance_name in sorted(all_results.keys(), key=lambda x: all_results[x]['n']):
        data = all_results[instance_name]
        n = data['n']
        optimum = data['optimum']
        results = data['results']

        for method in ['NN+2opt', 'ILP', 'GA', 'SA', 'AMSH']:
            if method in results:
                r = results[method]
                length = r['length']
                gap = (length - optimum) / optimum * 100
                time_key = 'total_time' if 'total_time' in r else 'solve_time'
                runtime = r.get(time_key, 0)

                print(f"{instance_name:<12} {n:<6} {optimum:<8} {method:<10} {length:<10} {gap:<8.2f} {runtime:<10.3f}")

    # Save results
    output_file = os.path.join(output_dir, 'scalability_results.json')

    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    with open(output_file, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    # ========================================================================
    # Generate Scalability Plots
    # ========================================================================
    plot_scalability(all_results, output_dir)

    print("="*80)


def plot_scalability(all_results: dict, output_dir: str):
    """Generate scalability analysis plots."""

    instances = sorted(all_results.keys(), key=lambda x: all_results[x]['n'])
    n_values = [all_results[inst]['n'] for inst in instances]

    methods = ['NN+2opt', 'GA', 'SA', 'AMSH']
    if any('ILP' in all_results[inst]['results'] for inst in instances):
        methods.insert(1, 'ILP')

    # Plot 1: Solution quality (gap) vs instance size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for method in methods:
        gaps = []
        times = []
        ns = []

        for inst in instances:
            if method in all_results[inst]['results']:
                r = all_results[inst]['results'][method]
                optimum = all_results[inst]['optimum']
                gap = (r['length'] - optimum) / optimum * 100
                time_key = 'total_time' if 'total_time' in r else 'solve_time'
                runtime = r.get(time_key, 0)

                gaps.append(gap)
                times.append(runtime)
                ns.append(all_results[inst]['n'])

        if gaps:
            ax1.plot(ns, gaps, 'o-', label=method, linewidth=2, markersize=8)
            ax2.plot(ns, times, 'o-', label=method, linewidth=2, markersize=8)

    ax1.set_xlabel('Number of Cities (n)', fontsize=12)
    ax1.set_ylabel('Optimality Gap (%)', fontsize=12)
    ax1.set_title('Solution Quality vs Problem Size', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    ax2.set_xlabel('Number of Cities (n)', fontsize=12)
    ax2.set_ylabel('Runtime (seconds, log scale)', fontsize=12)
    ax2.set_title('Runtime vs Problem Size', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scalability_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Scalability plot saved")


if __name__ == '__main__':
    data_dir = '..' if len(sys.argv) < 2 else sys.argv[1]
    output_dir = '../results_multi' if len(sys.argv) < 3 else sys.argv[2]

    print("Multi-Instance TSP Scalability Analysis")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    run_multi_instance_experiments(data_dir, output_dir)
