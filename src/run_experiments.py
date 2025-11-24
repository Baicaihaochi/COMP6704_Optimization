"""
Main experimental runner for Berlin52 TSP
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tsp_data import TSPData
from nn_2opt import NearestNeighbor2Opt
from ilp_solver import ILPLazySEC, ILPMTZ
from genetic_algorithm import GeneticAlgorithm
from simulated_annealing import SimulatedAnnealing
from adaptive_hybrid import AdaptiveMultiStrategyHybrid
from visualization import TSPVisualizer
import json
import time
import numpy as np


def run_all_experiments(data_path: str = 'berlin52.tsp', output_dir: str = 'results'):
    """Run all TSP solution methods and generate comprehensive results."""

    print("="*80)
    print("Berlin52 TSP Comprehensive Evaluation")
    print("="*80)

    # Load data
    print("\n[1/7] Loading Berlin52 data...")
    tsp_data = TSPData(data_path)
    print(f"  - Loaded {tsp_data.name} with {tsp_data.dimension} cities")

    # Verify known optimum
    opt_length, is_correct = tsp_data.verify_known_optimum()
    print(f"  - Known optimal tour verification: {opt_length} (Expected: 7542)")
    if not is_correct:
        print(f"  - WARNING: Distance calculation may be incorrect!")

    dist_matrix = tsp_data.distance_matrix
    coords = tsp_data.coordinates

    # Initialize visualizer
    visualizer = TSPVisualizer(coords, output_dir)

    # Results storage
    all_results = {}
    all_histories = {}

    # ========================================================================
    # Method 1: Nearest Neighbor + 2-opt
    # ========================================================================
    print("\n[2/7] Running Method 1: Nearest Neighbor + 2-opt...")
    nn_solver = NearestNeighbor2Opt(dist_matrix, max_passes=100)

    # Try multiple start cities
    print("  - Running multi-start (10 different starting cities)...")
    nn_result = nn_solver.multi_start(num_starts=10)
    nn_best = nn_result['best_solution']

    print(f"  - Best tour length: {nn_best['length']}")
    print(f"  - NN initial length: {nn_best['nn_length']}")
    print(f"  - Improvement: {nn_best['improvement']} ({nn_best['improvement_pct']:.2f}%)")
    print(f"  - 2-opt passes: {nn_best['passes']}")
    print(f"  - Total time: {nn_best['total_time']:.3f}s")

    all_results['NN+2opt'] = nn_best
    visualizer.plot_tour(nn_best['tour'], nn_best['length'],
                         'Nearest Neighbor + 2-opt', 'tour_nn_2opt.png')

    # ========================================================================
    # Method 2: ILP with Lazy SEC (Gurobi)
    # ========================================================================
    print("\n[3/7] Running Method 2: ILP with Lazy SEC...")
    print("  - Using Gurobi with lazy subtour elimination constraints")
    print("  - Time limit: 900 seconds (15 minutes)")

    try:
        ilp_solver = ILPLazySEC(dist_matrix, time_limit=900)
        # Warm start with NN+2opt solution
        ilp_result = ilp_solver.solve(warm_start_tour=nn_best['tour'])

        print(f"  - Tour length: {ilp_result['length']}")
        print(f"  - Status: {ilp_result['status']}")
        print(f"  - MIP gap: {ilp_result['mip_gap_pct']:.4f}%")
        print(f"  - Lower bound: {ilp_result['lower_bound']:.2f}")
        print(f"  - Lazy constraints added: {ilp_result['callback_count']}")
        print(f"  - Solve time: {ilp_result['solve_time']:.3f}s")

        all_results['ILP_LazySEC'] = ilp_result
        visualizer.plot_tour(ilp_result['tour'], ilp_result['length'],
                             'ILP with Lazy SEC (Gurobi)', 'tour_ilp_sec.png')

    except Exception as e:
        print(f"  - ERROR: {e}")
        print(f"  - Skipping ILP method (Gurobi not available?)")

    # ========================================================================
    # Method 3: Genetic Algorithm
    # ========================================================================
    print("\n[4/7] Running Method 3: Genetic Algorithm...")
    print("  - Population: 100, Generations: 500, Tournament: 3")
    print("  - Memetic: 2-opt on top 10% every 10 generations")
    print("  - Running 10 independent runs...")

    ga_solver = GeneticAlgorithm(
        dist_matrix,
        population_size=100,
        generations=500,
        tournament_size=3,
        crossover_rate=0.9,
        mutation_rate=0.2,
        elitism=2,
        memetic_2opt=True,
        memetic_fraction=0.1,
        memetic_frequency=10,
        early_stop=50
    )

    ga_results = ga_solver.solve(num_runs=10)
    ga_best = ga_results['best_solution']

    print(f"  - Best tour length: {ga_best['length']}")
    print(f"  - Mean ± Std: {ga_results['mean_length']:.2f} ± {ga_results['std_length']:.2f}")
    print(f"  - Range: [{ga_results['min_length']}, {ga_results['max_length']}]")
    print(f"  - Generations (best run): {ga_best['generations']}")
    print(f"  - Time (best run): {ga_best['total_time']:.3f}s")

    all_results['GeneticAlgorithm'] = ga_best
    all_results['GeneticAlgorithm_stats'] = ga_results
    all_histories['GeneticAlgorithm'] = ga_best['history']

    visualizer.plot_tour(ga_best['tour'], ga_best['length'],
                         'Genetic Algorithm', 'tour_ga.png')

    # ========================================================================
    # Method 4: Simulated Annealing
    # ========================================================================
    print("\n[5/7] Running Method 4: Simulated Annealing...")
    print("  - Cooling rate: 0.99, Moves per temp: 100*n")
    print("  - High initial temperature (T0≥200, 90% acceptance)")
    print("  - NN initialization with extensive exploration")
    print("  - Running 10 independent runs...")

    sa_solver = SimulatedAnnealing(
        dist_matrix,
        initial_temp=None,  # Auto-compute: T0≥200 for high acceptance
        cooling_rate=0.99,  # Geometric cooling
        moves_per_temp=100*tsp_data.dimension,  # Extensive exploration per level
        min_temp=1e-3,
        max_no_improve=20  # Patience before termination
    )

    sa_results = sa_solver.solve(num_runs=10)
    sa_best = sa_results['best_solution']

    print(f"  - Best tour length: {sa_best['length']}")
    print(f"  - Mean ± Std: {sa_results['mean_length']:.2f} ± {sa_results['std_length']:.2f}")
    print(f"  - Range: [{sa_results['min_length']}, {sa_results['max_length']}]")
    print(f"  - Temperature levels (best run): {sa_best['temperature_levels']}")
    print(f"  - Time (best run): {sa_best['total_time']:.3f}s")

    all_results['SimulatedAnnealing'] = sa_best
    all_results['SimulatedAnnealing_stats'] = sa_results
    all_histories['SimulatedAnnealing'] = sa_best['history']

    visualizer.plot_tour(sa_best['tour'], sa_best['length'],
                         'Simulated Annealing', 'tour_sa.png')

    # ========================================================================
    # Method 5: Adaptive Multi-Strategy Hybrid (Novel)
    # ========================================================================
    print("\n[6/7] Running Method 5: Adaptive Multi-Strategy Hybrid (AMSH)...")
    print("  - Novel adaptive operator selection method")
    print("  - Pool size: 10, Iterations: 5000")
    print("  - Operators: 2-opt, 3-opt, Or-opt, Swap, Insert")
    print("  - Running 10 independent runs...")

    amsh_best_overall = None
    amsh_best_length = float('inf')
    amsh_all_results = []

    for run in range(10):
        amsh_solver = AdaptiveMultiStrategyHybrid(
            dist_matrix,
            pool_size=10,
            iterations=5000,
            intensification_freq=100,
            diversification_freq=500,
            learning_rate=0.1,
            min_operator_prob=0.05
        )

        amsh_result = amsh_solver.solve()
        amsh_all_results.append(amsh_result)

        if amsh_result['length'] < amsh_best_length:
            amsh_best_length = amsh_result['length']
            amsh_best_overall = amsh_result

    # Compute statistics
    amsh_lengths = [r['length'] for r in amsh_all_results]
    amsh_stats = {
        'best_solution': amsh_best_overall,
        'mean_length': np.mean(amsh_lengths),
        'std_length': np.std(amsh_lengths),
        'min_length': np.min(amsh_lengths),
        'max_length': np.max(amsh_lengths),
        'num_runs': 10
    }

    print(f"  - Best tour length: {amsh_best_overall['length']}")
    print(f"  - Mean ± Std: {amsh_stats['mean_length']:.2f} ± {amsh_stats['std_length']:.2f}")
    print(f"  - Range: [{amsh_stats['min_length']}, {amsh_stats['max_length']}]")
    print(f"  - Time (best run): {amsh_best_overall['total_time']:.3f}s")
    print(f"  - Final operator weights: {amsh_best_overall['final_operator_weights']}")

    all_results['AMSH'] = amsh_best_overall
    all_results['AMSH_stats'] = amsh_stats
    all_histories['AMSH'] = amsh_best_overall['history']

    visualizer.plot_tour(amsh_best_overall['tour'], amsh_best_overall['length'],
                         'Adaptive Multi-Strategy Hybrid', 'tour_amsh.png')

    visualizer.plot_operator_adaptation(amsh_best_overall['history'],
                                        'operator_adaptation.png')

    # ========================================================================
    # Generate Comparative Visualizations
    # ========================================================================
    print("\n[7/7] Generating comparative visualizations...")

    # Convergence plots
    visualizer.plot_convergence(all_histories, 'convergence_comparison.png')

    # Solution quality comparison
    comparison_results = {
        'NN+2opt': all_results['NN+2opt'],
        'GeneticAlgorithm': all_results['GeneticAlgorithm'],
        'SimulatedAnnealing': all_results['SimulatedAnnealing'],
        'AMSH': all_results['AMSH']
    }

    if 'ILP_LazySEC' in all_results:
        comparison_results['ILP_LazySEC'] = all_results['ILP_LazySEC']

    visualizer.plot_comparison_bar(comparison_results, 'quality_comparison.png')
    visualizer.plot_runtime_comparison(comparison_results, 'runtime_comparison.png')

    # Statistical analysis for stochastic methods
    stochastic_results = {
        'GeneticAlgorithm': ga_results,
        'SimulatedAnnealing': sa_results,
        'AMSH': amsh_stats
    }
    visualizer.plot_statistical_analysis(stochastic_results, 'statistical_analysis.png')

    # ========================================================================
    # Save results to JSON
    # ========================================================================
    print("\nSaving results to JSON...")

    # Convert results to JSON-serializable format
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    json_results = {}
    for method, result in all_results.items():
        # Skip nested dicts like 'history'
        json_results[method] = {
            k: convert_to_json_serializable(v)
            for k, v in result.items()
            if not isinstance(v, dict) and k != 'history'
        }

    with open(os.path.join(output_dir, 'results_summary.json'), 'w') as f:
        json.dump(json_results, f, indent=2)

    # ========================================================================
    # Print Summary Table
    # ========================================================================
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'Best Length':<12} {'Gap (%)':<10} {'Time (s)':<10} {'Status':<15}")
    print("-"*80)

    for method_name, result in comparison_results.items():
        length = result['length']
        gap = (length - 7542) / 7542 * 100 if length else 0
        runtime = result.get('total_time', result.get('solve_time', 0))
        status = result.get('status', 'completed')

        print(f"{method_name:<25} {length:<12} {gap:<10.4f} {runtime:<10.3f} {status:<15}")

    print("="*80)
    print(f"\nOptimal solution (7542): {'FOUND' if any(r['length'] == 7542 for r in comparison_results.values()) else 'NOT FOUND'}")
    print(f"All results saved to: {output_dir}/")
    print("="*80)


if __name__ == '__main__':
    import sys

    data_path = '../berlin52.tsp' if len(sys.argv) < 2 else sys.argv[1]
    output_dir = '../results' if len(sys.argv) < 3 else sys.argv[2]

    run_all_experiments(data_path, output_dir)
