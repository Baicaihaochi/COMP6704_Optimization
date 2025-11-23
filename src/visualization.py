"""
Visualization utilities for TSP solutions
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os


class TSPVisualizer:
    """Visualization tools for TSP solutions."""

    def __init__(self, coordinates: Dict[int, tuple], output_dir: str = 'results'):
        """
        Args:
            coordinates: Dict mapping city_id (1-based) to (x, y)
            output_dir: Directory to save plots
        """
        self.coordinates = coordinates
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_tour(self, tour: List[int], length: int, title: str, filename: str):
        """Plot a single tour.

        Args:
            tour: List of city indices (1-based)
            length: Tour length
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(10, 8))

        # Extract coordinates
        x_coords = [self.coordinates[city][0] for city in tour] + [self.coordinates[tour[0]][0]]
        y_coords = [self.coordinates[city][1] for city in tour] + [self.coordinates[tour[0]][1]]

        # Plot tour edges
        plt.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=1.5)

        # Plot cities
        for city in tour:
            x, y = self.coordinates[city]
            plt.plot(x, y, 'ro', markersize=6)

        # Highlight start city
        x_start, y_start = self.coordinates[tour[0]]
        plt.plot(x_start, y_start, 'go', markersize=10, label=f'Start (City {tour[0]})')

        plt.title(f'{title}\nTour Length: {length}', fontsize=14, fontweight='bold')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_convergence(self, histories: Dict[str, dict], filename: str):
        """Plot convergence curves for multiple methods.

        Args:
            histories: Dict mapping method name to history dict
            filename: Output filename
        """
        plt.figure(figsize=(12, 6))

        for method_name, history in histories.items():
            if 'best_length' in history and 'iteration' in history:
                plt.plot(history['iteration'], history['best_length'],
                         label=method_name, linewidth=2)

        plt.axhline(y=7542, color='r', linestyle='--', label='Known Optimum (7542)', linewidth=2)
        plt.xlabel('Iteration / Generation', fontsize=12)
        plt.ylabel('Best Tour Length', fontsize=12)
        plt.title('Convergence Comparison', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_comparison_bar(self, results: Dict[str, dict], filename: str):
        """Create bar chart comparing solution quality.

        Args:
            results: Dict mapping method name to result dict
            filename: Output filename
        """
        methods = list(results.keys())
        lengths = [results[m]['length'] for m in methods]
        gaps = [(l - 7542) / 7542 * 100 for l in lengths]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Tour lengths
        colors = ['green' if l == 7542 else 'orange' if l < 8000 else 'red' for l in lengths]
        bars1 = ax1.bar(methods, lengths, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=7542, color='r', linestyle='--', label='Optimum', linewidth=2)
        ax1.set_ylabel('Tour Length', fontsize=12)
        ax1.set_title('Solution Quality Comparison', fontsize=13, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, length in zip(bars1, lengths):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(length)}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Optimality gaps
        colors_gap = ['green' if g == 0 else 'orange' if g < 5 else 'red' for g in gaps]
        bars2 = ax2.bar(methods, gaps, color=colors_gap, alpha=0.7, edgecolor='black')
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_ylabel('Optimality Gap (%)', fontsize=12)
        ax2.set_title('Optimality Gap Comparison', fontsize=13, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, gap in zip(bars2, gaps):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{gap:.2f}%',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_runtime_comparison(self, results: Dict[str, dict], filename: str):
        """Plot runtime comparison.

        Args:
            results: Dict mapping method name to result dict
            filename: Output filename
        """
        methods = list(results.keys())
        runtimes = []

        for m in methods:
            if 'total_time' in results[m]:
                runtimes.append(results[m]['total_time'])
            elif 'solve_time' in results[m]:
                runtimes.append(results[m]['solve_time'])
            else:
                runtimes.append(0)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, runtimes, color='skyblue', alpha=0.7, edgecolor='black')
        plt.ylabel('Runtime (seconds)', fontsize=12)
        plt.title('Runtime Comparison', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, runtime in zip(bars, runtimes):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{runtime:.2f}s',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_statistical_analysis(self, results_multiple_runs: Dict[str, dict], filename: str):
        """Plot statistical analysis for stochastic methods.

        Args:
            results_multiple_runs: Dict with statistics from multiple runs
            filename: Output filename
        """
        methods = []
        means = []
        stds = []
        mins = []
        maxs = []

        for method_name, result in results_multiple_runs.items():
            if 'mean_length' in result:
                methods.append(method_name)
                means.append(result['mean_length'])
                stds.append(result['std_length'])
                mins.append(result['min_length'])
                maxs.append(result['max_length'])

        if not methods:
            return

        x = np.arange(len(methods))
        width = 0.6

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot bars with error bars
        bars = ax.bar(x, means, width, yerr=stds, capsize=5,
                      color='lightblue', alpha=0.7, edgecolor='black',
                      label='Mean ± Std')

        # Plot min and max as scatter
        ax.scatter(x, mins, color='green', s=100, zorder=3, label='Best')
        ax.scatter(x, maxs, color='red', s=100, zorder=3, label='Worst')

        # Optimum line
        ax.axhline(y=7542, color='purple', linestyle='--', linewidth=2, label='Optimum')

        ax.set_ylabel('Tour Length', fontsize=12)
        ax.set_title('Statistical Analysis (Multiple Runs)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_operator_adaptation(self, history: dict, filename: str):
        """Plot operator probability evolution for adaptive methods.

        Args:
            history: History dict containing operator_probs
            filename: Output filename
        """
        if 'operator_probs' not in history:
            return

        plt.figure(figsize=(12, 6))

        for operator, probs in history['operator_probs'].items():
            if len(probs) > 0:
                iterations = history['iteration'][:len(probs)]
                plt.plot(iterations, probs, label=operator, linewidth=2, marker='o', markersize=3)

        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Operator Probability', fontsize=12)
        plt.title('Adaptive Operator Selection Evolution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
