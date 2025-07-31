"""
Plotting Module for Calvano et al. (2020) Replication

This module contains functions to generate all figures from the paper,
including price dynamics, punishment strategies, convergence analysis,
and robustness results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import os
from pathlib import Path


# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Paper-style colors
COLORS = {
    'agent_0': '#1f77b4',  # Blue
    'agent_1': '#ff7f0e',  # Orange
    'agent_2': '#2ca02c',  # Green
    'agent_3': '#d62728',  # Red
    'nash': '#8c564b',     # Brown
    'monopoly': '#e377c2', # Pink
    'competitive': '#7f7f7f', # Gray
    'punishment': '#bcbd22'    # Olive
}


class CalvanoPlotter:
    """Main plotting class for Calvano et al. (2020) replication."""
    
    def __init__(self, output_dir: str = "figures"):
        """Initialize plotter with output directory."""
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Figure settings
        self.figsize_single = (10, 6)
        self.figsize_double = (15, 6)
        self.figsize_multi = (12, 8)
        self.dpi = 300
        
    def plot_price_dynamics(
        self,
        results: Dict[str, Any],
        window_size: int = 1000,
        title: str = "Price Dynamics Over Time",
        filename: Optional[str] = None
    ):
        """
        Plot price dynamics over time for all agents.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Experiment results containing price history
        window_size : int
            Rolling window size for smoothing
        title : str
            Plot title
        filename : str, optional
            Output filename
        """
        if 'runs' not in results or not results['runs']:
            return
        
        # Use first run for plotting
        run_data = results['runs'][0]
        prices = np.array(run_data['prices'])
        n_agents = prices.shape[1]
        
        fig, ax = plt.subplots(figsize=self.figsize_single)
        
        # Plot price series for each agent
        episodes = np.arange(len(prices))
        for i in range(n_agents):
            # Apply rolling average for smoothing
            if len(prices) > window_size:
                price_smooth = pd.Series(prices[:, i]).rolling(window_size, center=True).mean()
            else:
                price_smooth = prices[:, i]
            
            ax.plot(episodes, price_smooth, 
                   color=COLORS[f'agent_{i}'], 
                   label=f'Agent {i+1}', 
                   linewidth=2)
        
        # Add theoretical benchmarks
        env_params = results.get('parameters', {}).get('env_params', {})
        marginal_costs = results.get('parameters', {}).get('marginal_costs', [1.0] * n_agents)
        
        # Nash price (approximate)
        nash_price = marginal_costs[0] + 1.0 / (1.0 - 0.25 * (n_agents - 1))
        ax.axhline(y=nash_price, color=COLORS['nash'], 
                  linestyle='--', label='Nash Equilibrium', alpha=0.7)
        
        # Monopoly price
        monopoly_price = marginal_costs[0] + 1.0
        ax.axhline(y=monopoly_price, color=COLORS['monopoly'], 
                  linestyle='--', label='Monopoly Price', alpha=0.7)
        
        # Competitive price
        ax.axhline(y=marginal_costs[0], color=COLORS['competitive'], 
                  linestyle='--', label='Marginal Cost', alpha=0.7)
        
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Price')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        if filename is None:
            filename = "price_dynamics.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_punishment_strategy(
        self,
        deviation_results: Dict[str, Any],
        title: str = "Punishment Strategy Response",
        filename: Optional[str] = None
    ):
        """
        Plot the punishment strategy response to deviation.
        
        Parameters:
        -----------
        deviation_results : Dict[str, Any]
            Results from deviation analysis
        title : str
            Plot title
        filename : str, optional
            Output filename
        """
        if not deviation_results:
            return
        
        # Find a representative deviation case
        key = list(deviation_results.keys())[0]
        deviation_data = deviation_results[key]
        
        prices = np.array(deviation_data['prices'])
        deviating_agent = deviation_data['deviating_agent']
        deviation_price = deviation_data['deviation_price']
        n_periods = len(prices)
        n_agents = prices.shape[1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize_double)
        
        periods = np.arange(n_periods)
        
        # Plot 1: Price responses
        for i in range(n_agents):
            if i == deviating_agent:
                ax1.plot(periods, prices[:, i], 
                        color=COLORS['punishment'], 
                        label=f'Deviating Agent {i+1}', 
                        linewidth=3, marker='o', markersize=4)
            else:
                ax1.plot(periods, prices[:, i], 
                        color=COLORS[f'agent_{i}'], 
                        label=f'Agent {i+1}', 
                        linewidth=2, marker='s', markersize=3)
        
        # Mark deviation period
        ax1.axvline(x=0, color='red', linestyle=':', alpha=0.7, label='Deviation')
        
        # Add benchmark prices
        nash_price = 1.0 + 1.0 / (1.0 - 0.25 * (n_agents - 1))
        ax1.axhline(y=nash_price, color=COLORS['nash'], 
                   linestyle='--', alpha=0.5, label='Nash')
        
        ax1.set_xlabel('Periods After Deviation')
        ax1.set_ylabel('Price')
        ax1.set_title('Price Response to Deviation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Profit evolution
        profits = np.array(deviation_data['profits'])
        for i in range(n_agents):
            if i == deviating_agent:
                ax2.plot(periods, profits[:, i], 
                        color=COLORS['punishment'], 
                        label=f'Deviating Agent {i+1}', 
                        linewidth=3, marker='o', markersize=4)
            else:
                ax2.plot(periods, profits[:, i], 
                        color=COLORS[f'agent_{i}'], 
                        label=f'Agent {i+1}', 
                        linewidth=2, marker='s', markersize=3)
        
        ax2.axvline(x=0, color='red', linestyle=':', alpha=0.7, label='Deviation')
        ax2.set_xlabel('Periods After Deviation')
        ax2.set_ylabel('Profit')
        ax2.set_title('Profit Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save figure
        if filename is None:
            filename = "punishment_strategy.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_convergence_analysis(
        self,
        results: Dict[str, Any],
        title: str = "Convergence Analysis",
        filename: Optional[str] = None
    ):
        """
        Plot convergence metrics over time.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Experiment results
        title : str
            Plot title
        filename : str, optional
            Output filename
        """
        if 'runs' not in results or not results['runs']:
            return
        
        # Collect convergence data from all runs
        all_convergence = []
        for run in results['runs']:
            if 'convergence_metrics' in run:
                convergence_data = []
                for cm in run['convergence_metrics']:
                    episode = cm['episode']
                    metrics = cm['metrics']
                    convergence_data.append({
                        'episode': episode,
                        'mean_price': metrics.get('mean_price', 0),
                        'std_price': metrics.get('std_price', 0),
                        'mean_profit': metrics.get('mean_profit', 0),
                        'price_correlation': metrics.get('price_correlation', 0)
                    })
                if convergence_data:
                    all_convergence.append(pd.DataFrame(convergence_data))
        
        if not all_convergence:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize_multi)
        
        # Average across runs
        episodes = all_convergence[0]['episode'].values
        
        metrics_to_plot = ['mean_price', 'std_price', 'mean_profit', 'price_correlation']
        axes = [ax1, ax2, ax3, ax4]
        titles = ['Mean Price Convergence', 'Price Volatility', 'Mean Profit', 'Price Correlation']
        
        for metric, ax, subtitle in zip(metrics_to_plot, axes, titles):
            # Calculate mean and std across runs
            metric_values = np.array([df[metric].values for df in all_convergence])
            mean_values = np.mean(metric_values, axis=0)
            std_values = np.std(metric_values, axis=0)
            
            ax.plot(episodes, mean_values, color='blue', linewidth=2)
            ax.fill_between(episodes, 
                           mean_values - std_values, 
                           mean_values + std_values, 
                           alpha=0.3, color='blue')
            
            ax.set_xlabel('Episodes')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(subtitle)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save figure
        if filename is None:
            filename = "convergence_analysis.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_robustness_comparison(
        self,
        suite_results: Dict[str, Any],
        comparison_type: str = "asymmetric_costs",
        title: Optional[str] = None,
        filename: Optional[str] = None
    ):
        """
        Plot robustness comparison across different parameter values.
        
        Parameters:
        -----------
        suite_results : Dict[str, Any]
            Results from experiment suite
        comparison_type : str
            Type of comparison ('asymmetric_costs', 'multiplayer', 'uncertainty')
        title : str, optional
            Plot title
        filename : str, optional
            Output filename
        """
        if 'summary' not in suite_results:
            return
        
        summary = suite_results['summary']
        
        if comparison_type == "asymmetric_costs":
            x_data = summary.get('cost_differences', [])
            y_data = summary.get('average_prices', [])
            x_label = 'Cost Difference'
            y_label = 'Average Price'
            if title is None:
                title = 'Robustness to Asymmetric Costs'
            
        elif comparison_type == "multiplayer":
            x_data = summary.get('n_agents', [])
            y_data = summary.get('average_prices', [])
            x_label = 'Number of Agents'
            y_label = 'Average Price'
            if title is None:
                title = 'Robustness to Number of Players'
            
        elif comparison_type == "uncertainty":
            x_data = summary.get('sigma_values', [])
            y_data = summary.get('average_prices', [])
            x_label = 'Demand Volatility (σ)'
            y_label = 'Average Price'
            if title is None:
                title = 'Robustness to Demand Uncertainty'
        
        else:
            return
        
        if not x_data or not y_data:
            return
        
        fig, ax = plt.subplots(figsize=self.figsize_single)
        
        # Plot main data
        ax.plot(x_data, y_data, 'o-', color='blue', linewidth=2, markersize=8, label='Algorithm Prices')
        
        # Add theoretical benchmarks
        marginal_cost = 1.0
        nash_price = marginal_cost + 1.0 / (1.0 - 0.25 * (2 - 1))  # For 2-player baseline
        monopoly_price = marginal_cost + 1.0
        
        ax.axhline(y=nash_price, color=COLORS['nash'], 
                  linestyle='--', label='Nash Equilibrium', alpha=0.7)
        ax.axhline(y=monopoly_price, color=COLORS['monopoly'], 
                  linestyle='--', label='Monopoly Price', alpha=0.7)
        ax.axhline(y=marginal_cost, color=COLORS['competitive'], 
                  linestyle='--', label='Marginal Cost', alpha=0.7)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        if filename is None:
            filename = f"robustness_{comparison_type}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_price_distribution(
        self,
        results: Dict[str, Any],
        last_n_episodes: int = 5000,
        title: str = "Price Distribution (Final Episodes)",
        filename: Optional[str] = None
    ):
        """
        Plot distribution of prices in final episodes.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Experiment results
        last_n_episodes : int
            Number of final episodes to include
        title : str
            Plot title
        filename : str, optional
            Output filename
        """
        if 'runs' not in results or not results['runs']:
            return
        
        # Collect price data from all runs
        all_prices = []
        n_agents = None
        
        for run in results['runs']:
            if 'prices' in run and run['prices']:
                prices = np.array(run['prices'])
                if n_agents is None:
                    n_agents = prices.shape[1]
                
                # Take last episodes
                final_prices = prices[-last_n_episodes:]
                all_prices.append(final_prices)
        
        if not all_prices or n_agents is None:
            return
        
        # Combine all price data
        combined_prices = np.vstack(all_prices)
        
        fig, axes = plt.subplots(1, n_agents, figsize=(5*n_agents, 6))
        if n_agents == 1:
            axes = [axes]
        
        for i in range(n_agents):
            agent_prices = combined_prices[:, i]
            
            axes[i].hist(agent_prices, bins=30, alpha=0.7, 
                        color=COLORS[f'agent_{i}'], density=True)
            
            # Add vertical lines for benchmarks
            marginal_cost = 1.0
            nash_price = marginal_cost + 1.0 / (1.0 - 0.25 * (n_agents - 1))
            monopoly_price = marginal_cost + 1.0
            
            axes[i].axvline(x=marginal_cost, color=COLORS['competitive'], 
                           linestyle='--', alpha=0.7, label='Marginal Cost')
            axes[i].axvline(x=nash_price, color=COLORS['nash'], 
                           linestyle='--', alpha=0.7, label='Nash')
            axes[i].axvline(x=monopoly_price, color=COLORS['monopoly'], 
                           linestyle='--', alpha=0.7, label='Monopoly')
            
            # Add statistics
            mean_price = np.mean(agent_prices)
            axes[i].axvline(x=mean_price, color='red', 
                           linewidth=2, label=f'Mean: {mean_price:.3f}')
            
            axes[i].set_xlabel('Price')
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'Agent {i+1} Price Distribution')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save figure
        if filename is None:
            filename = "price_distribution.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_profit_evolution(
        self,
        results: Dict[str, Any],
        window_size: int = 1000,
        title: str = "Profit Evolution Over Time",
        filename: Optional[str] = None
    ):
        """
        Plot profit evolution over time.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Experiment results
        window_size : int
            Rolling window size for smoothing
        title : str
            Plot title
        filename : str, optional
            Output filename
        """
        if 'runs' not in results or not results['runs']:
            return
        
        # Use first run for plotting
        run_data = results['runs'][0]
        profits = np.array(run_data['profits'])
        n_agents = profits.shape[1]
        
        fig, ax = plt.subplots(figsize=self.figsize_single)
        
        episodes = np.arange(len(profits))
        for i in range(n_agents):
            # Apply rolling average for smoothing
            if len(profits) > window_size:
                profit_smooth = pd.Series(profits[:, i]).rolling(window_size, center=True).mean()
            else:
                profit_smooth = profits[:, i]
            
            ax.plot(episodes, profit_smooth, 
                   color=COLORS[f'agent_{i}'], 
                   label=f'Agent {i+1}', 
                   linewidth=2)
        
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Profit')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        if filename is None:
            filename = "profit_evolution.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def create_summary_figure(
        self,
        results: Dict[str, Any],
        filename: str = "summary_figure.png"
    ):
        """
        Create a comprehensive summary figure with multiple subplots.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Experiment results
        filename : str
            Output filename
        """
        if 'runs' not in results or not results['runs']:
            return
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplots
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Price dynamics
        ax1 = fig.add_subplot(gs[0, :])
        run_data = results['runs'][0]
        prices = np.array(run_data['prices'])
        n_agents = prices.shape[1]
        
        episodes = np.arange(len(prices))
        for i in range(n_agents):
            price_smooth = pd.Series(prices[:, i]).rolling(1000, center=True).mean()
            ax1.plot(episodes, price_smooth, 
                    color=COLORS[f'agent_{i}'], 
                    label=f'Agent {i+1}', linewidth=2)
        
        # Add benchmarks
        marginal_cost = 1.0
        nash_price = marginal_cost + 1.0 / (1.0 - 0.25 * (n_agents - 1))
        monopoly_price = marginal_cost + 1.0
        
        ax1.axhline(y=nash_price, color=COLORS['nash'], linestyle='--', alpha=0.7)
        ax1.axhline(y=monopoly_price, color=COLORS['monopoly'], linestyle='--', alpha=0.7)
        ax1.axhline(y=marginal_cost, color=COLORS['competitive'], linestyle='--', alpha=0.7)
        
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Price')
        ax1.set_title('Price Dynamics Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Price distribution
        ax2 = fig.add_subplot(gs[1, 0])
        final_prices = prices[-5000:].flatten()
        ax2.hist(final_prices, bins=30, alpha=0.7, density=True, color='blue')
        ax2.axvline(x=np.mean(final_prices), color='red', linewidth=2, 
                   label=f'Mean: {np.mean(final_prices):.3f}')
        ax2.axvline(x=nash_price, color=COLORS['nash'], linestyle='--', alpha=0.7)
        ax2.axvline(x=monopoly_price, color=COLORS['monopoly'], linestyle='--', alpha=0.7)
        ax2.set_xlabel('Price')
        ax2.set_ylabel('Density')
        ax2.set_title('Final Price Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Profit evolution
        ax3 = fig.add_subplot(gs[1, 1])
        profits = np.array(run_data['profits'])
        total_profits = np.sum(profits, axis=1)
        profit_smooth = pd.Series(total_profits).rolling(1000, center=True).mean()
        ax3.plot(episodes, profit_smooth, color='green', linewidth=2)
        ax3.set_xlabel('Episodes')
        ax3.set_ylabel('Total Profit')
        ax3.set_title('Total Profit Evolution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Calculate summary statistics
        if 'summary_statistics' in results:
            stats = results['summary_statistics']
            
            summary_text = "EXPERIMENT SUMMARY\n\n"
            
            if 'final_prices' in stats:
                prices_stats = stats['final_prices']
                for i in range(n_agents):
                    markup = (prices_stats['mean'][i] - marginal_cost) / marginal_cost * 100
                    summary_text += f"Agent {i+1} Final Price: {prices_stats['mean'][i]:.3f} ± {prices_stats['std'][i]:.3f} (markup: {markup:.1f}%)\n"
                
                avg_price = sum(prices_stats['mean']) / len(prices_stats['mean'])
                summary_text += f"\nAverage Final Price: {avg_price:.3f}\n"
                summary_text += f"Nash Equilibrium Price: {nash_price:.3f}\n"
                summary_text += f"Monopoly Price: {monopoly_price:.3f}\n"
                summary_text += f"Marginal Cost: {marginal_cost:.3f}\n\n"
            
            if 'collusion_metrics' in stats:
                collusion = stats['collusion_metrics']
                summary_text += "COLLUSION METRICS:\n"
                for key, value in collusion.items():
                    if isinstance(value, dict) and 'mean' in value:
                        summary_text += f"{key.replace('_', ' ').title()}: {value['mean']:.3f} ± {value['std']:.3f}\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Calvano et al. (2020) Replication Results', fontsize=16, fontweight='bold')
        
        # Save figure
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        plt.close()


def generate_all_figures(results_dict: Dict[str, Any], output_dir: str = "figures"):
    """
    Generate all figures for the paper replication.
    
    Parameters:
    -----------
    results_dict : Dict[str, Any]
        Dictionary containing all experiment results
    output_dir : str
        Output directory for figures
    """
    plotter = CalvanoPlotter(output_dir)
    
    print("Generating figures...")
    
    # Main results figures
    if 'baseline' in results_dict:
        baseline_results = results_dict['baseline']
        
        plotter.plot_price_dynamics(baseline_results, title="Figure 1: Price Dynamics (Baseline)")
        plotter.plot_convergence_analysis(baseline_results, title="Figure 2: Convergence Analysis")
        plotter.plot_price_distribution(baseline_results, title="Figure 3: Price Distribution")
        plotter.create_summary_figure(baseline_results, "figure_1_main_results.png")
        
        if 'deviation_analysis' in baseline_results:
            plotter.plot_punishment_strategy(
                baseline_results['deviation_analysis'], 
                title="Figure 4: Punishment Strategy",
                filename="figure_4_punishment_strategy.png"
            )
    
    # Robustness figures
    if 'asymmetric_costs' in results_dict:
        plotter.plot_robustness_comparison(
            results_dict['asymmetric_costs'],
            comparison_type="asymmetric_costs",
            title="Figure 5: Robustness to Asymmetric Costs",
            filename="figure_5_asymmetric_costs.png"
        )
    
    if 'multiplayer' in results_dict:
        plotter.plot_robustness_comparison(
            results_dict['multiplayer'],
            comparison_type="multiplayer", 
            title="Figure 6: Robustness to Number of Players",
            filename="figure_6_multiplayer.png"
        )
    
    if 'uncertainty' in results_dict:
        plotter.plot_robustness_comparison(
            results_dict['uncertainty'],
            comparison_type="uncertainty",
            title="Figure 7: Robustness to Demand Uncertainty", 
            filename="figure_7_uncertainty.png"
        )
    
    print(f"All figures saved to {output_dir}/")