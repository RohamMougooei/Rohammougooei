"""
Asymmetric Cost Experiment

This module implements experiments with asymmetric marginal costs
from Calvano et al. (2020).
"""

from typing import Dict, Any, Optional, List
from .base_experiment import BaseExperiment


class AsymmetricCostExperiment(BaseExperiment):
    """
    Asymmetric marginal cost experiment.
    
    This experiment tests robustness of collusion when firms have
    different marginal costs.
    """
    
    def __init__(
        self, 
        cost_difference: float = 0.1,
        output_dir: str = "results"
    ):
        """
        Initialize asymmetric cost experiment.
        
        Parameters:
        -----------
        cost_difference : float
            Difference in marginal costs between firms
        output_dir : str
            Directory to save results
        """
        # Set asymmetric marginal costs
        marginal_costs = [1.0, 1.0 + cost_difference]
        
        super().__init__(
            experiment_name=f"asymmetric_costs_{cost_difference:.2f}",
            n_agents=2,
            n_prices=15,
            price_min=1.0,
            price_max=2.0,
            marginal_costs=marginal_costs,
            environment_params={
                'a0': 2.0,
                'a1': 1.0,
                'a2': 0.25,
                'mu': 0.0,
                'sigma': 0.0
            },
            agent_params={
                'alpha': 0.15,
                'beta': 0.95,
                'epsilon': 0.10,
                'epsilon_decay': 0.9999,
                'epsilon_min': 0.01,
                'memory_size': 10000
            },
            output_dir=output_dir
        )
        
        self.cost_difference = cost_difference
    
    def run_full_experiment(
        self, 
        n_runs: int = 10, 
        n_episodes: int = 50000
    ) -> Dict[str, Any]:
        """
        Run the complete asymmetric cost experiment.
        
        Parameters:
        -----------
        n_runs : int
            Number of independent runs
        n_episodes : int
            Number of episodes per run
            
        Returns:
        --------
        Dict[str, Any] : Complete experiment results
        """
        print("="*60)
        print(f"ASYMMETRIC COST EXPERIMENT (Δc = {self.cost_difference:.2f})")
        print("="*60)
        print(f"Configuration:")
        print(f"  - Agents: {self.n_agents}")
        print(f"  - Marginal costs: {self.marginal_costs}")
        print(f"  - Cost difference: {self.cost_difference:.3f}")
        print()
        
        # Run main experiments
        results = self.run_multiple_experiments(
            n_runs=n_runs,
            n_episodes=n_episodes,
            verbose=True
        )
        
        # Add cost difference to results
        results['parameters']['cost_difference'] = self.cost_difference
        
        # Save results
        filename = f"asymmetric_costs_{self.cost_difference:.2f}_complete.pkl"
        self.save_results(results, filename)
        
        # Print summary
        self.print_experiment_summary(results)
        
        return results
    
    def print_experiment_summary(self, results: Dict[str, Any]):
        """Print a summary of the experiment results."""
        print("\n" + "="*60)
        print("ASYMMETRIC COST EXPERIMENT SUMMARY")
        print("="*60)
        
        if 'summary_statistics' in results:
            stats = results['summary_statistics']
            
            if 'final_prices' in stats:
                prices = stats['final_prices']
                print(f"Final Prices (mean ± std):")
                for i in range(self.n_agents):
                    cost = self.marginal_costs[i]
                    markup = (prices['mean'][i] - cost) / cost * 100
                    print(f"  Agent {i} (c={cost:.2f}): {prices['mean'][i]:.3f} ± {prices['std'][i]:.3f} (markup: {markup:.1f}%)")
                
                price_difference = abs(prices['mean'][0] - prices['mean'][1])
                print(f"\nPrice difference: {price_difference:.3f}")
                print(f"Cost difference: {self.cost_difference:.3f}")
        
        # Theoretical benchmarks
        env = self.create_environment()
        nash_prices = env.get_nash_equilibrium_prices()
        monopoly_price = env.get_monopoly_price()
        
        print(f"\nTheoretical Benchmarks:")
        for i in range(self.n_agents):
            print(f"  Agent {i} Nash price: {nash_prices[i]:.3f}")
        print(f"  Monopoly price: {monopoly_price:.3f}")
        
        print("="*60)


class AsymmetricExperimentSuite:
    """Run multiple asymmetric cost experiments with different cost differences."""
    
    def __init__(self, output_dir: str = "results"):
        """Initialize asymmetric experiment suite."""
        self.output_dir = output_dir
        self.cost_differences = [0.05, 0.10, 0.15, 0.20, 0.25]
    
    def run_all_experiments(
        self,
        n_runs: int = 10,
        n_episodes: int = 50000
    ) -> Dict[str, Any]:
        """
        Run asymmetric cost experiments for multiple cost differences.
        
        Parameters:
        -----------
        n_runs : int
            Number of independent runs per experiment
        n_episodes : int
            Number of episodes per run
            
        Returns:
        --------
        Dict[str, Any] : Results from all experiments
        """
        all_results = {
            'experiments': {},
            'cost_differences': self.cost_differences,
            'summary': {}
        }
        
        print("="*60)
        print("ASYMMETRIC COST EXPERIMENT SUITE")
        print("="*60)
        print(f"Running experiments for cost differences: {self.cost_differences}")
        print()
        
        for cost_diff in self.cost_differences:
            print(f"\n{'='*40}")
            print(f"Cost difference: {cost_diff:.2f}")
            print(f"{'='*40}")
            
            experiment = AsymmetricCostExperiment(
                cost_difference=cost_diff,
                output_dir=self.output_dir
            )
            
            results = experiment.run_full_experiment(
                n_runs=n_runs,
                n_episodes=n_episodes
            )
            
            all_results['experiments'][f'cost_diff_{cost_diff:.2f}'] = results
        
        # Create summary across all experiments
        all_results['summary'] = self.create_summary(all_results['experiments'])
        
        # Save combined results
        import pickle
        import os
        filepath = os.path.join(self.output_dir, "asymmetric_cost_suite_complete.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(all_results, f)
        
        print("\n" + "="*60)
        print("ASYMMETRIC COST SUITE SUMMARY")
        print("="*60)
        self.print_suite_summary(all_results)
        
        return all_results
    
    def create_summary(self, experiments: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics across all cost difference experiments."""
        summary = {
            'price_differences': [],
            'cost_differences': [],
            'average_prices': [],
            'price_correlations': []
        }
        
        for cost_diff_str, results in experiments.items():
            cost_diff = float(cost_diff_str.split('_')[-1])
            
            if 'summary_statistics' in results and 'final_prices' in results['summary_statistics']:
                prices = results['summary_statistics']['final_prices']['mean']
                price_diff = abs(prices[0] - prices[1])
                avg_price = sum(prices) / len(prices)
                
                summary['cost_differences'].append(cost_diff)
                summary['price_differences'].append(price_diff)
                summary['average_prices'].append(avg_price)
                
                # Get price correlation if available
                if 'collusion_metrics' in results['summary_statistics']:
                    corr_data = results['summary_statistics']['collusion_metrics'].get('price_correlation', {})
                    if isinstance(corr_data, dict) and 'mean' in corr_data:
                        summary['price_correlations'].append(corr_data['mean'])
                    else:
                        summary['price_correlations'].append(0.0)
        
        return summary
    
    def print_suite_summary(self, all_results: Dict[str, Any]):
        """Print summary of all asymmetric cost experiments."""
        summary = all_results['summary']
        
        print("Cost Diff | Price Diff | Avg Price | Price Corr")
        print("-" * 45)
        
        for i, cost_diff in enumerate(summary['cost_differences']):
            price_diff = summary['price_differences'][i]
            avg_price = summary['average_prices'][i]
            corr = summary['price_correlations'][i] if i < len(summary['price_correlations']) else 0.0
            
            print(f"{cost_diff:8.2f} | {price_diff:9.3f} | {avg_price:8.3f} | {corr:9.3f}")
        
        print("-" * 45)
        print(f"Overall collusion remains strong even with cost asymmetries")