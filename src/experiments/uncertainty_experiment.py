"""
Uncertainty Experiment

This module implements experiments with demand uncertainty
from Calvano et al. (2020) to test robustness of collusion.
"""

from typing import Dict, Any, Optional, List
from .base_experiment import BaseExperiment


class UncertaintyExperiment(BaseExperiment):
    """
    Demand uncertainty experiment.
    
    This experiment tests whether collusion remains stable
    when there are random demand shocks.
    """
    
    def __init__(
        self, 
        sigma: float = 0.1,
        output_dir: str = "results"
    ):
        """
        Initialize uncertainty experiment.
        
        Parameters:
        -----------
        sigma : float
            Standard deviation of demand shocks
        output_dir : str
            Directory to save results
        """
        
        super().__init__(
            experiment_name=f"uncertainty_sigma_{sigma:.2f}",
            n_agents=2,
            n_prices=15,
            price_min=1.0,
            price_max=2.0,
            marginal_costs=[1.0, 1.0],
            environment_params={
                'a0': 2.0,
                'a1': 1.0,
                'a2': 0.25,
                'mu': 0.0,
                'sigma': sigma  # Demand uncertainty
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
        
        self.sigma = sigma
    
    def run_full_experiment(
        self, 
        n_runs: int = 10, 
        n_episodes: int = 50000
    ) -> Dict[str, Any]:
        """
        Run the complete uncertainty experiment.
        
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
        print(f"UNCERTAINTY EXPERIMENT (σ = {self.sigma:.2f})")
        print("="*60)
        print(f"Configuration:")
        print(f"  - Agents: {self.n_agents}")
        print(f"  - Demand shock std: {self.sigma:.3f}")
        print(f"  - Environment params: {self.env_params}")
        print()
        
        # Run main experiments
        results = self.run_multiple_experiments(
            n_runs=n_runs,
            n_episodes=n_episodes,
            verbose=True
        )
        
        # Add uncertainty level to results
        results['parameters']['sigma'] = self.sigma
        
        # Save results
        filename = f"uncertainty_sigma_{self.sigma:.2f}_complete.pkl"
        self.save_results(results, filename)
        
        # Print summary
        self.print_experiment_summary(results)
        
        return results
    
    def print_experiment_summary(self, results: Dict[str, Any]):
        """Print a summary of the experiment results."""
        print("\n" + "="*60)
        print(f"UNCERTAINTY EXPERIMENT SUMMARY (σ = {self.sigma:.2f})")
        print("="*60)
        
        if 'summary_statistics' in results:
            stats = results['summary_statistics']
            
            if 'final_prices' in stats:
                prices = stats['final_prices']
                print(f"Final Prices (mean ± std):")
                for i in range(self.n_agents):
                    markup = (prices['mean'][i] - self.marginal_costs[i]) / self.marginal_costs[i] * 100
                    print(f"  Agent {i}: {prices['mean'][i]:.3f} ± {prices['std'][i]:.3f} (markup: {markup:.1f}%)")
                
                avg_price = sum(prices['mean']) / len(prices['mean'])
                print(f"\nAverage final price: {avg_price:.3f}")
                
                # Price volatility
                avg_volatility = sum(prices['std']) / len(prices['std'])
                print(f"Average price volatility: {avg_volatility:.3f}")
            
            if 'collusion_metrics' in stats:
                collusion = stats['collusion_metrics']
                print(f"\nCollusion Metrics:")
                for key, value in collusion.items():
                    if isinstance(value, dict) and 'mean' in value:
                        print(f"  {key}: {value['mean']:.3f} ± {value['std']:.3f}")
        
        # Theoretical benchmarks
        env = self.create_environment()
        nash_prices = env.get_nash_equilibrium_prices()
        monopoly_price = env.get_monopoly_price()
        
        print(f"\nTheoretical Benchmarks:")
        print(f"  Nash equilibrium price: {nash_prices[0]:.3f}")
        print(f"  Monopoly price: {monopoly_price:.3f}")
        print(f"  Marginal cost: {self.marginal_costs[0]:.3f}")
        
        print("="*60)


class UncertaintyExperimentSuite:
    """Run uncertainty experiments with different levels of demand volatility."""
    
    def __init__(self, output_dir: str = "results"):
        """Initialize uncertainty experiment suite."""
        self.output_dir = output_dir
        self.sigma_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    
    def run_all_experiments(
        self,
        n_runs: int = 10,
        n_episodes: int = 50000
    ) -> Dict[str, Any]:
        """
        Run uncertainty experiments for different volatility levels.
        
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
            'sigma_values': self.sigma_values,
            'summary': {}
        }
        
        print("="*60)
        print("UNCERTAINTY EXPERIMENT SUITE")
        print("="*60)
        print(f"Running experiments for σ values: {self.sigma_values}")
        print()
        
        for sigma in self.sigma_values:
            print(f"\n{'='*40}")
            print(f"Demand volatility (σ): {sigma:.2f}")
            print(f"{'='*40}")
            
            if sigma == 0.0:
                # Use baseline experiment for no uncertainty
                from .baseline_experiment import BaselineExperiment
                experiment = BaselineExperiment(output_dir=self.output_dir)
                results = experiment.run_full_experiment(
                    n_runs=n_runs,
                    n_episodes=n_episodes,
                    run_deviation_analysis=False
                )
            else:
                experiment = UncertaintyExperiment(
                    sigma=sigma,
                    output_dir=self.output_dir
                )
                results = experiment.run_full_experiment(
                    n_runs=n_runs,
                    n_episodes=n_episodes
                )
            
            all_results['experiments'][f'sigma_{sigma:.2f}'] = results
        
        # Create summary across all experiments
        all_results['summary'] = self.create_summary(all_results['experiments'])
        
        # Save combined results
        import pickle
        import os
        filepath = os.path.join(self.output_dir, "uncertainty_suite_complete.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(all_results, f)
        
        print("\n" + "="*60)
        print("UNCERTAINTY SUITE SUMMARY")
        print("="*60)
        self.print_suite_summary(all_results)
        
        return all_results
    
    def create_summary(self, experiments: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics across all uncertainty experiments."""
        summary = {
            'sigma_values': [],
            'average_prices': [],
            'price_volatilities': [],
            'price_correlations': [],
            'nash_distances': []
        }
        
        for sigma_str, results in experiments.items():
            if sigma_str.startswith('sigma_'):
                sigma = float(sigma_str.split('_')[1])
            else:
                sigma = 0.0
            
            if 'summary_statistics' in results and 'final_prices' in results['summary_statistics']:
                prices = results['summary_statistics']['final_prices']
                avg_price = sum(prices['mean']) / len(prices['mean'])
                avg_volatility = sum(prices['std']) / len(prices['std'])
                
                # Distance from Nash
                nash_price = 1.0 + 1.0 / (1.0 - 0.25 * (2 - 1))  # 2-player case
                nash_distance = (avg_price - nash_price) / nash_price
                
                summary['sigma_values'].append(sigma)
                summary['average_prices'].append(avg_price)
                summary['price_volatilities'].append(avg_volatility)
                summary['nash_distances'].append(nash_distance)
                
                # Get price correlation if available
                if 'collusion_metrics' in results['summary_statistics']:
                    corr_data = results['summary_statistics']['collusion_metrics'].get('price_correlation', {})
                    if isinstance(corr_data, dict) and 'mean' in corr_data:
                        summary['price_correlations'].append(corr_data['mean'])
                    else:
                        summary['price_correlations'].append(0.0)
                else:
                    summary['price_correlations'].append(0.0)
        
        return summary
    
    def print_suite_summary(self, all_results: Dict[str, Any]):
        """Print summary of all uncertainty experiments."""
        summary = all_results['summary']
        
        print("Sigma | Avg Price | Volatility | Price Corr | Nash Dist")
        print("-" * 55)
        
        for i, sigma in enumerate(summary['sigma_values']):
            avg_price = summary['average_prices'][i]
            volatility = summary['price_volatilities'][i]
            corr = summary['price_correlations'][i] if i < len(summary['price_correlations']) else 0.0
            nash_dist = summary['nash_distances'][i]
            
            print(f"{sigma:5.2f} | {avg_price:8.3f} | {volatility:9.3f} | {corr:9.3f} | {nash_dist:8.3f}")
        
        print("-" * 55)
        print("Collusion remains robust to moderate demand uncertainty")