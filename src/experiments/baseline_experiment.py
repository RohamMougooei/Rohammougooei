"""
Baseline Experiment

This module implements the baseline symmetric 2-player experiment
from Calvano et al. (2020).
"""

from typing import Dict, Any, Optional, List
from .base_experiment import BaseExperiment


class BaselineExperiment(BaseExperiment):
    """
    Baseline symmetric 2-player experiment.
    
    This is the main experiment from the paper showing that Q-learning
    algorithms learn to collude in a symmetric duopoly setting.
    """
    
    def __init__(self, output_dir: str = "results"):
        """Initialize baseline experiment with paper parameters."""
        super().__init__(
            experiment_name="baseline_symmetric",
            n_agents=2,
            n_prices=15,
            price_min=1.0,
            price_max=2.0,
            marginal_costs=[1.0, 1.0],  # Symmetric marginal costs
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
    
    def run_full_experiment(
        self, 
        n_runs: int = 10, 
        n_episodes: int = 50000,
        run_deviation_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete baseline experiment.
        
        Parameters:
        -----------
        n_runs : int
            Number of independent runs
        n_episodes : int
            Number of episodes per run  
        run_deviation_analysis : bool
            Whether to run deviation analysis
            
        Returns:
        --------
        Dict[str, Any] : Complete experiment results
        """
        print("="*60)
        print("BASELINE SYMMETRIC EXPERIMENT")
        print("="*60)
        print(f"Configuration:")
        print(f"  - Agents: {self.n_agents}")
        print(f"  - Price grid: {self.n_prices} points from {self.price_min} to {self.price_max}")
        print(f"  - Marginal costs: {self.marginal_costs}")
        print(f"  - Environment params: {self.env_params}")
        print(f"  - Agent params: {self.agent_params}")
        print()
        
        # Run main experiments
        results = self.run_multiple_experiments(
            n_runs=n_runs,
            n_episodes=n_episodes,
            verbose=True
        )
        
        # Run deviation analysis on trained agents from first run
        if run_deviation_analysis and results['runs']:
            print("\nRunning deviation analysis...")
            
            # Get trained agents from first run
            first_run = results['runs'][0]
            if hasattr(self, 'agents') and self.agents:
                deviation_prices = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
                deviation_results = self.run_deviation_analysis(
                    self.agents,
                    deviation_prices,
                    n_periods=25
                )
                results['deviation_analysis'] = deviation_results
        
        # Save results
        self.save_results(results, "baseline_symmetric_complete.pkl")
        
        # Print summary
        self.print_experiment_summary(results)
        
        return results
    
    def print_experiment_summary(self, results: Dict[str, Any]):
        """Print a summary of the experiment results."""
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        if 'summary_statistics' in results:
            stats = results['summary_statistics']
            
            if 'final_prices' in stats:
                prices = stats['final_prices']
                print(f"Final Prices (mean ± std):")
                for i in range(self.n_agents):
                    print(f"  Agent {i}: {prices['mean'][i]:.3f} ± {prices['std'][i]:.3f}")
                
                print(f"\nAverage final price: {sum(prices['mean'])/len(prices['mean']):.3f}")
            
            if 'collusion_metrics' in stats:
                collusion = stats['collusion_metrics']
                print(f"\nCollusion Metrics:")
                for key, value in collusion.items():
                    if isinstance(value, dict) and 'mean' in value:
                        print(f"  {key}: {value['mean']:.3f} ± {value['std']:.3f}")
        
        # Theoretical benchmarks
        nash_prices = self.create_environment().get_nash_equilibrium_prices()
        monopoly_price = self.create_environment().get_monopoly_price()
        
        print(f"\nTheoretical Benchmarks:")
        print(f"  Nash equilibrium price: {nash_prices[0]:.3f}")
        print(f"  Monopoly price: {monopoly_price:.3f}")
        print(f"  Marginal cost: {self.marginal_costs[0]:.3f}")
        
        print("="*60)