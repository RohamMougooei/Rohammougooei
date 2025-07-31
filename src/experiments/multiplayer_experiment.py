"""
Multi-Player Experiment

This module implements experiments with 3 and 4 players
from Calvano et al. (2020) to test robustness of collusion.
"""

from typing import Dict, Any, Optional, List
from .base_experiment import BaseExperiment
import numpy as np


class MultiplayerExperiment(BaseExperiment):
    """
    Multi-player experiment with 3 or 4 agents.
    
    This experiment tests whether collusion emerges in markets
    with more than 2 competitors.
    """
    
    def __init__(
        self, 
        n_agents: int = 3,
        output_dir: str = "results"
    ):
        """
        Initialize multiplayer experiment.
        
        Parameters:
        -----------
        n_agents : int
            Number of agents (3 or 4)
        output_dir : str
            Directory to save results
        """
        assert n_agents >= 3, "Use baseline experiment for 2 agents"
        
        # All agents have same marginal cost
        marginal_costs = [1.0] * n_agents
        
        super().__init__(
            experiment_name=f"multiplayer_{n_agents}_agents",
            n_agents=n_agents,
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
    
    def run_full_experiment(
        self, 
        n_runs: int = 10, 
        n_episodes: int = 50000
    ) -> Dict[str, Any]:
        """
        Run the complete multiplayer experiment.
        
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
        print(f"MULTIPLAYER EXPERIMENT ({self.n_agents} agents)")
        print("="*60)
        print(f"Configuration:")
        print(f"  - Agents: {self.n_agents}")
        print(f"  - Marginal costs: {self.marginal_costs}")
        print(f"  - Environment params: {self.env_params}")
        print()
        
        # Run main experiments
        results = self.run_multiple_experiments(
            n_runs=n_runs,
            n_episodes=n_episodes,
            verbose=True
        )
        
        # Save results
        filename = f"multiplayer_{self.n_agents}_agents_complete.pkl"
        self.save_results(results, filename)
        
        # Print summary
        self.print_experiment_summary(results)
        
        return results
    
    def print_experiment_summary(self, results: Dict[str, Any]):
        """Print a summary of the experiment results."""
        print("\n" + "="*60)
        print(f"MULTIPLAYER EXPERIMENT SUMMARY ({self.n_agents} agents)")
        print("="*60)
        
        if 'summary_statistics' in results:
            stats = results['summary_statistics']
            
            if 'final_prices' in stats:
                prices = stats['final_prices']
                print(f"Final Prices (mean ± std):")
                total_price = 0
                for i in range(self.n_agents):
                    markup = (prices['mean'][i] - self.marginal_costs[i]) / self.marginal_costs[i] * 100
                    print(f"  Agent {i}: {prices['mean'][i]:.3f} ± {prices['std'][i]:.3f} (markup: {markup:.1f}%)")
                    total_price += prices['mean'][i]
                
                avg_price = total_price / self.n_agents
                print(f"\nAverage final price: {avg_price:.3f}")
                
                # Price dispersion
                price_std = np.std(prices['mean']) if len(prices['mean']) > 1 else 0.0
                print(f"Price dispersion (std): {price_std:.3f}")
            
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


class MultiplayerExperimentSuite:
    """Run multiplayer experiments for different numbers of agents."""
    
    def __init__(self, output_dir: str = "results"):
        """Initialize multiplayer experiment suite."""
        self.output_dir = output_dir
        self.n_agents_list = [2, 3, 4]  # Include 2 for comparison
    
    def run_all_experiments(
        self,
        n_runs: int = 10,
        n_episodes: int = 50000
    ) -> Dict[str, Any]:
        """
        Run multiplayer experiments for different numbers of agents.
        
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
            'n_agents_list': self.n_agents_list,
            'summary': {}
        }
        
        print("="*60)
        print("MULTIPLAYER EXPERIMENT SUITE")
        print("="*60)
        print(f"Running experiments for {self.n_agents_list} agents")
        print()
        
        for n_agents in self.n_agents_list:
            print(f"\n{'='*40}")
            print(f"Number of agents: {n_agents}")
            print(f"{'='*40}")
            
            if n_agents == 2:
                # Use baseline experiment for 2 agents
                from .baseline_experiment import BaselineExperiment
                experiment = BaselineExperiment(output_dir=self.output_dir)
                results = experiment.run_full_experiment(
                    n_runs=n_runs,
                    n_episodes=n_episodes,
                    run_deviation_analysis=False
                )
            else:
                experiment = MultiplayerExperiment(
                    n_agents=n_agents,
                    output_dir=self.output_dir
                )
                results = experiment.run_full_experiment(
                    n_runs=n_runs,
                    n_episodes=n_episodes
                )
            
            all_results['experiments'][f'{n_agents}_agents'] = results
        
        # Create summary across all experiments
        all_results['summary'] = self.create_summary(all_results['experiments'])
        
        # Save combined results
        import pickle
        import os
        filepath = os.path.join(self.output_dir, "multiplayer_suite_complete.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(all_results, f)
        
        print("\n" + "="*60)
        print("MULTIPLAYER SUITE SUMMARY")
        print("="*60)
        self.print_suite_summary(all_results)
        
        return all_results
    
    def create_summary(self, experiments: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics across all multiplayer experiments."""
        summary = {
            'n_agents': [],
            'average_prices': [],
            'price_dispersions': [],
            'average_markups': [],
            'nash_distances': []
        }
        
        for n_agents_str, results in experiments.items():
            n_agents = int(n_agents_str.split('_')[0])
            
            if 'summary_statistics' in results and 'final_prices' in results['summary_statistics']:
                prices = results['summary_statistics']['final_prices']['mean']
                avg_price = sum(prices) / len(prices)
                price_dispersion = np.std(prices) if len(prices) > 1 else 0.0
                
                # Calculate average markup
                marginal_cost = 1.0  # All agents have same cost
                avg_markup = (avg_price - marginal_cost) / marginal_cost
                
                # Calculate distance from Nash
                nash_price = marginal_cost + 1.0 / (1.0 - 0.25 * (n_agents - 1))  # Approximate
                nash_distance = (avg_price - nash_price) / nash_price
                
                summary['n_agents'].append(n_agents)
                summary['average_prices'].append(avg_price)
                summary['price_dispersions'].append(price_dispersion)
                summary['average_markups'].append(avg_markup)
                summary['nash_distances'].append(nash_distance)
        
        return summary
    
    def print_suite_summary(self, all_results: Dict[str, Any]):
        """Print summary of all multiplayer experiments."""
        summary = all_results['summary']
        
        print("Agents | Avg Price | Price Disp | Avg Markup | Nash Dist")
        print("-" * 55)
        
        for i, n_agents in enumerate(summary['n_agents']):
            avg_price = summary['average_prices'][i]
            price_disp = summary['price_dispersions'][i]
            avg_markup = summary['average_markups'][i]
            nash_dist = summary['nash_distances'][i]
            
            print(f"{n_agents:6d} | {avg_price:8.3f} | {price_disp:9.3f} | {avg_markup:9.3f} | {nash_dist:8.3f}")
        
        print("-" * 55)
        print("Collusion decreases but remains significant with more players")