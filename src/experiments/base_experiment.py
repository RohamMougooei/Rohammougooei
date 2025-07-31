"""
Base Experiment Class

This module provides the base experiment class for all experiments in the
Calvano et al. (2020) replication. It handles common functionality like
running experiments, collecting data, and saving results.
"""

import numpy as np
import pandas as pd
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import os
from pathlib import Path

from ..agents.q_learning_agent import QLearningAgent
from ..environment.bertrand_competition import BertrandEnvironment


class BaseExperiment:
    """
    Base class for all pricing experiments.
    
    This class provides common functionality for running experiments,
    including agent initialization, environment setup, data collection,
    and result saving.
    """
    
    def __init__(
        self,
        experiment_name: str,
        n_agents: int = 2,
        n_prices: int = 15,
        price_min: float = 1.0,
        price_max: float = 2.0,
        marginal_costs: Optional[List[float]] = None,
        environment_params: Optional[Dict[str, Any]] = None,
        agent_params: Optional[Dict[str, Any]] = None,
        output_dir: str = "results"
    ):
        """
        Initialize base experiment.
        
        Parameters:
        -----------
        experiment_name : str
            Name of the experiment
        n_agents : int
            Number of agents/firms
        n_prices : int
            Number of discrete price points
        price_min : float
            Minimum price
        price_max : float
            Maximum price
        marginal_costs : List[float] or None
            Marginal costs for each agent
        environment_params : Dict[str, Any] or None
            Additional environment parameters
        agent_params : Dict[str, Any] or None
            Additional agent parameters
        output_dir : str
            Directory to save results
        """
        self.experiment_name = experiment_name
        self.n_agents = n_agents
        self.n_prices = n_prices
        self.price_min = price_min
        self.price_max = price_max
        self.marginal_costs = marginal_costs or [1.0] * n_agents
        self.output_dir = output_dir
        
        # Default parameters
        self.default_env_params = {
            'a0': 2.0,
            'a1': 1.0,
            'a2': 0.25,
            'mu': 0.0,
            'sigma': 0.0
        }
        
        self.default_agent_params = {
            'alpha': 0.15,
            'beta': 0.95,
            'epsilon': 0.10,
            'epsilon_decay': 0.9999,
            'epsilon_min': 0.01,
            'memory_size': 10000
        }
        
        # Update with custom parameters
        self.env_params = self.default_env_params.copy()
        if environment_params:
            self.env_params.update(environment_params)
        
        self.agent_params = self.default_agent_params.copy()
        if agent_params:
            self.agent_params.update(agent_params)
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.environment = None
        self.agents = None
        
        # Results storage
        self.results = {}
        
    def create_environment(self) -> BertrandEnvironment:
        """Create and return the environment for this experiment."""
        return BertrandEnvironment(
            n_agents=self.n_agents,
            marginal_costs=self.marginal_costs,
            **self.env_params
        )
    
    def create_agents(self) -> List[QLearningAgent]:
        """Create and return the agents for this experiment."""
        agents = []
        for i in range(self.n_agents):
            agent = QLearningAgent(
                agent_id=i,
                n_prices=self.n_prices,
                price_min=self.price_min,
                price_max=self.price_max,
                marginal_cost=self.marginal_costs[i],
                **self.agent_params
            )
            agents.append(agent)
        return agents
    
    def run_single_experiment(
        self,
        n_episodes: int = 50000,
        convergence_window: int = 1000,
        save_frequency: int = 10000,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run a single experiment.
        
        Parameters:
        -----------
        n_episodes : int
            Number of episodes to run
        convergence_window : int
            Window size for convergence checking
        save_frequency : int
            How often to save intermediate results
        verbose : bool
            Whether to show progress bar
            
        Returns:
        --------
        Dict[str, Any] : Results from the experiment
        """
        # Initialize environment and agents
        self.environment = self.create_environment()
        self.agents = self.create_agents()
        
        # Track experiment data
        experiment_data = {
            'prices': [],
            'profits': [],
            'quantities': [],
            'episodes': [],
            'convergence_metrics': [],
            'agent_statistics': []
        }
        
        # Run episodes
        progress_bar = tqdm(range(n_episodes), desc=f"Running {self.experiment_name}") if verbose else range(n_episodes)
        
        for episode in progress_bar:
            # Get prices from each agent
            current_prices = []
            
            for i, agent in enumerate(self.agents):
                # Get previous market state
                if episode == 0:
                    prev_own_price = None
                    competitor_prices = []
                else:
                    prev_own_price = experiment_data['prices'][-1][i]
                    competitor_prices = [experiment_data['prices'][-1][j] 
                                       for j in range(self.n_agents) if j != i]
                
                # Get agent's price decision
                price = agent.get_price(prev_own_price, competitor_prices)
                current_prices.append(price)
            
            # Execute market step
            quantities, profits, info = self.environment.step(current_prices)
            
            # Update agents with profit information
            for i, agent in enumerate(self.agents):
                competitor_prices = [current_prices[j] for j in range(self.n_agents) if j != i]
                agent.update(profits[i], current_prices[i], competitor_prices)
            
            # Store episode data
            experiment_data['prices'].append(current_prices.copy())
            experiment_data['profits'].append(profits.copy())
            experiment_data['quantities'].append(quantities.copy())
            experiment_data['episodes'].append(episode)
            
            # Calculate convergence metrics periodically
            if episode > 0 and episode % convergence_window == 0:
                metrics = self.calculate_convergence_metrics(experiment_data, convergence_window)
                experiment_data['convergence_metrics'].append({
                    'episode': episode,
                    'metrics': metrics
                })
            
            # Save intermediate results
            if episode > 0 and episode % save_frequency == 0:
                self.save_intermediate_results(experiment_data, episode)
            
            # Update progress bar
            if verbose and episode % 1000 == 0:
                avg_price = np.mean(current_prices)
                avg_profit = np.mean(profits)
                progress_bar.set_postfix({
                    'avg_price': f"{avg_price:.3f}",
                    'avg_profit': f"{avg_profit:.4f}"
                })
        
        # Final agent statistics
        for i, agent in enumerate(self.agents):
            stats = agent.get_statistics()
            experiment_data['agent_statistics'].append({
                'agent_id': i,
                'statistics': stats
            })
        
        # Environment statistics
        experiment_data['environment_statistics'] = self.environment.get_statistics()
        
        # Collusion metrics
        experiment_data['collusion_metrics'] = self.environment.calculate_collusion_metrics()
        
        return experiment_data
    
    def run_multiple_experiments(
        self,
        n_runs: int = 10,
        n_episodes: int = 50000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run multiple independent experiments.
        
        Parameters:
        -----------
        n_runs : int
            Number of independent runs
        n_episodes : int
            Number of episodes per run
        **kwargs : Additional arguments for run_single_experiment
            
        Returns:
        --------
        Dict[str, Any] : Combined results from all runs
        """
        all_results = {
            'runs': [],
            'summary_statistics': {},
            'parameters': {
                'n_runs': n_runs,
                'n_episodes': n_episodes,
                'n_agents': self.n_agents,
                'marginal_costs': self.marginal_costs,
                'env_params': self.env_params,
                'agent_params': self.agent_params
            }
        }
        
        print(f"Running {n_runs} independent experiments for {self.experiment_name}")
        
        for run_id in range(n_runs):
            print(f"\nRun {run_id + 1}/{n_runs}")
            
            # Run single experiment
            run_results = self.run_single_experiment(n_episodes=n_episodes, **kwargs)
            run_results['run_id'] = run_id
            
            all_results['runs'].append(run_results)
        
        # Calculate summary statistics across runs
        all_results['summary_statistics'] = self.calculate_summary_statistics(all_results['runs'])
        
        return all_results
    
    def calculate_convergence_metrics(
        self,
        data: Dict[str, Any],
        window_size: int = 1000
    ) -> Dict[str, float]:
        """Calculate convergence metrics for the last window_size episodes."""
        if len(data['prices']) < window_size:
            return {}
        
        recent_prices = np.array(data['prices'][-window_size:])
        recent_profits = np.array(data['profits'][-window_size:])
        
        metrics = {
            'mean_price': np.mean(recent_prices),
            'std_price': np.std(recent_prices),
            'mean_profit': np.mean(recent_profits),
            'price_correlation': np.corrcoef(recent_prices.T)[0, 1] if self.n_agents == 2 else 0.0,
            'profit_margin': np.mean((recent_prices - np.array(self.marginal_costs)) / recent_prices)
        }
        
        return metrics
    
    def calculate_summary_statistics(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics across multiple runs."""
        # Collect final metrics from each run
        final_prices = []
        final_profits = []
        collusion_metrics = []
        
        for run in runs:
            if run['prices']:
                # Last 1000 episodes
                last_prices = np.array(run['prices'][-1000:])
                last_profits = np.array(run['profits'][-1000:])
                
                final_prices.append(np.mean(last_prices, axis=0))
                final_profits.append(np.mean(last_profits, axis=0))
                
                if run['collusion_metrics']:
                    collusion_metrics.append(run['collusion_metrics'])
        
        if not final_prices:
            return {}
        
        final_prices = np.array(final_prices)
        final_profits = np.array(final_profits)
        
        # Calculate statistics
        stats = {
            'final_prices': {
                'mean': np.mean(final_prices, axis=0).tolist(),
                'std': np.std(final_prices, axis=0).tolist(),
                'min': np.min(final_prices, axis=0).tolist(),
                'max': np.max(final_prices, axis=0).tolist()
            },
            'final_profits': {
                'mean': np.mean(final_profits, axis=0).tolist(),
                'std': np.std(final_profits, axis=0).tolist(),
                'min': np.min(final_profits, axis=0).tolist(),
                'max': np.max(final_profits, axis=0).tolist()
            }
        }
        
        # Average collusion metrics
        if collusion_metrics:
            avg_collusion = {}
            for key in collusion_metrics[0].keys():
                values = [cm[key] for cm in collusion_metrics if key in cm]
                if values:
                    avg_collusion[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
            stats['collusion_metrics'] = avg_collusion
        
        return stats
    
    def save_intermediate_results(self, data: Dict[str, Any], episode: int):
        """Save intermediate results."""
        filename = f"{self.experiment_name}_intermediate_ep{episode}.pkl"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save experiment results."""
        if filename is None:
            filename = f"{self.experiment_name}_results.pkl"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        # Also save summary as JSON
        json_filename = filename.replace('.pkl', '_summary.json')
        json_filepath = os.path.join(self.output_dir, json_filename)
        
        # Create JSON-serializable summary
        summary = {
            'experiment_name': self.experiment_name,
            'parameters': results.get('parameters', {}),
            'summary_statistics': results.get('summary_statistics', {})
        }
        
        with open(json_filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load experiment results."""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        return results
    
    def run_deviation_analysis(
        self,
        trained_agents: List[QLearningAgent],
        deviation_prices: List[float],
        n_periods: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze punishment strategies by simulating deviations.
        
        Parameters:
        -----------
        trained_agents : List[QLearningAgent]
            Pre-trained agents
        deviation_prices : List[float]
            Prices to test as deviations
        n_periods : int
            Number of periods to simulate after deviation
            
        Returns:
        --------
        Dict[str, Any] : Deviation analysis results
        """
        deviation_results = {}
        
        for dev_price in deviation_prices:
            for deviating_agent in range(self.n_agents):
                key = f"agent_{deviating_agent}_price_{dev_price:.2f}"
                
                # Create fresh environment for deviation test
                test_env = self.create_environment()
                
                # Simulate deviation
                result = test_env.simulate_deviation_response(
                    trained_agents,
                    deviating_agent,
                    dev_price,
                    n_periods
                )
                
                deviation_results[key] = result
        
        return deviation_results