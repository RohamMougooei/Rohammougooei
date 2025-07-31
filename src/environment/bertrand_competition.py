"""
Bertrand Competition Environment

This module implements the economic environment for the Calvano et al. (2020)
paper on algorithmic pricing. It includes the logit demand function, profit
calculations, and market dynamics for repeated price competition.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import random


class BertrandEnvironment:
    """
    Bertrand competition environment with logit demand.
    
    This class implements the economic environment described in Calvano et al. (2020),
    including the logit demand function, profit calculations, and market dynamics.
    """
    
    def __init__(
        self,
        n_agents: int = 2,
        a0: float = 2.0,
        a1: float = 1.0,
        a2: float = 0.25,
        mu: float = 0.0,
        sigma: float = 0.0,
        marginal_costs: Optional[List[float]] = None
    ):
        """
        Initialize Bertrand competition environment.
        
        Parameters:
        -----------
        n_agents : int
            Number of competing agents/firms (default: 2)
        a0 : float
            Demand intercept parameter (default: 2.0)
        a1 : float
            Own-price elasticity parameter (default: 1.0)
        a2 : float
            Cross-price elasticity parameter (default: 0.25)
        mu : float
            Mean of demand shock (default: 0.0)
        sigma : float
            Standard deviation of demand shock (default: 0.0)
        marginal_costs : List[float] or None
            Marginal costs for each agent (default: [1.0] * n_agents)
        """
        self.n_agents = n_agents
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.mu = mu
        self.sigma = sigma
        
        # Set marginal costs
        if marginal_costs is None:
            self.marginal_costs = [1.0] * n_agents
        else:
            assert len(marginal_costs) == n_agents, "Must provide marginal cost for each agent"
            self.marginal_costs = marginal_costs.copy()
        
        # Track market history
        self.price_history = []
        self.quantity_history = []
        self.profit_history = []
        self.demand_shock_history = []
        
        # Current period state
        self.current_prices = None
        self.current_quantities = None
        self.current_profits = None
        self.current_demand_shock = 0.0
        
        # Episode tracking
        self.episode = 0
        
    def logit_demand(self, prices: List[float], demand_shock: float = 0.0) -> List[float]:
        """
        Calculate quantities demanded using logit demand function.
        
        The demand function is:
        q_i = exp(a0 - a1*p_i + a2*sum(p_j for j≠i) + ε) / 
              (1 + sum_k exp(a0 - a1*p_k + a2*sum(p_j for j≠k) + ε))
        
        Parameters:
        -----------
        prices : List[float]
            Prices set by each agent
        demand_shock : float
            Random demand shock (default: 0.0)
            
        Returns:
        --------
        List[float] : Quantities demanded for each agent
        """
        prices = np.array(prices)
        n = len(prices)
        
        # Calculate the utilities for each firm
        utilities = np.zeros(n)
        for i in range(n):
            # Own price effect
            own_price_effect = -self.a1 * prices[i]
            # Cross price effects (competitors' prices)
            cross_price_effect = self.a2 * np.sum(prices) - self.a2 * prices[i]
            # Utility for firm i
            utilities[i] = self.a0 + own_price_effect + cross_price_effect + demand_shock
        
        # Calculate market shares using logit formula
        # Add outside option (normalized to 0)
        exp_utilities = np.exp(utilities)
        total_exp_utility = 1.0 + np.sum(exp_utilities)  # 1 for outside option
        
        # Market shares
        market_shares = exp_utilities / total_exp_utility
        
        # Convert to quantities (assuming market size = 1)
        quantities = market_shares.tolist()
        
        return quantities
    
    def calculate_profits(self, prices: List[float], quantities: List[float]) -> List[float]:
        """
        Calculate profits for each agent.
        
        Parameters:
        -----------
        prices : List[float]
            Prices set by each agent
        quantities : List[float]
            Quantities sold by each agent
            
        Returns:
        --------
        List[float] : Profits for each agent
        """
        profits = []
        for i in range(self.n_agents):
            profit = (prices[i] - self.marginal_costs[i]) * quantities[i]
            profits.append(profit)
        
        return profits
    
    def step(self, prices: List[float]) -> Tuple[List[float], List[float], Dict[str, Any]]:
        """
        Execute one step of the environment.
        
        Parameters:
        -----------
        prices : List[float]
            Prices set by each agent
            
        Returns:
        --------
        Tuple[List[float], List[float], Dict[str, Any]] : 
            (quantities, profits, info)
        """
        assert len(prices) == self.n_agents, f"Must provide {self.n_agents} prices"
        
        # Generate demand shock
        if self.sigma > 0:
            self.current_demand_shock = np.random.normal(self.mu, self.sigma)
        else:
            self.current_demand_shock = self.mu
        
        # Calculate quantities using logit demand
        quantities = self.logit_demand(prices, self.current_demand_shock)
        
        # Calculate profits
        profits = self.calculate_profits(prices, quantities)
        
        # Store current state
        self.current_prices = prices.copy()
        self.current_quantities = quantities.copy()
        self.current_profits = profits.copy()
        
        # Update history
        self.price_history.append(prices.copy())
        self.quantity_history.append(quantities.copy())
        self.profit_history.append(profits.copy())
        self.demand_shock_history.append(self.current_demand_shock)
        
        # Increment episode
        self.episode += 1
        
        # Create info dictionary
        info = {
            'episode': self.episode,
            'demand_shock': self.current_demand_shock,
            'total_quantity': sum(quantities),
            'market_concentration': self.calculate_hhi(quantities),
            'average_price': np.mean(prices),
            'price_dispersion': np.std(prices)
        }
        
        return quantities, profits, info
    
    def calculate_hhi(self, quantities: List[float]) -> float:
        """Calculate Herfindahl-Hirschman Index."""
        total_quantity = sum(quantities)
        if total_quantity == 0:
            return 0.0
        
        hhi = sum((q / total_quantity) ** 2 for q in quantities)
        return hhi
    
    def get_nash_equilibrium_prices(self) -> List[float]:
        """
        Calculate Nash equilibrium prices analytically (for symmetric case).
        
        For symmetric firms with marginal cost c, the Nash price is:
        p_Nash = c + 1/(a1 - a2*(n-1))
        
        Returns:
        --------
        List[float] : Nash equilibrium prices
        """
        nash_prices = []
        for i in range(self.n_agents):
            c = self.marginal_costs[i]
            # Symmetric calculation (approximation for asymmetric case)
            denominator = self.a1 - self.a2 * (self.n_agents - 1)
            if denominator > 0:
                nash_price = c + 1.0 / denominator
            else:
                nash_price = c + 0.1  # Fallback
            nash_prices.append(nash_price)
        
        return nash_prices
    
    def get_monopoly_price(self) -> float:
        """
        Calculate monopoly price (same for all firms in symmetric case).
        
        Returns:
        --------
        float : Monopoly price
        """
        # For symmetric case with marginal cost c
        c = np.mean(self.marginal_costs)
        monopoly_price = c + 1.0 / self.a1
        return monopoly_price
    
    def get_competitive_price(self) -> List[float]:
        """
        Get competitive (marginal cost) prices.
        
        Returns:
        --------
        List[float] : Competitive prices (= marginal costs)
        """
        return self.marginal_costs.copy()
    
    def calculate_collusion_metrics(self, window_size: int = 1000) -> Dict[str, float]:
        """
        Calculate metrics to measure degree of collusion.
        
        Parameters:
        -----------
        window_size : int
            Number of recent episodes to consider
            
        Returns:
        --------
        Dict[str, float] : Collusion metrics
        """
        if len(self.price_history) < window_size:
            recent_prices = self.price_history
            recent_profits = self.profit_history
        else:
            recent_prices = self.price_history[-window_size:]
            recent_profits = self.profit_history[-window_size:]
        
        if len(recent_prices) == 0:
            return {}
        
        # Calculate average prices and profits
        avg_prices = np.mean(recent_prices, axis=0)
        avg_profits = np.mean(recent_profits, axis=0)
        
        # Get benchmark prices
        nash_prices = self.get_nash_equilibrium_prices()
        monopoly_price = self.get_monopoly_price()
        competitive_prices = self.get_competitive_price()
        
        # Calculate metrics
        metrics = {}
        
        for i in range(self.n_agents):
            # Price markup over marginal cost
            markup = (avg_prices[i] - self.marginal_costs[i]) / self.marginal_costs[i]
            metrics[f'markup_agent_{i}'] = markup
            
            # Distance from Nash
            nash_distance = (avg_prices[i] - nash_prices[i]) / nash_prices[i]
            metrics[f'nash_distance_agent_{i}'] = nash_distance
            
            # Distance from monopoly
            monopoly_distance = abs(avg_prices[i] - monopoly_price) / monopoly_price
            metrics[f'monopoly_distance_agent_{i}'] = monopoly_distance
        
        # Overall metrics
        metrics['average_price'] = np.mean(avg_prices)
        metrics['average_profit'] = np.mean(avg_profits)
        metrics['total_profit'] = np.sum(avg_profits)
        metrics['price_correlation'] = self.calculate_price_correlation(recent_prices)
        metrics['profit_margin'] = np.mean([(avg_prices[i] - self.marginal_costs[i]) / avg_prices[i] 
                                          for i in range(self.n_agents)])
        
        return metrics
    
    def calculate_price_correlation(self, price_history: List[List[float]]) -> float:
        """Calculate correlation between agents' prices."""
        if len(price_history) < 2 or self.n_agents < 2:
            return 0.0
        
        price_matrix = np.array(price_history)
        correlations = []
        
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                corr = np.corrcoef(price_matrix[:, i], price_matrix[:, j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def simulate_deviation_response(self, agents, deviating_agent: int, 
                                  deviation_price: float, n_periods: int = 20) -> Dict[str, Any]:
        """
        Simulate response to a price deviation.
        
        This method is used to analyze punishment strategies by forcing one agent
        to deviate and observing how others respond.
        
        Parameters:
        -----------
        agents : List[QLearningAgent]
            List of Q-learning agents
        deviating_agent : int
            Index of agent that will deviate
        deviation_price : float
            Price for the deviating agent
        n_periods : int
            Number of periods to simulate
            
        Returns:
        --------
        Dict[str, Any] : Results of deviation experiment
        """
        # Store original state
        original_episode = self.episode
        
        # Get initial state (last prices)
        if len(self.price_history) > 0:
            last_prices = self.price_history[-1].copy()
        else:
            last_prices = [1.5] * self.n_agents  # Default starting prices
        
        # Simulation results
        deviation_prices = []
        deviation_profits = []
        
        for period in range(n_periods):
            # Get prices from agents
            current_prices = []
            
            for i, agent in enumerate(agents):
                if i == deviating_agent and period == 0:
                    # Force deviation in first period
                    price = deviation_price
                else:
                    # Get price from agent (no exploration for cleaner results)
                    if period == 0:
                        prev_own = last_prices[i] if len(last_prices) > i else None
                        competitor_prices = [last_prices[j] for j in range(len(last_prices)) 
                                           if j != i and j < len(last_prices)]
                    else:
                        prev_own = deviation_prices[-1][i]
                        competitor_prices = [deviation_prices[-1][j] for j in range(self.n_agents) if j != i]
                    
                    price = agent.get_price(prev_own, competitor_prices, exploration=False)
                
                current_prices.append(price)
            
            # Execute market step
            quantities, profits, info = self.step(current_prices)
            
            # Store results
            deviation_prices.append(current_prices.copy())
            deviation_profits.append(profits.copy())
        
        # Restore original episode counter
        self.episode = original_episode
        
        return {
            'prices': deviation_prices,
            'profits': deviation_profits,
            'deviating_agent': deviating_agent,
            'deviation_price': deviation_price,
            'n_periods': n_periods
        }
    
    def reset(self):
        """Reset environment for new experiment."""
        self.price_history.clear()
        self.quantity_history.clear()
        self.profit_history.clear()
        self.demand_shock_history.clear()
        
        self.current_prices = None
        self.current_quantities = None
        self.current_profits = None
        self.current_demand_shock = 0.0
        self.episode = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get environment statistics."""
        if len(self.price_history) == 0:
            return {}
        
        price_matrix = np.array(self.price_history)
        profit_matrix = np.array(self.profit_history)
        
        stats = {
            'n_episodes': len(self.price_history),
            'n_agents': self.n_agents,
            'parameters': {
                'a0': self.a0,
                'a1': self.a1,
                'a2': self.a2,
                'mu': self.mu,
                'sigma': self.sigma
            },
            'marginal_costs': self.marginal_costs,
            'nash_prices': self.get_nash_equilibrium_prices(),
            'monopoly_price': self.get_monopoly_price(),
            'competitive_prices': self.get_competitive_price()
        }
        
        # Price statistics
        for i in range(self.n_agents):
            stats[f'agent_{i}_mean_price'] = np.mean(price_matrix[:, i])
            stats[f'agent_{i}_std_price'] = np.std(price_matrix[:, i])
            stats[f'agent_{i}_mean_profit'] = np.mean(profit_matrix[:, i])
        
        # Overall statistics
        stats['mean_price'] = np.mean(price_matrix)
        stats['mean_profit'] = np.mean(profit_matrix)
        stats['total_profit'] = np.sum(profit_matrix)
        
        return stats