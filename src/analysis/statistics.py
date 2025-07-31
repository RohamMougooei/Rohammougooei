"""
Statistical Analysis Tools for Calvano et al. (2020) Replication

This module provides statistical tools for analyzing collusion, testing hypotheses,
and measuring the significance of results in algorithmic pricing experiments.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
import warnings


class CollusionAnalyzer:
    """Statistical analysis tools for measuring collusion in pricing experiments."""
    
    def __init__(self):
        """Initialize the analyzer."""
        pass
    
    def calculate_collusion_index(
        self,
        prices: np.ndarray,
        marginal_costs: List[float],
        nash_prices: List[float],
        monopoly_price: float
    ) -> Dict[str, float]:
        """
        Calculate a collusion index based on price levels.
        
        The index measures how close prices are to monopoly vs Nash equilibrium:
        CI = (P - P_Nash) / (P_Monopoly - P_Nash)
        
        Parameters:
        -----------
        prices : np.ndarray
            Array of prices (episodes × agents)
        marginal_costs : List[float]
            Marginal costs for each agent
        nash_prices : List[float]
            Nash equilibrium prices
        monopoly_price : float
            Monopoly price level
            
        Returns:
        --------
        Dict[str, float] : Collusion index statistics
        """
        if prices.size == 0:
            return {}
        
        # Calculate average price per agent
        avg_prices = np.mean(prices, axis=0)
        
        # Calculate collusion index for each agent
        collusion_indices = []
        for i, avg_price in enumerate(avg_prices):
            nash_price = nash_prices[i] if i < len(nash_prices) else nash_prices[0]
            
            if monopoly_price > nash_price:
                ci = (avg_price - nash_price) / (monopoly_price - nash_price)
            else:
                ci = 0.0
            
            collusion_indices.append(ci)
        
        # Overall statistics
        avg_collusion_index = np.mean(collusion_indices)
        
        return {
            'average_collusion_index': avg_collusion_index,
            'agent_collusion_indices': collusion_indices,
            'interpretation': self._interpret_collusion_index(avg_collusion_index)
        }
    
    def test_price_convergence(
        self,
        prices: np.ndarray,
        window_size: int = 1000,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Test for price convergence using trend analysis.
        
        Parameters:
        -----------
        prices : np.ndarray
            Array of prices (episodes × agents)
        window_size : int
            Window size for trend analysis
        significance_level : float
            Significance level for statistical tests
            
        Returns:
        --------
        Dict[str, Any] : Convergence test results
        """
        if len(prices) < window_size:
            return {'error': 'Insufficient data for convergence test'}
        
        results = {}
        n_agents = prices.shape[1]
        
        for agent in range(n_agents):
            agent_prices = prices[:, agent]
            
            # Split into early and late periods
            early_prices = agent_prices[:len(agent_prices)//2]
            late_prices = agent_prices[-window_size:]
            
            # Test for mean difference
            t_stat, p_value = stats.ttest_ind(early_prices, late_prices)
            
            # Test for trend in late period
            episodes = np.arange(len(late_prices))
            slope, intercept, r_value, trend_p_value, std_err = stats.linregress(episodes, late_prices)
            
            # Variance ratio test (convergence should reduce variance)
            early_var = np.var(early_prices)
            late_var = np.var(late_prices)
            variance_ratio = late_var / early_var if early_var > 0 else np.inf
            
            results[f'agent_{agent}'] = {
                'mean_difference_pvalue': p_value,
                'trend_slope': slope,
                'trend_pvalue': trend_p_value,
                'variance_ratio': variance_ratio,
                'converged': (p_value > significance_level and 
                            abs(trend_p_value) > significance_level and 
                            variance_ratio < 1.0)
            }
        
        return results
    
    def compare_to_benchmarks(
        self,
        prices: np.ndarray,
        nash_prices: List[float],
        monopoly_price: float,
        marginal_costs: List[float]
    ) -> Dict[str, Any]:
        """
        Compare experimental prices to theoretical benchmarks.
        
        Parameters:
        -----------
        prices : np.ndarray
            Array of prices (episodes × agents)
        nash_prices : List[float]
            Nash equilibrium prices
        monopoly_price : float
            Monopoly price
        marginal_costs : List[float]
            Marginal costs
            
        Returns:
        --------
        Dict[str, Any] : Benchmark comparison results
        """
        if prices.size == 0:
            return {}
        
        # Calculate final period statistics
        final_window = min(1000, len(prices) // 4)
        final_prices = prices[-final_window:]
        avg_final_prices = np.mean(final_prices, axis=0)
        
        results = {}
        
        for i, agent_price in enumerate(avg_final_prices):
            nash_price = nash_prices[i] if i < len(nash_prices) else nash_prices[0]
            marginal_cost = marginal_costs[i] if i < len(marginal_costs) else marginal_costs[0]
            
            # Statistical tests
            agent_final_prices = final_prices[:, i]
            
            # Test against Nash equilibrium
            nash_tstat, nash_pvalue = stats.ttest_1samp(agent_final_prices, nash_price)
            
            # Test against monopoly price
            mono_tstat, mono_pvalue = stats.ttest_1samp(agent_final_prices, monopoly_price)
            
            # Test against marginal cost
            mc_tstat, mc_pvalue = stats.ttest_1samp(agent_final_prices, marginal_cost)
            
            # Calculate markups
            markup_over_mc = (agent_price - marginal_cost) / marginal_cost * 100
            markup_over_nash = (agent_price - nash_price) / nash_price * 100
            
            results[f'agent_{i}'] = {
                'final_price': agent_price,
                'markup_over_mc_percent': markup_over_mc,
                'markup_over_nash_percent': markup_over_nash,
                'significantly_above_nash': nash_pvalue < 0.05 and nash_tstat > 0,
                'significantly_below_monopoly': mono_pvalue < 0.05 and mono_tstat < 0,
                'significantly_above_mc': mc_pvalue < 0.05 and mc_tstat > 0,
                'nash_test_pvalue': nash_pvalue,
                'monopoly_test_pvalue': mono_pvalue,
                'mc_test_pvalue': mc_pvalue
            }
        
        return results
    
    def analyze_price_coordination(
        self,
        prices: np.ndarray,
        window_size: int = 1000
    ) -> Dict[str, float]:
        """
        Analyze price coordination between agents.
        
        Parameters:
        -----------
        prices : np.ndarray
            Array of prices (episodes × agents)
        window_size : int
            Window for analysis
            
        Returns:
        --------
        Dict[str, float] : Coordination analysis results
        """
        if prices.shape[1] < 2:
            return {'error': 'Need at least 2 agents for coordination analysis'}
        
        # Use final window for analysis
        final_window = min(window_size, len(prices))
        final_prices = prices[-final_window:]
        
        results = {}
        
        # Pairwise correlations
        n_agents = prices.shape[1]
        correlations = []
        
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                corr, p_value = stats.pearsonr(final_prices[:, i], final_prices[:, j])
                correlations.append(corr)
                results[f'correlation_agent_{i}_agent_{j}'] = corr
                results[f'correlation_pvalue_agent_{i}_agent_{j}'] = p_value
        
        # Overall coordination metrics
        results['average_correlation'] = np.mean(correlations)
        results['min_correlation'] = np.min(correlations)
        results['max_correlation'] = np.max(correlations)
        
        # Price dispersion
        price_dispersions = []
        for episode in final_prices:
            dispersion = np.std(episode) / np.mean(episode) if np.mean(episode) > 0 else 0
            price_dispersions.append(dispersion)
        
        results['average_price_dispersion'] = np.mean(price_dispersions)
        results['price_dispersion_trend'] = self._calculate_trend(price_dispersions)
        
        return results
    
    def measure_punishment_effectiveness(
        self,
        deviation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Measure the effectiveness of punishment strategies.
        
        Parameters:
        -----------
        deviation_results : Dict[str, Any]
            Results from deviation experiments
            
        Returns:
        --------
        Dict[str, Any] : Punishment effectiveness metrics
        """
        if not deviation_results:
            return {}
        
        punishment_metrics = []
        
        for key, deviation_data in deviation_results.items():
            prices = np.array(deviation_data['prices'])
            profits = np.array(deviation_data['profits'])
            deviating_agent = deviation_data['deviating_agent']
            deviation_price = deviation_data['deviation_price']
            
            # Analyze punishment pattern
            other_agents = [i for i in range(prices.shape[1]) if i != deviating_agent]
            
            if other_agents:
                other_agent = other_agents[0]  # Focus on first other agent
                other_prices = prices[:, other_agent]
                
                # Measure punishment severity
                pre_deviation_price = other_prices[0]
                min_punishment_price = np.min(other_prices[1:6])  # First 5 periods
                punishment_severity = (pre_deviation_price - min_punishment_price) / pre_deviation_price
                
                # Measure punishment duration
                punishment_duration = self._calculate_punishment_duration(other_prices, pre_deviation_price)
                
                # Measure profit impact on deviator
                deviator_profits = profits[:, deviating_agent]
                initial_profit = deviator_profits[0]
                min_profit = np.min(deviator_profits[1:6])
                profit_loss = (initial_profit - min_profit) / initial_profit if initial_profit > 0 else 0
                
                punishment_metrics.append({
                    'deviation_price': deviation_price,
                    'punishment_severity': punishment_severity,
                    'punishment_duration': punishment_duration,
                    'profit_loss': profit_loss,
                    'effective_punishment': punishment_severity > 0.05 and punishment_duration > 2
                })
        
        if punishment_metrics:
            return {
                'average_punishment_severity': np.mean([m['punishment_severity'] for m in punishment_metrics]),
                'average_punishment_duration': np.mean([m['punishment_duration'] for m in punishment_metrics]),
                'average_profit_loss': np.mean([m['profit_loss'] for m in punishment_metrics]),
                'punishment_effectiveness_rate': np.mean([m['effective_punishment'] for m in punishment_metrics]),
                'individual_punishments': punishment_metrics
            }
        
        return {}
    
    def _interpret_collusion_index(self, ci: float) -> str:
        """Interpret collusion index value."""
        if ci < 0.1:
            return "Competitive behavior (close to Nash)"
        elif ci < 0.3:
            return "Weak collusion"
        elif ci < 0.7:
            return "Moderate collusion"
        elif ci < 0.9:
            return "Strong collusion"
        else:
            return "Near-monopoly behavior"
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend slope."""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        slope, _, _, _, _ = stats.linregress(x, data)
        return slope
    
    def _calculate_punishment_duration(
        self,
        prices: np.ndarray,
        baseline_price: float,
        threshold: float = 0.05
    ) -> int:
        """Calculate how long punishment lasts."""
        for i in range(1, len(prices)):
            if abs(prices[i] - baseline_price) / baseline_price < threshold:
                return i
        return len(prices) - 1


def run_comprehensive_analysis(
    results: Dict[str, Any],
    experiment_name: str = "Experiment"
) -> Dict[str, Any]:
    """
    Run comprehensive statistical analysis on experiment results.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Experiment results
    experiment_name : str
        Name of the experiment
        
    Returns:
    --------
    Dict[str, Any] : Comprehensive analysis results
    """
    analyzer = CollusionAnalyzer()
    analysis_results = {'experiment_name': experiment_name}
    
    if 'runs' not in results or not results['runs']:
        return analysis_results
    
    # Collect price data from all runs
    all_prices = []
    for run in results['runs']:
        if 'prices' in run and run['prices']:
            prices = np.array(run['prices'])
            all_prices.append(prices)
    
    if not all_prices:
        return analysis_results
    
    # Get parameters
    params = results.get('parameters', {})
    marginal_costs = params.get('marginal_costs', [1.0, 1.0])
    n_agents = len(marginal_costs)
    
    # Theoretical benchmarks
    nash_prices = [mc + 1.0 / (1.0 - 0.25 * (n_agents - 1)) for mc in marginal_costs]
    monopoly_price = marginal_costs[0] + 1.0
    
    # Analyze each run
    run_analyses = []
    for i, prices in enumerate(all_prices):
        run_analysis = {}
        
        # Collusion index
        collusion_index = analyzer.calculate_collusion_index(
            prices, marginal_costs, nash_prices, monopoly_price
        )
        run_analysis['collusion_index'] = collusion_index
        
        # Convergence test
        convergence = analyzer.test_price_convergence(prices)
        run_analysis['convergence'] = convergence
        
        # Benchmark comparison
        benchmarks = analyzer.compare_to_benchmarks(
            prices, nash_prices, monopoly_price, marginal_costs
        )
        run_analysis['benchmarks'] = benchmarks
        
        # Coordination analysis
        coordination = analyzer.analyze_price_coordination(prices)
        run_analysis['coordination'] = coordination
        
        run_analyses.append(run_analysis)
    
    analysis_results['run_analyses'] = run_analyses
    
    # Summary across runs
    if run_analyses:
        # Average collusion index
        collusion_indices = [ra['collusion_index'].get('average_collusion_index', 0) 
                           for ra in run_analyses if 'collusion_index' in ra]
        if collusion_indices:
            analysis_results['average_collusion_index'] = {
                'mean': np.mean(collusion_indices),
                'std': np.std(collusion_indices),
                'min': np.min(collusion_indices),
                'max': np.max(collusion_indices)
            }
        
        # Convergence summary
        convergence_rates = []
        for ra in run_analyses:
            if 'convergence' in ra:
                converged_agents = sum(1 for agent_data in ra['convergence'].values() 
                                     if isinstance(agent_data, dict) and agent_data.get('converged', False))
                convergence_rates.append(converged_agents / n_agents)
        
        if convergence_rates:
            analysis_results['convergence_rate'] = {
                'mean': np.mean(convergence_rates),
                'std': np.std(convergence_rates)
            }
        
        # Coordination summary
        correlations = []
        for ra in run_analyses:
            if 'coordination' in ra and 'average_correlation' in ra['coordination']:
                correlations.append(ra['coordination']['average_correlation'])
        
        if correlations:
            analysis_results['average_price_correlation'] = {
                'mean': np.mean(correlations),
                'std': np.std(correlations)
            }
    
    # Punishment analysis (if available)
    if 'deviation_analysis' in results:
        punishment_analysis = analyzer.measure_punishment_effectiveness(
            results['deviation_analysis']
        )
        analysis_results['punishment_analysis'] = punishment_analysis
    
    return analysis_results