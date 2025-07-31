"""
Replication of "ARTIFICIAL INTELLIGENCE, ALGORITHMIC PRICING AND COLLUSION"
by Calvano et al. (2019)

This script implements the multi-agent reinforcement learning framework to study
how AI algorithms can lead to collusion in pricing without explicit communication.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class MarketEnvironment:
    """
    Market environment as described in Calvano et al. (2019)
    """
    
    def __init__(self, n_firms=2, demand_elasticity=1.5, cost=0.5, 
                 price_min=0.1, price_max=2.0, noise_std=0.1):
        """
        Initialize market environment
        
        Parameters:
        - n_firms: number of firms in the market
        - demand_elasticity: elasticity of demand
        - cost: marginal cost of production
        - price_min/max: bounds for prices
        - noise_std: standard deviation of demand noise
        """
        self.n_firms = n_firms
        self.elasticity = demand_elasticity
        self.cost = cost
        self.price_min = price_min
        self.price_max = price_max
        self.noise_std = noise_std
        
    def demand_function(self, prices):
        """
        Demand function: Q = (1/p)^elasticity + noise
        """
        if isinstance(prices, (int, float)):
            prices = [prices]
        
        demands = []
        for p in prices:
            # Base demand
            base_demand = (1 / p) ** self.elasticity
            # Add noise
            noise = np.random.normal(0, self.noise_std)
            demand = max(0, base_demand + noise)
            demands.append(demand)
        
        return demands[0] if len(demands) == 1 else demands
    
    def profit_function(self, price, quantity):
        """
        Profit function: π = (p - c) * q
        """
        return (price - self.cost) * quantity
    
    def get_market_equilibrium(self):
        """
        Calculate competitive Nash equilibrium
        """
        def objective(price):
            demand = self.demand_function(price)
            profit = self.profit_function(price, demand)
            return -profit  # Minimize negative profit
        
        # Find optimal price
        result = minimize(objective, x0=1.0, 
                        bounds=[(self.price_min, self.price_max)])
        return result.x[0]
    
    def get_collusive_equilibrium(self):
        """
        Calculate collusive equilibrium (monopoly)
        """
        def objective(price):
            demand = self.demand_function(price)
            profit = self.profit_function(price, demand)
            return -profit
        
        result = minimize(objective, x0=1.5, 
                        bounds=[(self.price_min, self.price_max)])
        return result.x[0]

class QLearningAgent:
    """
    Q-Learning agent for pricing decisions
    """
    
    def __init__(self, n_states=10, n_actions=10, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.Q = np.zeros((n_states, n_actions))
        
        # Price discretization
        self.price_min, self.price_max = 0.1, 2.0
        self.price_step = (self.price_max - self.price_min) / (n_actions - 1)
        self.prices = np.linspace(self.price_min, self.price_max, n_actions)
        
    def get_state(self, market_history, window=5):
        """
        Discretize market state based on recent history
        """
        if len(market_history) < window:
            return 0
        
        recent_prices = market_history[-window:]
        avg_price = np.mean(recent_prices)
        
        # Discretize state
        state = int((avg_price - self.price_min) / 
                   (self.price_max - self.price_min) * (self.n_states - 1))
        return max(0, min(self.n_states - 1, state))
    
    def choose_action(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def get_price(self, action):
        """
        Convert action to price
        """
        return self.prices[action]
    
    def update(self, state, action, reward, next_state):
        """
        Update Q-table using Q-learning update rule
        """
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.discount_factor * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * td_error

class MultiAgentSimulation:
    """
    Multi-agent simulation environment
    """
    
    def __init__(self, n_firms=2, n_periods=1000, n_episodes=100):
        self.n_firms = n_firms
        self.n_periods = n_periods
        self.n_episodes = n_episodes
        
        # Initialize market and agents
        self.market = MarketEnvironment(n_firms=n_firms)
        self.agents = [QLearningAgent() for _ in range(n_firms)]
        
        # Storage for results
        self.price_history = []
        self.profit_history = []
        self.concentration_history = []
        
    def run_simulation(self):
        """
        Run the multi-agent simulation
        """
        print("Running multi-agent simulation...")
        
        for episode in tqdm(range(self.n_episodes)):
            episode_prices = []
            episode_profits = []
            
            for period in range(self.n_periods):
                # Get current state for each agent
                states = []
                for agent in self.agents:
                    state = agent.get_state(episode_prices)
                    states.append(state)
                
                # Agents choose actions
                actions = []
                prices = []
                for i, agent in enumerate(self.agents):
                    action = agent.choose_action(states[i])
                    price = agent.get_price(action)
                    actions.append(action)
                    prices.append(price)
                
                # Market outcomes
                demands = self.market.demand_function(prices)
                profits = [self.market.profit_function(p, d) 
                          for p, d in zip(prices, demands)]
                
                # Update agents
                for i, agent in enumerate(self.agents):
                    next_state = agent.get_state(episode_prices + [prices[i]])
                    agent.update(states[i], actions[i], profits[i], next_state)
                
                # Store results
                episode_prices.append(prices)
                episode_profits.append(profits)
            
            # Store episode results
            self.price_history.append(episode_prices)
            self.profit_history.append(episode_profits)
            
            # Calculate concentration (HHI)
            avg_prices = np.mean(episode_prices, axis=0)
            market_shares = avg_prices / np.sum(avg_prices)
            hhi = np.sum(market_shares ** 2)
            self.concentration_history.append(hhi)
        
        return self.price_history, self.profit_history, self.concentration_history

def analyze_results(price_history, profit_history, concentration_history, market):
    """
    Analyze simulation results and create visualizations
    """
    print("\nAnalyzing results...")
    
    # Convert to numpy arrays
    price_array = np.array(price_history)
    profit_array = np.array(profit_history)
    
    # Calculate equilibrium prices
    competitive_price = market.get_market_equilibrium()
    collusive_price = market.get_collusive_equilibrium()
    
    # Calculate average prices over time
    avg_prices = np.mean(price_array, axis=0)  # Average across episodes
    final_prices = np.mean(avg_prices[-100:], axis=0)  # Last 100 periods
    
    # Calculate profits
    avg_profits = np.mean(profit_array, axis=0)
    final_profits = np.mean(avg_profits[-100:], axis=0)
    
    # Calculate price variance (measure of collusion)
    price_variance = np.var(price_array, axis=(0, 1))
    
    # Create results dictionary
    results = {
        'competitive_price': competitive_price,
        'collusive_price': collusive_price,
        'final_prices': final_prices,
        'final_profits': final_profits,
        'price_variance': price_variance,
        'concentration': np.mean(concentration_history),
        'price_history': price_array,
        'profit_history': profit_array
    }
    
    return results

def create_visualizations(results, market):
    """
    Create comprehensive visualizations
    """
    print("Creating visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('AI Pricing and Collusion: Replication of Calvano et al. (2019)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Price Evolution
    price_data = results['price_history']
    avg_prices = np.mean(price_data, axis=0)
    
    for i in range(market.n_firms):
        axes[0, 0].plot(avg_prices[:, i], label=f'Firm {i+1}', linewidth=2)
    
    axes[0, 0].axhline(y=results['competitive_price'], color='red', 
                       linestyle='--', label='Competitive', alpha=0.7)
    axes[0, 0].axhline(y=results['collusive_price'], color='green', 
                       linestyle='--', label='Collusive', alpha=0.7)
    axes[0, 0].set_title('Price Evolution Over Time')
    axes[0, 0].set_xlabel('Period')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Profit Evolution
    profit_data = results['profit_history']
    avg_profits = np.mean(profit_data, axis=0)
    
    for i in range(market.n_firms):
        axes[0, 1].plot(avg_profits[:, i], label=f'Firm {i+1}', linewidth=2)
    
    axes[0, 1].set_title('Profit Evolution Over Time')
    axes[0, 1].set_xlabel('Period')
    axes[0, 1].set_ylabel('Profit')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Price Distribution
    final_prices = price_data[:, -100:, :].flatten()
    axes[0, 2].hist(final_prices, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(results['competitive_price'], color='red', 
                       linestyle='--', label='Competitive', linewidth=2)
    axes[0, 2].axvline(results['collusive_price'], color='green', 
                       linestyle='--', label='Collusive', linewidth=2)
    axes[0, 2].set_title('Final Price Distribution')
    axes[0, 2].set_xlabel('Price')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Price Convergence Analysis
    window_size = 50
    price_convergence = []
    for t in range(window_size, len(avg_prices)):
        window_prices = avg_prices[t-window_size:t]
        convergence = np.std(window_prices)
        price_convergence.append(convergence)
    
    axes[1, 0].plot(price_convergence, linewidth=2)
    axes[1, 0].set_title('Price Convergence (Standard Deviation)')
    axes[1, 0].set_xlabel('Period')
    axes[1, 0].set_ylabel('Price Std Dev')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Market Concentration
    axes[1, 1].plot(results['concentration_history'], linewidth=2)
    axes[1, 1].set_title('Market Concentration (HHI)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('HHI')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Price vs Profit Scatter
    final_period_prices = price_data[:, -1, :].flatten()
    final_period_profits = profit_data[:, -1, :].flatten()
    
    axes[1, 2].scatter(final_period_prices, final_period_profits, alpha=0.6)
    axes[1, 2].set_title('Price vs Profit (Final Period)')
    axes[1, 2].set_xlabel('Price')
    axes[1, 2].set_ylabel('Profit')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ai_pricing_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_interactive_plots(results, market):
    """
    Create interactive Plotly visualizations
    """
    print("Creating interactive plots...")
    
    price_data = results['price_history']
    profit_data = results['profit_history']
    avg_prices = np.mean(price_data, axis=0)
    avg_profits = np.mean(profit_data, axis=0)
    
    # Create interactive price evolution plot
    fig = go.Figure()
    
    for i in range(market.n_firms):
        fig.add_trace(go.Scatter(
            x=list(range(len(avg_prices))),
            y=avg_prices[:, i],
            mode='lines',
            name=f'Firm {i+1}',
            line=dict(width=2)
        ))
    
    fig.add_hline(y=results['competitive_price'], line_dash="dash", 
                  line_color="red", annotation_text="Competitive")
    fig.add_hline(y=results['collusive_price'], line_dash="dash", 
                  line_color="green", annotation_text="Collusive")
    
    fig.update_layout(
        title="Interactive Price Evolution",
        xaxis_title="Period",
        yaxis_title="Price",
        template="plotly_white"
    )
    
    fig.write_html("interactive_price_evolution.html")
    
    # Create 3D plot of price evolution
    fig_3d = go.Figure()
    
    for episode in range(min(10, len(price_data))):
        fig_3d.add_trace(go.Scatter3d(
            x=list(range(len(price_data[episode]))),
            y=price_data[episode][:, 0],
            z=price_data[episode][:, 1],
            mode='lines',
            name=f'Episode {episode+1}',
            line=dict(width=2)
        ))
    
    fig_3d.update_layout(
        title="3D Price Evolution Across Episodes",
        scene=dict(
            xaxis_title="Period",
            yaxis_title="Firm 1 Price",
            zaxis_title="Firm 2 Price"
        )
    )
    
    fig_3d.write_html("3d_price_evolution.html")
    
    return fig, fig_3d

def generate_summary_statistics(results, market):
    """
    Generate summary statistics and tables
    """
    print("Generating summary statistics...")
    
    # Calculate key metrics
    competitive_profit = market.profit_function(
        results['competitive_price'], 
        market.demand_function(results['competitive_price'])
    )
    
    collusive_profit = market.profit_function(
        results['collusive_price'], 
        market.demand_function(results['collusive_price'])
    )
    
    final_avg_price = np.mean(results['final_prices'])
    final_avg_profit = np.mean(results['final_profits'])
    
    # Create summary table
    summary_data = {
        'Metric': [
            'Competitive Price',
            'Collusive Price',
            'Final Average Price',
            'Competitive Profit',
            'Collusive Profit',
            'Final Average Profit',
            'Price Variance',
            'Market Concentration (HHI)'
        ],
        'Value': [
            f"{results['competitive_price']:.3f}",
            f"{results['collusive_price']:.3f}",
            f"{final_avg_price:.3f}",
            f"{competitive_profit:.3f}",
            f"{collusive_profit:.3f}",
            f"{final_avg_profit:.3f}",
            f"{results['price_variance']:.3f}",
            f"{results['concentration']:.3f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    summary_df.to_csv('summary_statistics.csv', index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(summary_df.to_string(index=False))
    print("\n" + "="*50)
    
    return summary_df

def run_sensitivity_analysis(market):
    """
    Run sensitivity analysis with different parameters
    """
    print("Running sensitivity analysis...")
    
    # Test different parameters
    parameters = {
        'learning_rate': [0.05, 0.1, 0.2],
        'discount_factor': [0.9, 0.95, 0.99],
        'epsilon': [0.05, 0.1, 0.2],
        'n_firms': [2, 3, 4]
    }
    
    sensitivity_results = {}
    
    for param_name, param_values in parameters.items():
        print(f"Testing {param_name}...")
        param_results = []
        
        for value in param_values:
            # Create simulation with modified parameter
            sim = MultiAgentSimulation(n_firms=market.n_firms, 
                                     n_periods=500, n_episodes=50)
            
            # Modify agent parameters
            for agent in sim.agents:
                if param_name == 'learning_rate':
                    agent.learning_rate = value
                elif param_name == 'discount_factor':
                    agent.discount_factor = value
                elif param_name == 'epsilon':
                    agent.epsilon = value
            
            # Run simulation
            price_hist, profit_hist, conc_hist = sim.run_simulation()
            
            # Analyze results
            results = analyze_results(price_hist, profit_hist, conc_hist, market)
            
            # Store final average price
            final_avg_price = np.mean(results['final_prices'])
            param_results.append({
                'parameter_value': value,
                'final_avg_price': final_avg_price,
                'price_variance': results['price_variance'],
                'concentration': results['concentration']
            })
        
        sensitivity_results[param_name] = param_results
    
    # Create sensitivity plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    for i, (param_name, results) in enumerate(sensitivity_results.items()):
        row, col = i // 2, i % 2
        
        values = [r['parameter_value'] for r in results]
        prices = [r['final_avg_price'] for r in results]
        
        axes[row, col].plot(values, prices, 'o-', linewidth=2, markersize=8)
        axes[row, col].set_title(f'{param_name.replace("_", " ").title()}')
        axes[row, col].set_xlabel('Parameter Value')
        axes[row, col].set_ylabel('Final Average Price')
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return sensitivity_results

def main():
    """
    Main function to run the complete replication
    """
    print("="*60)
    print("AI PRICING AND COLLUSION REPLICATION")
    print("Calvano et al. (2019)")
    print("="*60)
    
    # Initialize market environment
    market = MarketEnvironment(n_firms=2, demand_elasticity=1.5, cost=0.5)
    
    print(f"Market Parameters:")
    print(f"- Number of firms: {market.n_firms}")
    print(f"- Demand elasticity: {market.elasticity}")
    print(f"- Marginal cost: {market.cost}")
    print(f"- Competitive price: {market.get_market_equilibrium():.3f}")
    print(f"- Collusive price: {market.get_collusive_equilibrium():.3f}")
    
    # Run main simulation
    simulation = MultiAgentSimulation(n_firms=2, n_periods=1000, n_episodes=100)
    price_history, profit_history, concentration_history = simulation.run_simulation()
    
    # Analyze results
    results = analyze_results(price_history, profit_history, concentration_history, market)
    
    # Create visualizations
    create_visualizations(results, market)
    create_interactive_plots(results, market)
    
    # Generate summary statistics
    summary_df = generate_summary_statistics(results, market)
    
    # Run sensitivity analysis
    sensitivity_results = run_sensitivity_analysis(market)
    
    print("\n" + "="*60)
    print("REPLICATION COMPLETE!")
    print("="*60)
    print("Generated files:")
    print("- ai_pricing_results.png (static plots)")
    print("- interactive_price_evolution.html (interactive plot)")
    print("- 3d_price_evolution.html (3D plot)")
    print("- sensitivity_analysis.png (sensitivity plots)")
    print("- summary_statistics.csv (summary table)")
    print("="*60)

if __name__ == "__main__":
    main()