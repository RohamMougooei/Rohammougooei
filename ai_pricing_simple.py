"""
Simplified AI Pricing and Collusion Replication
Calvano et al. (2019)

This version uses only basic Python packages available in most environments.
"""

import math
import random
import json
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)

class MarketEnvironment:
    """Market environment as described in Calvano et al. (2019)"""
    
    def __init__(self, n_firms=2, demand_elasticity=1.5, cost=0.5, 
                 price_min=0.1, price_max=2.0, noise_std=0.1):
        self.n_firms = n_firms
        self.elasticity = demand_elasticity
        self.cost = cost
        self.price_min = price_min
        self.price_max = price_max
        self.noise_std = noise_std
        
    def demand_function(self, prices):
        """Demand function: Q = (1/p)^elasticity + noise"""
        if isinstance(prices, (int, float)):
            prices = [prices]
        
        demands = []
        for p in prices:
            # Base demand
            base_demand = (1 / p) ** self.elasticity
            # Add noise
            noise = random.gauss(0, self.noise_std)
            demand = max(0, base_demand + noise)
            demands.append(demand)
        
        return demands[0] if len(demands) == 1 else demands
    
    def profit_function(self, price, quantity):
        """Profit function: π = (p - c) * q"""
        return (price - self.cost) * quantity
    
    def get_market_equilibrium(self):
        """Calculate competitive Nash equilibrium using simple grid search"""
        best_price = self.price_min
        best_profit = -float('inf')
        
        for price in [self.price_min + i * 0.01 for i in range(int((self.price_max - self.price_min) * 100))]:
            demand = self.demand_function(price)
            profit = self.profit_function(price, demand)
            if profit > best_profit:
                best_profit = profit
                best_price = price
        
        return best_price
    
    def get_collusive_equilibrium(self):
        """Calculate collusive equilibrium (monopoly)"""
        best_price = self.price_min
        best_profit = -float('inf')
        
        for price in [self.price_min + i * 0.01 for i in range(int((self.price_max - self.price_min) * 100))]:
            demand = self.demand_function(price)
            profit = self.profit_function(price, demand)
            if profit > best_profit:
                best_profit = profit
                best_price = price
        
        return best_price

class QLearningAgent:
    """Q-Learning agent for pricing decisions"""
    
    def __init__(self, n_states=10, n_actions=10, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.Q = [[0.0 for _ in range(n_actions)] for _ in range(n_states)]
        
        # Price discretization
        self.price_min, self.price_max = 0.1, 2.0
        self.prices = [self.price_min + i * (self.price_max - self.price_min) / (n_actions - 1) 
                      for i in range(n_actions)]
        
    def get_state(self, market_history, window=5):
        """Discretize market state based on recent history"""
        if len(market_history) < window:
            return 0
        
        # Extract individual prices from market history
        individual_prices = []
        for period_prices in market_history[-window:]:
            if isinstance(period_prices, list):
                # If it's a list of prices from all firms, take the average
                individual_prices.append(sum(period_prices) / len(period_prices))
            else:
                # If it's a single price
                individual_prices.append(period_prices)
        
        avg_price = sum(individual_prices) / len(individual_prices)
        
        # Discretize state
        state = int((avg_price - self.price_min) / 
                   (self.price_max - self.price_min) * (self.n_states - 1))
        return max(0, min(self.n_states - 1, state))
    
    def choose_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return max(range(self.n_actions), key=lambda a: self.Q[state][a])
    
    def get_price(self, action):
        """Convert action to price"""
        return self.prices[action]
    
    def update(self, state, action, reward, next_state):
        """Update Q-table using Q-learning update rule"""
        best_next_action = max(range(self.n_actions), key=lambda a: self.Q[next_state][a])
        td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.learning_rate * td_error

class MultiAgentSimulation:
    """Multi-agent simulation environment"""
    
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
        """Run the multi-agent simulation"""
        print("Running multi-agent simulation...")
        
        for episode in range(self.n_episodes):
            if episode % 10 == 0:
                print(f"Episode {episode}/{self.n_episodes}")
            
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
                    # Get the price for this specific firm
                    firm_price = prices[i]
                    next_state = agent.get_state(episode_prices + [firm_price])
                    agent.update(states[i], actions[i], profits[i], next_state)
                
                # Store results
                episode_prices.append(prices)
                episode_profits.append(profits)
            
            # Store episode results
            self.price_history.append(episode_prices)
            self.profit_history.append(episode_profits)
            
            # Calculate concentration (HHI)
            avg_prices = [sum(episode_prices[t][i] for t in range(len(episode_prices))) / len(episode_prices) 
                         for i in range(self.n_firms)]
            total_price = sum(avg_prices)
            market_shares = [p / total_price for p in avg_prices]
            hhi = sum(share ** 2 for share in market_shares)
            self.concentration_history.append(hhi)
        
        return self.price_history, self.profit_history, self.concentration_history

def analyze_results(price_history, profit_history, concentration_history, market):
    """Analyze simulation results"""
    print("\nAnalyzing results...")
    
    # Calculate equilibrium prices
    competitive_price = market.get_market_equilibrium()
    collusive_price = market.get_collusive_equilibrium()
    
    # Calculate average prices over time
    n_episodes = len(price_history)
    n_periods = len(price_history[0])
    
    # Calculate average prices across episodes
    avg_prices = []
    for period in range(n_periods):
        period_prices = [price_history[episode][period] for episode in range(n_episodes)]
        avg_period_prices = [sum(period_prices[episode][firm] for episode in range(n_episodes)) / n_episodes 
                           for firm in range(market.n_firms)]
        avg_prices.append(avg_period_prices)
    
    # Final prices (last 100 periods)
    final_prices = []
    for firm in range(market.n_firms):
        firm_final_prices = [avg_prices[period][firm] for period in range(max(0, n_periods-100), n_periods)]
        final_prices.append(sum(firm_final_prices) / len(firm_final_prices))
    
    # Calculate profits
    avg_profits = []
    for period in range(n_periods):
        period_profits = [profit_history[episode][period] for episode in range(n_episodes)]
        avg_period_profits = [sum(period_profits[episode][firm] for episode in range(n_episodes)) / n_episodes 
                             for firm in range(market.n_firms)]
        avg_profits.append(avg_period_profits)
    
    final_profits = []
    for firm in range(market.n_firms):
        firm_final_profits = [avg_profits[period][firm] for period in range(max(0, n_periods-100), n_periods)]
        final_profits.append(sum(firm_final_profits) / len(firm_final_profits))
    
    # Calculate price variance
    all_prices = []
    for episode in range(n_episodes):
        for period in range(n_periods):
            for firm in range(market.n_firms):
                all_prices.append(price_history[episode][period][firm])
    
    price_variance = sum((p - sum(all_prices) / len(all_prices)) ** 2 for p in all_prices) / len(all_prices)
    
    # Create results dictionary
    results = {
        'competitive_price': competitive_price,
        'collusive_price': collusive_price,
        'final_prices': final_prices,
        'final_profits': final_profits,
        'price_variance': price_variance,
        'concentration': sum(concentration_history) / len(concentration_history),
        'price_history': price_history,
        'profit_history': profit_history,
        'avg_prices': avg_prices,
        'avg_profits': avg_profits
    }
    
    return results

def generate_summary_statistics(results, market):
    """Generate summary statistics and tables"""
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
    
    final_avg_price = sum(results['final_prices']) / len(results['final_prices'])
    final_avg_profit = sum(results['final_profits']) / len(results['final_profits'])
    
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
    
    # Save to JSON (since we don't have pandas)
    with open('summary_statistics.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    for metric, value in zip(summary_data['Metric'], summary_data['Value']):
        print(f"{metric:<25}: {value}")
    print("\n" + "="*50)
    
    return summary_data

def create_simple_visualizations(results, market):
    """Create simple text-based visualizations"""
    print("Creating simple visualizations...")
    
    # Price evolution (last 20 periods)
    avg_prices = results['avg_prices']
    n_periods = len(avg_prices)
    
    print("\nPrice Evolution (Last 20 periods):")
    print("Period | Firm 1 | Firm 2 | Competitive | Collusive")
    print("-" * 50)
    
    for period in range(max(0, n_periods-20), n_periods):
        firm1_price = avg_prices[period][0]
        firm2_price = avg_prices[period][1]
        print(f"{period:6d} | {firm1_price:6.3f} | {firm2_price:6.3f} | "
              f"{results['competitive_price']:9.3f} | {results['collusive_price']:8.3f}")
    
    # Price distribution
    print("\nPrice Distribution Analysis:")
    all_prices = []
    for episode in range(len(results['price_history'])):
        for period in range(len(results['price_history'][episode])):
            for firm in range(market.n_firms):
                all_prices.append(results['price_history'][episode][period][firm])
    
    price_ranges = {
        'Very Low (0.1-0.5)': 0,
        'Low (0.5-0.8)': 0,
        'Medium (0.8-1.2)': 0,
        'High (1.2-1.5)': 0,
        'Very High (1.5-2.0)': 0
    }
    
    for price in all_prices:
        if price < 0.5:
            price_ranges['Very Low (0.1-0.5)'] += 1
        elif price < 0.8:
            price_ranges['Low (0.5-0.8)'] += 1
        elif price < 1.2:
            price_ranges['Medium (0.8-1.2)'] += 1
        elif price < 1.5:
            price_ranges['High (1.2-1.5)'] += 1
        else:
            price_ranges['Very High (1.5-2.0)'] += 1
    
    print("\nPrice Distribution:")
    total_prices = len(all_prices)
    for range_name, count in price_ranges.items():
        percentage = (count / total_prices) * 100
        print(f"{range_name:<20}: {count:6d} ({percentage:5.1f}%)")
    
    # Save detailed results
    detailed_results = {
        'competitive_price': results['competitive_price'],
        'collusive_price': results['collusive_price'],
        'final_prices': results['final_prices'],
        'final_profits': results['final_profits'],
        'price_variance': results['price_variance'],
        'concentration': results['concentration'],
        'price_distribution': price_ranges,
        'avg_prices_last_20': [avg_prices[i] for i in range(max(0, n_periods-20), n_periods)]
    }
    
    with open('detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nResults saved to 'summary_statistics.json' and 'detailed_results.json'")

def main():
    """Main function to run the complete replication"""
    print("="*60)
    print("AI PRICING AND COLLUSION REPLICATION")
    print("Calvano et al. (2019) - Simplified Version")
    print("="*60)
    
    # Initialize market environment
    market = MarketEnvironment(n_firms=2, demand_elasticity=1.5, cost=0.5)
    
    print(f"Market Parameters:")
    print(f"- Number of firms: {market.n_firms}")
    print(f"- Demand elasticity: {market.elasticity}")
    print(f"- Marginal cost: {market.cost}")
    print(f"- Competitive price: {market.get_market_equilibrium():.3f}")
    print(f"- Collusive price: {market.get_collusive_equilibrium():.3f}")
    
    # Run main simulation (reduced size for faster execution)
    simulation = MultiAgentSimulation(n_firms=2, n_periods=500, n_episodes=50)
    price_history, profit_history, concentration_history = simulation.run_simulation()
    
    # Analyze results
    results = analyze_results(price_history, profit_history, concentration_history, market)
    
    # Generate summary statistics
    summary_df = generate_summary_statistics(results, market)
    
    # Create simple visualizations
    create_simple_visualizations(results, market)
    
    print("\n" + "="*60)
    print("REPLICATION COMPLETE!")
    print("="*60)
    print("Generated files:")
    print("- summary_statistics.json (summary table)")
    print("- detailed_results.json (detailed analysis)")
    print("="*60)
    
    # Key findings summary
    print("\nKEY FINDINGS:")
    print(f"1. Competitive price: {results['competitive_price']:.3f}")
    print(f"2. Collusive price: {results['collusive_price']:.3f}")
    print(f"3. Final average price: {sum(results['final_prices'])/len(results['final_prices']):.3f}")
    print(f"4. Price variance: {results['price_variance']:.3f}")
    print(f"5. Market concentration (HHI): {results['concentration']:.3f}")
    
    # Interpretation
    final_avg_price = sum(results['final_prices']) / len(results['final_prices'])
    competitive_price = results['competitive_price']
    collusive_price = results['collusive_price']
    
    print("\nINTERPRETATION:")
    if final_avg_price > competitive_price * 1.1:
        print("✓ Evidence of collusion: Final prices are significantly above competitive levels")
    else:
        print("✗ No clear evidence of collusion: Final prices close to competitive levels")
    
    if final_avg_price < collusive_price * 0.9:
        print("✓ Prices below full collusion: Some competitive pressure remains")
    else:
        print("✗ Prices near full collusion: Strong coordination achieved")

if __name__ == "__main__":
    main()