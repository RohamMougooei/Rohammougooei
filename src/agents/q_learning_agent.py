"""
Q-Learning Agent for Algorithmic Pricing

This module implements the Q-learning algorithm used in Calvano et al. (2020)
for pricing in repeated Bertrand competition. The agent learns to set prices
through reinforcement learning with experience replay.
"""

import numpy as np
import random
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
import pickle


class QLearningAgent:
    """
    Q-Learning agent for algorithmic pricing.
    
    This agent learns to set prices in a repeated Bertrand competition
    using Q-learning with experience replay, following the specification
    in Calvano et al. (2020).
    """
    
    def __init__(
        self,
        agent_id: int,
        n_prices: int = 15,
        alpha: float = 0.15,
        beta: float = 0.95,
        epsilon: float = 0.10,
        epsilon_decay: float = 0.9999,
        epsilon_min: float = 0.01,
        memory_size: int = 10000,
        price_min: float = 1.0,
        price_max: float = 2.0,
        marginal_cost: float = 1.0
    ):
        """
        Initialize Q-learning agent.
        
        Parameters:
        -----------
        agent_id : int
            Unique identifier for the agent
        n_prices : int
            Number of discrete price points (default: 15)
        alpha : float
            Learning rate (default: 0.15)
        beta : float
            Discount factor (default: 0.95)
        epsilon : float
            Initial exploration rate (default: 0.10)
        epsilon_decay : float
            Decay rate for exploration (default: 0.9999)
        epsilon_min : float
            Minimum exploration rate (default: 0.01)
        memory_size : int
            Size of experience replay buffer (default: 10000)
        price_min : float
            Minimum price (default: 1.0)
        price_max : float
            Maximum price (default: 2.0)
        marginal_cost : float
            Marginal cost (default: 1.0)
        """
        self.agent_id = agent_id
        self.n_prices = n_prices
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size
        self.marginal_cost = marginal_cost
        
        # Create discrete price grid
        self.prices = np.linspace(price_min, price_max, n_prices)
        
        # Initialize Q-table: Q(state, action)
        # State includes own previous price and competitors' previous prices
        # For simplicity, we'll use a flattened state representation
        self.state_size = n_prices ** 2  # Own price × competitor price (for 2-player)
        self.q_table = np.zeros((self.state_size, n_prices))
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Track agent's history
        self.price_history = []
        self.profit_history = []
        self.action_history = []
        self.state_history = []
        
        # Current state tracking
        self.current_state = None
        self.current_action = None
        self.last_price = None
        self.last_profit = 0.0
        
    def get_price_index(self, price: float) -> int:
        """Convert continuous price to discrete index."""
        return np.argmin(np.abs(self.prices - price))
    
    def get_state(self, own_price: Optional[float], competitor_prices: List[float]) -> int:
        """
        Convert prices to state representation.
        
        Parameters:
        -----------
        own_price : float or None
            Agent's own previous price
        competitor_prices : List[float]
            List of competitors' previous prices
            
        Returns:
        --------
        int : State index
        """
        if own_price is None:
            own_idx = 0
        else:
            own_idx = self.get_price_index(own_price)
        
        if len(competitor_prices) == 0:
            comp_idx = 0
        else:
            # For multiple competitors, use average or first competitor
            comp_price = competitor_prices[0] if len(competitor_prices) > 0 else self.prices[0]
            comp_idx = self.get_price_index(comp_price)
        
        # Simple state encoding: own_price_idx * n_prices + competitor_price_idx
        state = own_idx * self.n_prices + comp_idx
        return min(state, self.state_size - 1)
    
    def choose_action(self, state: int, exploration: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Parameters:
        -----------
        state : int
            Current state
        exploration : bool
            Whether to use exploration (epsilon-greedy)
            
        Returns:
        --------
        int : Action index (price index)
        """
        if exploration and random.random() < self.epsilon:
            # Explore: choose random action
            return random.randint(0, self.n_prices - 1)
        else:
            # Exploit: choose action with highest Q-value
            return np.argmax(self.q_table[state])
    
    def get_price(self, own_previous_price: Optional[float], 
                  competitor_prices: List[float], 
                  exploration: bool = True) -> float:
        """
        Get pricing decision from the agent.
        
        Parameters:
        -----------
        own_previous_price : float or None
            Agent's previous price
        competitor_prices : List[float]
            Competitors' previous prices
        exploration : bool
            Whether to use exploration
            
        Returns:
        --------
        float : Chosen price
        """
        # Get current state
        state = self.get_state(own_previous_price, competitor_prices)
        
        # Choose action
        action = self.choose_action(state, exploration)
        
        # Store current state and action
        self.current_state = state
        self.current_action = action
        self.last_price = self.prices[action]
        
        # Record in history
        self.state_history.append(state)
        self.action_history.append(action)
        self.price_history.append(self.last_price)
        
        return self.last_price
    
    def update_q_value(self, state: int, action: int, reward: float, 
                      next_state: int, done: bool = False):
        """
        Update Q-value using Q-learning update rule.
        
        Parameters:
        -----------
        state : int
            Previous state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : int
            Current state
        done : bool
            Whether episode is done
        """
        # Q-learning update: Q(s,a) = Q(s,a) + α[r + β*max(Q(s',a')) - Q(s,a)]
        if done:
            target = reward
        else:
            target = reward + self.beta * np.max(self.q_table[next_state])
        
        current_q = self.q_table[state, action]
        self.q_table[state, action] = current_q + self.alpha * (target - current_q)
    
    def remember(self, state: int, action: int, reward: float, 
                next_state: int, done: bool = False):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay_experiences(self, batch_size: int = 32):
        """
        Learn from random batch of past experiences.
        
        Parameters:
        -----------
        batch_size : int
            Size of random batch to sample
        """
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            self.update_q_value(state, action, reward, next_state, done)
    
    def update(self, profit: float, own_price: Optional[float], 
               competitor_prices: List[float]):
        """
        Update agent with latest profit and market information.
        
        Parameters:
        -----------
        profit : float
            Profit received from last action
        own_price : float or None
            Current own price
        competitor_prices : List[float]
            Current competitor prices
        """
        self.last_profit = profit
        self.profit_history.append(profit)
        
        if self.current_state is not None and self.current_action is not None:
            # Get next state
            next_state = self.get_state(own_price, competitor_prices)
            
            # Store experience
            self.remember(
                self.current_state, 
                self.current_action, 
                profit, 
                next_state, 
                done=False
            )
            
            # Update Q-value
            self.update_q_value(
                self.current_state, 
                self.current_action, 
                profit, 
                next_state
            )
            
            # Experience replay
            self.replay_experiences()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_strategy_matrix(self) -> np.ndarray:
        """
        Get the learned strategy as a probability matrix.
        
        Returns:
        --------
        np.ndarray : Strategy matrix [state x action]
        """
        strategy = np.zeros_like(self.q_table)
        
        for state in range(self.state_size):
            # Softmax over Q-values to get action probabilities
            q_values = self.q_table[state]
            if np.max(q_values) > np.min(q_values):
                # Use temperature = 0.1 for near-greedy behavior
                temp = 0.1
                exp_q = np.exp(q_values / temp)
                strategy[state] = exp_q / np.sum(exp_q)
            else:
                # Uniform if all Q-values are equal
                strategy[state] = np.ones(self.n_prices) / self.n_prices
        
        return strategy
    
    def save_agent(self, filepath: str):
        """Save agent state to file."""
        agent_data = {
            'agent_id': self.agent_id,
            'q_table': self.q_table,
            'prices': self.prices,
            'price_history': self.price_history,
            'profit_history': self.profit_history,
            'action_history': self.action_history,
            'state_history': self.state_history,
            'parameters': {
                'n_prices': self.n_prices,
                'alpha': self.alpha,
                'beta': self.beta,
                'epsilon': self.epsilon,
                'marginal_cost': self.marginal_cost
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
    
    def load_agent(self, filepath: str):
        """Load agent state from file."""
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)
        
        self.agent_id = agent_data['agent_id']
        self.q_table = agent_data['q_table']
        self.prices = agent_data['prices']
        self.price_history = agent_data['price_history']
        self.profit_history = agent_data['profit_history']
        self.action_history = agent_data['action_history']
        self.state_history = agent_data['state_history']
        
        # Update parameters
        params = agent_data['parameters']
        self.n_prices = params['n_prices']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.epsilon = params['epsilon']
        self.marginal_cost = params['marginal_cost']
    
    def reset(self):
        """Reset agent for new experiment."""
        self.q_table = np.zeros((self.state_size, self.n_prices))
        self.memory.clear()
        self.price_history.clear()
        self.profit_history.clear()
        self.action_history.clear()
        self.state_history.clear()
        self.current_state = None
        self.current_action = None
        self.last_price = None
        self.last_profit = 0.0
        self.epsilon = 0.10  # Reset exploration rate
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        if len(self.price_history) == 0:
            return {}
        
        prices = np.array(self.price_history)
        profits = np.array(self.profit_history)
        
        return {
            'mean_price': np.mean(prices),
            'std_price': np.std(prices),
            'min_price': np.min(prices),
            'max_price': np.max(prices),
            'mean_profit': np.mean(profits),
            'total_profit': np.sum(profits),
            'price_markup': (np.mean(prices) - self.marginal_cost) / self.marginal_cost,
            'n_episodes': len(prices),
            'final_epsilon': self.epsilon
        }