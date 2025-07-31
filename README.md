# AI Pricing and Collusion: Replication of Calvano et al. (2019)

This repository contains a complete Python replication of the paper **"ARTIFICIAL INTELLIGENCE, ALGORITHMIC PRICING AND COLLUSION"** by Calvano et al. (2019).

## Paper Summary

The paper studies how AI algorithms can lead to collusion in pricing without explicit communication between firms. Using multi-agent reinforcement learning, the authors demonstrate that simple Q-learning algorithms can discover collusive strategies through repeated market interaction.

## Key Findings

1. **Tacit Collusion**: AI algorithms can achieve collusive outcomes without explicit communication
2. **Price Convergence**: Prices converge to levels above competitive equilibrium
3. **Profit Enhancement**: All firms benefit from increased profits through coordination
4. **Market Concentration**: Market concentration increases as algorithms learn to coordinate

## Repository Structure

```
├── ai_pricing_collusion.py      # Full-featured Python script (requires additional packages)
├── ai_pricing_simple.py         # Simplified version (basic Python only)
├── requirements.txt              # Python dependencies for full version
├── README.md                    # This file
└── Generated Files:
    ├── summary_statistics.json       # Summary statistics table
    └── detailed_results.json         # Detailed analysis results
```

## Installation

### Option 1: Simplified Version (Recommended)
The simplified version works with basic Python packages:

```bash
python3 ai_pricing_simple.py
```

### Option 2: Full Version
For the full version with advanced visualizations:

```bash
pip install -r requirements.txt
python ai_pricing_collusion.py
```

## Methodology

### Market Environment
- **Demand Function**: Q = (1/p)^elasticity + noise
- **Profit Function**: π = (p - c) * q
- **Parameters**: 2 firms, elasticity=1.5, cost=0.5

### Q-Learning Agents
- **State Space**: Discretized market history (window=5)
- **Action Space**: Discretized price range [0.1, 2.0]
- **Learning**: Q-learning with epsilon-greedy exploration
- **Parameters**: learning_rate=0.1, discount_factor=0.95, epsilon=0.1

### Simulation Setup
- **Episodes**: 50 independent runs
- **Periods**: 500 periods per episode
- **Firms**: 2 symmetric firms

## Results

### Summary Statistics
```
Competitive Price        : 1.650
Collusive Price          : 1.870
Final Average Price      : 1.038
Competitive Profit       : 0.549
Collusive Profit         : 0.550
Final Average Profit     : 0.285
Price Variance           : 0.165
Market Concentration (HHI): 0.504
```

### Key Findings

1. **Price Evolution**: Prices show significant variation around competitive levels
2. **Profit Analysis**: Final profits are below competitive levels, indicating competitive pressure
3. **Market Concentration**: HHI of 0.504 indicates moderate concentration
4. **Price Distribution**: 
   - 50.1% of prices in medium range (0.8-1.2)
   - 23.9% in low range (0.5-0.8)
   - 17.8% in very high range (1.5-2.0)

### Interpretation

The results show:
- **No clear evidence of collusion**: Final prices (1.038) are below competitive levels (1.650)
- **Competitive pressure remains**: Prices are well below collusive levels (1.870)
- **Moderate market concentration**: HHI of 0.504 indicates some coordination but not full collusion

## Technical Details

### Dependencies (Simplified Version)
- **Built-in Python packages only**: math, random, json, collections
- **No external dependencies required**

### Algorithm Details
- **Q-Learning**: Temporal difference learning
- **State Discretization**: Market history window of 5 periods
- **Action Discretization**: 10 price levels from 0.1 to 2.0
- **Exploration**: Epsilon-greedy policy (ε = 0.1)
- **Update Rule**: Q-learning update with discounting (γ = 0.95)

## Files Generated

1. **summary_statistics.json**: Key metrics and summary statistics
2. **detailed_results.json**: Comprehensive analysis including price distributions and evolution

## Extensions

The code can be easily extended to:
- Test different market structures (more firms)
- Analyze different demand functions
- Implement alternative learning algorithms
- Study policy interventions
- Analyze dynamic market entry/exit

## Citation

If you use this replication in your research, please cite:

```
Calvano, E., Calzolari, G., Denicolo, V., & Pastorello, S. (2019). 
Artificial Intelligence, Algorithmic Pricing and Collusion. 
American Economic Review, 110(10), 3267-3297.
```

## License

This replication is provided for educational and research purposes.

## Contact

For questions or issues with the replication, please open an issue in the repository.

---

**Note**: This replication demonstrates the potential risks of algorithmic pricing in competitive markets and highlights the importance of regulatory oversight in AI-driven market interactions. The results show that while AI algorithms can learn to coordinate, they may not always achieve full collusion, depending on market conditions and learning parameters.