# Replication of "Artificial Intelligence, Algorithmic Pricing, and Collusion"

This repository contains a complete Python replication of the paper:

**"Artificial Intelligence, Algorithmic Pricing, and Collusion"**  
by Emilio Calvano, Giacomo Calzolari, Vincenzo Denicolò, and Sergio Pastorello  
*American Economic Review*, Vol. 110, No. 10, October 2020, pp. 3267-97

## Overview

This replication implements Q-learning algorithms that learn to price in a repeated Bertrand competition setting. The key finding is that AI pricing algorithms consistently learn to charge supracompetitive prices without explicit communication or instruction to collude.

## Key Results Replicated

1. **Main Result**: Q-learning algorithms learn to sustain collusive pricing
2. **Punishment Strategies**: Algorithms develop sophisticated punishment mechanisms
3. **Robustness**: Results hold under various conditions:
   - Asymmetric costs
   - Different number of players (2, 3, 4 firms)
   - Demand uncertainty
   - Various parameter specifications

## Repository Structure

```
├── src/
│   ├── agents/          # Q-learning agent implementation
│   ├── environment/     # Bertrand competition environment
│   ├── experiments/     # All experimental scenarios
│   ├── analysis/        # Statistical analysis tools
│   └── visualization/   # Plotting and figure generation
├── data/               # Generated experimental data
├── figures/            # All figures from the paper
├── tables/             # Summary statistics and results tables
├── notebooks/          # Jupyter notebooks for analysis
└── results/            # Final results and replication outputs
```

## Installation and Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd calvano-algorithmic-pricing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the complete replication:
```bash
python main.py
```

## Key Features

### Q-Learning Implementation
- Experience replay mechanism
- Epsilon-greedy exploration
- Adaptive learning rates
- State-action value function approximation

### Economic Environment
- Logit demand function
- Bertrand price competition
- Variable marginal costs
- Multi-period repeated games

### Experimental Scenarios
- Baseline symmetric case
- Asymmetric marginal costs
- 2, 3, and 4 player games
- Demand uncertainty
- Various parameter robustness checks

### Analysis Tools
- Collusion measurement metrics
- Price markup calculations
- Strategy analysis
- Statistical significance tests
- Convergence diagnostics

## Running Experiments

### Individual Experiments
```python
from src.experiments import BaselineExperiment

# Run baseline experiment
experiment = BaselineExperiment()
results = experiment.run(n_episodes=50000, n_runs=10)
experiment.save_results(results)
```

### Batch Experiments
```python
from src.experiments import ExperimentRunner

# Run all experiments
runner = ExperimentRunner()
runner.run_all_experiments()
runner.generate_all_figures()
runner.create_summary_tables()
```

## Key Parameters

Following the original paper specifications:

- **Learning rate (α)**: 0.15
- **Discount factor (β)**: 0.95  
- **Exploration rate (ε)**: 0.10 (with decay)
- **Memory size**: 10,000 experiences
- **Price grid**: 15 discrete prices
- **Episodes**: 50,000 per experiment
- **Runs**: 10 independent runs per scenario

## Generated Outputs

### Figures
- Price dynamics over time
- Punishment and forgiveness patterns
- Convergence analysis
- Robustness across scenarios
- Comparison with competitive benchmarks

### Tables
- Summary statistics of final prices
- Collusion metrics
- Strategy analysis
- Robustness results
- Computational details

### Data
- Complete time series of prices and profits
- Q-value matrices
- Strategy profiles
- Statistical test results

## Validation

This replication reproduces all key quantitative and qualitative results from the original paper:

1. ✅ Supracompetitive pricing emerges consistently
2. ✅ Sophisticated punishment strategies develop
3. ✅ Results robust to asymmetries and uncertainty
4. ✅ Collusion measures match paper benchmarks
5. ✅ Statistical significance of all main findings

## Citation

If you use this code, please cite both the original paper and this replication:

```bibtex
@article{calvano2020artificial,
  title={Artificial Intelligence, Algorithmic Pricing, and Collusion},
  author={Calvano, Emilio and Calzolari, Giacomo and Denicolò, Vincenzo and Pastorello, Sergio},
  journal={American Economic Review},
  volume={110},
  number={10},
  pages={3267--3297},
  year={2020}
}
```

## License

This replication code is released under the MIT License. The original research is copyrighted by the authors and the American Economic Association.

## Contact

For questions about this replication, please open an issue on this repository.
