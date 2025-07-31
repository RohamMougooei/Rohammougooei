"""
Table Generator for Calvano et al. (2020) Replication

This module generates LaTeX and HTML tables summarizing the experimental results,
including price statistics, collusion metrics, and robustness analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import os
from pathlib import Path


class CalvanoTableGenerator:
    """Generate tables for Calvano et al. (2020) replication results."""
    
    def __init__(self, output_dir: str = "tables"):
        """Initialize table generator."""
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def create_summary_statistics_table(
        self,
        results_dict: Dict[str, Any],
        filename: str = "summary_statistics"
    ):
        """
        Create summary statistics table for all experiments.
        
        Parameters:
        -----------
        results_dict : Dict[str, Any]
            Dictionary containing all experiment results
        filename : str
            Output filename (without extension)
        """
        # Collect data for table
        table_data = []
        
        # Theoretical benchmarks
        marginal_cost = 1.0
        nash_price_2p = marginal_cost + 1.0 / (1.0 - 0.25 * (2 - 1))
        monopoly_price = marginal_cost + 1.0
        
        # Process each experiment
        experiments = [
            ('Baseline (2 agents)', 'baseline'),
            ('Asymmetric costs (Δc=0.10)', 'asymmetric_costs'),
            ('Three agents', 'multiplayer_3'),
            ('Four agents', 'multiplayer_4'),
            ('Uncertainty (σ=0.10)', 'uncertainty')
        ]
        
        for exp_name, exp_key in experiments:
            if exp_key in results_dict:
                results = results_dict[exp_key]
                
                if 'summary_statistics' in results and 'final_prices' in results['summary_statistics']:
                    prices = results['summary_statistics']['final_prices']
                    avg_price = np.mean(prices['mean'])
                    std_price = np.mean(prices['std'])
                    
                    # Calculate markup over marginal cost
                    markup = (avg_price - marginal_cost) / marginal_cost * 100
                    
                    # Distance from Nash (approximate)
                    nash_distance = ((avg_price - nash_price_2p) / nash_price_2p) * 100
                    
                    # Distance from monopoly
                    monopoly_distance = ((avg_price - monopoly_price) / monopoly_price) * 100
                    
                    # Price correlation (if available)
                    price_corr = 0.0
                    if 'collusion_metrics' in results['summary_statistics']:
                        corr_data = results['summary_statistics']['collusion_metrics'].get('price_correlation', {})
                        if isinstance(corr_data, dict) and 'mean' in corr_data:
                            price_corr = corr_data['mean']
                    
                    table_data.append({
                        'Experiment': exp_name,
                        'Mean Price': f"{avg_price:.3f}",
                        'Std Price': f"{std_price:.3f}",
                        'Markup (%)': f"{markup:.1f}",
                        'Nash Distance (%)': f"{nash_distance:.1f}",
                        'Monopoly Distance (%)': f"{monopoly_distance:.1f}",
                        'Price Correlation': f"{price_corr:.3f}"
                    })
        
        # Add theoretical benchmarks row
        table_data.append({
            'Experiment': 'Nash Equilibrium',
            'Mean Price': f"{nash_price_2p:.3f}",
            'Std Price': '—',
            'Markup (%)': f"{((nash_price_2p - marginal_cost) / marginal_cost * 100):.1f}",
            'Nash Distance (%)': '0.0',
            'Monopoly Distance (%)': f"{((nash_price_2p - monopoly_price) / monopoly_price * 100):.1f}",
            'Price Correlation': '—'
        })
        
        table_data.append({
            'Experiment': 'Monopoly Price',
            'Mean Price': f"{monopoly_price:.3f}",
            'Std Price': '—',
            'Markup (%)': f"{((monopoly_price - marginal_cost) / marginal_cost * 100):.1f}",
            'Nash Distance (%)': f"{((monopoly_price - nash_price_2p) / nash_price_2p * 100):.1f}",
            'Monopoly Distance (%)': '0.0',
            'Price Correlation': '—'
        })
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False)
        
        # Generate LaTeX table
        latex_table = self._generate_latex_table(df, "Summary Statistics of Experimental Results")
        latex_path = os.path.join(self.output_dir, f"{filename}.tex")
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        
        # Generate HTML table
        html_table = self._generate_html_table(df, "Summary Statistics of Experimental Results")
        html_path = os.path.join(self.output_dir, f"{filename}.html")
        with open(html_path, 'w') as f:
            f.write(html_table)
        
        print(f"Summary statistics table saved to {self.output_dir}/")
        return df
    
    def create_robustness_table(
        self,
        suite_results: Dict[str, Any],
        suite_type: str,
        filename: Optional[str] = None
    ):
        """
        Create robustness analysis table.
        
        Parameters:
        -----------
        suite_results : Dict[str, Any]
            Results from experiment suite
        suite_type : str
            Type of suite ('asymmetric_costs', 'multiplayer', 'uncertainty')
        filename : str, optional
            Output filename
        """
        if filename is None:
            filename = f"robustness_{suite_type}"
        
        if 'summary' not in suite_results:
            return None
        
        summary = suite_results['summary']
        table_data = []
        
        if suite_type == "asymmetric_costs":
            param_values = summary.get('cost_differences', [])
            param_name = 'Cost Difference'
            title = 'Robustness to Asymmetric Costs'
            
        elif suite_type == "multiplayer":
            param_values = summary.get('n_agents', [])
            param_name = 'Number of Agents'
            title = 'Robustness to Number of Players'
            
        elif suite_type == "uncertainty":
            param_values = summary.get('sigma_values', [])
            param_name = 'Demand Volatility (σ)'
            title = 'Robustness to Demand Uncertainty'
        
        else:
            return None
        
        # Extract data
        avg_prices = summary.get('average_prices', [])
        
        for i, param_val in enumerate(param_values):
            if i < len(avg_prices):
                avg_price = avg_prices[i]
                
                # Calculate metrics
                marginal_cost = 1.0
                markup = (avg_price - marginal_cost) / marginal_cost * 100
                
                # Nash distance (approximate)
                if suite_type == "multiplayer":
                    nash_price = marginal_cost + 1.0 / (1.0 - 0.25 * (param_val - 1))
                else:
                    nash_price = marginal_cost + 1.0 / (1.0 - 0.25 * (2 - 1))
                
                nash_distance = ((avg_price - nash_price) / nash_price) * 100
                
                table_data.append({
                    param_name: f"{param_val:.2f}" if isinstance(param_val, float) else str(param_val),
                    'Average Price': f"{avg_price:.3f}",
                    'Markup (%)': f"{markup:.1f}",
                    'Nash Distance (%)': f"{nash_distance:.1f}"
                })
        
        # Add additional metrics if available
        if 'price_correlations' in summary and len(summary['price_correlations']) > 0:
            for i, corr in enumerate(summary['price_correlations']):
                if i < len(table_data):
                    table_data[i]['Price Correlation'] = f"{corr:.3f}"
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Save files
        csv_path = os.path.join(self.output_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False)
        
        latex_table = self._generate_latex_table(df, title)
        latex_path = os.path.join(self.output_dir, f"{filename}.tex")
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        
        html_table = self._generate_html_table(df, title)
        html_path = os.path.join(self.output_dir, f"{filename}.html")
        with open(html_path, 'w') as f:
            f.write(html_table)
        
        print(f"Robustness table for {suite_type} saved to {self.output_dir}/")
        return df
    
    def create_convergence_table(
        self,
        results: Dict[str, Any],
        filename: str = "convergence_analysis"
    ):
        """
        Create convergence analysis table.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Experiment results with convergence metrics
        filename : str
            Output filename
        """
        if 'runs' not in results or not results['runs']:
            return None
        
        # Collect convergence data
        all_convergence = []
        for run in results['runs']:
            if 'convergence_metrics' in run and run['convergence_metrics']:
                convergence_data = []
                for cm in run['convergence_metrics']:
                    episode = cm['episode']
                    metrics = cm['metrics']
                    convergence_data.append({
                        'Episode': episode,
                        'Mean Price': metrics.get('mean_price', 0),
                        'Price Std': metrics.get('std_price', 0),
                        'Mean Profit': metrics.get('mean_profit', 0),
                        'Price Correlation': metrics.get('price_correlation', 0)
                    })
                if convergence_data:
                    all_convergence.append(pd.DataFrame(convergence_data))
        
        if not all_convergence:
            return None
        
        # Average across runs and select key episodes
        episodes_to_show = [1000, 5000, 10000, 20000, 30000, 40000, 50000]
        table_data = []
        
        for episode in episodes_to_show:
            # Find closest episode in data
            episode_data = []
            for run_df in all_convergence:
                closest_idx = np.argmin(np.abs(run_df['Episode'].values - episode))
                if closest_idx < len(run_df):
                    episode_data.append(run_df.iloc[closest_idx])
            
            if episode_data:
                # Average across runs
                avg_data = pd.DataFrame(episode_data).mean()
                
                table_data.append({
                    'Episode': f"{episode:,}",
                    'Mean Price': f"{avg_data['Mean Price']:.3f}",
                    'Price Std': f"{avg_data['Price Std']:.3f}",
                    'Mean Profit': f"{avg_data['Mean Profit']:.4f}",
                    'Price Correlation': f"{avg_data['Price Correlation']:.3f}"
                })
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Save files
        csv_path = os.path.join(self.output_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False)
        
        latex_table = self._generate_latex_table(df, "Convergence Analysis")
        latex_path = os.path.join(self.output_dir, f"{filename}.tex")
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        
        html_table = self._generate_html_table(df, "Convergence Analysis")
        html_path = os.path.join(self.output_dir, f"{filename}.html")
        with open(html_path, 'w') as f:
            f.write(html_table)
        
        print(f"Convergence table saved to {self.output_dir}/")
        return df
    
    def create_punishment_analysis_table(
        self,
        deviation_results: Dict[str, Any],
        filename: str = "punishment_analysis"
    ):
        """
        Create punishment strategy analysis table.
        
        Parameters:
        -----------
        deviation_results : Dict[str, Any]
            Results from deviation analysis
        filename : str
            Output filename
        """
        if not deviation_results:
            return None
        
        table_data = []
        
        # Analyze different deviation scenarios
        for key, deviation_data in deviation_results.items():
            prices = np.array(deviation_data['prices'])
            profits = np.array(deviation_data['profits'])
            deviating_agent = deviation_data['deviating_agent']
            deviation_price = deviation_data['deviation_price']
            
            # Calculate punishment severity and duration
            pre_deviation_price = prices[0, 1-deviating_agent]  # Other agent's initial price
            
            # Find when punishment ends (price returns to within 5% of initial)
            other_agent_prices = prices[:, 1-deviating_agent]
            punishment_duration = 0
            for t in range(1, len(other_agent_prices)):
                if abs(other_agent_prices[t] - pre_deviation_price) / pre_deviation_price < 0.05:
                    punishment_duration = t
                    break
            else:
                punishment_duration = len(other_agent_prices) - 1
            
            # Calculate profit loss for deviating agent
            initial_profit = profits[0, deviating_agent]
            min_profit = np.min(profits[1:6, deviating_agent])  # First 5 periods after deviation
            profit_loss = (initial_profit - min_profit) / initial_profit * 100 if initial_profit > 0 else 0
            
            table_data.append({
                'Deviation Price': f"{deviation_price:.2f}",
                'Deviating Agent': deviating_agent + 1,
                'Punishment Duration': punishment_duration,
                'Max Price Drop': f"{(pre_deviation_price - np.min(other_agent_prices[1:6])):.3f}",
                'Profit Loss (%)': f"{profit_loss:.1f}"
            })
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Save files
        csv_path = os.path.join(self.output_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False)
        
        latex_table = self._generate_latex_table(df, "Punishment Strategy Analysis")
        latex_path = os.path.join(self.output_dir, f"{filename}.tex")
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        
        html_table = self._generate_html_table(df, "Punishment Strategy Analysis")
        html_path = os.path.join(self.output_dir, f"{filename}.html")
        with open(html_path, 'w') as f:
            f.write(html_table)
        
        print(f"Punishment analysis table saved to {self.output_dir}/")
        return df
    
    def _generate_latex_table(self, df: pd.DataFrame, title: str) -> str:
        """Generate LaTeX table from DataFrame."""
        n_cols = len(df.columns)
        col_spec = 'l' + 'c' * (n_cols - 1)
        
        latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{title}}}
\\begin{{tabular}}{{{col_spec}}}
\\toprule
"""
        
        # Header
        header = " & ".join(df.columns) + " \\\\\n"
        latex += header
        latex += "\\midrule\n"
        
        # Data rows
        for _, row in df.iterrows():
            row_data = " & ".join([str(val) for val in row.values]) + " \\\\\n"
            latex += row_data
        
        latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
        
        return latex
    
    def _generate_html_table(self, df: pd.DataFrame, title: str) -> str:
        """Generate HTML table from DataFrame."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
        .title {{ text-align: center; font-size: 18px; font-weight: bold; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="title">{title}</div>
"""
        
        html += df.to_html(index=False, table_id="results-table")
        
        html += """
</body>
</html>"""
        
        return html


def generate_all_tables(results_dict: Dict[str, Any], output_dir: str = "tables"):
    """
    Generate all tables for the paper replication.
    
    Parameters:
    -----------
    results_dict : Dict[str, Any]
        Dictionary containing all experiment results
    output_dir : str
        Output directory for tables
    """
    generator = CalvanoTableGenerator(output_dir)
    
    print("Generating tables...")
    
    # Main summary table
    generator.create_summary_statistics_table(results_dict)
    
    # Convergence analysis (using baseline results)
    if 'baseline' in results_dict:
        generator.create_convergence_table(results_dict['baseline'])
        
        # Punishment analysis
        if 'deviation_analysis' in results_dict['baseline']:
            generator.create_punishment_analysis_table(
                results_dict['baseline']['deviation_analysis']
            )
    
    # Robustness tables
    robustness_experiments = [
        ('asymmetric_costs', 'asymmetric_costs'),
        ('multiplayer', 'multiplayer'),
        ('uncertainty', 'uncertainty')
    ]
    
    for results_key, suite_type in robustness_experiments:
        if results_key in results_dict:
            generator.create_robustness_table(
                results_dict[results_key],
                suite_type
            )
    
    print(f"All tables saved to {output_dir}/")