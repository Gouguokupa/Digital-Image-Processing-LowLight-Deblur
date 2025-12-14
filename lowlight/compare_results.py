"""
Compare results between baseline and trained model
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def compare_results(baseline_csv, model_csv, output_dir='./comparison'):
    """
    Compare baseline and model results
    
    Args:
        baseline_csv: Path to baseline results CSV
        model_csv: Path to model results CSV
        output_dir: Directory to save comparison plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    baseline_df = pd.read_csv(baseline_csv)
    model_df = pd.read_csv(model_csv)
    
    # Compute statistics
    metrics = ['psnr', 'ssim', 'niqe']
    baseline_stats = {}
    model_stats = {}
    
    for metric in metrics:
        if metric in baseline_df.columns:
            baseline_stats[metric] = {
                'mean': baseline_df[metric].mean(),
                'std': baseline_df[metric].std()
            }
        if metric in model_df.columns:
            model_stats[metric] = {
                'mean': model_df[metric].mean(),
                'std': model_df[metric].std()
            }
    
    # Print comparison table
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    print(f"{'Metric':<15} {'Baseline':<25} {'Your Model':<25} {'Improvement'}")
    print("-"*70)
    
    for metric in metrics:
        if metric in baseline_stats and metric in model_stats:
            baseline_val = baseline_stats[metric]['mean']
            baseline_std = baseline_stats[metric]['std']
            model_val = model_stats[metric]['mean']
            model_std = model_stats[metric]['std']
            
            # Calculate improvement
            if metric == 'niqe':  # Lower is better
                improvement = ((baseline_val - model_val) / baseline_val) * 100
                symbol = "↓" if model_val < baseline_val else "↑"
            else:  # Higher is better (PSNR, SSIM)
                improvement = ((model_val - baseline_val) / baseline_val) * 100
                symbol = "↑" if model_val > baseline_val else "↓"
            
            print(f"{metric.upper():<15} "
                  f"{baseline_val:.4f} ± {baseline_std:.4f}   "
                  f"{model_val:.4f} ± {model_std:.4f}   "
                  f"{symbol} {abs(improvement):.1f}%")
    
    print("="*70)
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, metric in enumerate(metrics):
        if metric in baseline_df.columns and metric in model_df.columns:
            ax = axes[idx]
            
            # Bar plot
            means = [baseline_stats[metric]['mean'], model_stats[metric]['mean']]
            stds = [baseline_stats[metric]['std'], model_stats[metric]['std']]
            labels = ['Baseline\n(CLAHE)', 'Your Model']
            colors = ['#ff7f0e', '#2ca02c']
            
            x = np.arange(len(labels))
            bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.7, 
                         capsize=5, edgecolor='black', linewidth=1.5)
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
            ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
            ax.set_title(f'{metric.upper()} Comparison', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.3f}',
                       ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'comparison_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {plot_path}")
    
    # Create distribution comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, metric in enumerate(metrics):
        if metric in baseline_df.columns and metric in model_df.columns:
            ax = axes[idx]
            
            # Histogram
            ax.hist(baseline_df[metric].dropna(), bins=10, alpha=0.6, 
                   label='Baseline', color='#ff7f0e', edgecolor='black')
            ax.hist(model_df[metric].dropna(), bins=10, alpha=0.6, 
                   label='Your Model', color='#2ca02c', edgecolor='black')
            
            ax.set_xlabel(metric.upper(), fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title(f'{metric.upper()} Distribution', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save distribution plot
    dist_path = os.path.join(output_dir, 'distribution_plot.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    print(f"✓ Distribution plot saved to: {dist_path}")
    
    # Save comparison table
    comparison_data = []
    for metric in metrics:
        if metric in baseline_stats and metric in model_stats:
            comparison_data.append({
                'Metric': metric.upper(),
                'Baseline_Mean': baseline_stats[metric]['mean'],
                'Baseline_Std': baseline_stats[metric]['std'],
                'Model_Mean': model_stats[metric]['mean'],
                'Model_Std': model_stats[metric]['std'],
                'Improvement_%': ((model_stats[metric]['mean'] - baseline_stats[metric]['mean']) / 
                                 baseline_stats[metric]['mean'] * 100)
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    table_path = os.path.join(output_dir, 'comparison_table.csv')
    comparison_df.to_csv(table_path, index=False)
    print(f"✓ Comparison table saved to: {table_path}")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Compare Baseline and Model Results')
    
    parser.add_argument('--baseline', type=str, 
                       default='./baseline_results/baseline_clahe_results.csv',
                       help='Path to baseline results CSV')
    parser.add_argument('--model', type=str,
                       default='./results/evaluation_results.csv',
                       help='Path to model results CSV')
    parser.add_argument('--output', type=str, default='./comparison',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.baseline):
        print(f"❌ Baseline results not found: {args.baseline}")
        print(f"\nRun baseline first:")
        print(f"  python baseline.py --data_root ./dataset --method clahe --save_csv")
        return
    
    if not os.path.exists(args.model):
        print(f"❌ Model results not found: {args.model}")
        print(f"\nEvaluate your model first:")
        print(f"  python evaluate.py --data_root ./dataset --checkpoint ./checkpoints/best_model.pth --save_csv")
        return
    
    compare_results(args.baseline, args.model, args.output)


if __name__ == '__main__':
    main()

