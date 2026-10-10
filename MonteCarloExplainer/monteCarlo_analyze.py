"""
Monte Carlo Shapley Analyzer - Load Results and Visualize

This script loads saved Monte Carlo Shapley results and creates bar-chart
visualizations of feature importance (full and top-20 views).

Usage:
    # Auto-load latest results for a dataset/task:
    python monteCarlo_analyze.py --dataset rel-arxiv --task paper-citation
    
    # Or specify exact results directory:
    python monteCarlo_analyze.py --dataset rel-arxiv --task paper-citation --results-dir path/to/results
    
    # Or specify exact results file:
    python monteCarlo_analyze.py --dataset rel-arxiv --task paper-citation \
        --results-file ../rel-arxiv/monteCarlo_explainer/monte_carlo_results_20260217_213253/monte_carlo_shapley_4.0h.csv

"""

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import re

# Local imports
from monteCarlo_explainer_config import get_montecarlo_output_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze Monte Carlo Shapley results and create visualizations'
    )
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., rel-arxiv)')
    parser.add_argument('--task', type=str, required=True,
                       help='Task name (e.g., paper-citation)')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Specific results directory (default: auto-find latest)')
    parser.add_argument('--results-file', type=str, default=None,
                       help='Path to a specific Monte Carlo results CSV file '
                            '(e.g. path/to/monte_carlo_results_20240315_142305/monte_carlo_shapley_4.0h.csv). '
                            'Overrides --results-dir.')
    parser.add_argument('--model_type', type=str, default='hgs',
                       choices=['hgs', 'transformer'],
                       help='Model type: hgs (default) or transformer (RelGT)')
    return parser.parse_args()


def find_latest_results_dir(output_base_dir):
    """Find the latest Monte Carlo results directory."""
    result_dirs = sorted(output_base_dir.glob("monte_carlo_results_*"))
    if not result_dirs:
        raise FileNotFoundError(
            f"No Monte Carlo results found in {output_base_dir}. "
            f"Run run_monteCarlo_explainer.py first."
        )
    return result_dirs[-1]


def extract_timestamp_from_results_dir(results_dir: Path) -> str:
    """
    Extract the timestamp from the results directory name.
    Expected format: monte_carlo_results_YYYYMMDD_HHMMSS
    Falls back to current time if no timestamp found.
    """
    dir_name = results_dir.name
    # Try to extract YYYYMMDD_HHMMSS pattern from directory name
    match = re.search(r'(\d{8}_\d{6})', dir_name)
    if match:
        return match.group(1)
    # Fallback: use current time
    print(f"  Warning: Could not extract timestamp from '{dir_name}', using current time.")
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def extract_hours_suffix(results_file: Path) -> str:
    """
    Extract the elapsed-hours suffix from a checkpoint filename.
    Expected format: monte_carlo_shapley_4.0h.csv  ->  '4.0h'
    Returns empty string if no hours suffix is found (e.g. final_results.csv).
    """
    match = re.search(r'_(\d+(?:\.\d+)?h)\.csv$', results_file.name)
    return match.group(1) if match else ""


def get_timestamped_filepath(output_dir: Path, base_name: str, ext: str,
                             timestamp: str, hours_suffix: str = "") -> Path:
    """
    Build an output filepath using the Monte Carlo run timestamp and optional hours suffix.
    Examples:
      hours_suffix=""    -> montecarlo_fidelity_plus_20240315_142305.csv
      hours_suffix="4.0h" -> montecarlo_fidelity_plus_20240315_142305_4.0h.csv
    """
    suffix = f"_{hours_suffix}" if hours_suffix else ""
    filename = f"{base_name}_{timestamp}{suffix}{ext}"
    return output_dir / filename


def load_latest_checkpoint(results_dir):
    """
    Load the latest checkpoint from a results directory.
    Returns (dataframe, file_path) so the caller always knows which file was loaded.
    """
    # Try final_results.csv first
    final_results = results_dir / "final_results.csv"
    if final_results.exists():
        print(f"✓ Loading final results: {final_results.name}")
        return pd.read_csv(final_results), final_results
    
    # Otherwise find latest checkpoint, sorted numerically by hours value
    # (alphabetical sort would rank 99.0h after 183.0h since '9' > '1')
    checkpoints = list(results_dir.glob("monte_carlo_shapley_*.csv"))
    if not checkpoints:
        raise FileNotFoundError(f"No results files found in {results_dir}")

    def parse_hours(p: Path) -> float:
        match = re.search(r'_(\d+(?:\.\d+)?)h\.csv$', p.name)
        return float(match.group(1)) if match else -1.0

    latest_checkpoint = max(checkpoints, key=parse_hours)
    print(f"✓ Loading latest checkpoint: {latest_checkpoint.name}")
    return pd.read_csv(latest_checkpoint), latest_checkpoint


def load_specific_file(results_file: Path):
    """
    Load a specific Monte Carlo results CSV file by its exact path.
    Returns (dataframe, file_path) for consistency with load_latest_checkpoint.
    """
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    if results_file.suffix != '.csv':
        raise ValueError(f"Expected a .csv file, got: {results_file.name}")
    print(f"✓ Loading specific results file: {results_file.name}")
    return pd.read_csv(results_file), results_file


def print_top_features(results_df, top_k=20):
    """Print top K features by absolute Shapley value."""
    print("\n" + "=" * 80)
    print(f"TOP {top_k} FEATURES BY ABSOLUTE SHAPLEY VALUE")
    print("=" * 80)
    
    # Sort by absolute value
    sorted_df = results_df.sort_values('abs_shapley_value', ascending=False).head(top_k)
    
    print(f"\n{'Rank':<6}{'Feature':<50}{'Shapley':<12}{'Std Error':<12}")
    print("-" * 80)
    for idx, row in sorted_df.iterrows():
        rank = idx + 1 if 'abs_shapley_value' in results_df.columns else "?"
        print(f"{rank:<6}{row['feature']:<50}{row['shapley_value']:<12.4f}{row.get('std_error', 0):<12.6f}")


def create_visualization(results_df, output_dir, timestamp, dataset_name, task_name, hours_suffix=""):
    """
    Create visualizations of Monte Carlo Shapley results (Absolute Values).
    Generates both a full feature plot and a Top 20 features plot.

    Bars show *absolute* Shapley values so features are ranked by magnitude
    regardless of sign.  The original signed value is preserved in the CSV.
    A dashed threshold line is drawn at 0.1 when any bar reaches that range.
    """
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    # Work on a copy; compute absolute importance and sort ascending so that
    # inverting the y-axis places the largest bar at the top of the chart.
    df_plot = results_df.copy()
    df_plot['Abs Shapley Value'] = df_plot['shapley_value'].abs()
    df_plot = df_plot.sort_values('Abs Shapley Value', ascending=True)

    # Build a subtitle string from available metadata columns
    meta_parts = []
    if 'num_permutations' in df_plot.columns:
        meta_parts.append(f"n_perms={int(df_plot['num_permutations'].iloc[0])}")
    if 'std_error' in df_plot.columns:
        mean_se = df_plot['std_error'].mean()
        meta_parts.append(f"mean_SE={mean_se:.2e}")
    if hours_suffix:
        meta_parts.append(f"elapsed={hours_suffix}")
    meta_str = "  |  ".join(meta_parts) if meta_parts else ""

    def draw_and_save_barh(df_subset, suffix_name, title_prefix):
        """Draw one horizontal bar chart and save it as a PNG."""
        if df_subset.empty:
            return None

        num_features = len(df_subset)
        fig_height = max(4, 0.35 * num_features)
        fig, ax = plt.subplots(figsize=(12, fig_height))

        bars = ax.barh(df_subset['feature'], df_subset['Abs Shapley Value'], color='skyblue')

        ax.set_xlabel('Absolute Shapley Value (Importance Score)')
        full_title = f'{title_prefix} — {dataset_name}/{task_name}'
        if meta_str:
            full_title += f'\n{meta_str}'
        ax.set_title(full_title, pad=15)
        ax.invert_yaxis()

        max_val = df_subset['Abs Shapley Value'].max()

        # Draw threshold line at 0.1 only when values reach that scale
        if max_val >= 0.1:
            ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold (0.1)')
            ax.legend(loc='lower right')

        # Value labels: scientific notation for tiny values, fixed decimals otherwise
        x_padding = max_val * 0.015 if max_val > 0 else 0.01
        for bar in bars:
            width = bar.get_width()
            if width > 0 and width < 0.0001:
                label_fmt = f'{width:.2e}'
            else:
                label_fmt = f'{width:.4f}'

            ax.text(
                width + x_padding,
                bar.get_y() + bar.get_height() / 2,
                label_fmt,
                va='center',
                ha='left',
                fontsize=9,
                fontweight='semibold',
            )

        ax.set_xlim(0, max_val * 1.15 if max_val > 0 else 1.0)
        plt.tight_layout()

        plot_path = get_timestamped_filepath(
            output_dir,
            f'montecarlo_shapley_plot_{suffix_name}',
            '.png',
            timestamp,
            hours_suffix,
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ {title_prefix} plot saved: {plot_path.name}")
        return plot_path

    # --- Plot 1: all features ---
    full_plot_path = draw_and_save_barh(df_plot, 'full', 'Monte Carlo Shapley (Full)')

    # --- Plot 2: Top 20 only (skip if fewer than 20 features) ---
    top20_plot_path = None
    if len(df_plot) > 20:
        df_top20 = df_plot.tail(20)   # tail = largest 20 after ascending sort
        top20_plot_path = draw_and_save_barh(df_top20, 'top20', 'Monte Carlo Shapley (Top 20)')

    return full_plot_path


def main():
    """Main analysis function."""
    # Parse arguments
    args = parse_args()
    DATASET_NAME = args.dataset
    TASK_NAME    = args.task
    MODEL_TYPE   = args.model_type

    print("=" * 80)
    print("MONTE CARLO SHAPLEY - ANALYSIS")
    print("=" * 80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Task: {TASK_NAME}")
    print("=" * 80)

    output_base_dir = get_montecarlo_output_dir(DATASET_NAME, TASK_NAME, MODEL_TYPE)
    
    # -----------------------------------------------------------------------
    # Resolve which results to load:
    #   1. --results-file  -> load that exact CSV; infer results_dir from parent
    #   2. --results-dir   -> load latest checkpoint inside that directory
    #   3. (default)       -> auto-find latest results directory
    # -----------------------------------------------------------------------
    if args.results_file:
        results_file = Path(args.results_file)
        results_dir  = results_file.parent
        print(f"\n✓ Using specific results file: {results_file.name}")
        print(f"  (results directory inferred as: {results_dir})")
    elif args.results_dir:
        results_dir  = Path(args.results_dir)
        results_file = None
        print(f"\n✓ Using results directory: {results_dir}")
    else:
        print(f"\nAuto-finding latest results in: {output_base_dir}")
        results_dir  = find_latest_results_dir(output_base_dir)
        results_file = None
        print(f"✓ Found results directory: {results_dir}")

    # Extract timestamp from the Monte Carlo results directory name.
    # All output files (fidelity, plot) will share this timestamp so they
    # are clearly linked back to their source Monte Carlo run.
    mc_timestamp = extract_timestamp_from_results_dir(results_dir)
    print(f"✓ Monte Carlo run timestamp: {mc_timestamp}")

    # Hours suffix is always extracted from whichever file was actually loaded
    # (works for both --results-file and auto/directory modes)
    # e.g. monte_carlo_shapley_4.0h.csv -> '4.0h', final_results.csv -> ''

    # ========================================================================
    # LOAD MONTE CARLO RESULTS
    # ========================================================================

    print("\n" + "=" * 80)
    print("LOADING MONTE CARLO RESULTS")
    print("=" * 80)

    if results_file:
        results_df, loaded_file = load_specific_file(results_file)
    else:
        results_df, loaded_file = load_latest_checkpoint(results_dir)
    
    # Extract hours suffix from the file that was actually loaded
    mc_hours_suffix = extract_hours_suffix(loaded_file)
    if mc_hours_suffix:
        print(f"✓ Hours suffix detected: {mc_hours_suffix}")
    else:
        print(f"  (No hours suffix in filename: {loaded_file.name})")

    # Add abs_shapley_value if not present
    if 'abs_shapley_value' not in results_df.columns:
        results_df['abs_shapley_value'] = results_df['shapley_value'].abs()
    
    print(f"\n✓ Loaded results with {len(results_df)} features")
    if 'num_permutations' in results_df.columns:
        print(f"  Permutations: {results_df['num_permutations'].iloc[0]}")
    if 'elapsed_hours' in results_df.columns:
        print(f"  Time: {results_df['elapsed_hours'].iloc[0]:.2f} hours")
    if 'std_error' in results_df.columns:
        mean_se = results_df['std_error'].mean()
        print(f"  Mean std error: {mean_se:.6f}")
    
    # Print top features
    print_top_features(results_df, top_k=20)

    # Create visualization
    plot_path = create_visualization(
        results_df, output_base_dir, mc_timestamp, DATASET_NAME, TASK_NAME, mc_hours_suffix
    )
    
    # Final summary
    print("\n" + "=" * 80)
    print("MONTE CARLO ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nDataset: {DATASET_NAME}")
    print(f"Task: {TASK_NAME}")
    print(f"Monte Carlo Timestamp: {mc_timestamp}")
    print(f"\nAnalyzed results from: {results_dir}")
    print(f"\nOutput directory: {output_base_dir.absolute()}")
    print(f"\nSaved files:")
    if plot_path is not None:
        print(f"  - {plot_path.name}  (full feature plot)")
        top20_name = plot_path.name.replace('_full_', '_top20_')
        top20_candidate = output_base_dir / top20_name
        if top20_candidate.exists():
            print(f"  - {top20_name}  (top 20 features plot)")
    print("\n✅ All Monte Carlo analysis outputs ready!")

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)