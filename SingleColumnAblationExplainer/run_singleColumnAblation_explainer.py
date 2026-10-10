"""
Main explainer script for RelBench datasets.

Usage:
    python run_singleColumnAblation_explainer.py --dataset <dataset> --task <task> [--model_type hgs|transformer] [--seed <seed>] [--strategies all|fast]

Examples:
    python run_singleColumnAblation_explainer.py --dataset rel-f1 --task driver-dnf --model_type hgs --seed 42
    python run_singleColumnAblation_explainer.py --dataset rel-f1 --task driver-dnf --model_type transformer --seed 42
    python run_singleColumnAblation_explainer.py --dataset rel-trial --task study-outcome --model_type transformer --strategies fast
        # ↑ 'fast' computes ONLY the ablation method per feature type whose
        #   masking logic matches Kernel SHAP / Monte Carlo Shapley's own
        #   coalition masking (numerical/text/timestamp→Mean, categorical→MFV).
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sys
import time
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Resolve all paths relative to this file's own location (SingleColumnAblationExplainer/)
# so imports work regardless of which directory you run from.
_THIS_DIR     = Path(os.path.dirname(os.path.abspath(__file__)))
_RESEARCH_DIR = _THIS_DIR.parent   # research-clean/

# Add research-clean/ so Utils, SingleColumnAblationExplainer etc. are importable
if str(_RESEARCH_DIR) not in sys.path:
    sys.path.append(str(_RESEARCH_DIR))

# Models source
model_src_path = str(_RESEARCH_DIR / "MaskLearningExplainer" / "src")
if model_src_path not in sys.path:
    sys.path.insert(0, model_src_path)

# Local imports
from singleColumnAblation_explainer_config import (
    get_explainer_config,
    get_saved_model_dir,
    get_explainer_output_dir,
    DEFAULT_STRATEGIES,
)

from Utils.Utils import (
    load_research_components,
    get_trained_model,
    fix_randomness,
    create_consistent_loader,
    get_next_run_index,
    get_indexed_filepath,
    get_latest_indexed_file,
    load_test_subset,
    subset_table
)

from singleColumnAblation_explainer import (
    AblationExplainer, RelGTAblationExplainer, timed_section, KERNEL_SHAP_STRATEGY,
)
from rdl_explain_files.model.gnn.model import HGSModel

    
    
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run explainer on trained RelBench model')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., rel-f1)')
    parser.add_argument('--task', type=str, required=True,
                       help='Task name (e.g., driver-position)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--strategies', type=str, default='all',
                       choices=['all', 'fast'],
                       help="Ablation strategy preset: 'all' (comprehensive — every "
                            "strategy per feature type) or 'fast' (restricts to ONLY "
                            "the one ablation method per feature type that matches "
                            "Kernel SHAP's / Monte Carlo Shapley's own coalition "
                            "masking — numerical/text/timestamp→Mean, categorical→MFV "
                            "— i.e. use this to get Kernel-SHAP-comparable results "
                            "without computing the other strategies at all).")
    parser.add_argument('--model_type', type=str, default='hgs',
                       choices=['hgs', 'transformer'],
                       help='Model type to explain: hgs (HGS/GraphSAGE) or transformer (RelGT)')
    parser.add_argument('--resume', type=int, default=None,
                       help='Run index to resume (e.g. --resume 3). If not set, starts fresh.')
    return parser.parse_args()


def get_ablation_strategies(preset='all'):
    """Get ablation strategies based on preset.

    'fast' restricts to exactly one ablation method per feature type,
    derived from the single KERNEL_SHAP_STRATEGY mapping in
    singleColumnAblation_explainer.py — the one method per type whose masking
    logic matches Kernel SHAP / Monte Carlo Shapley's own coalition
    masking. Use this preset to get Kernel-SHAP-comparable ablation results
    without computing the other (Zeroing/Permutation/etc.) strategies at all.
    """
    if preset == 'fast':
        # One strategy per feature type — the Kernel-SHAP-aligned method.
        return {feat_type: [method] for feat_type, method in KERNEL_SHAP_STRATEGY.items()}
    else:
        # Comprehensive version - all strategies
        return DEFAULT_STRATEGIES


def build_ablation_feature_map(data, dataset, config):
    """
    Build the ablation feature map for the dataset.
    
    Args:
        data: Graph data object
        dataset: Dataset object
        config: Explainer configuration for this dataset/task
    
    Returns:
        tuple: (node_to_col_names_dict, key_columns_to_exclude)
    """
    print("\n" + "=" * 80)
    print("BUILDING ABLATION FEATURE MAP")
    print("=" * 80)
    
    # Get db from dataset
    db = dataset.get_db()
    
    # Build node_to_col_names_dict from data
    node_to_col_names_dict = {}
    stype_mapping = config['stype_mapping']
    
    for node_type in data.node_types:
        if hasattr(data[node_type], 'tf') and data[node_type].tf is not None:
            tf = data[node_type].tf
            node_to_col_names_dict[node_type] = {}
            
            for stype_key, cols in tf.col_names_dict.items():
                stype_full = str(stype_key).lower()
                
                # Map stype to categories using config
                col_type = None
                for feature_type, stype_patterns in stype_mapping.items():
                    if any(pattern in stype_full for pattern in stype_patterns):
                        col_type = feature_type
                        break
                
                if col_type is None:
                    continue
                
                if col_type not in node_to_col_names_dict[node_type]:
                    node_to_col_names_dict[node_type][col_type] = []
                
                node_to_col_names_dict[node_type][col_type].extend(cols)
    
    print(f"✓ Built node_to_col_names_dict for {len(node_to_col_names_dict)} node types")
    
    # Identify primary and foreign keys to exclude
    key_columns_to_exclude = set()
    
    for table_name, table_object in db.table_dict.items():
        # Detect foreign keys from the DB schema
        if hasattr(table_object, 'fkey_col_to_pkey_table'):
            fkeys_dict = table_object.fkey_col_to_pkey_table
            if isinstance(fkeys_dict, dict):
                for fk_col_name in fkeys_dict.keys():
                    key_columns_to_exclude.add((table_name, fk_col_name))
        
        # Standard PK pattern for RelBench
        pk_name = table_object.pkey_col
        if pk_name:
            key_columns_to_exclude.add((table_name, pk_name))
    
    # Add dataset-specific key columns from config
    for key_col in config.get('key_columns_to_exclude', []):
        key_columns_to_exclude.add(tuple(key_col))

    # Put back columns that RelBench declares as foreign keys but that also
    # carry real information a model can learn from, so they stay explainable
    # here just as they are in Kernel SHAP and Monte Carlo Shapley (neither of
    # which consults the schema at all). Without this, such a column silently
    # disappears from the SingleColumnAblation results and the methods no longer compare
    # over the same feature set.
    keys_to_keep = {tuple(k) for k in config.get('keys_to_explain', [])}
    removed = key_columns_to_exclude & keys_to_keep
    key_columns_to_exclude -= keys_to_keep
    for table_name, col_name in sorted(removed):
        print(f"  \u21b3 keeping key column as an explainable feature: {table_name}.{col_name}")

    print(f"✓ Identified {len(key_columns_to_exclude)} primary and foreign key columns to exclude")
    
    # Count features by type for reporting
    feature_counts = {'numerical': 0, 'text': 0, 'timestamp': 0, 'categorical': 0}
    for node_type, cols_by_type in node_to_col_names_dict.items():
        for col_type in feature_counts.keys():
            if col_type in cols_by_type:
                for col_name in cols_by_type[col_type]:
                    feature_key = (node_type, col_name)
                    if feature_key not in key_columns_to_exclude:
                        feature_counts[col_type] += 1
    
    total_features = sum(feature_counts.values())
    print(f"✓ Ablation Feature Map built with {total_features} safe features")
    print(f"\nFeatures by type:")
    for col_type, count in feature_counts.items():
        print(f"   {col_type}: {count} features")
    
    return node_to_col_names_dict, key_columns_to_exclude


def run_ablation_experiments(explainer, strategies, test_table, baseline_predictions,
                             node_to_col_names_dict, key_columns_to_exclude,
                             tune_metric, threshold=0.5):
    """
    Run ablation experiments with the configured strategies.
    
    Returns:
        DataFrame with ablation results
    """
    print("\n" + "=" * 80)
    print("STARTING ABLATION EXPERIMENTS")
    print("=" * 80)
    
    results_df = explainer.run_ablation_experiments(
        strategies=strategies,
        test_loader_or_table=test_table,
        baseline_predictions=baseline_predictions,
        node_to_col_names_dict=node_to_col_names_dict,
        exclude_keys=key_columns_to_exclude,
        threshold=threshold,
        tune_metric=tune_metric
    )
    
    return results_df


def save_results(results_df, output_dir, dataset_name, task_name):
    """
    Save ablation results to CSV with automatic indexing.
    
    Returns:
        Tuple of (Path to saved results file, run_index)
    """
    run_index = get_next_run_index(output_dir)
    results_path = get_indexed_filepath(output_dir, 'singleColumnAblation_results', '.csv', run_index)
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    return results_path, run_index


def create_visualization(results_df, strategies, output_dir, run_index,
                         dataset_name, task_name, test_table, config, threshold=0.5):
    """
    Create visualization of ablation results as a single flat bar chart.

    For classification → bars show number of label flips (max across strategies) per feature.
    For regression    → bars show importance_score = mean(|ablated_pred - baseline_pred|)
                        (max across strategies) per feature.

    Returns:
        Path to saved plot, or None if no data.
    """
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION")
    print("=" * 80)

    task_type = config.get('task_type', 'classification')

    viz_config = config.get('visualization', {})
    top_n_features = viz_config.get('top_n_features', 20)
    bar_color = viz_config.get('bar_color', 'skyblue')

    TOTAL_RECORDS = len(test_table.df)

    if task_type == 'regression':
        importance_col = 'importance_score'
        x_label = 'Importance Score (mean |Δpred|)'
        title_metric = 'Mean |Δpred|'
        scale_note = "Scale: data-driven"
    else:
        importance_col = 'num_label_flips'
        x_label = 'Label Flips'
        title_metric = 'Label Flips'
        scale_note = f"Scale: 0 to {TOTAL_RECORDS} samples (threshold: {threshold:.4f})"

    if results_df.empty:
        print("⚠️  No data found. Skipping visualization.")
        return None

    # Aggregate: for each feature take the max importance across all strategies
    df_plot = (
        results_df
        .groupby('feature_name', as_index=False)[importance_col]
        .max()
        .rename(columns={importance_col: 'Abs Importance Score'})
    )
    df_plot['Abs Importance Score'] = df_plot['Abs Importance Score'].abs()

    # Sort descending, take top N
    df_plot = df_plot.sort_values(by='Abs Importance Score', ascending=False).head(top_n_features)

    if df_plot.empty:
        print("⚠️  No features with importance > 0. Skipping visualization.")
        return None

    num_features = len(df_plot)
    fig_height = max(4, 0.35 * num_features)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    bars = ax.barh(df_plot['feature_name'], df_plot['Abs Importance Score'], color=bar_color)

    ax.set_xlabel(x_label)
    ax.set_title(
        f'Feature Importance: {title_metric} — {dataset_name}/{task_name}\n{scale_note}',
        pad=15
    )
    ax.invert_yaxis()  # highest importance at top

    max_val = df_plot['Abs Importance Score'].max()

    x_padding = max_val * 0.015 if max_val > 0 else 0.01
    for bar in bars:
        width = bar.get_width()
        label_fmt = f'{width:.2e}' if 0 < width < 0.0001 else (
            f'{int(width)}' if task_type == 'classification' else f'{width:.4f}'
        )
        ax.text(
            width + x_padding,
            bar.get_y() + bar.get_height() / 2,
            label_fmt,
            va='center', ha='left', fontsize=9, fontweight='semibold'
        )

    ax.set_xlim(0, max_val * 1.15 if max_val > 0 else 1.0)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()

    plot_path = get_indexed_filepath(output_dir, 'ablation_importance_plot', '.png', run_index)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved to: {plot_path.name}")
    return plot_path

def extract_strategy_results(explainer, output_path, output_dir, run_index):
    """
    Extract results using the standard strategy protocol.
    Uses explainer.extract_strategy_results() method.
    
    Returns:
        Tuple of (Path to saved filtered results, filtered DataFrame)
    """
    print("\n" + "=" * 80)
    print("EXTRACTING STRATEGY RESULTS")
    print("=" * 80)
    
    # Standard protocol — one method per feature type. Sourced from
    # singleColumnAblation_explainer.KERNEL_SHAP_STRATEGY so this can't drift
    # out of sync with the --strategies fast preset.
    my_strategy = KERNEL_SHAP_STRATEGY
    
    filtered_path = get_indexed_filepath(
        output_dir, 
        'singleColumnAblation_results_as_kernel_shap', 
        '.csv', 
        run_index
    )
    
    explainer.extract_strategy_results(output_path, filtered_path, my_strategy)
    print(f"✓ Filtered results saved to: {filtered_path}")
    
    filtered_df = pd.read_csv(filtered_path)
    return filtered_path, filtered_df


# ── my_strategy method config ────────────────────────────────────────────────
# Sourced from singleColumnAblation_explainer.KERNEL_SHAP_STRATEGY — see comment
# in extract_strategy_results() above.
_MY_STRATEGY = KERNEL_SHAP_STRATEGY

# Each entry: (metric_column, y_axis_label, file_suffix)
_STRATEGY_METHODS = [
    ('label_flip_rate',      'Label Flip Rate (fraction)',           'label_flip'),
    ('importance_score',     'Importance Score (mean |Δprob|)',       'importance_score'),
]



def create_strategy_visualization(
    results_df: pd.DataFrame,
    output_dir,
    run_index: int,
    dataset_name: str,
    task_name: str,
    config: dict,
    threshold: float = 0.5,
) -> list:
    """
    Create visualizations per importance method, showing features selected by my_strategy.
    Generates horizontal bar charts using absolute metrics. Produces both a full features
    plot and a Top 20 features plot for each available metric.

    Args:
        results_df:    Full ablation results DataFrame from run_ablation_experiments().
        output_dir:    Directory in which to save the PNGs.
        run_index:     Integer run index used for file naming.
        dataset_name:  Dataset name for the plot title.
        task_name:     Task name for the plot title.
        config:        Explainer configuration dict (used for viz overrides).
        threshold:     The optimized decision threshold used for classification.

    Returns:
        List of Path objects for each saved PNG.
    """
    print("\n" + "=" * 80)
    print("CREATING STRATEGY VISUALIZATIONS (my_strategy - Absolute Values)")
    print("=" * 80)

    task_type  = config.get('task_type', 'classification')
    viz_config = config.get('visualization', {})
    bar_color  = viz_config.get('bar_color', '#4682B4')
    bar_height = viz_config.get('bar_width', 0.6)

    # ── Filter to my_strategy rows ────────────────────────────────────────────
    masks = []
    for feat_type, method in _MY_STRATEGY.items():
        masks.append(
            (results_df['feature_type'] == feat_type) &
            (results_df['ablation_method_used'] == method)
        )
    combined = masks[0]
    for m in masks[1:]:
        combined |= m
    strategy_df = results_df[combined].copy()

    if strategy_df.empty:
        print("⚠️  No data matching my_strategy — skipping strategy visualizations.")
        return []

    # Identify existing feature types in the data
    feature_types = [
        ft for ft in ['numerical', 'text', 'timestamp', 'categorical']
        if ft in strategy_df['feature_type'].unique()
    ]
    num_rows = len(feature_types)  # FIX 1: was dedented one level too far
    saved_paths = []

    def draw_strategy_grid(df_source, max_features_per_type, file_suffix_type, title_prefix):
        fig, axes = plt.subplots(
            num_rows, 1,
            figsize=(
                14,
                max(5, 5 * num_rows)
                if max_features_per_type <= 20
                else max(6, 0.3 * max_features_per_type * num_rows)
            ),
            squeeze=False,
        )

        for row_idx, feat_type in enumerate(feature_types):
            ax = axes[row_idx, 0]

            type_data = df_source[df_source['feature_type'] == feat_type].copy()
            type_data['Abs_Metric'] = type_data[metric_col].abs()

            # FIX 2: sort descending so .head() gives the highest-importance features
            type_data = (
                type_data
                .sort_values('Abs_Metric', ascending=False)
                .head(max_features_per_type)  # FIX 3: now correctly takes the top N after descending sort
                .reset_index(drop=True)
            )

            if type_data.empty:
                ax.axis('off')
                ax.text(0.5, 0.5, f'No data\n({feat_type})',
                        ha='center', va='center', transform=ax.transAxes, fontsize=14)
                continue

            y_pos = np.arange(len(type_data))
            rects = ax.barh(y_pos, type_data['Abs_Metric'], bar_height, color=bar_color)

            max_val = type_data['Abs_Metric'].max()
            x_padding = max_val * 0.015 if max_val > 0 else 0.01

            for rect in rects:
                w = rect.get_width()
                label_fmt = f'{int(w)}' if metric_col == 'num_label_flips' else f'{w:.4f}'
                ax.text(
                    w + x_padding,
                    rect.get_y() + rect.get_height() / 2.,
                    label_fmt,
                    ha='left', va='center', fontsize=9, fontweight='bold',
                )

            ablation_method = _MY_STRATEGY.get(feat_type, '?')
            ax.set_title(
                f"{feat_type.upper()}  [ablation: {ablation_method}]",
                fontsize=15, fontweight='bold', pad=10,
            )
            ax.set_xlabel(f"Absolute {y_label}", fontsize=12)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(type_data['feature_name'], fontsize=10)
            ax.set_xlim(0, max_val * 1.18 if max_val > 0 else 1.0)
            ax.invert_yaxis()  # highest value at top
            ax.grid(axis='x', linestyle='--', alpha=0.4)

        title_suffix = (
            f" (at Threshold: {threshold:.4f})"
            if (task_type == 'classification' and 'flip' in metric_col)
            else ""
        )
        fig.suptitle(
            f"my_strategy — {title_prefix} {y_label}{title_suffix}\n"
            f"{dataset_name} / {task_name}  (run {run_index})",
            fontsize=20, fontweight='bold',
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        plot_path = get_indexed_filepath(
            output_dir, f'singleColumnAblation_strategy_{file_suffix}_{file_suffix_type}', '.png', run_index
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths.append(plot_path)
        print(f"  ✓ [{file_suffix}_{file_suffix_type}] saved → {plot_path.name}")

    # ── Loop through metric configurations ────────────────────────────────────
    for metric_col, y_label, file_suffix in _STRATEGY_METHODS:

        if metric_col not in strategy_df.columns:
            continue

        plot_df = strategy_df.dropna(subset=[metric_col])
        if plot_df.empty:
            print(f"  ⚠️  All NaN for '{metric_col}' — skipping {file_suffix} plot.")
            continue

        max_feats_in_groups = plot_df.groupby('feature_type').size().max()

        # 1. Full features plot
        draw_strategy_grid(plot_df, max_features_per_type=max_feats_in_groups, file_suffix_type='full', title_prefix='Full')

        # 2. Top 20 plot (only if any category has more than 20 features)
        if max_feats_in_groups > 20:
            draw_strategy_grid(plot_df, max_features_per_type=20, file_suffix_type='top20', title_prefix='Top 20')
        else:
            print(f"  i Skipping Top 20 plot for {file_suffix}: all category feature counts are <= 20.")

    return saved_paths



def main():
    """Main explainer pipeline."""
    args = parse_args()
    script_start_time = time.time()
    DATASET_NAME = args.dataset
    TASK_NAME    = args.task
    SEED         = args.seed
    MODEL_TYPE   = args.model_type   # 'hgs' or 'transformer'
    RESUME_INDEX = args.resume

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:      {device}")
    print(f"Model type:  {MODEL_TYPE.upper()}")

    fix_randomness(SEED)

    # Get configuration — routes to HGS or transformer config block
    config      = get_explainer_config(DATASET_NAME, TASK_NAME, MODEL_TYPE)
    tune_metric = config['tune_metric']
    task_type   = config['task_type']

    print(f"\nTask type:   {task_type.upper()}")
    print(f"Tune metric: {tune_metric}")

    # Get directories — each model type has its own saved_model and explainer dirs
    saved_model_dir = get_saved_model_dir(DATASET_NAME, TASK_NAME, MODEL_TYPE)
    output_dir      = get_explainer_output_dir(DATASET_NAME, TASK_NAME, MODEL_TYPE)
    
    print("\n" + "=" * 80)
    print("LOADING RESEARCH COMPONENTS")
    print("=" * 80)
    print(f"Looking for latest saved model in: {saved_model_dir}")

    with timed_section("Loading research components (data, task, metadata)"):
        (data, col_stats_dict, task, dataset,
         metadata, loader_config, test_predictions) = load_research_components(
            saved_model_dir, device
        )

        fix_randomness(metadata['seed'])

        print(f"✓ Data, Task, and Metadata loaded from: {saved_model_dir}")
        print(f"✓ Randomness fixed with seed: {metadata.get('seed', 42)}")

        train_table = task.get_table("train")
        val_table = task.get_table("val")
        test_table = task.get_table("test")

        print(f"✓ Tables loaded: Train({len(train_table.df)}), "
              f"Val({len(val_table.df)}), Test({len(test_table.df)})")

        # Tasks with a saved explanation subset (rel-arxiv, whose test set is far
        # too large to explain in full) are explained on exactly those rows, so
        # every explainer explains the same records. Tasks without a subset file
        # keep explaining the full test set.
        subset_idx, subset_meta = load_test_subset(DATASET_NAME, TASK_NAME, task)
        if subset_idx is not None:
            test_table = subset_table(test_table, subset_idx)
            print(f"✓ Explaining a subset: Test({len(test_table.df)}) rows")

    with timed_section("Creating data loaders"):
        loader_dict = {}
        for split, table in [
            ("train", train_table),
            ("val", val_table),
            ("test", test_table),
        ]:
            loader_dict[split] = create_consistent_loader(
                data, loader_config, task, table, split
            )
        print("✓ Data loaders created consistently")
    
    print("\n" + "=" * 80)
    print(f"LOADING TRAINED MODEL  [{MODEL_TYPE.upper()}]")
    print("=" * 80)

    with timed_section(f"Loading trained model [{MODEL_TYPE.upper()}]"):
        weights_file = get_latest_indexed_file(saved_model_dir, 'model_weights', '.pt')
        if weights_file is None:
            raise FileNotFoundError(f"No model weights found in {saved_model_dir}")

        print(f"Using model weights: {weights_file.name}")

        if MODEL_TYPE == "hgs":
            # ── HGS (GraphSAGE) ───────────────────────────────────────────────────
            model = get_trained_model(
                model_class=HGSModel,
                data=data,
                col_stats_dict=col_stats_dict,
                metadata=metadata,
                weights_path=weights_file,
                device=device,
            )
            entity_table = task.entity_table

        else:
            # ── RelGT (Transformer) ───────────────────────────────────────────────
            from rdl_explain_files.model.relgt import RelGT, RelGTConfig
            checkpoint = torch.load(weights_file, map_location="cpu", weights_only=False)
            mcfg = checkpoint["model_config"]
            relgt_cfg = RelGTConfig(
                local_num_layers=mcfg["local_num_layers"],
                channels=mcfg["channels"],
                out_channels=mcfg["out_channels"],
                global_dim=mcfg["global_dim"],
                heads=mcfg["heads"],
                ff_dropout=mcfg.get("ff_dropout", 0.1),
                attn_dropout=mcfg.get("attn_dropout", 0.1),
                conv_type=mcfg.get("conv_type", "full"),
                num_centroids=mcfg.get("num_centroids", 4096),
                sample_node_len=mcfg["K"],
            )
            # RelGT needs node_type_map and max_neighbor_hop from checkpoint
            model = RelGT(
                num_nodes=checkpoint["num_nodes"],
                max_neighbor_hop=checkpoint["max_neighbor_hop"],
                node_type_map=checkpoint["node_type_map"],
                col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
                col_stats_dict=col_stats_dict,
                **relgt_cfg.model_dump(),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device).eval()

        baseline_predictions = test_predictions
        # test_predictions_N.npy covers the FULL test set — keep only the subset
        # rows so it lines up with the (subset) test table. For HGS the baseline
        # is recomputed from test_table further below anyway.
        if subset_idx is not None:
            baseline_predictions = test_predictions[subset_idx]
        entity_table = task.entity_table

        print(f"✓ Model loaded and set to EVAL mode")
        print(f"✓ Baseline predictions ready: {len(baseline_predictions)} samples")
        print(f"  min: {baseline_predictions.min():.4f}  "
              f"max: {baseline_predictions.max():.4f}  "
              f"mean: {baseline_predictions.mean():.4f}")

        # Verify model works (HGS only — RelGT loader has a different interface)
        if MODEL_TYPE == "hgs":
            print("\nVerifying model...")
            with torch.no_grad():
                sample_batch = next(iter(loader_dict['test'])).to(device)
                sample_pred = model(sample_batch, entity_table)
                print(f"✓ Forward pass successful: {sample_pred.shape}")

    print(f"\nSETUP COMPLETE - READY FOR EXPLAINER!")
    
    # Initialize explainer — model-type-specific subclass
    with timed_section("Initializing ablation explainer"):
        if MODEL_TYPE == "hgs":
            explainer = AblationExplainer(
                model=model,
                data=data,
                task=task,
                col_stats_dict=col_stats_dict,
                device=device,
                loader_config=loader_config,
                entity_table=entity_table,
                task_type=task_type,
                test_subset_idx=subset_idx,
            )
        else:
            # RelGTAblationExplainer overrides create_loader() and test() to use
            # RelGTTokens + DataLoader and the RelGT forward pass interface.
            relgt_tokens_kwargs = {
                "data":            data,
                "task":            task,
                "K":               loader_config["K"],
                "precomputed_dir": loader_config["precomputed_dir"],
                "undirected":      True,
                "num_workers":     loader_config.get("num_workers_precompute", 4),
            }
            explainer = RelGTAblationExplainer(
                model=model,
                data=data,
                task=task,
                col_stats_dict=col_stats_dict,
                device=device,
                loader_config=loader_config,
                entity_table=entity_table,
                task_type=task_type,
                relgt_tokens_kwargs=relgt_tokens_kwargs,
                test_subset_idx=subset_idx,
            )
        print(f"✅ Ablation explainer initialized for {MODEL_TYPE.upper()}!")

        # baseline_predictions above came from the on-disk test_predictions_N.npy
        # (computed by train.py's test() with the old, randomly-sampled neighbor
        # budget), while every ablated prediction from here on uses
        # explainer.test()'s deterministic temporal_strategy="last" sampling.
        # Diffing predictions from two different sampling policies would inject
        # spurious per-feature noise into every importance score, so recompute
        # baseline the same way ablated predictions will be computed. HGS only
        # for now.
        if MODEL_TYPE == "hgs":
            baseline_predictions = explainer.test(explainer.create_loader(test_table))
            print(f"✓ Baseline predictions recomputed via explainer.test() for consistency: "
                  f"{len(baseline_predictions)} samples")
            print(f"  min: {baseline_predictions.min():.4f}  "
                  f"max: {baseline_predictions.max():.4f}  "
                  f"mean: {baseline_predictions.mean():.4f}")

    # --------------------------------------------------------
    # Decision threshold (classification only) — read from metadata, NOT
    # tuned here. The threshold must be computed exactly once, via
    # Utils/compute_and_save_threshold.py, and stored in metadata_<N>.json.
    # Every explanation method (SingleColumnAblation, Kernel SHAP, Monte Carlo) reads the
    # same saved value so they never disagree on which threshold to use.
    # --------------------------------------------------------
    chosen_threshold = None if task_type == 'regression' else metadata.get('threshold')
    if task_type == 'classification':
        if chosen_threshold is None:
            print(
                "[Warning] No 'threshold' found in metadata.json for this run. "
                "Run Utils/compute_and_save_threshold.py first so all explanation "
                "methods use the same value. Falling back to 0.5 for now.\n"
            )
            chosen_threshold = 0.5
        else:
            print(f"[Threshold] Using saved decision threshold from metadata: {chosen_threshold:.4f}\n")
            
            
    strategies = get_ablation_strategies(args.strategies)
    print(f"\nStrategy preset: {args.strategies}")
    print("Configured ablation strategies:")
    for data_type, methods in strategies.items():
        print(f"  {data_type}: {methods}")
    
    with timed_section("Building ablation feature map"):
        node_to_col_names_dict, key_columns_to_exclude = build_ablation_feature_map(
            data, dataset, config
        )
    
    # ── Resolve run index ──────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    RESUME_DIR = output_dir / "resume"
    RESUME_DIR.mkdir(parents=True, exist_ok=True)
 
    if RESUME_INDEX is not None:
        run_index = RESUME_INDEX
        print(f"\n⟳  Resuming run index {run_index}")
    else:
        run_index = get_next_run_index(output_dir)
        print(f"\n Starting fresh run — index {run_index}")
 
    RESUME_FILE = RESUME_DIR / f"resume_{run_index}.json"
 
    import json as _json
    resume_state = {}
    if RESUME_INDEX is not None:
        if not RESUME_FILE.exists():
            raise FileNotFoundError(
                f"Resume file not found: {RESUME_FILE}\n"
                f"Make sure run index {run_index} exists in {RESUME_DIR}"
            )
        with open(RESUME_FILE) as f:
            resume_state = _json.load(f)
        print(f"   Completed phases: {resume_state.get('completed', [])}")
 
    def _save_resume(state: dict):
        with open(RESUME_FILE, 'w') as f:
            _json.dump(state, f, indent=2)
 
    # ── PHASE 1: Ablation experiments ─────────────────────────────────────────
    if "ablation" not in resume_state.get("completed", []):
        with timed_section("PHASE 1: Ablation experiments"):
            results_df = run_ablation_experiments(
                explainer,
                strategies,
                test_table,
                baseline_predictions,
                node_to_col_names_dict,
                key_columns_to_exclude,
                tune_metric,
                threshold=chosen_threshold,
            )
            # Add the tuned threshold as a column so the value is self-documenting
            # alongside the per-feature scores. For regression chosen_threshold is
            # None (threshold is irrelevant) so we skip the column entirely.
            if chosen_threshold is not None:
                results_df['decision_threshold'] = chosen_threshold
            results_path = get_indexed_filepath(output_dir, 'singleColumnAblation_results', '.csv', run_index)
            results_df.to_csv(results_path, index=False)
            print(f"\n✓ Results saved to: {results_path}")
            if chosen_threshold is not None:
                print(f"   decision_threshold = {chosen_threshold:.4f}")
        resume_state["results_path"] = str(results_path)
        resume_state.setdefault("completed", []).append("ablation")
        _save_resume(resume_state)
        print("✓ Phase 1 (ablation) complete — state saved")
    else:
        results_path = Path(resume_state["results_path"])
        results_df   = pd.read_csv(results_path)
        print(f"\n⟳  Skipping ablation (loaded from {results_path.name})")
 

    with timed_section("PHASE 2: Extract strategy results"):
        filtered_path, filtered_df = extract_strategy_results(
            explainer, results_path, output_dir, run_index
        )

    # ── PHASE 2b: Per-method strategy visualizations ──────────────────────────
    # Produces 4 separate PNGs (label_flip, importance_score, corr_spearman,
    # corr_kendall) showing only my_strategy features.
    with timed_section("PHASE 2b: Strategy visualizations"):
        strategy_plot_paths = create_strategy_visualization(
            results_df, output_dir, run_index,
            DATASET_NAME, TASK_NAME, config, chosen_threshold
        )
 
    if RESUME_FILE.exists():
        RESUME_FILE.unlink()
        print("\n✓ Resume file removed (all phases completed successfully)")
 
    print("\n" + "=" * 80)
    print("EXPLAINER COMPLETE!")
    print("=" * 80)
    print(f"\nDataset:     {DATASET_NAME}")
    print(f"Task:        {TASK_NAME}")
    print(f"Model type:  {MODEL_TYPE.upper()}")
    print(f"Task type:   {task_type.upper()}")
    print(f"Strategies:  {args.strategies}")
    print(f"Run Index:   {run_index}")
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"\nSaved files:")
    print(f"  - {results_path.name}")
    print(f"  - {filtered_path.name}")
    for sp in strategy_plot_paths:
        print(f"  - {sp.name}")
    

    script_total_elapsed = time.time() - script_start_time
    print(f"\n⏱  TOTAL RUNTIME (start to finish): {script_total_elapsed:.2f}s "
          f"({script_total_elapsed/60:.2f} min)")
    print("\n✓ All explainer outputs ready for analysis!")
 
 
if __name__ == "__main__":
    main()