"""
Main RDL explainer script for RelBench datasets.

Usage:
    python run_rdl_explainer.py --dataset <dataset> --task <task> [--model_type hgs|transformer] [--seed <seed>]

Examples:
    python run_rdl_explainer.py --dataset rel-trial --task study-outcome --model_type hgs --seed 42
    python run_rdl_explainer.py --dataset rel-trial --task study-outcome --model_type transformer --seed 42
    python run_rdl_explainer.py --dataset rel-f1 --task driver-position --model_type transformer

Note: RDL learns masks on GNN layers. For transformer, it operates on the
local GNN sub-network inside RelGT.
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
import copy
import pickle
import json
from pathlib import Path
from contextlib import contextmanager
from tqdm.auto import tqdm


@contextmanager
def timed_section(label: str):
    """Time a block of code and print how long it took.

    Usage:
        with timed_section("PHASE 4: Final mask learning"):
            ... do work ...

    Prints elapsed wall-clock time in seconds (and minutes) once the block
    exits, including if it raises, so every major stage of the pipeline is
    self-documenting about how much of the total runtime it consumed.
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"⏱  [{label}] completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")

    
# Resolve all paths relative to this file's own location so imports work
# regardless of which directory you run from.
_THIS_DIR     = Path(os.path.dirname(os.path.abspath(__file__)))
_RESEARCH_DIR = _THIS_DIR.parent   # research-clean/

if str(_RESEARCH_DIR) not in sys.path:
    sys.path.append(str(_RESEARCH_DIR))

# HGS model source (rdl-explain-main)
model_src_path = str(_RESEARCH_DIR / "MaskLearningExplainer" / "src")
if model_src_path not in sys.path:
    sys.path.insert(0, model_src_path)

# Local imports
from rdl_explainer_config import (
    get_rdl_config,
    get_rdl_saved_model_dir,
    get_rdl_output_dir,
    get_rdl_inference_dir,
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

from rdl_explain_files.model.gnn.model import HGSModel
from rdl_explain_files.explain.config import ExplainerConfig
from rdl_explain_files.explain.explainer import RDLExplainer

def apply_sigmoid(preds):
    """Convert raw logits to probabilities via sigmoid.
    Accepts either a numpy array or a torch.Tensor.
    Consistent with train.py and train_trans.py.
    """
    if isinstance(preds, torch.Tensor):
        return torch.sigmoid(preds)
    import numpy as np
    return 1.0 / (1.0 + np.exp(-preds))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run RDL explainer on trained RelBench model')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., rel-arxiv)')
    parser.add_argument('--task', type=str, required=True,
                       help='Task name (e.g., paper-citation)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--model_type', type=str, default='hgs',
                       choices=['hgs', 'transformer'],
                       help='Model type to explain: hgs (default) or transformer (RelGT)')
    parser.add_argument('--resume', type=int, default=None,
                       help='Run index to resume (e.g. --resume 3). If not set, starts fresh.')
    return parser.parse_args()


def create_inference_df_relgt(model, task, relgt_loaders, split_name, device,
                              inference_dir, task_type='classification'):
    """
    Create inference dataframe for RelGT model.
    Mirrors the notebook's generate_predictions() — returns raw logits in
    'output' and sigmoid probabilities in 'processed_output'.
    """
    import numpy as _np

    _infer_start = time.time()
    logit_list, idx_list = [], []
    with torch.no_grad():
        for batch in tqdm(relgt_loaders[split_name],
                          desc=f"Inference [{split_name}]", leave=False):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            grouped_tf_dict = {
                "grouped_tfs":     batch["grouped_tfs"],
                "grouped_indices": batch["grouped_indices"],
                "flat_batch_idx":  batch["flat_batch_idx"],
                "flat_nbr_idx":    batch["flat_nbr_idx"],
            }
            pred = model(
                batch["neighbor_types"], batch["node_indices"],
                batch["neighbor_hops"],  batch["neighbor_times"],
                grouped_tf_dict,
                edge_index=batch["edge_index"], batch=batch["batch"],
            )
            pred = pred.view(-1) if pred.size(-1) == 1 else pred
            logit_list.append(pred.detach().cpu().numpy())
            idx_list.append(batch["global_idx"].cpu().numpy())

    logits = _np.concatenate(logit_list)
    idxs   = _np.concatenate(idx_list)
    ordered_logits = _np.empty_like(logits)
    ordered_logits[idxs] = logits

    if task_type == 'classification':
        probs = apply_sigmoid(ordered_logits)
        hard  = (probs >= 0.5).astype(float)
    else:
        probs = ordered_logits
        hard  = ordered_logits

    df_base = task.get_table(split_name).df.copy()
    time_col   = 'date' if 'date' in df_base.columns else 'timestamp'
    target_col = task.target_col
    targets    = df_base[target_col].fillna(0).values if target_col in df_base.columns                  else _np.full(len(df_base), _np.nan)

    inference_data = {
        task.entity_col:    df_base[task.entity_col],
        time_col:           df_base[time_col],
        'output':           ordered_logits,
        'processed_output': probs,
        'predictions':      hard,
        target_col:         targets,
        'targets':          targets,
    }
    df_inference = pd.DataFrame(inference_data)

    os.makedirs(inference_dir, exist_ok=True)
    file_path = os.path.join(inference_dir, f"predictions_{split_name}.parquet")
    df_inference.to_parquet(file_path)
    print(f"     Saved to: {file_path}")
    _infer_elapsed = time.time() - _infer_start
    print(f"⏱  [Inference (RelGT) | {split_name}] completed in "
          f"{_infer_elapsed:.2f}s ({_infer_elapsed/60:.2f} min)")
    return df_inference


def create_inference_df(model, task, loader_dict, split_name, entity_table, device,
                        inference_dir, dataset, task_type='classification'):
    """
    Create inference dataframe for the HGS model.
    Uses the NeighborLoader batch interface: model(batch, entity_table).

    For classification: stores raw logits in 'output', sigmoid probabilities
                        in 'processed_output', thresholded labels in 'predictions'.
    For regression:     stores raw values in all three columns.
    """
    # Explanation methods compare predictions across many masking variants,
    # so they need bit-exact reproducibility. GPU scatter/aggregation ops in
    # message passing aren't covered by fix_randomness/manual_seed, so force
    # deterministic kernels here (paired with num_neighbors=-1 on loader_dict
    # above, and CUBLAS_WORKSPACE_CONFIG set at the top of this script).
    torch.use_deterministic_algorithms(True)
    all_preds = []
    _infer_start = time.time()

    # Run inference using HGS batch interface
    for batch in tqdm(loader_dict[split_name], desc=f"Inference on {split_name}", leave=False):
        batch = batch.to(device)
        pred  = model(batch, entity_table)
        all_preds.append(pred.detach().cpu())

    all_preds = torch.cat(all_preds, dim=0).view(-1)
    if task_type == 'regression':
        processed_output = all_preds
        predictions      = all_preds
    else:
        # HGS returns raw logits — apply sigmoid
        processed_output = apply_sigmoid(all_preds)
        predictions      = (processed_output > 0.5).long()

    df_base = task.get_table(split_name).df.copy()
    time_col = 'date' if 'date' in df_base.columns else 'timestamp'

    inference_data = {
        task.entity_col: df_base[task.entity_col],
        time_col: df_base[time_col],
        'output': all_preds.numpy(),
        'processed_output': processed_output.numpy(),
        'predictions': predictions.numpy(),
    }
    
    target_col = task.target_col
    
    if target_col in df_base.columns:
        targets_array = df_base[target_col].fillna(0).values
        inference_data[target_col] = targets_array
        # Also add a generic 'targets' key for the explainer's internal logic
        inference_data['targets'] = targets_array
        print(f"      Loaded {len(targets_array)} true outcomes from column: {target_col}")
    else:
        print(f"      No outcomes available for {split_name}, filling with NaN")
        inference_data[target_col] = np.full(len(df_base), np.nan)
        inference_data['targets'] = np.full(len(df_base), np.nan)

    df_inference = pd.DataFrame(inference_data)

    os.makedirs(inference_dir, exist_ok=True)
    file_path = os.path.join(inference_dir, f"predictions_{split_name}.parquet")
    df_inference.to_parquet(file_path)
    print(f"     Saved to: {file_path}\n")
    _infer_elapsed = time.time() - _infer_start
    print(f"⏱  [Inference (HGS) | {split_name}] completed in "
          f"{_infer_elapsed:.2f}s ({_infer_elapsed/60:.2f} min)")
    
    return df_inference


def make_explanation_task(prediction_task, path_to_inference_results, explanation_target_type='soft'):
    """
    Create an explanation task from the prediction task by loading inference results.
    """
    explanation_task = copy.deepcopy(prediction_task)

    # Load the inference results (which contain the original column names)
    train_predictions = pd.read_parquet(os.path.join(path_to_inference_results, "predictions_train.parquet"))
    val_predictions = pd.read_parquet(os.path.join(path_to_inference_results, "predictions_val.parquet"))
    test_predictions = pd.read_parquet(os.path.join(path_to_inference_results, "predictions_test.parquet"))
    
    train_table = prediction_task.get_table("train")
    val_table = prediction_task.get_table("val")
    test_table = prediction_task.get_table("test")

    train_table.df = train_predictions
    val_table.df = val_predictions
    test_table.df = test_predictions

    explanation_task.train_table = train_table
    explanation_task.val_table = val_table
    explanation_task.test_table = test_table

    assert all(col in table.df.columns for col in ['predictions', 'processed_output']
              for table in [train_table, val_table, test_table])

    target_col = prediction_task.target_col
    
    for table in [explanation_task.train_table, explanation_task.val_table, explanation_task.test_table]:
        if target_col in table.df.columns:
            table.df['targets'] = table.df[target_col]
        elif 'targets' in table.df.columns:
            pass 
        else:
            table.df['targets'] = np.nan

    explanation_task.explanation_target_column = 'predictions'
    explanation_task.explanation_soft_target_column = 'processed_output'
    
    # The 'target_col' of the task is redirected to the model output we want to explain
    explanation_task.target_col = (
        explanation_task.explanation_soft_target_column 
        if explanation_target_type == 'soft' 
        else explanation_task.explanation_target_column
    )

    return explanation_task


def make_explanation_task_relgt(prediction_task, split_predictions,
                                explanation_target_type='soft'):
    """
    Create explanation task for RelGT following the notebook exactly.
    Injects model predictions directly into the table DFs and fixes
    entity_table capitalisation to match data.node_types.
    """
    import copy as _copy
    explanation_task = _copy.deepcopy(prediction_task)

    for split in ["train", "val", "test"]:
        table = explanation_task.get_table(split)
        preds = split_predictions[split]
        table.df["output"]           = preds["output"]
        table.df["processed_output"] = preds["processed_output"]
        table.df["predictions"]      = preds["predictions"]
        target_col = prediction_task.target_col
        if target_col in table.df.columns:
            table.df["targets"] = table.df[target_col]
        else:
            table.df["targets"] = np.nan
        # Inject directly as attribute (required by RelGTExplainer)
        setattr(explanation_task, f"{split}_table", table)

    explanation_task.explanation_target_type = explanation_target_type
    if explanation_target_type == 'soft':
        explanation_task.target_col = 'processed_output'
    else:
        explanation_task.target_col = 'predictions'

    return explanation_task


def _fix_entity_table_case(explanation_task, data):
    """Fix entity_table capitalisation to match data.node_types (RelGT lowercases them)."""
    et_lower = explanation_task.entity_table.lower()
    for key in data.node_types:
        if key.lower() == et_lower:
            explanation_task.entity_table = key
            break
    return explanation_task


def learn_final_mask(explainer, selected_eps, dataset_name, task_name):
    """
    Learn the final mask using the selected epsilon.
    """
    print("\n" + "=" * 80)
    print("LEARNING FINAL MASK")
    print("=" * 80)
    
    mask_type = 'column'
    print(f'Learning mask of type: {mask_type}')
    
    mask_learning_config = {
        'reg_eps': selected_eps,
        'learning_rate': 0.05,
        'epochs': 50,
        'explanation_type': 'column',
        'elimination_strategy': 'permutation_joint'
    }
    
    print(f"\nConfiguration:")
    print(f"  reg_eps: {mask_learning_config['reg_eps']}")
    print(f"  learning_rate: {mask_learning_config['learning_rate']}")
    print(f"  epochs: {mask_learning_config['epochs']}")
    print(f"  explanation_type: {mask_learning_config['explanation_type']}")
    print(f"  elimination_strategy: {mask_learning_config['elimination_strategy']}")
    
    _mask_start = time.time()
    mask, mask_vals, metrics = explainer.learn_masks(
        eps=mask_learning_config['reg_eps'],
        explanation_type=mask_learning_config['explanation_type'], 
        elimination_strategy=mask_learning_config['elimination_strategy'],
        n_epochs=mask_learning_config['epochs'],
        lr=mask_learning_config['learning_rate'],
        # learn_masks() defaults to split='train'; pass split='test' explicitly
        # so mask learning explains the test set, like Kernel SHAP and Monte
        # Carlo Shapley.
        split='test',
    )
    _mask_elapsed = time.time() - _mask_start
    
    print("\n✓ Final mask learning complete")
    print(f"⏱  [learn_masks() | {mask_learning_config['epochs']} epochs] completed in "
          f"{_mask_elapsed:.2f}s ({_mask_elapsed/60:.2f} min)")
    
    return mask, mask_vals, metrics, mask_learning_config


def prepare_importance_dataframe(mask, data, dataset_name, task_name):
    """
    Prepare importance dataframe from mask with metadata.
    """
    print("\n" + "=" * 80)
    print("PREPARING IMPORTANCE DATAFRAME")
    print("=" * 80)
    _prep_start = time.time()

    explanation_elements = []
    feature_formatted = []
    importance_scores = []
    table_names = []
    column_names = []
    feature_types = []
    
    for explanation_element, mask_value in mask.items():
        # explanation_element is the tuple, e.g., ('studies', 'phase')
        table_name, col_name = explanation_element
        
        stype_str = "unknown"
        
        if table_name in data.node_types:
            if hasattr(data[table_name], 'tf') and data[table_name].tf is not None:
                tf = data[table_name].tf

                found = False
                for stype_key, cols in tf.col_names_dict.items():
                    if col_name in cols:
                        stype_full = str(stype_key)

                        if 'numerical' in stype_full.lower():
                            stype_str = 'numerical'
                        elif 'categorical' in stype_full.lower():
                            stype_str = 'categorical'
                        elif 'text' in stype_full.lower() or 'embedding' in stype_full.lower():
                            stype_str = 'text'
                        elif 'timestamp' in stype_full.lower():
                            stype_str = 'timestamp'
                        else:
                            if '.' in stype_full:
                                stype_str = stype_full.split('.')[-1].split(':')[0].replace("'", "").replace(">", "").strip().lower()
                                # embedding -> text
                                if stype_str == 'embedding':
                                    stype_str = 'text'
                        
                        found = True
                        break
                
                if not found and hasattr(tf, 'feat_dict'):
                    for stype_key in tf.feat_dict.keys():
                        stype_full = str(stype_key)
                        if 'numerical' in stype_full.lower():
                            potential_cols = tf.col_names_dict.get(stype_key, [])
                            if col_name in potential_cols:
                                stype_str = 'numerical'
                                break
                        elif 'categorical' in stype_full.lower():
                            potential_cols = tf.col_names_dict.get(stype_key, [])
                            if col_name in potential_cols:
                                stype_str = 'categorical'
                                break
                        elif 'text' in stype_full.lower() or 'embedding' in stype_full.lower():
                            potential_cols = tf.col_names_dict.get(stype_key, [])
                            if col_name in potential_cols:
                                stype_str = 'text'
                                break
        
        formatted_name = f"{table_name}.{col_name}.{stype_str}"
        
        explanation_elements.append(str(explanation_element))
        feature_formatted.append(formatted_name)
        importance_scores.append(apply_sigmoid(mask_value).item())
        table_names.append(table_name)
        column_names.append(col_name)
        feature_types.append(stype_str)
    
    importance_df = pd.DataFrame({
        'Explanation Element': explanation_elements,
        'feature': feature_formatted,
        'Importance Score': importance_scores,
        'table_name': table_names,
        'column_name': column_names,
        'feature_type': feature_types
    })

    importance_df = importance_df.sort_values(by='Importance Score', ascending=False)

    unknown_count = sum(importance_df['feature_type'] == 'unknown')
    if unknown_count > 0:
        print(f"\n⚠️  Warning: {unknown_count} features with 'unknown' type")
        print("Unknown features:")
        print(importance_df[importance_df['feature_type'] == 'unknown'][['feature', 'table_name', 'column_name']].head())
    
    print(f"\n✓ Importance dataframe prepared with {len(importance_df)} features")
    print(f"\nTop 10 features:")
    print(importance_df[['feature', 'Importance Score']].head(10))
    
    _prep_elapsed = time.time() - _prep_start
    print(f"⏱  [prepare_importance_dataframe] completed in {_prep_elapsed:.2f}s "
          f"({_prep_elapsed/60:.2f} min)")
    
    return importance_df


def save_mask_learning_outputs(mask, mask_vals, metrics, mask_learning_config,
                                output_dir, run_index):
    """
    Persist everything learn_final_mask() returns, permanently, to output_dir.

    Previously only `mask` survived (indirectly, via importance_df/rdl_results
    CSV). `mask_vals` and `metrics` were never saved anywhere at all, and
    `mask_learning_config` only lived inside resume_state.json — which gets
    deleted (RESUME_FILE.unlink()) once the run completes successfully, so
    after a clean run there was no permanent record of it either.

    mask/mask_vals are pickled since their internal structure (dict keyed by
    (table, column) tuples, values likely tensors/arrays) isn't reliably
    JSON-safe. metrics/mask_learning_config are plain dicts of
    floats/strings, so saved as human-readable JSON (with a fallback
    encoder for any stray numpy/torch values).
    """
    def _json_default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        return str(o)

    mask_path = get_indexed_filepath(output_dir, 'mask', '.pkl', run_index)
    with open(mask_path, 'wb') as f:
        pickle.dump(mask, f)

    mask_vals_path = get_indexed_filepath(output_dir, 'mask_vals', '.pkl', run_index)
    with open(mask_vals_path, 'wb') as f:
        pickle.dump(mask_vals, f)

    metrics_path = get_indexed_filepath(output_dir, 'mask_metrics', '.json', run_index)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    config_path = get_indexed_filepath(output_dir, 'mask_learning_config', '.json', run_index)
    with open(config_path, 'w') as f:
        json.dump(mask_learning_config, f, indent=2, default=_json_default)

    print(f"✓ Saved: {mask_path.name}")
    print(f"✓ Saved: {mask_vals_path.name}")
    print(f"✓ Saved: {metrics_path.name}")
    print(f"✓ Saved: {config_path.name}")

    return {
        'mask_path': mask_path,
        'mask_vals_path': mask_vals_path,
        'metrics_path': metrics_path,
        'config_path': config_path,
    }


def save_rdl_results(importance_df, output_dir, run_index, dataset_name, task_name):
    """
    Save RDL explanation results to CSV with automatic indexing.
    """
    results_path = get_indexed_filepath(output_dir, 'rdl_results', '.csv', run_index)
    importance_df.to_csv(results_path, index=False)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    return results_path


def create_visualization(importance_df, explainer, mask_learning_config, output_dir, run_index,
                         dataset_name, task_name):
    """
    Create visualizations of RDL results (Absolute Values).
    Generates both a full feature plot and a Top 20 features plot.
    """
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    _viz_start = time.time()

    df_plot = importance_df.copy()
    df_plot['Abs Importance Score'] = df_plot['Importance Score'].abs()

    # Sort descending: highest importance at the top of the horizontal bar chart
    df_plot = df_plot.sort_values(by='Abs Importance Score', ascending=False)

    reg_eps = mask_learning_config.get("reg_eps", "auto")
    if reg_eps == 'auto':
        if hasattr(explainer, 'reg_eps'):
            reg_eps = explainer.reg_eps
        elif hasattr(explainer, 'config') and isinstance(explainer.config, dict):
            reg_eps = explainer.config.get('reg_eps', 'auto')

    if isinstance(reg_eps, (int, float)):
        eps_str = f'{reg_eps:.2e}' if reg_eps < 0.001 else f'{reg_eps:.4f}'
    else:
        eps_str = str(reg_eps)

    def draw_and_save_barh(df_subset, suffix_name, title_prefix):
        if df_subset.empty:
            return None

        num_features = len(df_subset)
        fig_height = max(4, 0.35 * num_features)
        fig, ax = plt.subplots(figsize=(12, fig_height))

        bars = ax.barh(df_subset['feature'], df_subset['Abs Importance Score'], color='skyblue')

        ax.set_xlabel('Absolute Importance Score')
        ax.set_title(f'{title_prefix} - {dataset_name}/{task_name} (reg_eps={eps_str})', pad=15)

        ax.invert_yaxis()

        max_val = df_subset['Abs Importance Score'].max()
        if max_val >= 0.1:
            ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold (0.1)')
            ax.legend(loc='lower right')

        x_padding = max_val * 0.015 if max_val > 0 else 0.01
        for bar in bars:
            width = bar.get_width()
            label_fmt = f'{width:.2e}' if 0 < width < 0.0001 else f'{width:.4f}'
            ax.text(
                width + x_padding,
                bar.get_y() + bar.get_height() / 2,
                label_fmt,
                va='center', ha='left', fontsize=9, fontweight='semibold'
            )

        ax.set_xlim(0, max_val * 1.15 if max_val > 0 else 1.0)
        plt.tight_layout()

        plot_file_name = f'rdl_importance_plot_{suffix_name}'
        plot_path = get_indexed_filepath(output_dir, plot_file_name, '.png', run_index)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ {title_prefix} plot saved to: {plot_path.name}")
        return plot_path

    # Plot 1: Full features
    full_plot_path = draw_and_save_barh(df_plot, 'full', 'RDL Explanation (Full)')

    # Plot 2: Top 20 only — head(20) on a descending-sorted df gives the best 20
    if len(df_plot) > 20:
        df_top20 = df_plot.head(20)
        draw_and_save_barh(df_top20, 'top20', 'RDL Explanation (Top 20)')

    _viz_elapsed = time.time() - _viz_start
    print(f"⏱  [create_visualization] completed in {_viz_elapsed:.2f}s "
          f"({_viz_elapsed/60:.2f} min)")

    return full_plot_path



def main():
    """Main RDL explainer pipeline."""
    args = parse_args()
    script_start_time = time.time()
    DATASET_NAME = args.dataset
    TASK_NAME    = args.task
    SEED         = args.seed
    MODEL_TYPE   = args.model_type   # 'hgs' or 'transformer'
    RESUME_INDEX = args.resume

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:     {device}")
    print(f"Model type: {MODEL_TYPE.upper()}")

    fix_randomness(SEED)

    # Get configuration — routes to HGS or transformer config block
    config = get_rdl_config(DATASET_NAME, TASK_NAME, MODEL_TYPE)

    # Get directories — each model type has its own saved_model and output dirs
    saved_model_dir = get_rdl_saved_model_dir(DATASET_NAME, TASK_NAME, MODEL_TYPE)
    output_dir      = get_rdl_output_dir(DATASET_NAME, TASK_NAME, MODEL_TYPE)
    inference_dir   = get_rdl_inference_dir(DATASET_NAME, TASK_NAME, MODEL_TYPE)
    
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

    with timed_section("Creating data loaders"):
        # Explanation methods compare predictions across many masking
        # variants, so they need bit-exact reproducibility. NeighborLoader's
        # default temporal_strategy ("uniform") picks a RANDOM subset of
        # eligible neighbors whenever a node exceeds the num_neighbors
        # budget, using its own internal RNG that fix_randomness/use_
        # deterministic_algorithms can't reach. Removing the budget entirely
        # (num_neighbors=-1) is unsafe — some datasets have nodes with huge
        # true degree (e.g. a placeholder join-table row referenced 200k+
        # times), and that caused real CUDA OOM crashes. temporal_strategy=
        # "last" fixes this properly instead: keeps the original,
        # memory-safe budget, but selects the K most-recent eligible
        # neighbors by a deterministic rule (recency) rather than a random
        # draw — same bound as training, no RNG involved. This applies for
        # the HGS path; RelGT uses its own separate relgt_loaders, not
        # loader_dict, so this is a no-op for it.
        deterministic_loader_config = dict(loader_config)
        deterministic_loader_config["temporal_strategy"] = "last"
        loader_dict = {}
        for split, table in [
            ("train", train_table),
            ("val", val_table),
            ("test", test_table),
        ]:
            loader_dict[split] = create_consistent_loader(
                data, deterministic_loader_config, task, table, split
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
            model = get_trained_model(
                model_class=HGSModel,
                data=data,
                col_stats_dict=col_stats_dict,
                metadata=metadata,
                weights_path=weights_file,
                device=device,
            )
        else:
            # RelGT — load from checkpoint using saved architecture config
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

        entity_table = task.entity_table

        # NOTE: test_predictions (loaded from test_predictions_N.npy) was
        # computed back in train.py using loader_dict["test"] built via the
        # plain create_consistent_loader(), whose temporal_strategy defaults
        # to "uniform" (random neighbor sampling) unless loader_config.json
        # explicitly overrides it — train.py never does. That's INCONSISTENT
        # with everything else in this script: loader_dict here (line ~980)
        # is deliberately rebuilt with temporal_strategy="last" for
        # deterministic, reproducible neighbor selection, and Phase 1's
        # fresh inference dataframes (predictions/processed_output) are
        # computed through that same deterministic loader_dict. So reusing
        # the old uniform-sampled test_predictions as "baseline_predictions"
        # compares against a different neighbor-sampling run than the rest
        # of the explanation pipeline. Recompute it fresh here, through the
        # same deterministic loader_dict["test"], for HGS only (RelGT's
        # baseline comes from split_predictions, freshly computed in Phase 1
        # through its own deterministic RelGTTokens precompute path already).
        if MODEL_TYPE == "hgs":
            task_type_for_baseline = config.get('task_type', 'classification')
            baseline_preds_list = []
            with torch.no_grad():
                for batch in loader_dict['test']:
                    batch = batch.to(device)
                    out = model(batch, entity_table)
                    if task_type_for_baseline == "classification":
                        out = torch.sigmoid(out)
                    baseline_preds_list.append(out.detach().cpu().numpy().reshape(-1))
            baseline_predictions = np.concatenate(baseline_preds_list)
            print(f"✓ Recomputed baseline predictions via deterministic "
                  f"(temporal_strategy='last') loader — {len(baseline_predictions)} samples "
                  f"(previously reused train.py's uniform-sampled test_predictions.npy).")
        else:
            baseline_predictions = test_predictions

        print(f"✓ Model loaded and set to EVAL mode")
        print(f"✓ Baseline predictions ready: {len(baseline_predictions)} samples")

        # Verify model works (HGS only — RelGT uses a different batch interface)
        if MODEL_TYPE == "hgs":
            print("\nVerifying model...")
            with torch.no_grad():
                sample_batch = next(iter(loader_dict['test'])).to(device)
                sample_pred = model(sample_batch, entity_table)
                print(f"✓ Forward pass successful: {sample_pred.shape}")
    
    print(f"\nSETUP COMPLETE - READY FOR RDL EXPLAINER!")
    
    # ── Resolve run index ─────────────────────────────────────────────────────
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
        print(f"   Loaded resume state: completed phases = {list(resume_state.get('completed', []))}")

    def _save_resume(state: dict):
        with open(RESUME_FILE, 'w') as f:
            _json.dump(state, f, indent=2)

    # ── PHASE 1: Create inference dataframes ──────────────────────────────────
    if "inference" not in resume_state.get("completed", []):
        print("\n" + "=" * 80)
        print("CREATING INFERENCE DATAFRAMES")
        print("=" * 80)

        with timed_section("PHASE 1: Creating inference dataframes"):
            if MODEL_TYPE == "hgs":
                for split_name in ['train', 'val', 'test']:
                    print(f"\nProcessing {split_name} split...")
                    create_inference_df(
                        model, task, loader_dict, split_name, entity_table,
                        device, inference_dir, dataset,
                        task_type=config.get('task_type', 'classification'),
                    )
            else:
                # RelGT: build RelGTTokens loaders then use RelGT-specific inference
                from rdl_explain_files.model.relgt import RelGTTokens
                from torch.utils.data import DataLoader as _DL
                relgt_loaders = {}
                for split_name in ['train', 'val', 'test']:
                    ds = RelGTTokens(
                        data=data, task=task,
                        K=loader_config["K"],
                        split=split_name,
                        undirected=True,
                        precompute=True,
                        precomputed_dir=loader_config["precomputed_dir"],
                        num_workers=loader_config.get("num_workers_precompute", 4),
                    )
                    relgt_loaders[split_name] = _DL(
                        ds,
                        batch_size=loader_config.get("batch_size", 64),
                        shuffle=False,
                        collate_fn=ds.collate,
                        num_workers=loader_config.get("num_workers_loader", 2),
                        persistent_workers=True,
                        pin_memory=(device.type == "cuda"),
                    )
                split_predictions = {}
                for split_name in ['train', 'val', 'test']:
                    print(f"\nProcessing {split_name} split...")
                    df = create_inference_df_relgt(
                        model, task, relgt_loaders, split_name, device,
                        inference_dir,
                        task_type=config.get('task_type', 'classification'),
                    )
                    split_predictions[split_name] = {
                        "output":           df["output"].values,
                        "processed_output": df["processed_output"].values,
                        "predictions":      df["predictions"].values,
                    }

        resume_state.setdefault("completed", []).append("inference")
        _save_resume(resume_state)
        print("\n✓ Phase 1 (inference) complete — state saved")
    else:
        # Resuming — read predictions from the parquet files saved in Phase 1
        if MODEL_TYPE == "transformer":
            split_predictions = {}
            for split_name in ['train', 'val', 'test']:
                df = pd.read_parquet(
                    os.path.join(inference_dir, f"predictions_{split_name}.parquet")
                )
                split_predictions[split_name] = {
                    "output":           df["output"].values,
                    "processed_output": df["processed_output"].values,
                    "predictions":      df["predictions"].values,
                }
        print("\n⟳  Skipping inference (already completed)")

    # Verify baseline metrics
    try:
        baseline_metrics = task.evaluate(baseline_predictions)
        tune_metric = config['tune_metric']
        print(f"✓ Baseline {tune_metric.upper()}: {baseline_metrics[tune_metric]:.4f}")
    except Exception as e:
        print(f"Could not evaluate baseline: {e}")

    # ── PHASE 2: Build explanation task + explainer (fast, always rebuilt) ────
    with timed_section("PHASE 2: Build explanation task + explainer"):
        if MODEL_TYPE == "hgs":
            explanation_task = make_explanation_task(
                prediction_task=task,
                path_to_inference_results=inference_dir,
                explanation_target_type='soft',
            )
            gnn_layers = metadata['model_layers']
            # temporal_strategy="last" for consistency with the deterministic
            # (temporal_strategy="last") loader_dict already built above for
            # inference — otherwise the explainer's own internal neighbor
            # sampling (e.g. for any loaders it builds itself) would default
            # to ExplainerConfig's "uniform", reintroducing the same kind of
            # random/deterministic mismatch fixed for baseline_predictions.
            explainer_config = ExplainerConfig(device=device, gnn_layers=gnn_layers,
                                                temporal_strategy="last")
            # Subsetting the explanation task's test_table is enough for HGS:
            # RDLExplainer reads the *_table attributes directly (never through
            # task.get_table()), so mask learning and fidelity both see only
            # these rows. The graph in `data` stays complete, as it must.
            if subset_idx is not None:
                explanation_task.test_table = subset_table(
                    explanation_task.test_table, subset_idx)
                print(f"✓ Explaining a subset: "
                      f"Test({len(explanation_task.test_table.df)}) rows")
            explainer = RDLExplainer(explainer_config, model, data, explanation_task)
        else:
            # RelGT: split_predictions already populated in Phase 1 (or resume block above)
            from rdl_explain_files.explain.relgt_explainer import RelGTExplainer
            explanation_task = make_explanation_task_relgt(
                prediction_task=task,
                split_predictions=split_predictions,
                explanation_target_type='soft',
            )
            _fix_entity_table_case(explanation_task, data)

            gnn_layers = metadata.get('local_num_layers', 1)
            explainer_config = ExplainerConfig(
                device=device,
                gnn_layers=gnn_layers,
                inference_batch_size=loader_config.get("batch_size", 64),
                temporal_strategy="last",
            )
            explainer = RelGTExplainer(
                config=explainer_config,
                model=model,
                data=data,
                task=explanation_task,
                K=loader_config["K"],
                precomputed_dir=loader_config["precomputed_dir"],
                num_dataset_workers=loader_config.get("num_workers_precompute", 4),
                num_loader_workers=loader_config.get("num_workers_loader", 2),
                test_subset_idx=subset_idx,
            )

        print(f"✓ Explanation task built. Target column: {explanation_task.target_col}")
        print(f"✓ Explainer initialized  [{MODEL_TYPE.upper()}]  (GNN layers: {gnn_layers})")

    # ── PHASE 3: Epsilon tuning ────────────────────────────────────────────────
    if "epsilon_tuning" not in resume_state.get("completed", []):
        with timed_section("PHASE 3: Epsilon tuning"):
            selected_reg_eps = 'auto'
            best_eps_middle  = 'auto'

        resume_state["selected_reg_eps"] = selected_reg_eps
        resume_state["best_eps_middle"]  = best_eps_middle
        resume_state.setdefault("completed", []).append("epsilon_tuning")
        _save_resume(resume_state)
        print("\n✓ Phase 3 (epsilon tuning set to 'auto') complete — state saved")
    else:
        selected_reg_eps = resume_state["selected_reg_eps"]
        best_eps_middle  = resume_state["best_eps_middle"]
        print(f"\n⟳  Skipping epsilon tuning (loaded configuration: {best_eps_middle})")

    # ── PHASE 4: Final mask learning ──────────────────────────────────────────
    if "mask_learning" not in resume_state.get("completed", []):
        with timed_section("PHASE 4: Final mask learning"):
            # Run mask learning (receives 'auto' on first run, or a float during resume)
            mask, mask_vals, metrics, mask_learning_config = learn_final_mask(
                explainer, best_eps_middle, DATASET_NAME, TASK_NAME
            )
            
            # Extract calculated epsilon if available in config, otherwise fallback to the current best_eps_middle
            real_eps_used = mask_learning_config.get('reg_eps', best_eps_middle)

            # Permanently persist everything learn_final_mask() returned.
            # Previously mask_vals/metrics were never saved anywhere, and
            # mask_learning_config only lived in resume_state.json — which
            # gets deleted once the run finishes successfully, so a clean
            # run left no permanent record of any of them.
            mask_output_paths = save_mask_learning_outputs(
                mask, mask_vals, metrics, mask_learning_config,
                output_dir, run_index,
            )

            importance_df = prepare_importance_dataframe(mask, data, DATASET_NAME, TASK_NAME)
            imp_path = RESUME_DIR / f"importance_df_{run_index}.csv"
            importance_df.to_csv(imp_path, index=False)
        
        # Overwrite 'auto' in resume state with the actual float value used, ensuring consistency for subsequent resumes
        resume_state["selected_reg_eps"] = real_eps_used
        resume_state["best_eps_middle"]  = real_eps_used
        
        resume_state["mask_learning_config"] = mask_learning_config
        resume_state["importance_df_path"]   = str(imp_path)
        resume_state["mask_output_paths"]    = {k: str(v) for k, v in mask_output_paths.items()}
        resume_state.setdefault("completed", []).append("mask_learning")
        _save_resume(resume_state)
        print(f"\n✓ Phase 4 complete — Mask trained and saved with real eps: {real_eps_used}")
    else:
        import pandas as _pd
        importance_df        = _pd.read_csv(resume_state["importance_df_path"])
        mask_learning_config = resume_state["mask_learning_config"]
        mask_output_paths    = resume_state.get("mask_output_paths", {})
        print(f"\n⟳  Skipping mask learning (loaded from resume)")

    # Save results + visualisation (idempotent, fast — always run)
    with timed_section("Saving results + visualization"):
        results_path = save_rdl_results(
            importance_df, output_dir, run_index, DATASET_NAME, TASK_NAME
        )
        plot_path = create_visualization(
            importance_df, explainer, mask_learning_config, output_dir, run_index,
            DATASET_NAME, TASK_NAME
        )


    if RESUME_FILE.exists():
        RESUME_FILE.unlink()
        print("\n✓ Resume file removed (all phases completed successfully)")

    print("\n" + "=" * 80)
    print("RDL EXPLAINER COMPLETE!")
    print("=" * 80)
    print(f"\nDataset:    {DATASET_NAME}")
    print(f"Task:       {TASK_NAME}")
    print(f"Model type: {MODEL_TYPE.upper()}")
    print(f"Run Index:  {run_index}")
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"\nSaved files:")
    print(f"  - {results_path.name}")
    print(f"  - {plot_path.name}")
    for _key in ('mask_path', 'mask_vals_path', 'metrics_path', 'config_path'):
        if _key in mask_output_paths:
            print(f"  - {Path(mask_output_paths[_key]).name}")
    print(f"\nInference files saved in: {inference_dir}")

    script_total_elapsed = time.time() - script_start_time
    print(f"\n⏱  TOTAL RUNTIME (start to finish): {script_total_elapsed:.2f}s "
          f"({script_total_elapsed/60:.2f} min)")

    print("\n✓ All RDL explainer outputs ready for analysis!")


if __name__ == "__main__":
    main()