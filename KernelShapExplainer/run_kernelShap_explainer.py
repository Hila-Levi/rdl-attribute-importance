"""
Main Kernel SHAP explainer script for RelBench datasets.

Usage:
    python run_kernelShap_explainer.py --dataset <dataset> --task <task> [--model_type hgs|transformer] [--seed <seed>]

Examples:
    python run_kernelShap_explainer.py --dataset rel-trial --task study-outcome --model_type hgs --seed 42
    python run_kernelShap_explainer.py --dataset rel-trial --task study-outcome --model_type transformer --seed 42
    python run_kernelShap_explainer.py --dataset rel-f1 --task driver-position --model_type transformer
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


# Resolve all paths relative to this file's own location so imports work
# regardless of which directory you run from.
_THIS_DIR     = Path(os.path.dirname(os.path.abspath(__file__)))
_RESEARCH_DIR = _THIS_DIR.parent   # research-clean/

if str(_RESEARCH_DIR) not in sys.path:
    sys.path.append(str(_RESEARCH_DIR))

# Models source
model_src_path = str(_RESEARCH_DIR / "MaskLearningExplainer" / "src")
if model_src_path not in sys.path:
    sys.path.insert(0, model_src_path)

# Local imports
from kernelShap_explainer_config import (
    get_kernelshap_config,
    get_kernelshap_saved_model_dir,
    get_kernelshap_output_dir,
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
from KernelShapHetero import GlobalKernelSHAP, RelGTGlobalKernelSHAP, timed_section


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Kernel SHAP explainer on trained RelBench model')
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


def run_kernel_shap_explanation(shap_explainer, loader_dict, config, dataset_name, task_name,
                                chosen_threshold: float = None, output_dir=None, run_index=None,
                                checkpoint_every: int = 30):
    """
    Run Kernel SHAP explanation using importance-score aggregation
    (mean absolute prediction shift — same formula for classification and regression).

    Args:
        chosen_threshold: No longer used by the importance-score calculation
                          (compute_importance_score() ignores it — classification
                          and regression are now computed identically). Retained
                          only so it's still recorded in the output CSV for
                          reference. Pass None for regression.
        output_dir:       Base output directory for this dataset/task/model.
                          Checkpoints are written to a "checkpoints" subfolder
                          of this directory. Required if checkpoint_every is set.
        run_index:        Run index used to namespace checkpoint filenames so
                          they don't collide across runs (e.g. resumed runs).
        checkpoint_every: Save an intermediate SHAP-values CSV every this many
                          coalitions (default 30). Pass None to disable.
    Returns:
        shap_values: Dictionary mapping feature names to SHAP values
    """
    checkpoint_dir = None
    checkpoint_prefix = "kernelShap_checkpoint"
    if checkpoint_every is not None:
        checkpoint_dir = Path(output_dir) / "checkpoints"
        checkpoint_prefix = f"kernelShap_checkpoint_run{run_index}"
    # For regression, chosen_threshold is None — fall back to the config value.
    effective_threshold = chosen_threshold if chosen_threshold is not None         else config.get('disagreement_threshold', 0.5)

    print("\n" + "=" * 70)
    print("RUNNING GLOBAL KERNEL SHAP EXPLANATION")
    print("=" * 70)
    if chosen_threshold is not None:
        print(f"  Decision threshold: {effective_threshold:.4f} (tuned on val set)")
    else:
        print(f"  Decision threshold: {effective_threshold:.4f} (config default, regression)")

    # Run the explanation — always importance-score aggregation (mean |Δpred|,
    # same formula for classification and regression). `disagreement_threshold`
    # is kept as the keyword name for backward compatibility but is no longer
    # used in the calculation itself.
    shap_values_importance_score = shap_explainer.explain(
        loader=loader_dict['test'],
        num_coalitions=config['num_coalitions'],
        max_batches=config['max_batches'],
        disagreement_threshold=effective_threshold,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=checkpoint_prefix,
    )

    return shap_values_importance_score


def save_shap_results(shap_explainer, shap_values, output_dir, run_index, dataset_name,
                      task_name, chosen_threshold: float = None):
    """
    Save Kernel SHAP results to CSV with automatic indexing.

    Args:
        chosen_threshold: Decision threshold forwarded to save_results so it is
                          recorded as a column in the output CSV. Pass None for
                          regression tasks (column is omitted).

    Returns:
        results_path: Path to saved results file
    """
    results_path = get_indexed_filepath(
        output_dir,
        'kernelShap_disagreement_count_results',
        '.csv',
        run_index
    )

    shap_explainer.save_results(shap_values, results_path,
                                disagreement_threshold=chosen_threshold)

    print(f"\n✓ Results saved to: {results_path}")

    return results_path


def print_top_features(shap_values, dataset_name, task_name, top_k=10):
    """
    Print top K most important features by absolute SHAP value.
    """
    print("\n" + "=" * 70)
    print(f"IMPORTANCE SCORE - TOP {top_k} MOST UNSTABLE FEATURES")
    print("=" * 70)
    print("(High positive SHAP = removing this feature increases avg |Δpred|)")

    sorted_features_count = sorted(
        shap_values.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    
    for i, (feat, val) in enumerate(sorted_features_count[:top_k], 1):
        print(f"{i:2d}. {feat:60s} | SHAP = {val:+.6f} (±{abs(val):.6f} avg |Δpred|)")


def create_visualization(shap_values, output_dir, run_index, dataset_name, task_name,
                         top_n_features=20):
    """
    Create visualization of Kernel SHAP results.

    Returns:
        plot_path: Path to saved plot
    """
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION")
    print("=" * 80)

    # Build DataFrame, sort by absolute SHAP value descending, take top N
    df_plot = (
        pd.DataFrame(
            list(shap_values.items()),
            columns=['feature', 'SHAP Value']
        )
        .assign(**{'Abs SHAP Value': lambda d: d['SHAP Value'].abs()})
        .sort_values(by='Abs SHAP Value', ascending=False)
        .head(top_n_features)
    )

    if df_plot.empty:
        print("⚠️  No SHAP values to plot. Skipping visualization.")
        return None

    num_features = len(df_plot)
    fig_height = max(4, 0.35 * num_features)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    bars = ax.barh(df_plot['feature'], df_plot['Abs SHAP Value'], color='skyblue')

    ax.set_xlabel('Absolute SHAP Value (Importance Score, avg |Δpred|)')
    ax.set_title(
        f'Kernel SHAP — {dataset_name}/{task_name}\nImportance Score Aggregation (Top {num_features})',
        pad=15
    )
    ax.invert_yaxis()  # highest importance at top

    max_val = df_plot['Abs SHAP Value'].max()

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

    plot_path = get_indexed_filepath(
        output_dir, 'kernelShap_disagreement_plot', '.png', run_index
    )
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved to: {plot_path.name}")
    return plot_path


def main():
    """Main Kernel SHAP explainer pipeline."""
    script_start_time = time.time()
    
    args = parse_args()
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
    config = get_kernelshap_config(DATASET_NAME, TASK_NAME, MODEL_TYPE)

    # Get directories — each model type has its own saved_model and output dirs
    saved_model_dir = get_kernelshap_saved_model_dir(DATASET_NAME, TASK_NAME, MODEL_TYPE)
    output_dir      = get_kernelshap_output_dir(DATASET_NAME, TASK_NAME, MODEL_TYPE)
    
    print("\n" + "=" * 80)
    print("LOADING RESEARCH COMPONENTS")
    print("=" * 80)
    
    with timed_section("Loading research components (data, task, metadata)"):
        print(f"Looking for latest saved model in: {saved_model_dir}")

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
        # Create data loaders with temporal_strategy="last" (deterministic
        # neighbor selection), same as MonteCarlo, RDL and SingleColumnAblation explainers,
        # so every coalition and every explanation method sees the same
        # neighbors. The default "uniform" draws a random neighbor subset with
        # an RNG that fix_randomness can't reach. No-op for RelGT, which
        # builds its own RelGTTokens loader.
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

        baseline_predictions = test_predictions
        # test_predictions_N.npy covers the FULL test set — keep only the subset
        # rows so it lines up with the (subset) test table. For HGS the baseline is
        # recomputed via f_model_all_predictions() further below anyway.
        if subset_idx is not None:
            baseline_predictions = test_predictions[subset_idx]
        entity_table = task.entity_table

        print(f"✓ Model loaded and set to EVAL mode")
        print(f"✓ Baseline predictions ready: {len(baseline_predictions)} samples")

        # Verify model works (HGS only — RelGT uses a different batch interface)
        if MODEL_TYPE == "hgs":
            print("\nVerifying model...")
            with torch.no_grad():
                sample_batch = next(iter(loader_dict['test'])).to(device)
                sample_pred = model(sample_batch, entity_table)
                print(f"✓ Forward pass successful: {sample_pred.shape}")

    print(f"\nSETUP COMPLETE - READY FOR KERNEL SHAP EXPLAINER!")
    
    print("\n" + "=" * 80)
    print("INITIALIZING KERNEL SHAP EXPLAINER")
    print("=" * 80)

    with timed_section("Initializing Kernel SHAP explainer"):
        print(f"✓ Using loaded model (already in eval mode)")

        # Get test AUC from metadata (try both possible keys)
        test_auc = metadata.get('test_auc', metadata.get('test_roc_auc', 0.0))
        if test_auc > 0:
            print(f"  Test AUC (from metadata): {test_auc:.4f}")

        # Initialize the explainer — model-type-specific subclass
        if MODEL_TYPE == "hgs":
            shap_explainer = GlobalKernelSHAP(
                model=model,
                data=data,
                col_stats_dict=col_stats_dict,
                entity_table=entity_table,
                device=device,
                loader_config=loader_config,
                task_type=config.get("task_type", "classification"),
                task=task,
                test_subset_idx=subset_idx,
            )
        else:
            # RelGTGlobalKernelSHAP overrides f_model_all_predictions() to use
            # RelGTTokens + DataLoader and the RelGT forward pass interface.
            from rdl_explain_files.model.relgt import RelGTTokens
            relgt_tokens_kwargs = {
                "RelGTTokens":     RelGTTokens,   # passed here where sys.path is set up
                "data":            data,
                "task":            task,
                "K":               loader_config["K"],
                "precomputed_dir": loader_config["precomputed_dir"],
                "undirected":      True,
                "num_workers":     loader_config.get("num_workers_precompute", 4),
            }
            shap_explainer = RelGTGlobalKernelSHAP(
                model=model,
                data=data,
                col_stats_dict=col_stats_dict,
                entity_table=entity_table,
                device=device,
                loader_config=loader_config,
                task_type=config.get("task_type", "classification"),
                task=task,
                relgt_tokens_kwargs=relgt_tokens_kwargs,
                test_subset_idx=subset_idx,
            )

        print(f"✓ Kernel SHAP explainer initialized  [{MODEL_TYPE.upper()}]")
        print(f"  Dataset: {DATASET_NAME}")
        print(f"  Task: {TASK_NAME}")
        print(f"  Num coalitions: {config['num_coalitions']}")
        print(f"  Max batches: {config['max_batches']}")

        # baseline_predictions above came from the on-disk test_predictions_N.npy
        # (computed by train.py's test() with the old, randomly-sampled neighbor
        # budget), while every coalition prediction from here on uses
        # f_model_all_predictions()'s deterministic temporal_strategy="last"
        # sampling. Diffing predictions from two different sampling policies
        # would inject spurious noise into every Shapley value, so recompute
        # baseline the same way coalition predictions will be computed (the
        # all-present coalition = no features masked). HGS only for now.
        if MODEL_TYPE == "hgs":
            all_present_coalition = shap_explainer.create_coalition_data(np.ones(shap_explainer.M))
            baseline_loader = shap_explainer.create_loader(test_table)
            baseline_predictions = shap_explainer.f_model_all_predictions(
                baseline_loader, all_present_coalition, max_batches=len(baseline_loader)
            )
            print(f"✓ Baseline predictions recomputed via f_model_all_predictions() for consistency: "
                  f"{len(baseline_predictions)} samples")

    # ── Decision threshold (classification only) — read from metadata, NOT
    # tuned here. The threshold must be computed exactly once, via
    # Utils/compute_and_save_threshold.py, and stored in metadata_<N>.json.
    # Every explanation method (SingleColumnAblation, Kernel SHAP, Monte Carlo) reads the
    # same saved value so they never disagree on which threshold to use.
    task_type   = config.get("task_type", "classification")

    chosen_threshold = None if task_type == 'regression' else metadata.get('threshold')
    if task_type == "classification":
        if chosen_threshold is None:
            print(
                "[Warning] No 'threshold' found in metadata.json for this run. "
                "Run Utils/compute_and_save_threshold.py first so all explanation "
                "methods use the same value. Falling back to 0.5 for now.\n"
            )
            chosen_threshold = 0.5
        else:
            print(f"[Threshold] Using saved decision threshold from metadata: {chosen_threshold:.4f}\n")


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
 
    # ── PHASE 1: SHAP explanation ──────────────────────────────────────────────
    if "shap" not in resume_state.get("completed", []):
        with timed_section("PHASE 1: Kernel SHAP explanation"):
            shap_values = run_kernel_shap_explanation(
                shap_explainer, loader_dict, config, DATASET_NAME, TASK_NAME,
                chosen_threshold=chosen_threshold,
                output_dir=output_dir, run_index=run_index,
                checkpoint_every=30,
            )
            results_path = save_shap_results(
                shap_explainer, shap_values, output_dir, run_index, DATASET_NAME, TASK_NAME,
                chosen_threshold=chosen_threshold,
            )
            resume_state["results_path"] = str(results_path)
            resume_state.setdefault("completed", []).append("shap")
            _save_resume(resume_state)
            print("✓ Phase 1 (SHAP explanation) complete — state saved")
    else:
        results_path = Path(resume_state["results_path"])
        import pandas as _pd
        _df = _pd.read_csv(results_path)
        shap_values = dict(zip(_df.iloc[:, 0], _df.iloc[:, 1]))
        print(f"\n⟳  Skipping SHAP explanation (loaded from {results_path.name})")
 
    # ── PHASE 2: Visualise + top features (fast, always run) ──────────────────
    with timed_section("PHASE 2: Top features + visualization"):
        print_top_features(shap_values, DATASET_NAME, TASK_NAME, top_k=10)
        plot_path = create_visualization(
            shap_values, output_dir, run_index, DATASET_NAME, TASK_NAME
        )
 
    if RESUME_FILE.exists():
        RESUME_FILE.unlink()
        print("\n✓ Resume file removed (all phases completed successfully)")
 
    total_elapsed = time.time() - script_start_time
    print("\n" + "=" * 80)
    print("KERNEL SHAP EXPLAINER COMPLETE!")
    print("=" * 80)
    print(f"\nDataset:    {DATASET_NAME}")
    print(f"Task:       {TASK_NAME}")
    print(f"Model type: {MODEL_TYPE.upper()}")
    print(f"Run Index:  {run_index}")
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"\nSaved files:")
    print(f"  - {results_path.name}")
    print(f"  - {plot_path.name}")
    print(f"\n⏱  TOTAL RUNTIME (start to finish): {total_elapsed:.2f}s "
          f"({total_elapsed/60:.2f} min)")
    print("\n✓ All Kernel SHAP explainer outputs ready for analysis!")
    
 
if __name__ == "__main__":
    main()