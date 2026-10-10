#!/usr/bin/env python
"""
Monte Carlo Shapley Explainer - Long-running Computation

This script runs Monte Carlo Shapley value calculation which can take hours or days.
It automatically saves checkpoints and can be interrupted at any time.

Usage:
    python run_monteCarlo_explainer.py --dataset <dataset_name> --task <task_name> [--seed <seed>]
    python run_monteCarlo_explainer.py --dataset rel-arxiv --task paper-citation --model_type hgs --chernoff
    python run_monteCarlo_explainer.py --dataset rel-arxiv --task paper-citation --model_type hgs --chernoff --chernoff-epsilon 0.005 --chernoff-delta 0.01
Run in background:
    nohup python run_monteCarlo_explainer.py --dataset rel-arxiv --task paper-citation > monte_carlo.log 2>&1 &
    
Monitor progress:
    tail -f monte_carlo.log
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sys
import time
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

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
from monteCarlo_explainer_config import (
    get_montecarlo_config,
    get_montecarlo_saved_model_dir,
    get_montecarlo_output_dir
)

from Utils.Utils import (
    load_research_components,
    get_trained_model,
    fix_randomness,
    create_consistent_loader,
    get_latest_indexed_file,
    load_test_subset,
    subset_table
)

from rdl_explain_files.model.gnn.model import HGSModel
from MonteCarloShapley import MonteCarloShapley, MonteCarloShapleyRelGT, timed_section
from monteCarlo_analyze import create_visualization



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Monte Carlo Shapley explainer on trained RelBench model (long-running)'
    )
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., rel-arxiv)')
    parser.add_argument('--task', type=str, required=True,
                       help='Task name (e.g., paper-citation)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--model_type', type=str, default='hgs',
                       choices=['hgs', 'transformer'],
                       help='Model type to explain: hgs (default) or transformer (RelGT)')
    parser.add_argument('--resume', type=str, default=None,
                       help="Continue a previous run instead of starting from zero. "
                            "Pass a path to one of its monte_carlo_shapley_*.csv checkpoints, "
                            "or 'auto' to pick the most recent checkpoint of this "
                            "dataset/task/model. The permutations already averaged into that "
                            "checkpoint are carried over.")
    parser.add_argument('--chernoff', action='store_true', default=False,
                       help='Print Chernoff / Hoeffding bound estimates before and after running. '
                            'Shows how many permutations are theoretically required.')
    parser.add_argument('--chernoff-epsilon', type=float, default=0.01,
                       help='Desired accuracy (ε) for the Chernoff bound (default: 0.01).')
    parser.add_argument('--chernoff-delta', type=float, default=0.05,
                       help='Failure probability (δ) for the Chernoff bound (default: 0.05).')
    return parser.parse_args()


def main():
    """Main function to run Monte Carlo Shapley."""
    
    script_start_time = time.time()
    args = parse_args()
    DATASET_NAME = args.dataset
    TASK_NAME    = args.task
    SEED         = args.seed
    MODEL_TYPE   = args.model_type
    RUN_CHERNOFF    = args.chernoff
    CHERNOFF_EPS    = args.chernoff_epsilon
    CHERNOFF_DELTA  = args.chernoff_delta
    
    print("=" * 80)
    print("MONTE CARLO SHAPLEY - SERVER EXECUTION")
    print("=" * 80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Task: {TASK_NAME}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("This script will run independently until completion.")
    print("=" * 80)
    
    config = get_montecarlo_config(DATASET_NAME, TASK_NAME, MODEL_TYPE)

    saved_model_dir = get_montecarlo_saved_model_dir(DATASET_NAME, TASK_NAME, MODEL_TYPE)
    output_base_dir = get_montecarlo_output_dir(DATASET_NAME, TASK_NAME, MODEL_TYPE)
    
    print(f"\nConfiguration:")
    print(f"  Model directory: {saved_model_dir}")
    print(f"  Convergence threshold: {config['convergence_threshold']}")
    print(f"  Checkpoint interval: {config['auto_checkpoint_interval']}h")
    print(f"  Max permutations: {config['max_permutations'] if config['max_permutations'] else 'unlimited'}")
    print(f"  Task type: {config.get('task_type', 'classification')}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base_dir / f"monte_carlo_results_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    # ========================================================================
    # LOAD DATA AND MODEL
    # ========================================================================
    
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

        print(f"✓ Data loaded from: {saved_model_dir}")
        print(f"✓ Randomness fixed with seed: {metadata['seed']}")

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
        print("\nCreating data loaders...")
        # Explanation methods compare predictions across many coalitions, so
        # they need bit-exact reproducibility. NeighborLoader's default
        # temporal_strategy ("uniform") picks a RANDOM subset of eligible
        # neighbors whenever a node exceeds the num_neighbors budget, using
        # its own internal RNG that fix_randomness/use_deterministic_
        # algorithms can't reach. Removing the budget entirely
        # (num_neighbors=-1) is unsafe — some datasets have nodes with huge
        # true degree (e.g. a placeholder join-table row referenced 200k+
        # times), and that caused real CUDA OOM crashes. temporal_strategy=
        # "last" fixes this properly instead: keeps the original,
        # memory-safe budget, but selects the K most-recent eligible
        # neighbors by a deterministic rule (recency) rather than a random
        # draw — same bound as training, no RNG involved. This applies for
        # the HGS path; RelGT ignores this loader_dict entirely (rebuilds
        # its own RelGTTokens loader in f_model_all_predictions), so this is
        # a no-op for it.
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
        print("✓ Loaders created")
    
    # Load model
    print("\n" + "=" * 80)
    print("LOADING TRAINED MODEL")
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

        print(f"✓ Model loaded and set to EVAL mode  [{MODEL_TYPE.upper()}]")
        print(f"✓ Baseline predictions: {len(baseline_predictions)} samples")
        print(f"✓ Entity table: {entity_table}")
    
    # ========================================================================
    # INITIALIZE MONTE CARLO SHAPLEY
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("INITIALIZING MONTE CARLO SHAPLEY")
    print("=" * 80)

    with timed_section("Initializing Monte Carlo Shapley"):
        if MODEL_TYPE == "hgs":
            mc_shapley = MonteCarloShapley(
                model=model,
                data=data,
                col_stats_dict=col_stats_dict,
                entity_table=entity_table,
                device=device,
                loader_config=loader_config,
                task_type=config.get('task_type', 'classification'),
                test_subset_idx=subset_idx,
            )
        else:
            # RelGT — use the subclass that overrides f_model_all_predictions
            # to rebuild a RelGTTokens DataLoader over the ablated coalition data.
            from rdl_explain_files.model.relgt import RelGTTokens
            relgt_tokens_kwargs = {
                "RelGTTokens":     RelGTTokens,
                "data":            data,
                "task":            task,
                "K":               loader_config["K"],
                "precomputed_dir": loader_config["precomputed_dir"],
                "undirected":      True,
                "num_workers":     loader_config.get("num_workers_precompute", 4),
                # Authoritative node_type -> integer ID mapping, exactly as used to
                # build the model. This is what the integer keys in `grouped_tfs`
                # actually correspond to at runtime (NOT a plain enumerate() of
                # some list of node type names) - without this, masking silently
                # no-ops and every Shapley value comes out as 0.0.
                "node_type_map":   checkpoint["node_type_map"],
            }
            mc_shapley = MonteCarloShapleyRelGT(
                model=model,
                data=data,
                col_stats_dict=col_stats_dict,
                entity_table=entity_table,
                device=device,
                loader_config=loader_config,
                task_type=config.get('task_type', 'classification'),
                relgt_tokens_kwargs=relgt_tokens_kwargs,
                test_subset_idx=subset_idx,
            )
            mc_shapley.sanity_check_masking(max_batches=1, reference_predictions=baseline_predictions)
            mc_shapley.sanity_check_per_feature_masking(max_batches=1)

        print(f"\n✓ Monte Carlo Shapley initialized")
        print(f"  Dataset: {DATASET_NAME}")
        print(f"  Task: {TASK_NAME}")
        print(f"  Total features (M): {mc_shapley.M}")

        # baseline_predictions above came from the on-disk test_predictions_N.npy
        # (computed by train.py's test() with the old, randomly-sampled neighbor
        # budget), while every coalition prediction from here on uses
        # f_model_all_predictions()'s deterministic temporal_strategy="last"
        # sampling (loader_dict was built with it earlier). Diffing predictions
        # from two different sampling policies would inject spurious noise into
        # every Shapley value, so recompute baseline the same way coalition
        # predictions will be computed (the all-present coalition = no features
        # masked). HGS only for now.
        if MODEL_TYPE == "hgs":
            all_present_coalition = np.ones(mc_shapley.M)
            baseline_predictions = mc_shapley.f_model_all_predictions(
                loader_dict['test'], all_present_coalition, max_batches=len(loader_dict['test'])
            )
            print(f"✓ Baseline predictions recomputed via f_model_all_predictions() for consistency: "
                  f"{len(baseline_predictions)} samples")

    # ========================================================================
    # Decision threshold (classification only) — read from metadata, NOT
    # tuned here. The threshold must be computed exactly once, via
    # Utils/compute_and_save_threshold.py, and stored in metadata_<N>.json.
    # Every explanation method (SingleColumnAblation, Kernel SHAP, Monte Carlo) reads the
    # same saved value so they never disagree on which threshold to use.
    # ========================================================================
    task_type   = config.get("task_type", "classification")
    tune_metric = config.get("tune_metric", "roc_auc")

    chosen_threshold = None if task_type == "regression" else metadata.get("threshold")
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


    
    # ========================================================================
    # RUN MONTE CARLO SHAPLEY
    # ========================================================================

    # Determine max_permutations:
    #   --chernoff  →  compute a fixed a-priori n_required via the Hoeffding
    #                  bound (range_val=1.0, conservative unit range) and run
    #                  for exactly that many permutations with no early-stopping.
    #   (default)   →  use whatever max_permutations is set in the config and
    #                  stop on SE convergence as usual.

    print("\n" + "=" * 80)
    print("RUNNING MONTE CARLO SHAPLEY")
    print("=" * 80)

    if RUN_CHERNOFF:
        from MonteCarloShapley import compute_chernoff_n
        n_required = compute_chernoff_n(
            epsilon=CHERNOFF_EPS,
            delta=CHERNOFF_DELTA,
            range_val=1.0,
            M=mc_shapley.M,
        )
        print(f"\nMode: FIXED CHERNOFF")
        print(f"  ε={CHERNOFF_EPS},  δ={CHERNOFF_DELTA},  M={mc_shapley.M},  range=1.0")
        print(f"  n_required = {n_required:,} permutations")
        print(f"  convergence_threshold: disabled (running full n_required)")
        run_max_permutations   = n_required
        run_convergence_threshold = None
    else:
        print(f"\nMode: STANDARD (SE convergence < {config['convergence_threshold']})")
        run_max_permutations      = config['max_permutations']
        run_convergence_threshold = config['convergence_threshold']

    print(f"Checkpoints every {config['auto_checkpoint_interval']}h  →  {output_dir}")
    print("\nPress Ctrl+C to stop and save current results\n")

    # ========================================================================
    # RESUME FROM A PREVIOUS CHECKPOINT
    # ========================================================================
    # A run carries only three pieces of state: the running mean, Welford's
    # running sum of squared deviations (m2) and the permutation count. Every
    # hourly checkpoint CSV already stores all three - m2 == variance * (n - 1)
    # - so a run continues from any checkpoint with no extra state file.
    # Resuming is exact: Welford accumulation is additive and permutations are
    # i.i.d., so n + m permutations split across two runs equal n + m in one.
    resume_offset_hours = 0.0
    if args.resume:
        if args.resume == 'auto':
            candidates = sorted(
                output_base_dir.glob('monte_carlo_results_*/monte_carlo_shapley_*.csv'),
                key=lambda f: f.stat().st_mtime,
            )
            if not candidates:
                raise FileNotFoundError(
                    f"--resume auto found no checkpoint under {output_base_dir}")
            resume_path = candidates[-1]
        else:
            resume_path = Path(args.resume)
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        prev = pd.read_csv(resume_path).set_index('feature')
        missing = [f for f in mc_shapley.feature_names if f not in prev.index]
        if missing:
            raise ValueError(
                f"{resume_path} is missing {len(missing)} of this model's features "
                f"(e.g. {missing[:3]}) - it belongs to a different task or model.")
        prev = prev.loc[mc_shapley.feature_names]   # align CSV rows to internal order

        n_prev = int(prev['num_permutations'].iloc[0])
        resume_offset_hours = float(prev['elapsed_hours'].iloc[0])

        mc_shapley._initialize_running_stats()
        mc_shapley.running_mean = prev['shapley_value'].to_numpy(dtype=np.float64)
        mc_shapley.running_m2 = prev['variance'].to_numpy(dtype=np.float64) * max(n_prev - 1, 0)
        mc_shapley.num_permutations = n_prev

        # No reseeding is needed here: the permutation sampler uses its own
        # np.random.default_rng() seeded from OS entropy at construction, so a
        # resumed run never replays the permutations already accounted for.
        print("\n" + "=" * 80)
        print("RESUMING PREVIOUS RUN")
        print("=" * 80)
        print(f"  Checkpoint:   {resume_path}")
        print(f"  Carried over: {n_prev} permutations, {resume_offset_hours:.2f} hours")

    stop_reason = "unknown"   # filled in at each exit point for the summary
    mc_run_start_time = time.time()
    
    try:
        results = mc_shapley.explain(
            loader=loader_dict['test'],
            convergence_threshold=run_convergence_threshold,
            auto_checkpoint_interval=config['auto_checkpoint_interval'],
            max_permutations=run_max_permutations,
            max_batches=config['max_batches'],
            output_dir=str(output_dir),
            base_filename="monte_carlo_shapley",
            disagreement_threshold=chosen_threshold if chosen_threshold is not None else 0.5,
            verbose=True,
            reset_stats=not args.resume,
            elapsed_offset_hours=resume_offset_hours,
        )
        if RUN_CHERNOFF:
            stop_reason = f"Chernoff n_required reached  (n={n_required:,}, ε={CHERNOFF_EPS}, δ={CHERNOFF_DELTA})"
        else:
            stop_reason = "SE convergence / max_permutations"

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user! Saving current results...")
        results = mc_shapley.get_current_results_with_uncertainty()
        stop_reason = f"user interrupt  (perms={results['num_permutations']})"
        print(f"   Captured results from {results['num_permutations']} permutations")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        try:
            results = mc_shapley.get_current_results_with_uncertainty()
            stop_reason = f"exception — partial results  (perms={results['num_permutations']})"
            print(f"   Saving emergency results from {results['num_permutations']} permutations")
        except Exception:
            print("   Could not save emergency results")
            raise

    mc_run_elapsed = time.time() - mc_run_start_time
    print(f"\n  Stop reason: {stop_reason}")
    print(f"⏱  [Monte Carlo Shapley run (mc_shapley.explain call)] completed in "
          f"{mc_run_elapsed:.2f}s ({mc_run_elapsed/60:.2f} min)")
    
    # ========================================================================
    # SAVE FINAL RESULTS
    # ========================================================================

    # ── A-posteriori Chernoff bound (data-driven, uses empirical variance) ──
    # Always recompute at the end so the report reflects the final state,
    # regardless of which stopping path was taken.
    if RUN_CHERNOFF:
        print("\n" + "=" * 80)
        print("A-POSTERIORI CHERNOFF BOUND (data-driven, empirical σ)")
        print("=" * 80)
        chernoff_report = mc_shapley.chernoff_bound(
            epsilon=CHERNOFF_EPS,
            delta=CHERNOFF_DELTA,
            verbose=True,
        )
    
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    with timed_section("Saving results, metadata, and top-features print"):
        final_path = output_dir / "final_results.csv"
        mc_shapley.save_results(results, str(final_path),
                                disagreement_threshold=chosen_threshold)

        mc_shapley.print_top_features(results, top_k=15)

        metadata_dict = {
            'dataset': DATASET_NAME,
            'task': TASK_NAME,
            'timestamp': timestamp,
            'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_permutations': results['num_permutations'],
            'elapsed_hours': results.get('elapsed_hours', 0),
            'model_type': MODEL_TYPE,
            'task_type': config.get('task_type', 'classification'),
            'disagreement_threshold': chosen_threshold,
            'convergence_threshold': config['convergence_threshold'],
            'auto_checkpoint_interval': config['auto_checkpoint_interval'],
            'num_features': mc_shapley.M,
            'entity_table': entity_table,
            'device': str(device),
            'seed': metadata['seed'],
        }

        if RUN_CHERNOFF:
            metadata_dict['chernoff_epsilon']    = CHERNOFF_EPS
            metadata_dict['chernoff_delta']      = CHERNOFF_DELTA
            metadata_dict['chernoff_n_required'] = chernoff_report['n_required']
            metadata_dict['chernoff_worst_feature'] = chernoff_report['worst_feature']
            metadata_dict['chernoff_worst_range']   = float(chernoff_report['worst_range'])
            metadata_dict['chernoff_already_sufficient'] = bool(chernoff_report['already_sufficient'])

        import yaml
        metadata_path = output_dir / "metadata.yaml"
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata_dict, f)

        print(f"\n✓ Metadata saved: {metadata_path}")

    # ========================================================================
    # CREATE VISUALIZATION
    # ========================================================================
    
    with timed_section("Creating visualization"):
        # Build a results DataFrame from the saved CSV so create_visualization
        # gets the same input it would receive from monteCarlo_analyze.py.
        viz_df = pd.read_csv(final_path)
        elapsed_h = results.get("elapsed_hours", 0)
        hours_suffix = f"{elapsed_h:.1f}h" if elapsed_h > 0 else ""

        full_plot_path = create_visualization(
            results_df=viz_df,
            output_dir=output_dir,
            timestamp=timestamp,
            dataset_name=DATASET_NAME,
            task_name=TASK_NAME,
            hours_suffix=hours_suffix,
        )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("MONTE CARLO SHAPLEY COMPLETE")
    print("=" * 80)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n📊 Results:")
    print(f"   Dataset:    {DATASET_NAME}")
    print(f"   Task:       {TASK_NAME}")
    print(f"   Model type: {MODEL_TYPE.upper()}")
    print(f"   Permutations: {results['num_permutations']}")
    print(f"   Time: {results.get('elapsed_hours', 0):.2f} hours")
    print(f"   Features: {mc_shapley.M}")
    print(f"   Stop reason: {stop_reason}")
    
    if 'std_errors' in results:
        mean_se = np.mean(list(results['std_errors'].values()))
        max_se = np.max(list(results['std_errors'].values()))
        print(f"   Mean std error: {mean_se:.6f}")
        print(f"   Max std error: {max_se:.6f}")
    
    print(f"\n📄 Files created:")
    print(f"   Final results:  {final_path.name}")
    print(f"   Metadata:       {metadata_path.name}")
    print(f"   Checkpoints:    {len(list(output_dir.glob('*checkpoint*.csv')))} files")
    if full_plot_path is not None:
        print(f"   Plot (full):    {full_plot_path.name}")
        top20_name = full_plot_path.name.replace('_full_', '_top20_')
        if (output_dir / top20_name).exists():
            print(f"   Plot (top 20):  {top20_name}")
    
    print(f"\n✅ COMPLETE!")
    print(f"   Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    script_total_elapsed = time.time() - script_start_time
    print(f"\n⏱  TOTAL SCRIPT RUNTIME (start to finish): {script_total_elapsed:.2f}s "
          f"({script_total_elapsed/60:.2f} min / {script_total_elapsed/3600:.3f} h)")
    
    print(f"\n💡 Next steps:")
    print(f"   Run fidelity analysis with:")
    print(f"   python montecarlo_analyze.py --dataset {DATASET_NAME} --task {TASK_NAME} --results-dir {output_dir}")
    
    return results, output_dir


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MONTE CARLO SHAPLEY - STARTING")
    print("=" * 80)
    
    try:
        results, output_dir = main()
        sys.exit(0)  # Success
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  # Failure