"""
run_evaluation_pipeline.py  —  Centralised XAI Evaluation Pipeline

Performs, in order:
  1. Data acquisition  — locate result CSVs for every available explainer
  2. Profile generation — deterministic full ranking via generate_full_profile
  3. Stability analysis — F_prof, Jaccard, Overlap@K for all method pairs
  4. Fidelity calculation — Fidelity+ and Fidelity- for three k-ranges

Stability runs BEFORE fidelity: it only needs the profile CSVs from Phase 2
and involves no model inference, so it's fast and its results are written
to disk (and usable) well before the much slower fidelity phase — which
reloads the full model/graph and runs inference for every k value — even
starts computing.

All outputs are written to:
    EvaluationMethods/{DATASET_NAME}/{TASK_NAME}/{MODEL_TYPE}/run_{N}/

Usage:
    python run_evaluation_pipeline.py --dataset rel-trial --task study-outcome
    python run_evaluation_pipeline.py --dataset rel-f1 --task driver-position --model_type transformer --monte_carlo_file --classification_metric
    python run_evaluation_pipeline.py --dataset rel-trial --task study-outcome --resume 3
    python run_evaluation_pipeline.py --dataset rel-trial --task study-outcome --skip_monte_carlo
    python run_evaluation_pipeline.py --dataset rel-trial --task study-outcome --monte_carlo_file /path/to/specific/monte_carlo_results.csv
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sys
import re
import json
import time
import argparse
import itertools
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.special import expit

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Make sure Utils and fidelity_calculator are importable
# ----------------------------------------------------------------------------
_THIS_DIR     = Path(os.path.dirname(os.path.abspath(__file__)))
_RESEARCH_DIR = _THIS_DIR.parent   # research-clean/
if str(_RESEARCH_DIR) not in sys.path:
    sys.path.append(str(_RESEARCH_DIR))

# HGS model source (needed by FidelityCalculator)
_model_src = str(_RESEARCH_DIR / "MaskLearningExplainer" / "src")
if _model_src not in sys.path:
    sys.path.insert(0, _model_src)

from evaluation_pipeline_config import (
    EVALUATION_CONFIG,
    FIDELITY_K_FIXED, FIDELITY_K_STEP_SHORT, FIDELITY_K_PERCENT,
    get_eval_config,
    get_task_dir,
    get_eval_output_dir,
    get_feature_score_cols,
)
from ComparingTopK import (
    generate_full_profile,
    compute_f_prof_distance,
    compute_score_distance,
    compute_jaccard_similarity,
    compute_overlap_at_k,
)
from Utils.Utils import (
    get_next_run_index, get_indexed_filepath,
    load_research_components, get_trained_model, fix_randomness,
    create_consistent_loader, get_latest_indexed_file,
    load_test_subset, subset_table,
)
from EvaluationMethods.fidelity_calculator import FidelityCalculator, timed_section


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="XAI Evaluation Pipeline")
    p.add_argument("--dataset",    type=str, required=True,
                   help="Dataset name (e.g. rel-trial)")
    p.add_argument("--task",       type=str, required=True,
                   help="Task name (e.g. study-outcome)")
    p.add_argument("--model_type", type=str, default="hgs",
                   choices=["hgs", "transformer"],
                   help="Model type: hgs (default) or transformer")
    p.add_argument("--monte_carlo_file", type=str, default=None,
                   help="Path to a specific Monte Carlo results CSV "
                        "(default: auto-find latest)")
    p.add_argument("--skip_monte_carlo", action="store_true",
                   help="Skip Monte Carlo Shapley — do not discover or calculate it.")
    p.add_argument("--resume", type=int, default=None,
                   help="Run index to resume. Omit to start fresh.")
    p.add_argument("--classification_metric", type=str, default="importance_score",
                   choices=["label_flip", "importance_score"],
                   help="Controls TWO things for classification tasks (ignored for "
                        "regression, which always uses the importance_score-equivalent "
                        "formula): (1) how SingleColumnAblation Strategy features are ranked — "
                        "'importance_score' (default) ranks by continuous mean |Δpred|; "
                        "'label_flip' ranks by num_label_flips. Only affects "
                        "SingleColumnAblation Strategy — Kernel SHAP, RDL Explainer, and Monte Carlo Shapley "
                        "are unaffected. (2) which fidelity secondary metrics get "
                        "computed in FidelityCalculator — 'importance_score' (default) "
                        "computes only the threshold-free mean |Δprob|; 'label_flip' "
                        "ALSO computes accuracy-drop/label-flip-rate at a fixed "
                        "threshold (opt-in, since a single fixed threshold doesn't "
                        "reflect these tasks' actual decision boundaries well).")
    return p.parse_args()


# ============================================================================
# PHASE 1 HELPERS  —  file discovery
# ============================================================================

def get_latest_file(directory: Path, prefix: str, suffix: str = ".csv") -> Path:
    """Return the highest-indexed file matching prefix_N.suffix in directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    pattern = f"{prefix}_*{suffix}"
    files = list(directory.glob(pattern))
    if not files:
        fallback = directory / f"{prefix}{suffix}"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    indexed = []
    for f in files:
        try:
            indexed.append((int(f.stem.split("_")[-1]), f))
        except ValueError:
            pass
    if not indexed:
        return files[0]
    return max(indexed, key=lambda x: x[0])[1]


def find_latest_results_dir(base_dir: Path) -> Path:
    dirs = sorted(base_dir.glob("monte_carlo_results_*"))
    if not dirs:
        raise FileNotFoundError(f"No Monte Carlo results in {base_dir}")
    return dirs[-1]


def get_latest_checkpoint(results_dir: Path) -> Path:
    final = results_dir / "final_results.csv"
    if final.exists():
        return final
    checkpoints = sorted(results_dir.glob("monte_carlo_shapley_*.csv"))
    if not checkpoints:
        raise FileNotFoundError(f"No results files in {results_dir}")
    return checkpoints[-1]


def resolve_monte_carlo_source(base_dir: Path, results_file_path=None) -> Path:
    """Return the Monte Carlo results CSV to use.

    Fidelity files are no longer looked up here — the pipeline computes
    fidelity itself from the raw results CSV.
    """
    if results_file_path:
        mc_csv = Path(results_file_path)
        if not mc_csv.exists():
            raise FileNotFoundError(f"Specified Monte Carlo file not found: {mc_csv}")
        return mc_csv
    results_dir = find_latest_results_dir(base_dir)
    return get_latest_checkpoint(results_dir)


def _extract_index(path: Path):
    """Extract the trailing integer index from a filename stem, or None.

    Examples:
        kernelShap_disagreement_count_results_3.csv  ->  3
        rdl_results_12.csv                           ->  12
        monte_carlo_shapley_184.0h.csv               ->  None  (no plain int suffix)
        final_results.csv                            ->  None
    """
    try:
        return int(path.stem.split("_")[-1])
    except (ValueError, IndexError):
        return None


def _source_meta(path: Path, threshold=None) -> dict:
    """Build the source-metadata dict stored in data_sources.json."""
    return {
        "source_file":     path.name,
        "source_path":     str(path.resolve()),
        "explainer_index": _extract_index(path),
        "threshold":       threshold,
    }

def read_saved_threshold(saved_model_dir: Path) -> float | None:
    """Read the 'threshold' field directly from the model's metadata_<N>.json
    (the highest-index file in saved_model_dir).

    This is deliberately lightweight — it does NOT go through
    load_research_components (which loads the full graph, model weights,
    etc.); we only need this one scalar. The threshold itself must already
    have been computed once via Utils/compute_and_save_threshold.py, which
    all explanation methods (SingleColumnAblation, Kernel SHAP, Monte Carlo) also read
    from — so this pipeline sees exactly the same value they used.

    Returns None if no metadata file is found, or if the field is absent
    (e.g. a regression task, or threshold computation hasn't been run yet).
    """
    candidates = sorted(saved_model_dir.glob("metadata_*.json"))
    if not candidates:
        print(f"  ⚠  No metadata_*.json found in {saved_model_dir}")
        return None

    def _index(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return -1

    latest = max(candidates, key=_index)
    try:
        with open(latest) as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"  ⚠  Could not read {latest.name}: {e}")
        return None

    threshold = metadata.get("threshold")
    return float(threshold) if threshold is not None else None


def discover_available_methods(task_dir: Path,
                               monte_carlo_file=None,
                               skip_monte_carlo: bool = False) -> dict:
    """Return dict {method_name: {"results": Path, "source_meta": dict}}
    for every explainer whose raw results CSV is present.

    ``source_meta`` captures the filename, resolved path, and integer index
    of the source file so the pipeline can write data_sources.json and the
    notebook can display which explainer run each evaluation belongs to.

    This pipeline is responsible for computing fidelity — the individual
    explainer scripts no longer produce fidelity files.  We only need the
    raw importance scores here.

    If ``skip_monte_carlo`` is True, the Monte Carlo Shapley method is
    skipped entirely and will not appear in the returned dict.
    """
    single_column_ablation_dir = task_dir / "singleColumnAblation_explainer"
    kernelshap_dir  = task_dir / "kernelShap_explainer"
    rdl_dir         = task_dir / "rdl_explainer"
    monte_carlo_dir = task_dir / "monteCarlo_explainer"

    available = {}

    def _try_add(name, results_fn):
        try:
            p = results_fn()
            # Threshold is assigned after discovery (from the SingleColumnAblation file).
            available[name] = {"results": p, "source_meta": _source_meta(p),
                               "threshold": None}
            print(f"  \u2713 {name}: {p.name}  (index: {available[name]['source_meta']['explainer_index']})")
        except FileNotFoundError as e:
            print(f"  \u26a0\ufe0f  {name} — skipping ({e})")

    print("\nSearching for explainer results...")
    _try_add("SingleColumnAblation Strategy",
             lambda: get_latest_file(single_column_ablation_dir, "singleColumnAblation_results_as_kernel_shap"))
    _try_add("Kernel SHAP",
             lambda: get_latest_file(kernelshap_dir, "kernelShap_disagreement_count_results"))
    _try_add("RDL Explainer",
             lambda: get_latest_file(rdl_dir,        "rdl_results"))

    # Monte Carlo only needs its results CSV — no fidelity files expected
    if skip_monte_carlo:
        print("  ⊘  Monte Carlo Shapley — skipped (--skip_monte_carlo)")
    else:
        try:
            mc_csv = resolve_monte_carlo_source(monte_carlo_dir, monte_carlo_file)
            available["Monte Carlo Shapley"] = {
                "results":     mc_csv,
                "source_meta": _source_meta(mc_csv),
                "threshold":   None,
            }
            print(f"  \u2713 Monte Carlo Shapley: {mc_csv.name}"
                  f"  (index: {available['Monte Carlo Shapley']['source_meta']['explainer_index']})")
        except FileNotFoundError as e:
            print(f"  \u26a0\ufe0f  Monte Carlo Shapley — skipping ({e})")

    # Extract the task-level decision threshold directly from metadata.json
    # (classification tasks only — None for regression). This is read
    # independently of which explainer methods happen to be available, so it
    # no longer depends on "SingleColumnAblation Strategy" being present.
    saved_model_dir = task_dir / "saved_model"
    task_threshold = read_saved_threshold(saved_model_dir)
    if task_threshold is not None:
        print(f"\n  ✓ Task threshold (from metadata): {task_threshold}")
    else:
        print("\n  ℹ  No threshold found in metadata.json — regression task assumed, "
              "or Utils/compute_and_save_threshold.py hasn't been run for this model yet.")

    # Propagate the shared threshold to every method entry
    for info in available.values():
        info["threshold"] = task_threshold
        info["source_meta"]["threshold"] = task_threshold

    if len(available) < 2:
        raise ValueError(
            f"Need at least 2 explainer methods. Found: {list(available.keys())}"
        )
    return available


# ============================================================================
# PHASE 2  —  Profile generation
# ============================================================================

def build_profiles(available_methods: dict, profiles_dir: Path,
                   run_index: int, task_type: str,
                   classification_metric: str = "label_flip") -> tuple:
    """Generate, save, and return full ranking profiles for each method.

    Args:
        classification_metric: Only affects the "SingleColumnAblation Strategy" method on
            classification tasks — 'label_flip' (default) ranks by
            num_label_flips; 'importance_score' ranks by the continuous mean
            |Δpred| metric instead (same formula used for regression).
            Ignored for regression and for named methods (Kernel SHAP / RDL /
            Monte Carlo), whose score columns don't depend on task type.

    Returns:
        all_profiles:   {method_name: pd.Series(rank, index=feature)}
                        — loaded from the canonical CSV so the in-memory
                          object is always identical to what is on disk.
        profile_csvs:   {method_name: Path}  — path to each profile CSV.
        total_features: int
        thresholds:     {method_name: float | None}
    """
    profiles_dir.mkdir(parents=True, exist_ok=True)
    all_profiles = {}
    profile_csvs = {}
    thresholds   = {}

    for method_name, files in available_methods.items():
        try:
            df          = pd.read_csv(files["results"])
            feat_col, score_col = get_feature_score_cols(
                method_name, task_type=task_type,
                classification_metric=classification_metric,
            )
            weights     = dict(zip(df[feat_col], df[score_col]))
            out_name    = method_name.replace(" ", "_")
            profile_csv = profiles_dir / f"profile_{out_name}_{run_index}.csv"

            # generate_full_profile writes the CSV and returns the Series
            profile     = generate_full_profile(weights, output_path=profile_csv)

            # Re-read from CSV so in-memory object is byte-for-byte identical
            # to what every other phase will read from disk.
            saved_df    = pd.read_csv(profile_csv)
            profile     = pd.Series(saved_df["rank"].values,
                                    index=saved_df["feature"].values)

            # Annotate the profile CSV with the threshold from the source file
            threshold = files.get("threshold")
            thresholds[method_name] = threshold
            if threshold is not None:
                saved_df["threshold"] = threshold
                saved_df.to_csv(profile_csv, index=False)

            all_profiles[method_name] = profile
            profile_csvs[method_name] = profile_csv
            thr_str = f", threshold={threshold}" if threshold is not None else ""
            score_str = f", scored by '{score_col}'" if method_name == "SingleColumnAblation Strategy" else ""
            print(f"  ✓ {method_name}: {len(profile)} features → {profile_csv.name}{thr_str}{score_str}")
        except Exception as e:
            print(f"  ⚠  {method_name}: could not generate profile — {e}")

    if len(all_profiles) < 2:
        raise ValueError("Not enough profiles generated.")

    sizes = {m: len(p) for m, p in all_profiles.items()}
    unique = set(sizes.values())
    total_features = min(unique)
    if len(unique) > 1:
        print(f"  ⚠  Inconsistent profile sizes {sizes}. Using minimum: {total_features}")
    else:
        print(f"  ✓ All profiles: {total_features} features")

    return all_profiles, profile_csvs, total_features, thresholds


# ============================================================================
# PHASE 4  —  Fidelity (centralised, profile-driven)
# ============================================================================
 
def build_k_ranges(total_features: int, k_step_long: int = 10) -> dict:
    """Return the k-range lists specified in the requirements."""
    quarter = max(1, total_features // 4)
    percent_ks = sorted(set(
        min(total_features, max(1, round(total_features * pct / 100.0)))
        for pct in FIDELITY_K_PERCENT
    ))
    return {
        "fixed":   sorted(set(k for k in FIDELITY_K_FIXED if k <= total_features)),
        "short":   list(range(1, quarter + 1, FIDELITY_K_STEP_SHORT)),
        "long":    list(range(1, total_features + 1, k_step_long)),
        "percent": percent_ks,
    }
 
 
def run_fidelity(
    available_methods: dict,
    profile_csvs: dict,
    total_features: int,
    model, data, task, col_stats_dict, loader_config,
    test_table, test_predictions,
    out_dir: Path, run_index: int,
    task_type: str,
    tune_metric: str = "roc_auc",
    k_step_long: int = 10,
    model_type: str = "hgs",
    test_subset_idx=None,
    thresholds: dict = None,
    classification_metric: str = "importance_score",
) -> dict:
    """
    Run Fidelity+ and Fidelity- for all methods across all k-ranges.

    Uses FidelityCalculator.calculate_fidelity_plus_from_profile /
    calculate_fidelity_minus_from_profile which read directly from the
    canonical profile CSV and select top-k by row order — no re-ranking
    by score, no column manipulation.

    CACHING: the four named ranges ("fixed", "short", "long", "percent")
    overlap heavily in which k values they contain (e.g. k=5 typically
    appears in "fixed" AND "short" AND sometimes "percent"). This used to
    call calculate_fidelity_plus_from_profile / _minus_from_profile once
    PER RANGE, so the same (method, k) pair was recomputed 2-3x — each
    recomputation re-running ablation + inference from scratch. Now each
    unique (method, k) pair across ALL ranges is computed exactly ONCE,
    cached into one "master" CSV per method per test_type (the template
    file), and each named range's CSV is produced by filtering that
    master cache on the 'k' column — no recomputation, and every named
    range is guaranteed to report identical numbers for any k value they
    share.

    ``thresholds`` maps method_name -> float | None.  When provided, each
    method's threshold is forwarded to the FidelityCalculator calls and
    written as a column into the fidelity output CSVs for traceability.

    Returns {method_name: {range_name: {plus_path, minus_path}}}.
    """
    if thresholds is None:
        thresholds = {}
    _fidelity_start_time = time.time()
    k_ranges = build_k_ranges(total_features, k_step_long=k_step_long)

    # Union of every k value needed across every named range — each one
    # gets computed exactly once per method.
    all_k_union = sorted(set(k for k_list in k_ranges.values() for k in k_list))

    # Build relgt_tokens_kwargs if transformer model
    relgt_tokens_kwargs = None
    if model_type == "transformer":
        from rdl_explain_files.model.relgt import RelGTTokens
        relgt_tokens_kwargs = {
            "RelGTTokens":     RelGTTokens,
            "data":            data,
            "task":            task,
            "K":               loader_config["K"],
            "precomputed_dir": loader_config["precomputed_dir"],
            "undirected":      True,
            "num_workers":     loader_config.get("num_workers_precompute", 4),
        }

    fidelity_calc = FidelityCalculator(
        explainer=None,
        model=model,
        data=data,
        task=task,
        col_stats_dict=col_stats_dict,
        loader_config=loader_config,
        model_type=model_type,
        relgt_tokens_kwargs=relgt_tokens_kwargs,
        test_subset_idx=test_subset_idx,
    )
    
    # Fresh, zero-ablation inference pass on the currently loaded graph —
    # self.data is guaranteed untouched here, before any method/k loop.
    _fresh_baseline = fidelity_calc._run_inference(test_table)

    # test_predictions (loaded from test_predictions_N.npy) was computed by
    # train.py's test() — a different process/run than _run_inference now
    # performs. Comparing ablated/suffix predictions against that on-disk
    # baseline can never hit zero gap even with a correct ablate/restore
    # round trip, since it isn't the same computation. Use the
    # freshly-computed, same-method baseline for every fidelity call instead
    # so baseline and ablated/suffix predictions are always measured the
    # same way.
    #
    # _run_inference() always returns raw logits (verified directly — never
    # applies sigmoid, in either the HGS or RelGT branch), but train.py's
    # test() DID apply sigmoid for classification tasks, so the on-disk
    # test_predictions_N.npy is on a probability scale for those tasks.
    # calculate_fidelity_plus/minus_from_profile() also sigmoid the
    # suffix/ablated predictions for classification (resolved_suffix_preds =
    # expit(suffix_preds)) before comparing against baseline_predictions via
    # a fixed threshold and raw differences (fidelity_acc, fidelity_flip,
    # fidelity_prob) — those would silently compare two different scales if
    # baseline_predictions were left as raw logits. The rank-based metric
    # (fidelity_metric / ROC-AUC) can't catch this, since AUC is invariant
    # to any monotonic transform — sigmoid included. Match what the on-disk
    # baseline used to provide.
    test_predictions = _fresh_baseline
    if task_type == "classification":
        test_predictions = expit(test_predictions)

    results = {}

    for method_name, profile_csv in profile_csvs.items():
        results[method_name] = {}
        threshold = thresholds.get(method_name)
        thr_str = f", threshold={threshold}" if threshold is not None else ""

        safe_name = method_name.replace(' ', '_')

        # --- Compute every unique k ONCE, cache into the master/template file ---
        print(f"\n  {method_name} | computing {len(all_k_union)} unique k values once{thr_str}")
        plus_rows, minus_rows = [], []

        with timed_section(f"Fidelity (Plus+Minus, all k) | {method_name}"):
            for k in all_k_union:
                try:
                    f_plus = fidelity_calc.calculate_fidelity_plus_from_profile(
                        profile_csv=profile_csv,
                        baseline_predictions=test_predictions,
                        test_table=test_table,
                        method_name=method_name,
                        top_k=k,
                        metric=tune_metric,
                        task_type=task_type,
                        threshold=threshold,
                        classification_metric=classification_metric,
                    )
                    f_plus.setdefault('k', k)
                    f_plus['method_name'] = method_name
                    f_plus['test_type']   = 'Plus'
                    f_plus['timestamp']   = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    if threshold is not None:
                        f_plus['threshold'] = threshold
                    plus_rows.append(f_plus)
                except Exception as e:
                    import traceback
                    print(f"    ⚠  Fidelity+ k={k}: {e}")
                    traceback.print_exc()

                try:
                    f_minus = fidelity_calc.calculate_fidelity_minus_from_profile(
                        profile_csv=profile_csv,
                        baseline_predictions=test_predictions,
                        test_table=test_table,
                        method_name=method_name,
                        top_k=k,
                        metric=tune_metric,
                        task_type=task_type,
                        threshold=threshold,
                        classification_metric=classification_metric,
                    )
                    f_minus.setdefault('k', k)
                    f_minus['method_name'] = method_name
                    f_minus['test_type']  = 'Minus'
                    f_minus['timestamp']  = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    if threshold is not None:
                        f_minus['threshold'] = threshold
                    minus_rows.append(f_minus)
                except Exception as e:
                    import traceback
                    print(f"    ⚠  Fidelity- k={k}: {e}")
                    traceback.print_exc()

        plus_df  = pd.DataFrame(plus_rows)
        minus_df = pd.DataFrame(minus_rows)

        master_plus_path  = out_dir / f"fidelity_plus_{safe_name}_all_k_{run_index}.csv"
        master_minus_path = out_dir / f"fidelity_minus_{safe_name}_all_k_{run_index}.csv"
        plus_df.to_csv(master_plus_path, index=False)
        minus_df.to_csv(master_minus_path, index=False)
        print(f"    ✓ master cache saved: {master_plus_path.name}, {master_minus_path.name}")

        # --- Slice the cache per named range — no recomputation ---
        for range_name, k_list in k_ranges.items():
            plus_path  = out_dir / f"fidelity_plus_{safe_name}_{range_name}_{run_index}.csv"
            minus_path = out_dir / f"fidelity_minus_{safe_name}_{range_name}_{run_index}.csv"

            if not plus_df.empty:
                plus_df[plus_df['k'].isin(k_list)].to_csv(plus_path, index=False)
            else:
                plus_df.to_csv(plus_path, index=False)

            if not minus_df.empty:
                minus_df[minus_df['k'].isin(k_list)].to_csv(minus_path, index=False)
            else:
                minus_df.to_csv(minus_path, index=False)

            results[method_name][range_name] = {
                "fidelity_plus":  plus_path,
                "fidelity_minus": minus_path,
                "threshold":      threshold,
            }
            print(f"    ✓ {range_name}: {plus_path.name}, {minus_path.name} ({len(k_list)} k values, from cache)")

    _fidelity_elapsed = time.time() - _fidelity_start_time
    print(f"\n⏱  [run_fidelity TOTAL] completed in {_fidelity_elapsed:.2f}s "
          f"({_fidelity_elapsed/60:.2f} min)")
    return results
 
 
# ============================================================================
# PHASE 3  —  Stability analysis
# ============================================================================
 
def run_stability(profile_csvs: dict, total_features: int,
                  out_dir: Path, run_index: int,
                  k_step_long: int = 10) -> dict:
    """Compute F_prof, Score distance, Jaccard, Overlap@K for all method pairs.
 
    Loads each method's ranking (rank, position, normalized_score) from its
    saved profile CSV (same source of truth as fidelity) and uses the same
    k-ranges as fidelity (fixed, short, long) so that stability and fidelity
    are directly comparable. F_prof measures order agreement (via rank);
    the Score distance measures magnitude agreement (via normalized_score) —
    see compute_score_distance in ComparingTopK.py.

    CACHING: same pattern as run_fidelity — each unique k across all named
    ranges is computed once (cheap here, pure rank comparison, but kept
    consistent with fidelity's caching so both phases follow one pattern
    and stay trivially comparable), cached to one master/template CSV, and
    each named range's CSV is a filtered slice of that cache.
    """
    # Load rank, position, and normalized_score from each profile CSV
    _stability_start_time = time.time()
    loaded_ranks = {}
    loaded_positions = {}
    loaded_scores = {}
    for method_name, profile_csv in profile_csvs.items():
        df = pd.read_csv(profile_csv)
        loaded_ranks[method_name] = pd.Series(
            df["rank"].values, index=df["feature"].values
        )
        loaded_positions[method_name] = pd.Series(
            df["position"].values, index=df["feature"].values
        )
        loaded_scores[method_name] = pd.Series(
            df["normalized_score"].values, index=df["feature"].values
        )
 
    method_names = list(loaded_ranks.keys())
    num_features = min(len(p) for p in loaded_ranks.values())
 
    # Use the exact same k-ranges as fidelity for direct comparability
    k_ranges = build_k_ranges(num_features, k_step_long=k_step_long)
    all_k_union = sorted(set(k for k_list in k_ranges.values() for k in k_list))

    # --- Compute every unique k ONCE across all method pairs ---
    all_rows = []
    for k_val in all_k_union:
        for m1, m2 in itertools.combinations(method_names, 2):
            r1, r2 = loaded_ranks[m1], loaded_ranks[m2]
            pos1, pos2 = loaded_positions[m1], loaded_positions[m2]
            s1, s2 = loaded_scores[m1], loaded_scores[m2]
            f_res = compute_f_prof_distance(r1, r2, pos1, pos2, k_val, total_features)
            score_res = compute_score_distance(s1, s2, pos1, pos2, k_val)
            # Top-k sets determined by position, not rank
            top1 = set(pos1[pos1 <= k_val].index)
            top2 = set(pos2[pos2 <= k_val].index)
            all_rows.append({
                "k":                            k_val,
                "Comparison":                   f"{m1} vs {m2}",
                "Normalized Distance (F_prof)": f_res["normalized_distance"],
                "Similarity (F_prof)":          f_res["similarity"],
                "Normalized Distance (Score)":  score_res["normalized_distance"],
                "Similarity (Score)":           score_res["similarity"],
                "Jaccard Similarity":           compute_jaccard_similarity(top1, top2),
                "Overlap@K":                    compute_overlap_at_k(top1, top2, k_val),
            })

    master_df   = pd.DataFrame(all_rows)
    master_path = out_dir / f"stability_all_k_{run_index}.csv"
    master_df.to_csv(master_path, index=False)
    print(f"  ✓ master cache: {len(master_df)} rows ({len(all_k_union)} unique k values) → {master_path.name}")

    # --- Slice the cache per named range — no recomputation ---
    saved = {}
    for range_name, k_list in k_ranges.items():
        df = master_df[master_df["k"].isin(k_list)]
        if df.empty:
            continue
        p = out_dir / f"stability_{range_name}_{run_index}.csv"
        df.to_csv(p, index=False)
        saved[range_name] = p
        print(f"  ✓ {range_name}: {len(df)} rows → {p.name} (from cache)")

    _stability_elapsed = time.time() - _stability_start_time
    print(f"\n⏱  [run_stability TOTAL] completed in {_stability_elapsed:.2f}s "
          f"({_stability_elapsed/60:.2f} min)")
    return saved
 
# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    script_start_time = time.time()

    DATASET_NAME = args.dataset
    TASK_NAME    = args.task
    MODEL_TYPE   = args.model_type
    CLASSIFICATION_METRIC = args.classification_metric

    # Resolve config for this run
    cfg        = get_eval_config(DATASET_NAME, TASK_NAME, MODEL_TYPE)
    TASK_TYPE  = cfg["task_type"]
    TASK_DIR   = Path(cfg["task_dir"])
    EVAL_BASE_DIR = get_eval_output_dir(DATASET_NAME, TASK_NAME, MODEL_TYPE)

    # Override FIDELITY_K_STEP_LONG from per-task config if present
    FIDELITY_K_STEP_LONG = cfg.get("fidelity_k_step_long", 10)

    print("=" * 70)
    print("XAI EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Dataset:    {DATASET_NAME}")
    print(f"Task:       {TASK_NAME}")
    print(f"Task type:  {TASK_TYPE}")
    print(f"Model type: {MODEL_TYPE}")
    if TASK_TYPE == "classification":
        print(f"Classification metric (SingleColumnAblation Strategy): {CLASSIFICATION_METRIC}")
    print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Output directory + resume state ──────────────────────────────────────
    EVAL_BASE_DIR.mkdir(parents=True, exist_ok=True)
    RESUME_DIR = EVAL_BASE_DIR / "resume"
    RESUME_DIR.mkdir(parents=True, exist_ok=True)

    if args.resume is not None:
        run_index = args.resume
        print(f"\n⟳  Resuming run index {run_index}")
    else:
        # Determine the next run index by scanning existing run_* directories.
        # We do this here rather than relying on get_next_run_index so the
        # logic is self-contained: any directory whose name matches run_{int}
        # counts as a taken slot, regardless of whether it has a run_index.json.
        existing = []
        for d in EVAL_BASE_DIR.glob("run_*"):
            try:
                existing.append(int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
        run_index = max(existing) + 1 if existing else 1

    RUN_DIR      = EVAL_BASE_DIR / f"run_{run_index}"
    PROFILES_DIR = RUN_DIR / "profiles"
    FIDELITY_DIR = RUN_DIR / "fidelity"
    STABILITY_DIR = RUN_DIR / "stability"
    for d in [RUN_DIR, PROFILES_DIR, FIDELITY_DIR, STABILITY_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Write a stub run_index.json immediately so that a subsequent pipeline
    # invocation (or a crash + restart) sees this slot as taken and picks
    # run_{N+1} instead of colliding with this directory.
    stub = {
        "run_index":  run_index,
        "dataset":    DATASET_NAME,
        "task":       TASK_NAME,
        "model_type": MODEL_TYPE,
        "status":     "in_progress",
        "started_at": datetime.now().isoformat(),
    }
    index_path = RUN_DIR / "run_index.json"
    index_path.write_text(json.dumps(stub, indent=2))

    RESUME_FILE = RESUME_DIR / f"resume_{run_index}.json"
    resume_state: dict = {}
    if args.resume is not None:
        if not RESUME_FILE.exists():
            raise FileNotFoundError(
                f"Resume file not found: {RESUME_FILE}\n"
                f"Make sure run index {run_index} exists in {RESUME_DIR}"
            )
        resume_state = json.loads(RESUME_FILE.read_text())
        print(f"   Completed phases: {resume_state.get('completed', [])}")

    def _save_resume():
        RESUME_FILE.write_text(json.dumps(resume_state, indent=2, default=str))

    completed = resume_state.get("completed", [])

    # ── PHASE 1: Discover available method files ──────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1 — Data acquisition")
    print("=" * 70)
    with timed_section("Phase 1: Data acquisition"):
        available_methods = discover_available_methods(
            TASK_DIR, args.monte_carlo_file, skip_monte_carlo=args.skip_monte_carlo
        )
        print(f"\n✓ {len(available_methods)} methods: {list(available_methods.keys())}")

        # Persist source metadata so the notebook can link each evaluation run
        # back to the exact explainer output file that was used as input.
        data_sources = {
            method: {
                **info["source_meta"],
                "threshold": info.get("threshold"),
            }
            for method, info in available_methods.items()
        }
        DATA_SOURCES_PATH = RUN_DIR / "data_sources.json"
        DATA_SOURCES_PATH.write_text(json.dumps(data_sources, indent=2))
        print(f"\u2713 data_sources.json written — maps methods to their source explainer files")

    # ── PHASE 2: Profiles ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2 — Profile generation")
    print("=" * 70)

    with timed_section("Phase 2: Profile generation"):
        if "profiles" not in completed:
            all_profiles, profile_csvs, total_features, thresholds = build_profiles(
                available_methods, PROFILES_DIR, run_index, task_type=TASK_TYPE,
                classification_metric=CLASSIFICATION_METRIC,
            )
            resume_state["total_features"] = total_features
            resume_state["method_names"]   = list(all_profiles.keys())
            resume_state["classification_metric"] = CLASSIFICATION_METRIC
            resume_state.setdefault("completed", []).append("profiles")
            _save_resume()
            print("✓ Phase 2 complete — state saved")
        else:
            # Reconstruct profile_csvs paths from saved method names
            total_features = resume_state["total_features"]
            all_profiles   = {}
            profile_csvs   = {}
            thresholds = {}
            for method in resume_state["method_names"]:
                out_name    = method.replace(" ", "_")
                csv_path    = PROFILES_DIR / f"profile_{out_name}_{run_index}.csv"
                df          = pd.read_csv(csv_path)
                all_profiles[method]  = pd.Series(df["rank"].values, index=df["feature"].values)
                profile_csvs[method]  = csv_path
                # Re-read threshold that was stamped into the profile CSV
                if "threshold" in df.columns:
                    vals = df["threshold"].dropna().unique()
                    thresholds[method] = float(vals[0]) if len(vals) > 0 else None
                else:
                    thresholds[method] = None
            print(f"⟳  Profiles reloaded ({len(all_profiles)} methods, "
                  f"{total_features} features)")

            prior_metric = resume_state.get("classification_metric")
            if TASK_TYPE == "classification" and prior_metric is not None \
                    and prior_metric != CLASSIFICATION_METRIC:
                print(f"  ⚠  This run's profiles were built with "
                      f"classification_metric='{prior_metric}', but you passed "
                      f"'--classification_metric {CLASSIFICATION_METRIC}'. The "
                      f"already-completed profiles will be reused as-is — start "
                      f"a fresh run (no --resume) to rebuild with the new metric.")

    # ── PHASE 3: Stability ────────────────────────────────────────────────────
    # Runs BEFORE fidelity: only needs the profile CSVs from Phase 2 (no model
    # inference), so it's fast and its results are on disk and usable well
    # before the much slower fidelity phase below even starts.
    print("\n" + "=" * 70)
    print("PHASE 3 — Stability analysis")
    print("=" * 70)

    if "stability" not in completed:
        with timed_section("Phase 3: Stability analysis"):
            stability_paths = run_stability(
                profile_csvs, total_features, STABILITY_DIR, run_index,
                k_step_long=FIDELITY_K_STEP_LONG,
            )
        resume_state.setdefault("completed", []).append("stability")
        _save_resume()
        print("✓ Phase 3 complete — state saved")
    else:
        print("⟳  Skipping stability (already completed)")

    # ── Load model + data (needed for Fidelity) ───────────────────────────────
    if "fidelity" not in completed:
        with timed_section("Loading model and data for fidelity calculation"):
            print("\nLoading model and data for fidelity calculation...")
            saved_model_dir = TASK_DIR / "saved_model"
            device = __import__("torch").device(
                "cuda" if __import__("torch").cuda.is_available() else "cpu"
            )
 
            (data, col_stats_dict, task, dataset,
             metadata, loader_config, test_predictions) = load_research_components(
                saved_model_dir, device
            )
            fix_randomness(metadata["seed"])
            test_table = task.get_table("test")

            # Tasks with a saved explanation subset (rel-arxiv, whose test set is
            # far too large to explain in full) are EXPLAINED on exactly those
            # rows, so fidelity must be measured on the same rows — otherwise it
            # would score an explanation against records it never saw. The saved
            # test_predictions cover the full test set, so they are subset too.
            subset_idx, subset_meta = load_test_subset(DATASET_NAME, TASK_NAME, task)
            if subset_idx is not None:
                test_table = subset_table(test_table, subset_idx)
                test_predictions = test_predictions[subset_idx]
                print(f"✓ Evaluating a subset: Test({len(test_table.df)}) rows")
 
            weights_file = get_latest_indexed_file(saved_model_dir, "model_weights", ".pt")
            if weights_file is None:
                raise FileNotFoundError(f"No model weights in {saved_model_dir}")
 
            if MODEL_TYPE == "hgs":
                from rdl_explain_files.model.gnn.model import HGSModel
                model = get_trained_model(
                    model_class=HGSModel,
                    data=data, col_stats_dict=col_stats_dict,
                    metadata=metadata, weights_path=weights_file, device=device,
                )
            else:
                import torch
                from rdl_explain_files.model.relgt import RelGT, RelGTConfig
                ckpt = torch.load(weights_file, map_location="cpu", weights_only=False)
                mc   = ckpt["model_config"]
                relgt_cfg = RelGTConfig(
                    local_num_layers=mc["local_num_layers"],
                    channels=mc["channels"], out_channels=mc["out_channels"],
                    global_dim=mc["global_dim"], heads=mc["heads"],
                    ff_dropout=mc.get("ff_dropout", 0.1),
                    attn_dropout=mc.get("attn_dropout", 0.1),
                    conv_type=mc.get("conv_type", "full"),
                    num_centroids=mc.get("num_centroids", 4096),
                    sample_node_len=mc["K"],
                )
                model = RelGT(
                    num_nodes=ckpt["num_nodes"],
                    max_neighbor_hop=ckpt["max_neighbor_hop"],
                    node_type_map=ckpt["node_type_map"],
                    col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
                    col_stats_dict=col_stats_dict,
                    **relgt_cfg.model_dump(),
                )
                model.load_state_dict(ckpt["model_state_dict"])
                model.to(device).eval()
 
            print(f"✓ Model loaded [{MODEL_TYPE.upper()}]")
 
    # ── PHASE 4: Fidelity ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 4 — Fidelity calculation")
    print("=" * 70)
 
    if "fidelity" not in completed:
        with timed_section("Phase 4: Fidelity calculation"):
            fidelity_results = run_fidelity(
                available_methods, profile_csvs, total_features,
                model, data, task, col_stats_dict, loader_config,
                test_table, test_predictions,
                FIDELITY_DIR, run_index,
                task_type=TASK_TYPE,
                tune_metric=cfg["tune_metric"],
                k_step_long=FIDELITY_K_STEP_LONG,
                model_type=MODEL_TYPE,
                test_subset_idx=subset_idx,
                thresholds=thresholds,
                classification_metric=CLASSIFICATION_METRIC,
            )
        resume_state.setdefault("completed", []).append("fidelity")
        _save_resume()
        print("\n✓ Phase 4 complete — state saved")
    else:
        print("⟳  Skipping fidelity (already completed)")
 
    # ── Save index file for the notebook ─────────────────────────────────────
    # Overwrite the in-progress stub with the full completed record.
    index = {
        "run_index":            run_index,
        "dataset":              DATASET_NAME,
        "task":                 TASK_NAME,
        "task_type":            TASK_TYPE,
        "model_type":           MODEL_TYPE,
        "classification_metric": CLASSIFICATION_METRIC if TASK_TYPE == "classification" else None,
        "total_features":       total_features,
        "method_names":         list(all_profiles.keys()),
        "profiles_dir":         str(PROFILES_DIR),
        "fidelity_dir":         str(FIDELITY_DIR),
        "stability_dir":        str(STABILITY_DIR),
        "eval_base_dir":        str(EVAL_BASE_DIR),
        "run_dir":              str(RUN_DIR),
        "status":               "completed",
        "started_at":           stub["started_at"],
        "completed_at":         datetime.now().isoformat(),
        "data_sources_path":    str(DATA_SOURCES_PATH),
        "thresholds":           thresholds,
    }
    index_path = RUN_DIR / "run_index.json"
    index_path.write_text(json.dumps(index, indent=2))
 
    # All done — remove resume file
    if RESUME_FILE.exists():
        RESUME_FILE.unlink()
 
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Output directory: {RUN_DIR}")
    print(f"  profiles/       — {len(list(PROFILES_DIR.glob('*.csv')))} CSVs")
    print(f"  fidelity/       — {len(list(FIDELITY_DIR.glob('*.csv')))} CSVs")
    print(f"  stability/      — {len(list(STABILITY_DIR.glob('*.csv')))} CSVs")
    print(f"  run_index.json  — notebook entry point")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    script_total_elapsed = time.time() - script_start_time
    print(f"\n⏱  TOTAL PIPELINE RUNTIME (start to finish): {script_total_elapsed:.2f}s "
          f"({script_total_elapsed/60:.2f} min / {script_total_elapsed/3600:.3f} h)")

    print(f"\nNext step:")
    print(f"  Open Visualization.ipynb — it will auto-load this run (index {run_index})")
    print(f"  Or force a specific run: set FORCE_RUN_INDEX = {run_index} in the notebook")
 
 
if __name__ == "__main__":
    main()