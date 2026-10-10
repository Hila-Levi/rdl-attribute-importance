"""
Fidelity Metrics Calculator for Feature Attribution Strategies

Core Objectives:
1. Process ablation results for strategy combinations (mapping attribution methods to feature types).
2. Utilize baseline model predictions as a reference point for performance evaluation.
3. Identify and manipulate the Top-K features to validate explanation faithfulness.

Calculated Metrics:
- Fidelity+ (Necessity): Measures the performance drop when the Top-K features are 
  removed (ablated). High drop indicates those features were necessary for the prediction.
- Fidelity- (Sufficiency): Measures the performance when ONLY the Top-K features 
  are present (restored). High retention indicates those features are sufficient on their own.

Supports both binary classification and regression tasks.
For regression, label-flip metrics are replaced by mean absolute prediction shift.
"""

import torch
import numpy as np
import pandas as pd
import time
from contextlib import contextmanager
from tqdm import tqdm
from scipy.special import expit
import os
import sys
from pathlib import Path
# Metrics are computed locally with sklearn instead of via task.evaluate()
# — see _compute_metrics_dict() for why.
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score

current_file_path = os.path.abspath(__file__)
eval_methods_dir = os.path.dirname(current_file_path)
research_root = os.path.dirname(eval_methods_dir)

# Make the repo root importable (for Utils.*)
if research_root not in sys.path:
    sys.path.append(research_root)

from Utils.Utils import backup_column, restore_column

try:
    from SingleColumnAblationExplainer.singleColumnAblation_explainer import AblationStrategies
    from Utils.Utils import load_research_components, get_trained_model, fix_randomness, create_consistent_loader
    print("Successfully imported AblationStrategies and Utils.")
except ImportError as e:
    print(f"Warning: Could not import required modules. Error: {e}")


@contextmanager
def timed_section(label: str):
    """Time a block of code and print how long it took.

    Usage:
        with timed_section("Fidelity+ | k=15 | Kernel SHAP"):
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


        
class FidelityCalculator:
    """
    Comprehensive Fidelity Metrics Calculator to validate explanation faithfulness.
    
    This class evaluates how well identified important features align with model behaviour:
    - Fidelity+ (Necessity): Quantifies the performance drop when top-K features are removed.
    - Fidelity- (Sufficiency): Quantifies performance retention when only top-K features are kept.

    Supports binary classification and regression tasks via the task_type parameter.
    Besides the primary metric change, each call reports fidelity_importance: the
    mean absolute prediction change (on the probability scale for classification,
    on the raw target scale for regression).
    
    The calculator operates on an existing Explainer instance, leveraging its pre-loaded 
    components (model, graph data, and task objects) to ensure consistency and efficiency.
    """
    
    def __init__(self, explainer, model=None, data=None, task=None, col_stats_dict=None,
                 test_loader=None, loader_config=None,
                 model_type: str = "hgs",
                 relgt_tokens_kwargs: dict = None,
                 test_subset_idx=None):
        """
        Initialize with an Explainer instance and explicit model/data components.
        
        Args:
            explainer: Explainer instance (should contain model, data, task, etc.)
            model: Model instance (optional, will use explainer.model if not provided)
            data: Graph data (optional, will use explainer.data if not provided)
            task: Task object (optional, will use explainer.task if not provided)
            col_stats_dict: Column statistics (optional, will use explainer.col_stats_dict
                            if not provided)
            test_loader: Test data loader (optional)
            loader_config: Loader configuration dict (optional, will use
                           explainer.loader_config if not provided)
        """
        self.explainer = explainer
        self.model = model or getattr(explainer, 'model', None)
        self.data = data or getattr(explainer, 'data', None)
        self.task = task or getattr(explainer, 'task', None)
        
        if self.model is None and hasattr(self.task, 'model'):
            self.model = self.task.model
            
        self.device = getattr(explainer, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.col_stats_dict = col_stats_dict or getattr(explainer, 'col_stats_dict', None)
        self.test_loader = test_loader or getattr(explainer, 'test_loader', None)
        self.loader_config = loader_config or getattr(explainer, 'loader_config', None)
        
        # Standard protocol for ablation (same for both task types)
        self.standard_protocol = {
            'numerical': 'Mean',
            'categorical': 'MFV',
            'timestamp': 'Mean',
            'text': 'Mean'
        }
        
        self.seed = self.loader_config.get('seed', 42) if self.loader_config else 42
        self.model_type = model_type
        self.relgt_tokens_kwargs = relgt_tokens_kwargs or {}
        # Optional array of test-table row positions to evaluate (from
        # Utils.load_test_subset). None evaluates the full test set. Set for tasks
        # whose test set is too large to explain in full, so that fidelity is
        # measured on exactly the rows the explanations were computed on.
        self.test_subset_idx = test_subset_idx

    def _get_eval_labels(self, test_table) -> np.ndarray:
        """Return ground-truth labels for `test_table`, guaranteed present.

        RelBench masks the label column on the default
        `task.get_table("test")` (mask_input_cols=True) to keep leaderboard
        submissions honest — this is original RelBench behavior, unchanged
        by the RelBench 3 migration, not a bug. Re-fetch unmasked when
        needed; same rows/order, just with the label column present.
        """
        if self.task.target_col in test_table.df.columns:
            return test_table.df[self.task.target_col].values
        unmasked_df = self.task.get_table("test", mask_input_cols=False).df
        # Keep the same rows as the (already subset) test_table, otherwise labels
        # would cover the full test set while predictions cover only the subset.
        if self.test_subset_idx is not None:
            unmasked_df = unmasked_df.iloc[self.test_subset_idx]
        return unmasked_df[self.task.target_col].values

    def _compute_metrics_dict(self, task_type: str, y_true: np.ndarray,
                              y_pred: np.ndarray, threshold: float = 0.5,
                              classification_metric: str = "importance_score") -> dict:
        """Local replacement for task.evaluate(pred).

        RelBench 3's relbench.metrics module was pruned down to only
        roc_auc, map, and make_nmae — the 'mae' key this file has always
        read from task.evaluate()'s result was removed outright. Computed
        directly here with sklearn instead — reproduces the same formulas
        RelBench used to report (verified against stored metadata from
        pre-migration runs, matching to within floating-point noise), so
        `metrics[metric]` lookups throughout this file keep working exactly
        as before.

        roc_auc is always computed for classification — it's a ranking
        metric, invariant to any monotonic transform of the scores (sigmoid
        included), so it needs no threshold. "accuracy" (thresholding
        y_pred at a fixed cutoff before comparing to y_true) is only
        computed when classification_metric == "label_flip" — a single
        fixed threshold doesn't reflect these tasks' actual decision
        boundaries well, so it's opt-in rather than the default. Default
        ("importance_score") skips it entirely in favor of the threshold-free
        fidelity_importance metric.
        """
        metrics = {}
        if task_type == "classification":
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred))
            if classification_metric == "label_flip":
                metrics["accuracy"] = float(accuracy_score(y_true, (y_pred >= threshold).astype(int)))
        else:
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        return metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_inference(self, test_table) -> np.ndarray:
        """Run model inference on the (possibly ablated) graph and return
        predictions as a CPU numpy array.

        Handles both HGS (NeighborLoader) and RelGT (RelGTTokens + DataLoader)
        based on self.model_type.
        """
        fix_randomness(self.seed)
        torch.use_deterministic_algorithms(True)
        _inference_start = time.time()

        if self.model_type == "hgs":
            # Fidelity comparisons need bit-exact reproducibility across
            # ablation variants. NeighborLoader's default temporal_strategy
            # ("uniform") samples a random subset of neighbors when a node
            # exceeds the num_neighbors budget, using an RNG that
            # fix_randomness can't reach. Removing the budget instead risks
            # CUDA OOM on high-degree nodes, so use temporal_strategy="last":
            # same memory bound as training, but neighbors are chosen
            # deterministically by recency.
            deterministic_loader_config = dict(self.loader_config)
            deterministic_loader_config["temporal_strategy"] = "last"
            test_loader = create_consistent_loader(
                data=self.data, loader_config=deterministic_loader_config,
                task=self.task, table=test_table, split_name="test"
            )
            preds = []
            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Inference"):
                    batch = batch.to(self.device)
                    pred  = self.model(batch, self.task.entity_table)
                    pred  = pred.view(-1) if pred.size(1) == 1 else pred
                    preds.append(pred.cpu())
            result = torch.cat(preds).numpy()
            _elapsed = time.time() - _inference_start
            print(f"⏱  [Inference pass (HGS)] completed in {_elapsed:.2f}s ({_elapsed/60:.2f} min)")
            return result

        else:
            # RelGT — use RelGTTokens + DataLoader
            RelGTTokens = self.relgt_tokens_kwargs.get("RelGTTokens")
            if RelGTTokens is None:
                raise ValueError(
                    "For model_type='transformer', pass RelGTTokens class via "
                    "relgt_tokens_kwargs={'RelGTTokens': RelGTTokens, ...}"
                )
            from torch.utils.data import DataLoader
            token_kwargs = {k: v for k, v in self.relgt_tokens_kwargs.items()
                            if k != "RelGTTokens"}
            ds = RelGTTokens(split="test", precompute=True, **token_kwargs)

            # RelGTTokens builds itself from the FULL test split (it reads the task
            # table itself and pulls neighbors from the precomputed HDF5 cache by
            # position), so restrict it to the evaluation subset here rather than
            # shortening the table — a shortened table would silently read the
            # first N cached rows instead of the subset's rows. Subset has no
            # .collate, hence capturing it first.
            collate_fn = ds.collate
            if self.test_subset_idx is not None:
                from torch.utils.data import Subset
                ds = Subset(ds, [int(i) for i in self.test_subset_idx])
            loader = DataLoader(
                ds,
                batch_size=self.loader_config.get("batch_size", 64),
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=self.loader_config.get("num_workers_loader", 2),
                persistent_workers=True,
                pin_memory=(self.device.type == "cuda"),
            )
            preds, idxs = [], []
            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(loader, desc="Inference"):
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(self.device)
                    grouped_tf_dict = {
                        "grouped_tfs":     batch["grouped_tfs"],
                        "grouped_indices": batch["grouped_indices"],
                        "flat_batch_idx":  batch["flat_batch_idx"],
                        "flat_nbr_idx":    batch["flat_nbr_idx"],
                    }
                    pred = self.model(
                        batch["neighbor_types"], batch["node_indices"],
                        batch["neighbor_hops"],  batch["neighbor_times"],
                        grouped_tf_dict,
                        edge_index=batch["edge_index"], batch=batch["batch"],
                    )
                    pred = pred.view(-1) if pred.size(-1) == 1 else pred
                    preds.append(pred.detach().cpu())
                    idxs.append(batch["global_idx"].cpu())
            preds = torch.cat(preds).numpy()
            idxs  = torch.cat(idxs).numpy()
            # global_idx is the row's position in the FULL test table. When
            # evaluating a subset, remap it to its position within the subset —
            # otherwise these indices run past the end of `preds` (IndexError, or
            # silent misordering).
            if self.test_subset_idx is not None:
                position_of = {int(g): i for i, g in enumerate(self.test_subset_idx)}
                idxs = np.array([position_of[int(i)] for i in idxs])
            ordered = np.empty_like(preds)
            ordered[idxs] = preds
            _elapsed = time.time() - _inference_start
            print(f"⏱  [Inference pass (RelGT)] completed in {_elapsed:.2f}s ({_elapsed/60:.2f} min)")
            return ordered

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _get_top_features_from_profile(self, profile_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """
        Select the top-k features from a profile DataFrame by row order.

        The profile CSV is already in canonical order (rank ascending, A-Z within
        ties) — so the first top_k rows ARE the top-k features. No score column
        needed, no re-ranking.

        profile_df must have columns: feature, abs_weight, bin, rank
        Returns a DataFrame with table_name, column_name, feature_type, feature_name.
        """
        top = profile_df.head(top_k).copy()

        def _parse(feat):
            parts = str(feat).split(".")
            table  = parts[0] if len(parts) >= 1 else "unknown"
            col    = parts[1] if len(parts) >= 2 else feat
            ftype  = parts[2] if len(parts) >= 3 else "numerical"
            return table, col, ftype

        top["table_name"]   = top["feature"].apply(lambda f: _parse(f)[0])
        top["column_name"]  = top["feature"].apply(lambda f: _parse(f)[1])
        top["feature_type"] = top["feature"].apply(lambda f: _parse(f)[2])
        top["feature_name"] = top["feature"]
        return top

    def calculate_fidelity_plus_from_profile(
        self,
        profile_csv: "str | Path",
        baseline_predictions: np.ndarray,
        test_table,
        method_name: str,
        top_k: int = 15,
        metric: str = "roc_auc",
        threshold: float = 0.5,
        task_type: str = "classification",
        classification_metric: str = "importance_score",
    ) -> dict:
        print(f"\n{'='*80}")
        print(f"FIDELITY+ from profile | Top-{top_k} | {method_name}")
        print(f"{'='*80}")
        _fpp_start_time = time.time()

        profile_df   = pd.read_csv(profile_csv)
        top_features = self._get_top_features_from_profile(profile_df, top_k)

        if len(top_features) == 0:
            raise ValueError(f"No features found in profile: {profile_csv}")

        print(f"Method: {method_name}")
        print(f"Top feature: {top_features.iloc[0]['feature_name']}")
        print(f"table: {top_features.iloc[0]['table_name']}, col: {top_features.iloc[0]['column_name']}")

        print(f"Top {len(top_features)} features (profile order):")
        for i, row in top_features.reset_index(drop=True).iterrows():
            print(f"  {i+1}. {row['feature_name']:50s} | rank={row['rank']:.1f}")

        backups = {}
        for _, row in top_features.iterrows():
            t, c = row["table_name"], row["column_name"]
            backups[(t, c)] = backup_column(self.data, t, c)

        print(f"\nAblating Top-{top_k} (standard protocol: Mean/MFV)...")
        for _, row in top_features.iterrows():
            try:
                strategy = self.standard_protocol.get(row["feature_type"], "Mean")
                _, ablation_func = AblationStrategies.get_strategy(row["feature_type"], strategy)
                ablation_func(data=self.data, node_type=row["table_name"],
                              col_name=row["column_name"],
                              col_stats_dict=self.col_stats_dict)
            except Exception as e:
                print(f"  Warning: Could not ablate {row['feature_name']}: {e}")

        # Without this try/finally, an exception in _run_inference() here
        # leaves self.data ablated for every later method/k/range evaluated
        # in run_fidelity()'s shared loop.
        try:
            print("Getting model predictions on ablated graph...")
            ablated_preds = self._run_inference(test_table)
        finally:
            print("Restoring original data...")
            for (t, c), bkp in backups.items():
                restore_column(self.data, t, c, bkp)

        if task_type == "classification":
            resolved_ablated_preds = expit(ablated_preds)
        else:
            resolved_ablated_preds = ablated_preds

        y_true = self._get_eval_labels(test_table)
        baseline_metrics = self._compute_metrics_dict(task_type, y_true, baseline_predictions, threshold, classification_metric)
        ablated_metrics  = self._compute_metrics_dict(task_type, y_true, resolved_ablated_preds, threshold, classification_metric)
        baseline_score   = baseline_metrics[metric]
        ablated_score    = ablated_metrics[metric]

        if task_type == "classification":
            # 1. fidelity_metric
            fidelity_metric     = baseline_score - ablated_score
            fidelity_metric_pct = (fidelity_metric / baseline_score * 100) if baseline_score != 0 else 0.0

            # 2. fidelity_importance — mean |Δprobability|, threshold-free;
            # same formula as the regression branch, on the probability scale.
            baseline_probs = baseline_predictions
            ablated_probs  = resolved_ablated_preds
            fidelity_importance = float(np.abs(ablated_probs - baseline_probs).mean())

            # 3. fidelity_prob
            fidelity_prob = float((baseline_probs - ablated_probs).mean())

            results = {
                "k":                    top_k,
                "num_features_ablated": len(top_features),
                "baseline_metric":      baseline_score,
                "ablated_metric":       ablated_score,
                "fidelity_metric":      fidelity_metric,
                "fidelity_metric_pct":  fidelity_metric_pct,
                "fidelity_importance":  fidelity_importance,
                "fidelity_prob":        fidelity_prob,
            }

            # 4. fidelity_acc / fidelity_flip — threshold-dependent, so
            # computed only when classification_metric == "label_flip".
            if classification_metric == "label_flip":
                baseline_acc = baseline_metrics["accuracy"]
                ablated_acc  = ablated_metrics["accuracy"]
                results["fidelity_acc"] = baseline_acc - ablated_acc

                baseline_labels = (baseline_probs >= threshold).astype(int)
                ablated_labels  = (ablated_probs  >= threshold).astype(int)
                label_flips     = (baseline_labels != ablated_labels).sum()
                results["fidelity_flip"] = label_flips / len(baseline_predictions)

        else:  # regression
            
            # 1. fidelity_metric
            fidelity_metric     = ablated_score - baseline_score
            fidelity_metric_pct = (fidelity_metric / baseline_score * 100) if baseline_score != 0 else 0.0

            # 2. fidelity_importance
            fidelity_importance = float(np.abs(ablated_preds - baseline_predictions).mean())

            # 3. fidelity_prob
            max_baseline      = float(baseline_predictions.max())
            fidelity_prob_raw = float((ablated_preds - baseline_predictions).mean())
            fidelity_prob     = fidelity_prob_raw / max_baseline if max_baseline else 0.0

            results = {
                "k":                    top_k,
                "num_features_ablated": len(top_features),
                "baseline_metric":      baseline_score,
                "ablated_metric":       ablated_score,
                "fidelity_metric":      fidelity_metric,
                "fidelity_metric_pct":  fidelity_metric_pct,
                "fidelity_importance":        fidelity_importance,
                "fidelity_prob_mean":   fidelity_prob,
                "fidelity_prob":        fidelity_prob_raw,
            }

        print(f"\nFIDELITY+ results:")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
        _fpp_elapsed = time.time() - _fpp_start_time
        print(f"⏱  [calculate_fidelity_plus_from_profile | k={top_k} | {method_name}] completed in "
              f"{_fpp_elapsed:.2f}s ({_fpp_elapsed/60:.2f} min)")
        return results


    def calculate_fidelity_minus_from_profile(
            self,
            profile_csv: "str | Path",
            baseline_predictions: np.ndarray,
            test_table,
            method_name: str,
            top_k: int = 15,
            metric: str = "roc_auc",
            threshold: float = 0.5,
            task_type: str = "classification",
            classification_metric: str = "importance_score",
        ) -> dict:
            print(f"\n{'='*80}")
            print(f"FIDELITY- from profile | Keeping Top-{top_k} | {method_name}")
            print(f"{'='*80}")
            _fmp_start_time = time.time()

            profile_df   = pd.read_csv(profile_csv)
            top_features = self._get_top_features_from_profile(profile_df, top_k)
            all_features = self._get_top_features_from_profile(profile_df, len(profile_df))

            all_backups = {}
            for _, row in all_features.iterrows():
                t, c = row["table_name"], row["column_name"]
                all_backups[(t, c)] = backup_column(self.data, t, c)

            try:
                print(f"Ablating ALL {len(all_features)} features...")
                for _, row in all_features.iterrows():
                    try:
                        strategy = self.standard_protocol.get(row["feature_type"], "Mean")
                        _, ablation_func = AblationStrategies.get_strategy(
                            row["feature_type"], strategy)
                        ablation_func(data=self.data, node_type=row["table_name"],
                                    col_name=row["column_name"],
                                    col_stats_dict=self.col_stats_dict)
                    except Exception as e:
                        print(f"  Warning: Could not ablate {row['table_name']}.{row['column_name']}: {e}")

                print(f"Restoring ONLY the top {len(top_features)} features (profile order)...")
                for _, row in top_features.iterrows():
                    t, c = row["table_name"], row["column_name"]
                    restore_column(self.data, t, c, all_backups[(t, c)])

                print(f"Getting predictions using ONLY Top-{top_k} features...")
                suffix_preds = self._run_inference(test_table)

            finally:
                print(f"Restoring ALL {len(all_features)} features...")
                for (t, c), bkp in all_backups.items():
                    restore_column(self.data, t, c, bkp)

            if task_type == "classification":
                resolved_suffix_preds = expit(suffix_preds)
            else:
                resolved_suffix_preds = suffix_preds

            y_true = self._get_eval_labels(test_table)
            baseline_metrics = self._compute_metrics_dict(task_type, y_true, baseline_predictions, threshold, classification_metric)
            suffix_metrics   = self._compute_metrics_dict(task_type, y_true, resolved_suffix_preds, threshold, classification_metric)
            baseline_score   = baseline_metrics[metric]
            suffix_score     = suffix_metrics[metric]

            if task_type == "classification":
                # 1. fidelity_metric
                fidelity_metric           = baseline_score - suffix_score
                fidelity_metric_pct       = (fidelity_metric / baseline_score * 100) if baseline_score != 0 else 0.0
                one_minus_fidelity_metric = 1.0 - fidelity_metric

                # 2. fidelity_importance — mean |Δprobability|, threshold-free;
                # same formula as the regression branch, on the probability scale.
                baseline_probs = baseline_predictions
                suffix_probs   = resolved_suffix_preds
                fidelity_importance           = float(np.abs(suffix_probs - baseline_probs).mean())
                one_minus_fidelity_importance = 1.0 - fidelity_importance

                # 3. fidelity_prob
                fidelity_prob           = float((baseline_probs - suffix_probs).mean())
                one_minus_fidelity_prob = 1.0 - fidelity_prob

                results = {
                    "k":                            top_k,
                    "num_features_kept":            len(top_features),
                    "baseline_metric":              baseline_score,
                    "sufficient_metric":            suffix_score,
                    "fidelity_metric":              fidelity_metric,
                    "fidelity_metric_pct":          fidelity_metric_pct,
                    "one_minus_fidelity_metric":    one_minus_fidelity_metric,
                    "fidelity_importance":                fidelity_importance,
                    "one_minus_fidelity_importance":      one_minus_fidelity_importance,
                    "fidelity_prob":                fidelity_prob,
                    "one_minus_fidelity_prob":      one_minus_fidelity_prob,
                }

                # 4. fidelity_acc / fidelity_flip — threshold-dependent, so
                # computed only when classification_metric == "label_flip".
                if classification_metric == "label_flip":
                    baseline_acc = baseline_metrics["accuracy"]
                    suffix_acc   = suffix_metrics["accuracy"]
                    fidelity_acc = baseline_acc - suffix_acc
                    results["fidelity_acc"] = fidelity_acc
                    results["one_minus_fidelity_acc"] = 1.0 - fidelity_acc

                    baseline_labels = (baseline_probs >= threshold).astype(int)
                    suffix_labels   = (suffix_probs   >= threshold).astype(int)
                    label_flips     = (baseline_labels != suffix_labels).sum()
                    fidelity_flip   = label_flips / len(baseline_predictions)
                    results["fidelity_flip"] = fidelity_flip
                    results["one_minus_fidelity_flip"] = 1.0 - fidelity_flip

            else:  # regression
                
                # 1. fidelity_metric
                fidelity_metric           = suffix_score - baseline_score
                fidelity_metric_pct       = (fidelity_metric / baseline_score * 100) if baseline_score != 0 else 0.0
                one_minus_fidelity_metric = 1.0 - fidelity_metric

                # 2. fidelity_importance
                fidelity_importance           = float(np.abs(suffix_preds - baseline_predictions).mean())
                one_minus_fidelity_importance = 1.0 - fidelity_importance

                # 3. fidelity_prob
                max_baseline                 = float(baseline_predictions.max())
                fidelity_prob_raw            = float((suffix_preds - baseline_predictions).mean())
                fidelity_prob                = fidelity_prob_raw / max_baseline if max_baseline else 0.0
                one_minus_fidelity_prob_mean = 1.0 - fidelity_prob
                one_minus_fidelity_prob      = 1.0 - fidelity_prob_raw

                results = {
                    "k":                                top_k,
                    "num_features_kept":                len(top_features),
                    "baseline_metric":                  baseline_score,
                    "sufficient_metric":                suffix_score,
                    "fidelity_metric":                  fidelity_metric,
                    "fidelity_metric_pct":              fidelity_metric_pct,
                    "one_minus_fidelity_metric":        one_minus_fidelity_metric,
                    "fidelity_importance":                    fidelity_importance,
                    "one_minus_fidelity_importance":          one_minus_fidelity_importance,
                    "fidelity_prob_mean":               fidelity_prob,
                    "fidelity_prob":                    fidelity_prob_raw,
                    "one_minus_fidelity_prob_mean":     one_minus_fidelity_prob_mean,
                    "one_minus_fidelity_prob":          one_minus_fidelity_prob,
                }

            print(f"\nFIDELITY- results:")
            for k, v in results.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
            _fmp_elapsed = time.time() - _fmp_start_time
            print(f"⏱  [calculate_fidelity_minus_from_profile | k={top_k} | {method_name}] completed in "
                  f"{_fmp_elapsed:.2f}s ({_fmp_elapsed/60:.2f} min)")
            return results