"""
Single-Column Ablation-Based Feature Importance Explainer

This module provides a single-column ablation method for explaining model predictions by ablating 
(modifying) one feature at a time and measuring the impact on model predictions.

Usage:
    from singleColumnAblation_explainer import AblationExplainer, AblationStrategies
    
    # Initialize the explainer
    explainer = AblationExplainer(
        model=model,
        data=data,
        task=task,
        col_stats_dict=col_stats_dict,
        device=device
    )
    
    # Configure ablation strategies
    strategies = {
        'numerical': ['zeroing', 'mean_imputation', 'permutation'],
        'text': ['zero_vector', 'random_noise'],
        'timestamp': ['max_time', 'permutation'],
        'categorical': ['zero_vector', 'mode_imputation', 'permutation']
    }
    
    # Run ablation experiments
    results = explainer.run_ablation_experiments(
        strategies=strategies,
        test_loader_or_table=test_table,
        baseline_predictions=baseline_preds
    )
    
    # Save results
    results.to_csv('ablation_results.csv', index=False)
"""

import os
import sys
import time
from contextlib import contextmanager

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from Utils.Utils import create_consistent_loader, fix_randomness, backup_column, restore_column

import torch
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Callable, Any
from torch_frame.data import StatType
# Metrics are computed locally with sklearn instead of via task.evaluate()
# — see _compute_metric() for why.
from sklearn.metrics import roc_auc_score, mean_absolute_error


@contextmanager
def timed_section(label: str):
    """Time a block of code and print how long it took.

    Usage:
        with timed_section("numerical Mean"):
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


def apply_sigmoid(preds):
    """Convert raw logits to probabilities via sigmoid.
    Accepts either a numpy array or a torch.Tensor.
    """
    if isinstance(preds, torch.Tensor):
        return torch.sigmoid(preds)
    return 1.0 / (1.0 + np.exp(-preds))

# ============================================================================
# ABLATION FUNCTIONS
# ============================================================================

def ablate_to_zero(data, node_type, col_name, col_stats_dict):
    """
    Replaces column values with 0.0 (for numerical/text embedding columns).
    Handles MultiEmbeddingTensor and regular tensors properly.
    """
    tf = data[node_type].tf
    stype_key = next(s for s, cols in tf.col_names_dict.items() if col_name in cols)
    tensor = tf.feat_dict[stype_key]
    idx = tf.col_names_dict[stype_key].index(col_name)

    # --- MultiEmbeddingTensor columns (e.g. embedding-backed stypes) ---
    # IMPORTANT: check for .offset/.values on the FULL stype tensor, not on
    # a per-column slice (tensor[:, idx]). torch_frame's MultiEmbeddingTensor
    # stores all columns for this stype concatenated along the last dim,
    # with `offset` (length num_cols + 1, offset[0] == 0) marking each
    # column's boundary within the shared `values` buffer. Slicing a single
    # column out via tensor[:, idx] returns a *derived* MultiEmbeddingTensor
    # whose own .offset is not guaranteed to satisfy the same invariant
    # (offset[0] == 0) that torch_frame asserts on later, e.g. when a
    # DataLoader worker does a row-wise index_select on the *original*
    # tensor. Mutating that derived slice's .values in place (the previous
    # approach) left the parent tensor's offset/values in a state that
    # triggered `assert self.offset[0] == 0` inside torch_frame during
    # subsequent batch fetches.
    #
    # The safe fix: never construct or mutate a derived per-column slice.
    # Zero this column's embedding block directly within the full tensor's
    # own values buffer, using the full tensor's own offset boundaries —
    # nothing about offset itself changes, only the values in that column's
    # range are zeroed, so torch_frame's invariants stay intact.
    if hasattr(tensor, 'offset') and hasattr(tensor, 'values'):
        start = int(tensor.offset[idx])
        end = int(tensor.offset[idx + 1])
        tensor.values[:, start:end] = 0.0
        return

    # --- Regular (non-embedding) tensors ---
    target_slice = tensor[:, idx]
    try:
        target_slice.fill_(0.0)
    except (AttributeError, RuntimeError):
        zeros = torch.zeros_like(target_slice)
        target_slice.copy_(zeros)

def ablate_by_permutation(data, node_type, col_name, col_stats_dict):
    """Randomly shuffles column values to break correlation."""
    # Case 1: Handle direct attributes (e.g., timestamps) not stored in the feature dict
    if col_name in data[node_type]:
        if hasattr(data[node_type], col_name) and isinstance(data[node_type][col_name], torch.Tensor):
            ts_tensor = data[node_type][col_name]
            ts_np = ts_tensor.cpu().numpy()
            np.random.shuffle(ts_np)
            # Use in-place copy to ensure the graph object maintains the modification
            ts_tensor.copy_(torch.from_numpy(ts_np).to(ts_tensor.device).to(ts_tensor.dtype))
            return
    
    # Case 2: Handle regular features stored within the TabularFrame (tf)
    tf = data[node_type].tf
    # Identify the semantic type (stype) and index of the target column
    stype_key = next(s for s, cols in tf.col_names_dict.items() if col_name in cols)
    tensor = tf.feat_dict[stype_key]
    idx = tf.col_names_dict[stype_key].index(col_name)

    # Some 'numerical'/'categorical'/'timestamp'-labeled columns can still
    # turn out to be embedding-backed (MultiEmbeddingTensor) under the hood
    # (same edge case handled defensively in ablate_to_zero above). Permuting
    # via a derived tensor[:, idx] slice risks the same offset/values
    # desync — write into the parent's own values buffer via its own
    # offset boundaries when that's the case.
    if hasattr(tensor, 'offset') and hasattr(tensor, 'values'):
        start = int(tensor.offset[idx])
        end = int(tensor.offset[idx + 1])
        column_values = tensor.values[:, start:end]
        perm = torch.randperm(column_values.size(0), device=column_values.device)
        tensor.values[:, start:end] = column_values[perm]
        return

    # Extract the specific column slice for shuffling
    target_slice = tensor[:, idx]
    original_dtype = target_slice.dtype
    np_arr = target_slice.cpu().numpy()
    np.random.shuffle(np_arr)
    
    # Re-inject the shuffled data back into the original tensor slice
    shuffled_tensor = torch.from_numpy(np_arr).to(target_slice.device).to(original_dtype)
    target_slice.copy_(shuffled_tensor)


def ablate_text_random_noise(data, node_type, col_name, col_stats_dict):
    """Replaces text embeddings with random noise."""
    tf = data[node_type].tf
    stype_key = next(s for s, cols in tf.col_names_dict.items() if col_name in cols)
    tensor = tf.feat_dict[stype_key]
    idx = tf.col_names_dict[stype_key].index(col_name)

    # Mirrors ablate_to_zero's safe pattern (see its comment above): never
    # mutate a derived tensor[:, idx] slice's .values, since that slice's own
    # .offset is not guaranteed to satisfy the offset[0] == 0 invariant
    # torch_frame relies on. Write directly into the parent's own values
    # buffer using its own offset boundaries instead.
    if hasattr(tensor, 'offset') and hasattr(tensor, 'values'):
        start = int(tensor.offset[idx])
        end = int(tensor.offset[idx + 1])
        noise = torch.randn(
            tensor.values.size(0), end - start,
            device=tensor.values.device, dtype=tensor.values.dtype,
        )
        tensor.values[:, start:end] = noise
        return

    # Regular (non-embedding) tensor
    target_slice = tensor[:, idx]
    if hasattr(target_slice, 'values'):
        underlying_tensor = target_slice.values
    else:
        underlying_tensor = target_slice

    noise = torch.randn_like(underlying_tensor)
    underlying_tensor.copy_(noise)

def ablate_timestamp_max_time(data, node_type, col_name, col_stats_dict):
    """
    Replaces all values in a timestamp column with the maximum observed value.
    
    This function handles the complex 3D structure of timestamp tensors in torch_frame,
    where each entry is typically a vector of length 7 [year, month, day, hour, etc.].
    """
    try:
        # 1. Retrieve the maximum value from statistics
        # We default to 2100.0 if the MAX stat is missing
        stats = col_stats_dict.get(node_type, {}).get(col_name, {})
        max_val = stats.get(StatType.MAX, 2100.0)
        
        # 2. Access the TensorFrame (tf) from the data object
        if not hasattr(data[node_type], 'tf'):
            return
            
        tf = data[node_type].tf
        # Identify which semantic type (stype) the column belongs to
        stype_key = next((s for s, cols in tf.col_names_dict.items() if col_name in cols), None)
        
        if stype_key is not None:
            tensor = tf.feat_dict[stype_key]
            idx = tf.col_names_dict[stype_key].index(col_name)
            
            # 3. Handle 3D Timestamp Tensors [Batch_Size, Num_Cols, 7]
            if tensor.dim() == 3:
                # Slice all rows for the specific column, keeping the 7-length vector dimension
                target_slice = tensor[:, idx, :] 
                
                # Check if the max_val is a scalar (just the year) or a full vector
                if isinstance(max_val, (int, float, torch.Tensor)) and (not hasattr(max_val, 'dim') or max_val.dim() == 0):
                    # Convert scalar max_year into a compatible 7-element vector [year, 1, 1, 0, 0, 0, 0]
                    fill_vector = torch.tensor([float(max_val), 1, 1, 0, 0, 0, 0], 
                                             device=target_slice.device, 
                                             dtype=target_slice.dtype)
                else:
                    # Use the provided max_val if it's already an array/tensor
                    fill_vector = torch.as_tensor(max_val, device=target_slice.device, dtype=target_slice.dtype)
                
                # 4. Use expand_as to broadcast the single fill_vector across all rows (N)
                # This ensures the entire column is filled with the max timestamp vector
                target_slice.copy_(fill_vector.expand_as(target_slice))
                
            else:
                # Fallback for 2D tensors (standard numerical representation of time)
                target_slice = tensor[:, idx]
                fill_scalar = float(max_val) if hasattr(max_val, 'item') else max_val
                target_slice.fill_(fill_scalar)

    except Exception as e:
        # Catch and log potential errors during the ablation process
        print(f"Warning: Could not ablate timestamp {col_name} to max in {node_type}: {e}")

# ============================================================================
# ABLATION FUNCTIONS ALIGNED WITH KERNEL SHAP
# ============================================================================

def ablate_numerical_mean(data, node_type, col_name, col_stats_dict):
    """
    Replaces numerical column with the mean from the training set.
    Matches Kernel SHAP logic: StatType.MEAN with NaN check.
    """
    tf = data[node_type].tf
    stype_key = next(s for s, cols in tf.col_names_dict.items() if col_name in cols)
    tensor = tf.feat_dict[stype_key]
    idx = tf.col_names_dict[stype_key].index(col_name)
    target_slice = tensor[:, idx]
    
    stats = col_stats_dict.get(node_type, {}).get(col_name, {})
    mask_val = stats.get(StatType.MEAN, 0.0)
    
    # NaN handling to match Kernel SHAP's approach of treating NaN means as zeros
    if isinstance(mask_val, float) and np.isnan(mask_val):
        mask_val = 0.0
    
    # Ensure it's a scalar for full_like
    if hasattr(mask_val, 'item'):
        mask_val = mask_val.item()
        
    fill_value = torch.full_like(target_slice, mask_val)
    target_slice.copy_(fill_value)


def ablate_categorical_mode(data, node_type, col_name, col_stats_dict):
    """
    Replaces categorical column with the most frequent value (Mode).
    Matches Kernel SHAP logic: Index of max count from StatType.COUNT.
    """
    tf = data[node_type].tf
    stype_key = next(s for s, cols in tf.col_names_dict.items() if col_name in cols)
    tensor = tf.feat_dict[stype_key]
    idx = tf.col_names_dict[stype_key].index(col_name)
    target_slice = tensor[:, idx]
    
    stats = col_stats_dict.get(node_type, {}).get(col_name, {})
    count_data = stats.get(StatType.COUNT)
    
    mode_val = 0 # Default fallback
    if count_data is not None and isinstance(count_data, tuple) and len(count_data) == 2:
        values, counts = count_data
        if len(counts) > 0:
            # Find the index corresponding to the max count (Kernel SHAP logic)
            mode_val = np.argmax(counts)
            
    fill_value = torch.full_like(target_slice, int(mode_val))
    target_slice.copy_(fill_value)


def ablate_timestamp_median(data, node_type, col_name, col_stats_dict):
    """
    Replaces timestamp with the global median time tensor.
    Matches Kernel SHAP logic: StatType.MEDIAN_TIME.
    """
    tf = data[node_type].tf
    stype_key = next(s for s, cols in tf.col_names_dict.items() if col_name in cols)
    tensor = tf.feat_dict[stype_key] # This is typically a 3D tensor [N, Num_Cols, 7]
    idx = tf.col_names_dict[stype_key].index(col_name)
    
    stats = col_stats_dict.get(node_type, {}).get(col_name, {})
    # Default is a tensor of zeros with length 7
    default_ts = torch.tensor([0]*7, dtype=torch.float32)
    mask_val = stats.get(StatType.MEDIAN_TIME, default_ts)
    
    # Target slice for timestamp is usually [N, 7]
    target_slice = tensor[:, idx, :]
    
    # Ensure mask_val is on the correct device and matches dimensions
    mask_tensor = mask_val.to(target_slice.device).to(target_slice.dtype)
    
    # Broadcasting the [7] mask across all [N] rows
    target_slice.copy_(mask_tensor.expand_as(target_slice))


def ablate_text_mean_embedding(data, node_type, col_name, col_stats_dict=None):
    """
    Replace text embeddings with their mean embedding.
    Handles both regular tensors and MultiEmbeddingTensor.
    """
    tf = data[node_type].tf
    stype_key = None
    for s, cols in tf.col_names_dict.items():
        if col_name in cols:
            stype_key = s
            break
    
    if stype_key is None:
        raise ValueError(f"Column {col_name} not found in {node_type}")
    
    tensor = tf.feat_dict[stype_key]
    idx = tf.col_names_dict[stype_key].index(col_name)

    try:
        # Never mutate a derived tensor[:, idx] slice's .values — the same
        # unsafe pattern explained in ablate_to_zero's comment above. That
        # slice's own .offset doesn't necessarily satisfy the offset[0] == 0
        # invariant torch_frame depends on. Write into the parent's own
        # values buffer via its own offset boundaries instead.
        if hasattr(tensor, 'offset') and hasattr(tensor, 'values'):
            start = int(tensor.offset[idx])
            end = int(tensor.offset[idx + 1])
            column_values = tensor.values[:, start:end]

            if column_values.numel() > 0:
                mean_embedding = column_values.mean(dim=0, keepdim=True)
                tensor.values[:, start:end] = mean_embedding.expand(
                    column_values.size(0), end - start
                )
            else:
                print(f"  Warning: Empty embedding column for {node_type}.{col_name}, using zeros")
                tensor.values[:, start:end] = 0.0

        else:
            # Regular tensor - compute mean and replicate
            target_slice = tensor[:, idx]
            if len(target_slice.shape) == 2:
                # Shape: [num_nodes, embedding_dim]
                mean_embedding = target_slice.mean(dim=0, keepdim=True)
                replacement = mean_embedding.expand_as(target_slice)
                target_slice.copy_(replacement)
            else:
                # Unexpected shape, zero it out
                print(f"  Warning: Unexpected tensor shape {target_slice.shape}, using zeros")
                target_slice.zero_()

    except Exception as e:
        print(f"  Warning: Failed to compute mean embedding for {node_type}.{col_name}: {e}")
        # Fallback: zero out on error — write via the parent's own
        # offset/values when embedding-backed, same safe pattern as above.
        if hasattr(tensor, 'offset') and hasattr(tensor, 'values'):
            start = int(tensor.offset[idx])
            end = int(tensor.offset[idx + 1])
            tensor.values[:, start:end] = 0.0
        else:
            tensor[:, idx].zero_()
            
# ============================================================================
# ABLATION STRATEGIES REGISTRY
# ============================================================================

class AblationStrategies:
    """Registry of available ablation strategies for each data type."""
    
    STRATEGIES = {
        'numerical': {
            'Zeroing': ('Zeroing', ablate_to_zero),
            'Mean': ('Mean', ablate_numerical_mean),
            'Permutation': ('Permutation', ablate_by_permutation)
        },
        'text': {
            'Zeroing': ('Zeroing', ablate_to_zero),
            'Random Noise': ('Random Noise', ablate_text_random_noise),
            'Mean': ('Mean', ablate_text_mean_embedding)
        },
        'timestamp': {
            'Max Time': ('Max Time', ablate_timestamp_max_time),
            'Permutation': ('Permutation', ablate_by_permutation),
            'Mean': ('Mean', ablate_timestamp_median)
        },
        'categorical': {
            'Zeroing': ('Zeroing', ablate_to_zero),
            'Permutation': ('Permutation', ablate_by_permutation),
            'MFV': ('MFV', ablate_categorical_mode)
        }
    }
    
    @classmethod
    def get_strategy(cls, data_type: str, strategy_name: str) -> Tuple[str, Callable]:
        """Get a specific ablation strategy."""
        if data_type not in cls.STRATEGIES:
            raise ValueError(f"Unknown data type: {data_type}")
        if strategy_name not in cls.STRATEGIES[data_type]:
            raise ValueError(f"Unknown strategy '{strategy_name}' for type '{data_type}'")
        return cls.STRATEGIES[data_type][strategy_name]
    
    @classmethod
    def list_strategies(cls, data_type: str) -> List[str]:
        """List available strategies for a data type."""
        if data_type not in cls.STRATEGIES:
            raise ValueError(f"Unknown data type: {data_type}")
        return list(cls.STRATEGIES[data_type].keys())


# ============================================================================
# KERNEL-SHAP-ALIGNED STRATEGY (single source of truth)
# ============================================================================
# The one ablation method per feature type whose masking logic exactly
# matches KernelSHAP's / MonteCarloShapley's own coalition-masking
# (see ablate_numerical_mean, ablate_text_mean_embedding,
# ablate_timestamp_median, ablate_categorical_mode above — each docstring
# says "Matches Kernel SHAP logic"). Both `run_singleColumnAblation_explainer.py`'s
# `--strategies fast` preset (computes ONLY these) and its
# post-hoc `extract_strategy_results` filter (extracts just these rows out
# of a full `--strategies all` run, for apples-to-apples comparison against
# Kernel SHAP / Monte Carlo Shapley) both derive from this single dict so
# they can never drift out of sync with each other.
KERNEL_SHAP_STRATEGY = {
    'numerical':   'Mean',
    'text':        'Mean',
    'timestamp':   'Mean',
    'categorical': 'MFV',
}


# ============================================================================
# MAIN EXPLAINER CLASS
# ============================================================================

class AblationExplainer:
    """
    Single-column ablation-based feature importance explainer.
    
    This class provides a single-column ablation method for explaining model predictions by ablating
    (modifying) one feature at a time and measuring the impact on predictions.

    Supports two task types:

    Binary classification (task_type='classification'):
        Importance is measured via multiple complementary signals:
          - label flips (num_label_flips, label_flip_rate)
          - mean absolute probability change (importance_score)
          - rank-order disruption via Spearman correlation (corr_spearman,
            spearman_importance = 1 - |corr|)
          - rank-order disruption via Kendall correlation (corr_kendall,
            kendall_importance = 1 - |corr|)
        Results columns: feature_name, table_name, column_name, feature_type,
                         ablation_method_used, importance_score, num_label_flips,
                         label_flip_rate, corr_spearman, corr_kendall,
                         spearman_importance, kendall_importance, metric_score.

    Regression (task_type='regression'):
        Importance is measured as mean(|ablated_pred - baseline_pred|).
        This is computed per-sample, then averaged across the test set.
        Results columns: feature_name, table_name, column_name, feature_type,
                         ablation_method_used, importance_score, metric_score.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        data: Any,
        task: Any,
        col_stats_dict: Dict[str, Dict[str, Dict]],
        device: torch.device,
        loader_config: Dict, 
        entity_table: str = None,
        task_type: str = 'classification',
        test_subset_idx=None,
    ):
        """
        Initialize the ablation explainer.
        
        Args:
            model: Trained PyTorch model
            data: HeteroData object containing the graph data
            task: RelBench task object
            col_stats_dict: Dictionary containing column statistics
            device: PyTorch device (cpu or cuda)
            loader_config: Configuration dictionary for data loaders (e.g., batch size, seed,
                           shuffling). Ensures consistent graph sampling during baseline and
                           ablated evaluations.
            entity_table: Name of the entity table (auto-detected if None)
            task_type: 'classification' for classification tasks, 'regression' for regression tasks
            test_subset_idx: Optional array of test-table row positions to explain
                             (from Utils.load_test_subset). None explains the full
                             test set. Used for tasks whose test set is too large
                             to explain in full, so that every explainer explains
                             exactly the same records.
        """
        
        self.model = model
        self.data = data
        self.task = task
        self.col_stats_dict = col_stats_dict
        self.device = device
        self.loader_config = loader_config 
        self.entity_table = entity_table or task.entity_table
        self.task_type = task_type
        self.test_subset_idx = test_subset_idx
        
        self.model.eval()

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Local replacement for task.evaluate(...)[tune_metric].

        RelBench 3's relbench.metrics module was pruned down to only
        roc_auc, map, and make_nmae — the 'mae' (and rmse/r2/f1/mrr/...)
        metrics this explainer has always tracked for regression tasks were
        removed outright, and task.evaluate()'s default metric set changed
        along with it (regression tasks now default to NMAE, a differently
        scaled metric, not MAE). Calling self.task.evaluate(pred)[tune_metric]
        now raises KeyError: 'mae' on every single call. Since the broad
        except Exception around each ablation's evaluation step silently
        swallows that and skips the row, this previously meant EVERY
        ablation in a regression run silently failed and no results were
        ever collected — but with no crash, so the failure only surfaced
        much later as an empty output CSV. Computed directly here with
        sklearn instead — reproduces the same formulas RelBench used to
        report (verified against stored metadata from pre-migration runs,
        matching to within floating-point noise).
        """
        if self.task_type == "classification":
            return float(roc_auc_score(y_true, y_pred))
        return float(mean_absolute_error(y_true, y_pred))

    def create_loader(self, table, split_name: str = "test") -> "NeighborLoader":
        """Create a NeighborLoader for the given table and split."""
        # Explanation methods compare predictions across many ablation
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
        # draw — same bound as training, no RNG involved.
        deterministic_loader_config = dict(self.loader_config)
        deterministic_loader_config["temporal_strategy"] = "last"
        return create_consistent_loader(
            data=self.data,
            loader_config=deterministic_loader_config,
            task=self.task,
            table=table,
            split_name=split_name
        )

    @torch.no_grad()
    def test(self, loader: "NeighborLoader") -> np.ndarray:
        """Run inference on a data loader and return predictions.
        For classification, returns probabilities (sigmoid applied).
        For regression, returns raw values.
        Consistent with train.py behaviour.
        """
        torch.use_deterministic_algorithms(True)
        self.model.eval()
        pred_list = []
        
        for batch in tqdm(loader, desc="Testing", leave=False):
            batch = batch.to(self.device)
            pred = self.model(batch, self.entity_table)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu())
        
        result = torch.cat(pred_list, dim=0).numpy()

        if self.task_type == "classification":
            result = apply_sigmoid(result)

        if np.isnan(result).any():
            print(f"WARNING: {np.isnan(result).sum()} NaN values detected in predictions!")
        
        return result
    
    def build_feature_map(
        self,
        node_to_col_names_dict: Dict[str, Dict[str, List[str]]],
        exclude_keys: set = None
    ) -> Dict[Tuple[str, str], str]:
        """
        Build a map of features to ablate.
        
        Args:
            node_to_col_names_dict: Dictionary mapping node types to column names by type
            exclude_keys: Set of (node_type, col_name) tuples to exclude
            
        Returns:
            Dictionary mapping (node_type, col_name) to column type
        """
        exclude_keys = exclude_keys or set()
        feature_map = {}
        
        for node_type, cols_by_type in node_to_col_names_dict.items():
            for col_type in ['numerical', 'text', 'timestamp', 'categorical']:
                if col_type in cols_by_type:
                    for col_name in cols_by_type[col_type]:
                        feature_key = (node_type, col_name)
                        if feature_key not in exclude_keys:
                            feature_map[feature_key] = col_type
        
        return feature_map
    
    def run_ablation_experiments(
        self,
        strategies: Dict[str, List[str]],
        test_loader_or_table: Any,
        baseline_predictions: np.ndarray,
        feature_map: Dict[Tuple[str, str], str] = None,
        node_to_col_names_dict: Dict[str, Dict[str, List[str]]] = None,
        exclude_keys: set = None,
        threshold: float = 0.5,
        tune_metric: str = 'roc_auc'
    ) -> pd.DataFrame:
        """
        Run ablation experiments for all features.
        
        Args:
            strategies: Dictionary mapping data types to list of strategy names
            test_loader_or_table: Test loader or test table object
            baseline_predictions: Baseline model predictions (logits for classification, raw values
                                  for regression)
            feature_map: Pre-built feature map (optional, auto-built if None)
            node_to_col_names_dict: Node to column names mapping (required if feature_map
                                    is None)
            exclude_keys: Set of (node_type, col_name) tuples to exclude
            threshold: Decision threshold for binary classification (ignored for regression)
            tune_metric: Metric to track (e.g., 'roc_auc' or 'mae')
            
        Returns:
            DataFrame with ablation results. Columns differ by task type:
              - classification:     feature_name, table_name, column_name, feature_type,
                            ablation_method_used, num_label_flips, label_flip_rate,
                            pos_to_neg_flips, neg_to_pos_flips, avg_prob_change, metric_score
              - regression: feature_name, table_name, column_name, feature_type,
                            ablation_method_used, importance_score, metric_score
        """
        # Build feature map if not provided
        if feature_map is None:
            if node_to_col_names_dict is None:
                raise ValueError("Either feature_map or node_to_col_names_dict must be provided")
            feature_map = self.build_feature_map(node_to_col_names_dict, exclude_keys)

        _ablation_start_time = time.time()
        
        # Resolve test table
        is_loader = type(test_loader_or_table).__name__ == "NeighborLoader"
        if not is_loader:
            test_table = test_loader_or_table
        else:
            test_table = self.task.get_table("test")
        
        # Build experiment combinations
        experiment_combinations = []
        for data_type, strategy_names in strategies.items():
            for strategy_name in strategy_names:
                strategy_display_name, ablation_func = AblationStrategies.get_strategy(
                    data_type, strategy_name
                )
                experiment_combinations.append({
                    'group_name': f"{data_type} {strategy_display_name}",
                    'data_type': data_type,
                    'strategy_name': strategy_name,
                    'ablation_func': ablation_func
                })
        
        # Pre-compute baseline quantities shared across all experiments.
        # baseline_predictions are already probabilities (sigmoid applied by
        # train.py / AblationExplainer.test() before saving / returning).
        if self.task_type == 'classification':
            baseline_probs = baseline_predictions
            baseline_labels = (baseline_probs >= threshold).astype(int)

        # Resolve ground-truth labels ONCE, up front — they don't change
        # across ablations, and re-fetching per iteration would be wasteful
        # (and, over hundreds of ablations, slow). RelBench masks the target
        # column on the default task.get_table("test") (mask_input_cols=
        # True) to keep leaderboard submissions honest — this is original
        # RelBench behavior, not something the RelBench 3 migration changed
        # — so re-fetch unmasked when needed.
        if self.task.target_col in test_table.df.columns:
            y_true = test_table.df[self.task.target_col].values
        else:
            unmasked_df = self.task.get_table("test", mask_input_cols=False).df
            # Keep the same rows as the (already subset) test_table, otherwise
            # y_true would cover the full test set while predictions cover only
            # the subset.
            if self.test_subset_idx is not None:
                unmasked_df = unmasked_df.iloc[self.test_subset_idx]
            y_true = unmasked_df[self.task.target_col].values
        
        # Run experiments
        all_results = []

        for combo in experiment_combinations:
            experiment_name = combo['group_name']
            print(f"\n{'='*60}")
            print(f"Running Group Experiment: {experiment_name}")
            print(f"{'='*60}")

            with timed_section(f"Group experiment: {experiment_name}"):
                for (node_type, col_name), col_type in feature_map.items():
                    # Skip if this column type doesn't match the current experiment
                    if col_type != combo['data_type']:
                        continue

                    ablation_func = combo['ablation_func']
                    strategy_name = combo['group_name'].split(' ', 1)[1]  # Get display name

                    print(f"   Executing Ablation: {node_type}.{col_name} using method {strategy_name}")

                    # Backup column
                    try:
                        column_backup = backup_column(self.data, node_type, col_name)
                    except Exception as e:
                        print(f"   Warning: Could not backup {col_name}: {e}. Skipping.")
                        continue

                    # Apply ablation
                    try:
                        ablation_func(
                            data=self.data,
                            node_type=node_type,
                            col_name=col_name,
                            col_stats_dict=self.col_stats_dict
                        )
                    except Exception as e:
                        print(f"   Warning: Ablation failed for {col_name}: {e}. Restoring and skipping.")
                        restore_column(self.data, node_type, col_name, column_backup)
                        continue

                    # Create loader — must see the freshly ablated data
                    ablation_test_loader = self.create_loader(test_table, "test")
                    print(f"      → Loader created with {len(ablation_test_loader)} batches")
                    fix_randomness(self.loader_config.get('seed', 42))

                    try:
                        ablation_pred = self.test(ablation_test_loader)
                        test_score = self._compute_metric(y_true, ablation_pred)

                        if self.task_type == 'regression':
                            # --------------------------------------------------------
                            # Regression importance:
                            #   importance_score = mean( |ablated_pred - baseline_pred| )
                            # --------------------------------------------------------
                            importance_score = np.abs(ablation_pred - baseline_predictions).mean()

                            print(f"      → Importance score (mean |Δpred|): {importance_score:.4f}")
                            print(f"      → {tune_metric}: {test_score:.4f}")

                            all_results.append({
                                'feature_name': f"{node_type}.{col_name}.{col_type}",
                                'table_name': node_type,
                                'column_name': col_name,
                                'feature_type': col_type,
                                'ablation_method_used': strategy_name,
                                'importance_score': float(importance_score),
                                'metric_score': float(test_score),
                            })

                        else:
                            # --------------------------------------------------------
                            # Binary classification importance: label flips, prob change,
                            # and rank-order disruption (Spearman / Kendall correlations).
                            # --------------------------------------------------------
                            importance_score = np.abs(ablation_pred - baseline_probs).mean()

                            # Label flips
                            ablation_labels = (ablation_pred >= threshold).astype(int)
                            label_flips_mask = (baseline_labels != ablation_labels)
                            num_label_flips = label_flips_mask.sum()
                            label_flip_rate = num_label_flips / len(baseline_predictions)

                            print(f"      → Importance score (mean |Δprob|): {importance_score:.4f}")
                            print(f"      → Label flip rate (at threshold {threshold:.4f}): {label_flip_rate:.3f} ({num_label_flips} flips)")
                            print(f"      → {tune_metric}: {test_score:.4f}")

                            all_results.append({
                                'feature_name': f"{node_type}.{col_name}.{col_type}",
                                'table_name': node_type,
                                'column_name': col_name,
                                'feature_type': col_type,
                                'ablation_method_used': strategy_name,
                                'importance_score': float(importance_score),
                                'num_label_flips': int(num_label_flips),
                                'label_flip_rate': float(label_flip_rate),
                                'metric_score': float(test_score),
                            })

                    except Exception as e:
                        print(f"      Evaluation failed: {type(e).__name__}: {e}. Skipping.")

                    finally:
                        # Always restore the column regardless of success/failure
                        restore_column(self.data, node_type, col_name, column_backup)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Sort by the appropriate importance column (descending for both:
        # more label flips = more important; higher mean |Δpred| = more important)
        if not results_df.empty:
            sort_col = 'num_label_flips' if self.task_type == 'classification' else 'importance_score'
            results_df = results_df.sort_values(sort_col, ascending=False)

        _ablation_elapsed = time.time() - _ablation_start_time
        print(f"\n⏱  [run_ablation_experiments TOTAL] completed in {_ablation_elapsed:.2f}s "
              f"({_ablation_elapsed/60:.2f} min)")
        return results_df
    
    def save_results(self, results_df: pd.DataFrame, output_path: str):
        """Save results to a CSV file."""
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    @staticmethod
    def extract_strategy_results(input_csv_path: str, output_csv_path: str, strategy_mapping: Dict[str, str]):
        """
        Filters the full ablation results CSV based on a specific strategy mapping
        and saves a simplified version for comparison.
        
        Args:
            input_csv_path: Path to the full results (all strategy combinations).
            output_csv_path: Where to save the filtered strategy file.
            strategy_mapping: A dict like {'numerical': 'Mean', 'text': 'Zero-Out', ...}
        """
        
        # 1. Load the full results
        df = pd.read_csv(input_csv_path)
        
        # 2. Build the filter mask
        masks = []
        for feat_type, method in strategy_mapping.items():
            masks.append((df['feature_type'] == feat_type) & (df['ablation_method_used'] == method))
        
        # Combine all masks with OR
        final_mask = masks[0]
        for m in masks[1:]:
            final_mask |= m
            
        filtered_df = df[final_mask].copy()
        
        # 3. Save the filtered file
        filtered_df.to_csv(output_csv_path, index=False)
        print(f"Strategy-specific file saved to: {output_csv_path}")
        return filtered_df


# ============================================================================
# RELGT ABLATION EXPLAINER
# ============================================================================

class RelGTAblationExplainer(AblationExplainer):
    """
    Ablation explainer for the RelGT (transformer) model.

    Overrides create_loader() and test() to use RelGTTokens + standard
    DataLoader instead of NeighborLoader, matching the RelGT forward
    pass interface.

    All ablation logic (feature backup/restore, importance scoring,
    result collection) is inherited unchanged from AblationExplainer.
    """

    def __init__(self, *args, relgt_tokens_kwargs: dict, **kwargs):
        """
        Args:
            relgt_tokens_kwargs: Dict passed directly to RelGTTokens.
                Must contain: data, task, K, precomputed_dir.
                Optional: undirected, num_workers.
            All other args/kwargs forwarded to AblationExplainer.
        """
        super().__init__(*args, **kwargs)
        self._relgt_tokens_kwargs = relgt_tokens_kwargs

    def create_loader(self, table, split_name: str = "test"):
        """Build a RelGTTokens DataLoader for the given split."""
        from torch.utils.data import DataLoader

        # Import here so the base class has no hard dependency on RelGT
        try:
            from rdl_explain_files.model.relgt import RelGTTokens
        except ImportError as e:
            raise ImportError(
                "RelGTTokens not found. Make sure rdl-explain-relgt-integration/src "
                "is on sys.path."
            ) from e

        # RelGTTokens derives its token representations from the raw feature
        # tensors at CONSTRUCTION time, so precompute=False risks reading a
        # stale representation from before the column was ablated — making
        # ablation a silent no-op for RelGT.
        #
        # Force RelGTTokens's OWN internal precompute parallelism to a real
        # worker count: RelGTTokens.__init__ guards _precompute_sampling()
        # behind `if os.path.exists(precomputed_path)`, so multiprocessing.Pool
        # is only ever created on a genuine cache miss — num_workers is unused
        # (harmless) when the cache exists, and must be >=1 when it doesn't
        # (Pool(processes=0) raises ValueError).
        token_kwargs = dict(self._relgt_tokens_kwargs)
        token_kwargs["num_workers"] = max(1, self._relgt_tokens_kwargs.get("num_workers", 4) or 4)

        ds = RelGTTokens(
            split=split_name,
            precompute=True,
            **token_kwargs,
        )

        batch_size = self.loader_config.get("batch_size", 64)

        # A nonzero worker count here is unsafe: run_ablation_experiments()
        # mutates self.data in place (ablate -> new loader -> infer ->
        # restore) and builds a BRAND NEW DataLoader (spawning fresh worker
        # processes) for every single feature. A worker forked while
        # self.data is mid-ablation (or racing the restore step for the
        # previous feature) can end up with a torn/inconsistent view of a
        # MultiEmbeddingTensor's values/offset pair, which is exactly what
        # surfaces downstream as AssertionError: offset[0] == 0 inside
        # torch_frame's validate().
        num_workers = 0

        # RelGTTokens always builds itself from the full split (it reads the task
        # table itself and ignores `table`), so restrict it to the explanation
        # subset here. Subset has no .collate, hence capturing it first.
        collate_fn = ds.collate
        if self.test_subset_idx is not None and split_name == "test":
            from torch.utils.data import Subset
            ds = Subset(ds, [int(i) for i in self.test_subset_idx])

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=False,
            pin_memory=(self.device.type == "cuda"),
        )
        return loader

    @torch.no_grad()
    def test(self, loader) -> np.ndarray:
        self.model.eval()
        pred_list, idx_list = [], []

        for batch in tqdm(loader, desc="Testing", leave=False):
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
                batch["neighbor_types"],
                batch["node_indices"],
                batch["neighbor_hops"],
                batch["neighbor_times"],
                grouped_tf_dict,
                edge_index=batch["edge_index"],
                batch=batch["batch"],
            )
            pred = pred.view(-1) if pred.size(-1) == 1 else pred

            if self.task_type == "classification":
                pred = torch.sigmoid(pred) 

            pred_list.append(pred.detach().cpu().numpy())
            idx_list.append(batch["global_idx"].detach().cpu().numpy())

        preds = np.concatenate(pred_list)
        idxs  = np.concatenate(idx_list)

        # global_idx is the row's position in the FULL test table. When explaining
        # a subset, remap it to its position within the subset — otherwise these
        # indices run past the end of `preds` (IndexError, or silent misordering).
        if self.test_subset_idx is not None:
            position_of = {int(g): i for i, g in enumerate(self.test_subset_idx)}
            idxs = np.array([position_of[int(i)] for i in idxs])

        ordered = np.empty_like(preds)
        ordered[idxs.astype(int)] = preds
        return ordered