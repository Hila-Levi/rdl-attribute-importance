import torch
import numpy as np
import pandas as pd
import time
import warnings
import copy
import os
import sys
import gc
from contextlib import contextmanager
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
from collections import OrderedDict
from torch_frame.data import StatType

# Get the root path to import Utils
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from Utils.Utils import create_consistent_loader, fix_randomness, backup_column, restore_column

def apply_sigmoid(preds):
    """Convert raw logits to probabilities via sigmoid.
    Accepts either a numpy array or a torch.Tensor.
    Consistent with train.py and train_trans.py.
    """
    if isinstance(preds, torch.Tensor):
        return torch.sigmoid(preds)
    return 1.0 / (1.0 + np.exp(-preds))

@contextmanager
def timed_section(label: str):
    """Time a block of code and print how long it took.

    Usage:
        with timed_section("Constructing feature space"):
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


class MonteCarloShapley:
    """
    Standalone Monte Carlo Shapley Value calculator using random permutation sampling.
    
    This class calculates Shapley values by:
    1. Sampling random permutations of features
    2. For each permutation, adding features one by one
    3. Computing marginal contribution of each feature
    4. Averaging contributions across all permutations
    
    Uses importance-score aggregation: marginal contribution = change in
    mean absolute prediction shift vs the original data. Uses the SAME
    formula for classification and regression — classification no longer
    counts label flips against a threshold.

    Key features:
    - Completely standalone (no external class dependencies)
    - Time-based checkpoints for controllable runtime
    - Can be stopped anytime (Ctrl+C) and returns current results
    - Incremental statistics (memory-efficient)
    - Shares infrastructure methods with GlobalKernelSHAP for consistency
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 data: Any,
                 col_stats_dict: Dict[str, Dict[str, Dict]],
                 entity_table: str,
                 device: torch.device,
                 loader_config: Dict,
                 task_type: str = 'classification',
                 test_subset_idx=None):
        """
        Initialize Monte Carlo Shapley calculator.
        
        Args:
            model: Trained GNN model
            data: HeteroData object containing the graph
            col_stats_dict: Dictionary containing column statistics
            entity_table: Name of the entity table (target nodes)
            device: torch device
            loader_config: Standardized loader configuration
            task_type: 'classification' or 'regression'
        """
        self.model = model
        self.data = data
        self.col_stats_dict = col_stats_dict
        self.entity_table = entity_table
        self.device = device
        self.loader_config = loader_config
        self.task_type = task_type
        # Optional array of test-table row positions to explain (from
        # Utils.load_test_subset). None explains the full test set. Used for tasks
        # whose test set is too large to explain in full, so that every explainer
        # explains exactly the same records.
        self.test_subset_idx = test_subset_idx
        
        # Internal storage
        self.feature_names = []
        self.feature_mapping = OrderedDict()
        self.M = 0
        self.global_masks = {}
        
        # Storage for running statistics
        self.running_mean = None
        self.running_m2 = None  # For variance computation (Welford's algorithm)
        self.num_permutations = 0
        
        # Dedicated RNG for permutation sampling, entropy-seeded (NOT the
        # fixed `loader_config['seed']`). `numpy.random.Generator` maintains
        # its own internal state, completely independent of the LEGACY
        # global RNG that `np.random.seed()` / `fix_randomness()` controls.
        # This matters because `fix_randomness(seed)` is called repeatedly
        # inside every f_model_all_predictions call (46+ times per
        # permutation) to keep model inference deterministic for fair
        # coalition comparisons - and since that resets the legacy global RNG
        # to the exact same fixed seed every time, drawing permutations via
        # the legacy `np.random.permutation(...)` would return to that same
        # reset state on every call too, making every "random" permutation
        # identical across the entire run (real, differentiated per-feature
        # values, but zero variance across permutations - exactly the
        # symptom this was built to fix). Using an isolated Generator here
        # means fix_randomness's resets elsewhere can never touch it.
        self._permutation_rng = np.random.default_rng()

        # Initialize feature space and masks
        with timed_section("Construct feature space"):
            self._construct_feature_space()
        with timed_section("Compute global masks"):
            self._compute_global_masks()
    
    def _construct_feature_space(self):
        """
        Construct M feature space.
        Format: table_name.column_name.type
        """
        print("\n" + "=" * 70)
        print("CONSTRUCTING FEATURE SPACE M")
        print("=" * 70)
        
        feature_idx = 0
        
        for node_type in self.data.node_types:
            if not hasattr(self.data[node_type], 'tf'):
                continue
            
            tf = self.data[node_type].tf
            if not hasattr(tf, 'col_names_dict'):
                continue
            
            print(f"\n Processing {node_type}:")
            
            for stype_key, col_names in tf.col_names_dict.items():
                stype_str = str(stype_key)
                
                # Determine column type
                if 'numerical' in stype_str.lower():
                    col_type = 'numerical'
                elif 'categorical' in stype_str.lower():
                    col_type = 'categorical'
                elif 'embedding' in stype_str.lower() or 'text' in stype_str.lower():
                    col_type = 'text'
                elif 'timestamp' in stype_str.lower():
                    col_type = 'timestamp'
                else:
                    continue
                
                for col_idx_in_stype, col_name in enumerate(col_names):
                    feature_name = f"{node_type}.{col_name}.{col_type}"
                    self.feature_names.append(feature_name)
                    self.feature_mapping[feature_idx] = {
                        'node_type': node_type,
                        'col_name': col_name,
                        'col_type': col_type,
                        'stype_key': stype_key,
                        'col_idx_in_stype': col_idx_in_stype,
                        'num_cols_in_stype': len(col_names)
                    }
                    print(f"   [{feature_idx}] {feature_name}")
                    feature_idx += 1
        
        self.M = len(self.feature_names)
        print(f"\n Total features M = {self.M}")
    
    def _compute_global_masks(self):
        """
        Compute global masking values for each column.
        """
        print("\n" + "=" * 70)
        print("COMPUTING GLOBAL MASKING VALUES")
        print("=" * 70)
        
        for node_type in self.data.node_types:
            if not hasattr(self.data[node_type], 'tf'):
                continue
            
            tf = self.data[node_type].tf
            if not hasattr(tf, 'col_names_dict'):
                continue
            
            self.global_masks[node_type] = {}
            print(f"\n {node_type}:")
            
            for stype_key, col_names in tf.col_names_dict.items():
                stype_str = str(stype_key).lower()
                
                # Determine column type and masking key
                col_type = None
                if 'numerical' in stype_str:
                    col_type = 'numerical'
                    stat_key, default_val = StatType.MEAN, 0.0
                elif 'timestamp' in stype_str:
                    col_type = 'timestamp'
                    stat_key, default_val = StatType.MEDIAN_TIME, torch.tensor([0]*7)
                elif 'categorical' in stype_str:
                    col_type = 'categorical'
                    stat_key, default_val = StatType.COUNT, 0
                elif 'embedding' in stype_str or 'text' in stype_str:
                    col_type = 'text'
                
                if col_type is None:
                    continue
                
                # Handle Numerical and Timestamp
                if col_type in ['numerical', 'timestamp']:
                    self.global_masks[node_type][col_type] = {}
                    
                    for col_name in col_names:
                        stats = self.col_stats_dict.get(node_type, {}).get(col_name, {})
                        mask_val = stats.get(stat_key, default_val)
                        
                        if isinstance(mask_val, float) and np.isnan(mask_val):
                            print("   Warning: NaN detected in stats, defaulting to 0.0")
                            mask_val = 0.0
                        
                        self.global_masks[node_type][col_type][col_name] = mask_val
                        
                        if col_type == 'timestamp':
                            print(f"   Timestamp '{col_name}': Global Median Time Tensor (Mask)")
                        else:
                            print(f"   Numerical '{col_name}': Global Mean = {mask_val:.4f}")
                
                # Handle Categorical
                elif col_type == 'categorical':
                    self.global_masks[node_type][col_type] = {}
                    
                    for col_name in col_names:
                        stats = self.col_stats_dict.get(node_type, {}).get(col_name, {})
                        count_data = stats.get(StatType.COUNT)
                        mode_val = 0
                        
                        if count_data is not None and isinstance(count_data, tuple) and len(count_data) == 2:
                            values, counts = count_data
                            if len(counts) > 0:
                                max_count_index = np.argmax(counts)
                                mode_val = max_count_index
                        
                        self.global_masks[node_type][col_type][col_name] = int(mode_val)
                        print(f"   Categorical '{col_name}': Global MFV (Index) = {int(mode_val)}")
                
                # Handle Text/Embedding
                elif col_type == 'text':
                    self.global_masks[node_type]['text'] = {}

                    if hasattr(tf, 'feat_dict') and stype_key in tf.feat_dict:
                        stype_tensor = tf.feat_dict[stype_key]

                        # Compute the mean PER COLUMN (using
                        # stype_tensor[:, col_idx], which gives each column
                        # its own, correctly-shaped embedding) instead of
                        # one mean for the whole stype_key group stored under
                        # every col_name — that single-column-wide vector
                        # only lines up with downstream re-slicing for
                        # col_idx == 0; every later column's slice would fall
                        # past the end.
                        for col_idx, col_name in enumerate(col_names):
                            try:
                                column_slice = stype_tensor[:, col_idx]
                                if hasattr(column_slice, 'values'):
                                    tensor_to_sample = column_slice.values.to(self.device)
                                elif isinstance(column_slice, torch.Tensor):
                                    tensor_to_sample = column_slice.to(self.device)
                                else:
                                    raise ValueError("Unsupported embedding format or missing data")

                                if tensor_to_sample.numel() == 0:
                                    raise ValueError("Embedding tensor is empty")

                                embed_dim = tensor_to_sample.shape[-1]
                                num_samples = min(tensor_to_sample.shape[0], 10000)

                                if num_samples > 0:
                                    fix_randomness(self.loader_config.get('seed', 42))
                                    indices = torch.randperm(tensor_to_sample.shape[0])[:num_samples].to(tensor_to_sample.device)
                                    mean_embedding = tensor_to_sample[indices].mean(dim=0).to(self.device)
                                else:
                                    print("   Warning: No samples available, using zero embedding")
                                    mean_embedding = torch.zeros(embed_dim, dtype=torch.float32).to(self.device)

                                self.global_masks[node_type]['text'][col_name] = mean_embedding
                                print(f"   Text '{col_name}': Global Mean Embedding shape = {mean_embedding.shape} (Computed)")

                            except Exception as e:
                                print(f"   Warning: Could not compute mean for {node_type}, {col_name}: {e}")

                                try:
                                    print("   Using fallback zero embedding")
                                    stats = self.col_stats_dict.get(node_type, {}).get(col_name, {})
                                    embed_dim = stats.get(StatType.EMB_DIM, 300)
                                    mean_embedding = torch.zeros(embed_dim, dtype=torch.float32).to(self.device)
                                    self.global_masks[node_type]['text'][col_name] = mean_embedding
                                    print(f"   Text '{col_name}': Using zero embedding (fallback) shape={mean_embedding.shape}")
                                except Exception as e_fallback:
                                    print(f"   FATAL: Failed to create fallback zero embedding for {node_type}, {col_name}: {e_fallback}")
                                    continue
    
    
    def create_coalition_data(self, z_prime: np.ndarray) -> np.ndarray:
        """
        Returns the coalition vector unchanged. Actual masking happens
        directly on `self.data` inside `f_model_all_predictions` via
        backup_column/restore_column (see class docstring notes on
        MonteCarloShapleyRelGT for the full rationale).

        NOTE: this used to `copy.deepcopy(self.data)` and mask the copy.
        That was found to silently corrupt multi-column `MultiEmbeddingTensor`
        objects (torch_frame's storage for concatenated text/embedding
        columns) - `copy.deepcopy` does not reliably preserve their full
        `.values` width, so slicing a later column (anything past index 0)
        after deepcopy could silently return an empty (N, 0) tensor instead
        of raising - and worse, this mechanism's batch-override step in the
        old f_model_all_predictions wrapped that lookup in a bare
        `except Exception: pass`, so the corruption could fail completely
        silently with no warning at all. Backup/restore never deep-copies
        anything, so it can't hit this failure mode.
        """
        return np.asarray(z_prime).copy()

    def _apply_feature_mask(self, data_obj: Any, feat_idx: int, is_present: Any) -> None:
        """
        Mutates data_obj IN PLACE, masking a single feature (by index) with
        its precomputed global mask value, if is_present indicates the
        feature is absent (0). No-op if the feature is present (1).

        Extracted from create_coalition_data so the exact same, already-
        validated masking logic can also be applied directly to `self.data`
        (the single canonical object) rather than only to a deep-copied one -
        see MonteCarloShapleyRelGT.f_model_all_predictions, which uses this
        via backup_column/restore_column instead of copy.deepcopy for
        performance (matches FidelityCalculator's proven ablation pattern).
        """
        if feat_idx not in self.feature_mapping:
            return

        feat_details = self.feature_mapping[feat_idx]
        node_type = feat_details['node_type']
        col_type = feat_details['col_type']
        col_name = feat_details['col_name']
        stype_key = feat_details['stype_key']
        col_idx_in_stype = feat_details['col_idx_in_stype']

        if node_type not in data_obj.node_types or \
        not hasattr(data_obj[node_type], 'tf') or \
        stype_key not in data_obj[node_type].tf.feat_dict:
            return
        
        tf = data_obj[node_type].tf
        tensor = tf.feat_dict[stype_key]
        
        # Get mask value
        mask_dict = self.global_masks.get(node_type, {}).get(col_type, {})
        mask_value = mask_dict.get(col_name)

        if mask_value is None:
            return

        z_scalar = float(is_present)
        if z_scalar == 1.0:
            return  # Feature present, do nothing

        # Text Feature Masking
        if col_type == 'text':
            try:
                # `tensor` is a MultiEmbeddingTensor with shape (N, num_text_cols).
                try:
                    from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
                    is_met = isinstance(tensor, MultiEmbeddingTensor)
                except ImportError:
                    is_met = False

                # mask_value is the correctly per-column-shaped mean embedding
                # computed in _compute_global_masks, so it's used directly
                # without re-slicing — write into the parent tensor's own
                # values buffer using its own offset boundaries.
                if is_met and hasattr(tensor, 'offset') and hasattr(tensor, 'values'):
                    # CRITICAL: never replace tensor.values with a new tensor object.
                    # Replacing it makes .values inconsistent with .offset, so
                    # _row_index_select computes a broken offset and validate()
                    # raises AssertionError(offset[0] == 0) in the DataLoader worker.
                    # Instead, modify .values IN-PLACE with copy_() — offset is
                    # never touched and stays consistent.
                    start = int(tensor.offset[col_idx_in_stype])
                    end = int(tensor.offset[col_idx_in_stype + 1])
                    vals = tensor.values  # live reference, do not reassign
                    mean_embedding_slice = mask_value.to(self.device).to(vals.dtype)
                    patch = mean_embedding_slice.unsqueeze(0).expand(vals.size(0), end - start)
                    vals[:, start:end].copy_(patch)
                else:
                    # HGS / plain 2D tensor — mirror the original working code exactly.
                    target_slice = tensor[:, col_idx_in_stype]
                    if hasattr(target_slice, 'values'):
                        underlying = target_slice.values
                    else:
                        underlying = target_slice
                    original_dtype = underlying.dtype
                    mean_embedding_slice = mask_value.to(self.device).to(original_dtype)
                    underlying.copy_(mean_embedding_slice.unsqueeze(0).expand_as(underlying))

            except Exception as e:
                warnings.warn(f"Failed to mask text column {col_name} (index {col_idx_in_stype}): {e}")

        # Non-Text Feature Masking
        elif col_type in ['numerical', 'categorical', 'timestamp']:
            # Same safeguard as the text branch above: some 'categorical'/
            # 'numerical'-labeled columns can turn out to be embedding-backed
            # (MultiEmbeddingTensor) rather than plain tensors — write into
            # the parent tensor's own values/offset rather than a derived
            # column slice whenever that's the case.
            if hasattr(tensor, 'offset') and hasattr(tensor, 'values'):
                start = int(tensor.offset[col_idx_in_stype])
                end = int(tensor.offset[col_idx_in_stype + 1])

                if isinstance(mask_value, torch.Tensor):
                    mask_vec = mask_value.to(self.device).to(tensor.values.dtype).reshape(-1)
                else:
                    mask_vec = torch.full((end - start,), float(mask_value),
                                        device=self.device, dtype=tensor.values.dtype)
                tensor.values[:, start:end] = mask_vec.unsqueeze(0).expand(
                    tensor.values.size(0), end - start
                )
            else:
                target_slice = tensor[:, col_idx_in_stype]

                if isinstance(mask_value, torch.Tensor):
                    mask_value = mask_value.to(target_slice.dtype).to(self.device)
                    target_slice.copy_(mask_value.unsqueeze(0).expand_as(target_slice))
                else:
                    target_slice.fill_(mask_value)

    
    def create_loader(self, table: str, split_name: str = "test"):
        """
        Create a DataLoader for the specified table and split.
        
        Args:
            table: Node type to create loader for
            split_name: Split name ("train", "val", "test")
        
        Returns:
            loader: NeighborLoader
        """
        return create_consistent_loader(
            data=self.data,
            loader_config=self.loader_config,
            task=None,
            table=table,
            split_name=split_name
        )
    
    
    def f_model_all_predictions(self, loader: Any, data_coalition: Any, max_batches: int) -> np.ndarray:
        """
        Get ALL predictions for a coalition.

        Args:
            loader: NeighborLoader for this split
            data_coalition: raw binary coalition vector (from
                create_coalition_data - NOT a HeteroData object anymore)
            max_batches: Maximum batches to process

        Returns:
            predictions: Array of shape (total_nodes,) with ALL predictions

        Ablates self.data directly in place (backup -> mask -> infer ->
        restore, matching FidelityCalculator's proven pattern and
        MonteCarloShapleyRelGT). `loader` samples feature values from
        self.data live at batch-construction time, so masking self.data
        BEFORE iterating is reflected in every sampled batch automatically -
        no manual per-batch feature override needed (the old version's
        override step is also what silently swallowed failures via a bare
        `except Exception: pass`).
        """
        z_prime = np.asarray(data_coalition)
        backups: Dict[Tuple[str, str], torch.Tensor] = {}

        try:
            for feat_idx, is_present in enumerate(z_prime):
                if float(is_present) == 1.0 or feat_idx not in self.feature_mapping:
                    continue
                feat = self.feature_mapping[feat_idx]
                key = (feat['node_type'], feat['col_name'])
                if key not in backups:
                    backups[key] = backup_column(self.data, key[0], key[1])
                self._apply_feature_mask(self.data, feat_idx, is_present)

            torch.use_deterministic_algorithms(True)
            self.model.eval()
            all_predictions = []

            with torch.no_grad():
                fix_randomness(self.loader_config.get('seed', 42))
                for batch_idx, batch in enumerate(loader):
                    if max_batches is not None and batch_idx >= max_batches:
                        break

                    batch = batch.to(self.device)

                    pred = self.model(batch, self.entity_table)
                    pred = pred.view(-1) if pred.size(1) == 1 else pred
                    if self.task_type == "classification":
                        pred = apply_sigmoid(pred)
                    all_predictions.append(pred.detach().cpu().numpy())

            if not all_predictions:
                raise RuntimeError("MonteCarloShapley.f_model_all_predictions: No predictions generated.")

            result = np.concatenate(all_predictions, axis=0)

        finally:
            # CRITICAL: always undo the ablation, even on exception - self.data
            # is the single shared canonical object, so leaving it mutated
            # would silently corrupt every subsequent evaluation for the rest
            # of the run with no error raised.
            for (node_type, col_name), backup_tensor in backups.items():
                restore_column(self.data, node_type, col_name, backup_tensor)

        return result
    
    
    def compute_importance_score(self, predictions_coalition: np.ndarray,
                                 predictions_original: np.ndarray,
                                 threshold: float = 0.5) -> float:
        """
        Compute the importance score between two sets of predictions.

        Uses the SAME calculation for both classification and regression:
            f(z') = mean absolute prediction shift across all nodes.

        For classification, predictions are already sigmoid-applied
        probabilities (see f_model_all_predictions), so this measures the
        average change in predicted probability caused by ablating a
        feature — rather than counting label flips against a threshold.
        This keeps classification and regression on the same continuous,
        threshold-independent scale.

        Args:
            predictions_coalition: Predictions from coalition data  (num_nodes,)
            predictions_original:  Predictions from original data   (num_nodes,)
            threshold:              Unused. Kept only for backward-compatible call
                                    signatures; no longer affects the calculation
                                    since label-flip counting was replaced with
                                    mean absolute shift for both task types.

        Returns:
            importance_score: scalar float
        """
        return float(np.abs(predictions_coalition - predictions_original).mean())

    # Backward-compatible alias — old callers/scripts may still reference
    # compute_disagreement() by name. Delegates to compute_importance_score().
    def compute_disagreement(self, predictions_coalition: np.ndarray,
                            predictions_original: np.ndarray,
                            threshold: float = 0.5) -> float:
        return self.compute_importance_score(predictions_coalition, predictions_original, threshold)
    
    def _initialize_running_stats(self):
        """Initialize running statistics for incremental updates."""
        self.running_mean = np.zeros(self.M, dtype=np.float64)
        self.running_m2 = np.zeros(self.M, dtype=np.float64)
        self.num_permutations = 0
        print(f"\n Initialized running statistics for {self.M} features")
    
    def _update_running_stats(self, new_shapley_values: np.ndarray):
        """
        Update running mean and variance using Welford's online algorithm.
        
        Args:
            new_shapley_values: Shapley values from one permutation (M,)
        """
        self.num_permutations += 1
        delta = new_shapley_values - self.running_mean
        self.running_mean += delta / self.num_permutations
        delta2 = new_shapley_values - self.running_mean
        self.running_m2 += delta * delta2
    
    def _get_current_stats(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get current mean, variance, and standard error.
        
        Returns:
            mean: Current mean Shapley values (M,)
            variance: Current variance (M,)
            std_error: Current standard error (M,)
        """
        if self.num_permutations < 2:
            return self.running_mean.copy(), np.zeros(self.M), np.zeros(self.M)
        
        variance = self.running_m2 / (self.num_permutations - 1)
        std_error = np.sqrt(variance / self.num_permutations)
        
        return self.running_mean.copy(), variance, std_error
    
    
    def _compute_shapley_from_permutation(self,
                                          permutation: np.ndarray,
                                          loader: Any,
                                          max_batches: int,
                                          predictions_original: np.ndarray,
                                          disagreement_threshold: float = 0.5) -> np.ndarray:
        """
        Compute Shapley values for a single random permutation using
        importance-score aggregation.

        f(z') = mean absolute prediction shift vs the original data — the
        SAME formula for classification and regression (see
        compute_importance_score()).

        Args:
            permutation:           Random ordering of features (M,)
            loader:                DataLoader for evaluation
            max_batches:           Number of batches to process
            predictions_original:  Baseline predictions from original data (num_nodes,)
            disagreement_threshold: Unused — retained only for backward-compatible
                call signatures (see compute_importance_score()).

        Returns:
            shapley_values: Shapley value for each feature (M,)
        """
        shapley_values = np.zeros(self.M, dtype=np.float64)

        # Start with empty coalition
        current_coalition = np.zeros(self.M, dtype=int)

        # Baseline value with no features
        data_empty = self.create_coalition_data(current_coalition)
        pred_current = self.f_model_all_predictions(loader, data_empty, max_batches)
        value_current = self.compute_importance_score(pred_current, predictions_original, threshold=disagreement_threshold)

        # Add features one by one according to permutation order
        for feature_idx in permutation:
            current_coalition[feature_idx] = 1

            data_with_feature = self.create_coalition_data(current_coalition)
            pred_with_feature = self.f_model_all_predictions(loader, data_with_feature, max_batches)
            value_with_feature = self.compute_importance_score(pred_with_feature, predictions_original, threshold=disagreement_threshold)

            shapley_values[feature_idx] = value_with_feature - value_current
            value_current = value_with_feature

        return shapley_values
    
    def get_current_results(self) -> Dict[str, float]:
        """
        Get current Shapley value estimates as a dictionary.
        
        Returns:
            results: Dictionary mapping feature names to Shapley values
        """
        mean_vals, _, _ = self._get_current_stats()
        
        results = OrderedDict()
        for idx, feature_name in enumerate(self.feature_names):
            results[feature_name] = mean_vals[idx]
        
        return results
    
    def get_current_results_with_uncertainty(self) -> Dict[str, Any]:
        """
        Get current results with uncertainty estimates.
        
        Returns:
            results: Dictionary with Shapley values, std errors, and variances
        """
        mean_vals, variance, std_error = self._get_current_stats()
        
        shap_values = OrderedDict()
        std_errors = OrderedDict()
        variances = OrderedDict()
        
        for idx, feature_name in enumerate(self.feature_names):
            shap_values[feature_name] = mean_vals[idx]
            std_errors[feature_name] = std_error[idx]
            variances[feature_name] = variance[idx]
        
        return {
            'shap_values': shap_values,
            'std_errors': std_errors,
            'variances': variances,
            'num_permutations': self.num_permutations
        }
    
    
    def explain(self,
                loader: Optional[Any] = None,
                split_name: str = "test",
                checkpoint_hours: Optional[List[float]] = None,
                auto_checkpoint_interval: Optional[float] = None,
                max_permutations: Optional[int] = None,
                convergence_threshold: Optional[float] = None,
                max_batches: int = 10,
                disagreement_threshold: float = 0.5,
                output_dir: Optional[str] = None,
                base_filename: str = "monte_carlo_shapley",
                verbose: bool = True,
                reset_stats: bool = True,
                elapsed_offset_hours: float = 0.0) -> Dict[str, Any]:
        """
        Compute Shapley values using Monte Carlo permutation sampling
        with importance-score aggregation.

        SAME calculation for both task types:
            marginal contribution = change in mean absolute prediction shift
            vs the original data (see compute_importance_score()).

        For classification, predictions are sigmoid-applied probabilities,
        so marginal contributions are in units of average probability
        shift. For regression they're in units of the target variable.
        (Classification previously counted label flips against a decision
        threshold instead — that has been replaced so both task types are
        computed identically.)

        Args:
            loader:                    DataLoader (if None, creates one from entity_table)
            split_name:                Split to use if creating loader ('train'/'val'/'test')
            checkpoint_hours:          Save checkpoints at these elapsed hours (e.g. [1, 2, 4])
            auto_checkpoint_interval:  Save every N hours; overrides checkpoint_hours
            max_permutations:          Stop after this many permutations (None = unlimited)
            convergence_threshold:     Stop when mean standard error < this value
            max_batches:               Batches to process per coalition evaluation
            disagreement_threshold: Unused — retained only for backward-compatible
                call signatures (see compute_importance_score()).
            output_dir:                Directory to save checkpoints
            base_filename:             Base name for checkpoint files
            verbose:                   Print progress
            reset_stats:               If True (default), reset Welford running statistics
                                       before sampling.  Pass False when calling explain()
                                       in a loop to accumulate state across calls.
            elapsed_offset_hours:      Hours already spent by an earlier run that this call
                                       continues (see --resume in
                                       run_monteCarlo_explainer.py).  The offset is added to
                                       every reported/saved elapsed time, so checkpoint files
                                       keep counting up (...53.0h, 54.0h) across a restart
                                       instead of starting over at 1.0h.  Default 0.0 keeps
                                       the original behaviour for a fresh run.

        Returns:
            results: Dict with keys shap_values, std_errors, variances,
                     num_permutations, elapsed_hours, checkpoint_files
        """
        explain_start_time = time.time()

        # Create loader if not provided
        if loader is None:
            if verbose:
                print(f"\n Creating loader for {self.entity_table} ({split_name} split)...")
            loader = self.create_loader(self.entity_table, split_name=split_name)

        # Store threshold so _save_checkpoint can embed it in every checkpoint CSV.
        # No longer used to compute the importance score itself (kept for
        # backward-compatible record-keeping only).
        self._disagreement_threshold = disagreement_threshold if self.task_type == "classification" else None
        
        # Setup checkpoints
        checkpoint_files = {}
        if auto_checkpoint_interval is not None:
            # Auto-checkpoint mode: save every N hours
            if output_dir is None:
                output_dir = "./monte_carlo_results"
            os.makedirs(output_dir, exist_ok=True)
            # Start from the first interval boundary after the offset so a resumed
            # run never rewrites checkpoint files the earlier run already produced.
            next_checkpoint_hour = (
                int(elapsed_offset_hours // auto_checkpoint_interval) + 1
            ) * auto_checkpoint_interval
            use_auto_checkpoint = True
            if verbose:
                print(f" Auto-checkpoint: every {auto_checkpoint_interval} hours")
        elif checkpoint_hours is not None:
            # Manual checkpoint mode: save at specified hours
            if output_dir is None:
                output_dir = "./monte_carlo_results"
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_hours_sorted = sorted(checkpoint_hours)
            next_checkpoint_idx = 0
            use_auto_checkpoint = False
        else:
            # No checkpoints
            use_auto_checkpoint = False
            checkpoint_hours_sorted = []
            next_checkpoint_idx = -1
        
        if verbose:
            print("\n" + "=" * 70)
            print("MONTE CARLO SHAPLEY VALUE CALCULATION")
            print("=" * 70)
            print(f"\n Task type: {self.task_type.upper()}")
            print(f" Aggregation: IMPORTANCE SCORE (mean |Δpred|, same formula for both task types)")
            print(f" Features (M): {self.M}")
            print(f" Max permutations: {max_permutations if max_permutations else 'unlimited'}")
            print(f" Max batches per evaluation: {max_batches}")
            if checkpoint_hours:
                print(f" Checkpoint hours: {checkpoint_hours}")

        # Initialize statistics (skip when called in an adaptive loop to preserve
        # accumulated Welford state from previous rounds)
        if reset_stats:
            self._initialize_running_stats()

        # Get baseline predictions from original data (all features present)
        with timed_section("Getting baseline predictions from original data"):
            if verbose:
                print("\n Getting baseline predictions from ORIGINAL data...")
            z_original = np.ones(self.M, dtype=int)
            data_original = self.create_coalition_data(z_original)
            predictions_original = self.f_model_all_predictions(loader, data_original, max_batches)
            if verbose:
                print(f"   Original predictions shape: {predictions_original.shape}")
                print(f"   Original predictions — min: {predictions_original.min():.4f}, "
                      f"max: {predictions_original.max():.4f}, mean: {predictions_original.mean():.4f}")
        
        start_time = time.time()
        
        if verbose:
            print("\n" + "=" * 70)
            print("STARTING PERMUTATION SAMPLING")
            print("=" * 70)
            print(" Press Ctrl+C to stop and get current results")
        
        try:
            pbar = tqdm(desc="Sampling permutations", unit="perm", disable=not verbose, dynamic_ncols=True)
            
            while True:
                elapsed_seconds = time.time() - start_time
                elapsed_hours = elapsed_seconds / 3600 + elapsed_offset_hours
                
                # Update progress
                if verbose:
                    mean_vals, _, std_errors = self._get_current_stats()
                    pbar.set_postfix({
                        'elapsed_h': f'{elapsed_hours:.2f}',
                        'perms': self.num_permutations,
                        'mean_se': f'{std_errors.mean():.6f}' if self.num_permutations > 0 else 'N/A'
                    })
                
                # Save checkpoint if needed
                if use_auto_checkpoint:
                    # Auto-checkpoint: save every N hours
                    if elapsed_hours >= next_checkpoint_hour:
                        checkpoint_filename = f"{base_filename}_{next_checkpoint_hour:.1f}h.csv"
                        checkpoint_path = os.path.join(output_dir, checkpoint_filename)
                        self._save_checkpoint(checkpoint_path, elapsed_hours)
                        checkpoint_files[next_checkpoint_hour] = checkpoint_path
                        next_checkpoint_hour += auto_checkpoint_interval
                        if verbose:
                            pbar.write(f"✓ Checkpoint saved: {checkpoint_filename}")
                elif checkpoint_hours is not None and next_checkpoint_idx >= 0 and next_checkpoint_idx < len(checkpoint_hours_sorted):
                    # Manual checkpoint: save at specified hours
                    target_hours = checkpoint_hours_sorted[next_checkpoint_idx]
                    if elapsed_hours >= target_hours:
                        checkpoint_filename = f"{base_filename}_{target_hours}h.csv"
                        checkpoint_path = os.path.join(output_dir, checkpoint_filename)
                        self._save_checkpoint(checkpoint_path, elapsed_hours)
                        checkpoint_files[target_hours] = checkpoint_path
                        next_checkpoint_idx += 1
                        if verbose:
                            pbar.write(f"✓ Checkpoint saved: {checkpoint_filename}")
                
                # Check if max permutations reached
                if max_permutations is not None and self.num_permutations >= max_permutations:
                    if verbose:
                        pbar.write(f"\n Reached max permutations ({max_permutations})")
                    break
                
                # Check convergence
                if convergence_threshold is not None and self.num_permutations >= 10:
                    mean_vals, _, std_errors = self._get_current_stats()
                    mean_se = std_errors.mean()
                    if mean_se < convergence_threshold:
                        if verbose:
                            pbar.write(f"\n ✓ Converged! Mean SE ({mean_se:.6f}) < threshold ({convergence_threshold})")
                        break
                
                # Generate random permutation - uses self._permutation_rng
                # (isolated from the legacy global RNG that fix_randomness
                # resets inside every f_model_all_predictions call). See
                # __init__ for why this matters.
                permutation = self._permutation_rng.permutation(self.M)

                # Compute Shapley values for this permutation
                shapley_for_perm = self._compute_shapley_from_permutation(
                    permutation=permutation,
                    loader=loader,
                    max_batches=max_batches,
                    predictions_original=predictions_original,
                    disagreement_threshold=disagreement_threshold,
                )
                
                # Update running statistics
                self._update_running_stats(shapley_for_perm)
                pbar.update(1)
            
            pbar.close()
        
        except KeyboardInterrupt:
            if verbose:
                print("\n\n⚠️  Interrupted by user! Returning current results...")
            pbar.close()
            
            # Save emergency checkpoint
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                elapsed_seconds = time.time() - start_time
                elapsed_hours = elapsed_seconds / 3600 + elapsed_offset_hours
                emergency_filename = f"{base_filename}_interrupted_{elapsed_hours:.2f}h.csv"
                emergency_path = os.path.join(output_dir, emergency_filename)
                self._save_checkpoint(emergency_path, elapsed_hours)
                checkpoint_files['interrupted'] = emergency_path
                if verbose:
                    print(f" Emergency checkpoint saved: {emergency_filename}")
        
        # Get final results
        elapsed_seconds = time.time() - start_time
        elapsed_hours = elapsed_seconds / 3600 + elapsed_offset_hours
        
        if verbose:
            print("\n" + "=" * 70)
            print("CALCULATION COMPLETE")
            print("=" * 70)
            print(f"\n Total permutations sampled: {self.num_permutations}")
            print(f" Total time: {elapsed_hours:.2f} hours ({elapsed_seconds/60:.1f} minutes)")
            
            mean_vals, _, std_errors = self._get_current_stats()
            print(f" Mean standard error: {std_errors.mean():.6f}")
            print(f" Max standard error: {std_errors.max():.6f}")
            print(f" Min standard error: {std_errors.min():.6f}")
        
        results = self.get_current_results_with_uncertainty()
        results['checkpoint_files'] = checkpoint_files
        results['elapsed_hours'] = elapsed_hours

        explain_total_elapsed = time.time() - explain_start_time
        print(f"\n⏱  [explain() TOTAL] completed in {explain_total_elapsed:.2f}s "
              f"({explain_total_elapsed/60:.2f} min)")

        return results
    
    def _save_checkpoint(self, checkpoint_path: str, elapsed_hours: float):
        """Save current Shapley value estimates to CSV."""
        mean_vals, variance, std_error = self._get_current_stats()
        
        rows = []
        for idx, feature_name in enumerate(self.feature_names):
            rows.append({
                'feature': feature_name,
                'shapley_value': mean_vals[idx],
                'std_error': std_error[idx],
                'variance': variance[idx],
                'abs_shapley_value': abs(mean_vals[idx]),
                'elapsed_hours': elapsed_hours,
                'num_permutations': self.num_permutations
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('abs_shapley_value', ascending=False)

        threshold = getattr(self, '_disagreement_threshold', None)
        if threshold is not None:
            df['disagreement_threshold'] = threshold

        df.to_csv(checkpoint_path, index=False)
    
    def print_top_features(self, results: Optional[Dict[str, Any]] = None, top_k: int = 20):
        """Print top-K most important features."""
        if results is None:
            results = self.get_current_results_with_uncertainty()
        
        shap_values = results['shap_values']
        std_errors = results.get('std_errors', {})
        has_uncertainty = len(std_errors) > 0
        
        print("\n" + "=" * 70)
        print(f"TOP {top_k} MOST IMPORTANT FEATURES (by absolute Shapley value)")
        print("=" * 70)
        
        sorted_features = sorted(shap_values.items(), 
                                key=lambda x: abs(x[1]), 
                                reverse=True)
        
        for rank, (feature_name, shap_val) in enumerate(sorted_features[:top_k], 1):
            if has_uncertainty:
                std_val = std_errors[feature_name]
                print(f"{rank:3d}. {feature_name:50s} | φ = {shap_val:+.6f} ± {std_val:.6f}")
            else:
                print(f"{rank:3d}. {feature_name:50s} | φ = {shap_val:+.6f}")
        
        if has_uncertainty and 'num_permutations' in results:
            print(f"\n Based on {results['num_permutations']} permutations")
    
    def save_results(self, results: Dict[str, Any], filepath: str,
                     disagreement_threshold: float = None):
        """Save Shapley values to CSV.

        Args:
            results:                  Output dict from explain().
            filepath:                 Output file path.
            disagreement_threshold:   No longer used to compute the importance
                                      score (classification and regression now
                                      share the same mean-|Δpred| calculation).
                                      Kept as an optional constant column,
                                      written for backward compatibility /
                                      record-keeping only. Pass None to omit
                                      the column.
        """
        shap_values = results['shap_values']
        std_errors = results.get('std_errors', {})
        variances = results.get('variances', {})
        
        rows = []
        for feature_name in shap_values.keys():
            row = {
                'feature': feature_name,
                'shapley_value': shap_values[feature_name],
                'abs_shapley_value': abs(shap_values[feature_name])
            }
            if std_errors:
                row['std_error'] = std_errors[feature_name]
            if variances:
                row['variance'] = variances[feature_name]
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('abs_shapley_value', ascending=False)

        if disagreement_threshold is not None:
            df['disagreement_threshold'] = disagreement_threshold

        df.to_csv(filepath, index=False)
        
        print(f"\n Results saved to: {filepath}")
        if disagreement_threshold is not None:
            print(f"   disagreement_threshold = {disagreement_threshold:.4f}")
        if 'num_permutations' in results:
            print(f" Permutations: {results['num_permutations']}")
        if 'elapsed_hours' in results:
            print(f" Time: {results['elapsed_hours']:.2f} hours")

    # =========================================================================
    # CHERNOFF / HOEFFDING BOUND
    # =========================================================================

    def chernoff_bound(
        self,
        epsilon: float = 0.01,
        delta: float = 0.05,
        range_multiplier: float = 6.0,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Estimate how many permutations are needed to guarantee that every
        Shapley value estimate is within *epsilon* of the truth with probability
        at least *1 - delta*, using the Hoeffding / Chernoff bound.

        Mathematical background
        -----------------------
        Each permutation produces a marginal-contribution sample  xᵢ  for
        feature  i.  The sample mean after  n  permutations converges to the
        true Shapley value φᵢ.  Hoeffding's inequality gives a per-feature
        guarantee:

            P( |φ̂ᵢ - φᵢ| > ε )  ≤  2 · exp( -2n ε² / (b-a)² )

        where  [a, b]  is the range of a single marginal contribution.  To
        hold *simultaneously* for all M features we apply a union bound,
        replacing  δ  with  δ/M:

            n  ≥  (b - a)²  · ln(2M / δ)  / ( 2 ε² )

        Range estimation
        ----------------
        If we have already collected some permutations the empirical variance
        σ² is available from Welford's running statistics.  We approximate
        the range as  b - a ≈ range_multiplier · σ  (default 6σ, which covers
        ±3σ for a roughly normal distribution).  The *worst-case* feature
        (largest σ) drives the bound.

        For a pure a-priori estimate (no permutations yet) the caller should
        pass an explicit *range_override* kwarg (see below).

        Args:
            epsilon:          Desired accuracy — maximum allowed error |φ̂ - φ| (default 0.01).
            delta:            Failure probability — bound holds with prob ≥ 1-delta (default 0.05).
            range_multiplier: How many σ to use as the effective range (default 6).
            verbose:          Print a formatted report (default True).

        Returns:
            Dict with keys:
                n_required        – permutations needed (int)
                n_required_float  – exact (un-rounded) value
                epsilon           – as passed
                delta             – as passed
                effective_range   – (b-a) used in the formula (per feature, array)
                worst_feature     – name of the feature that drives the bound
                worst_range       – range value for that feature
                current_n         – permutations collected so far
                already_sufficient – True if current_n >= n_required
        """
        # ------------------------------------------------------------------
        # 1. Gather the effective per-feature range
        # ------------------------------------------------------------------
        if self.num_permutations >= 2:
            # Use empirical standard deviation from Welford running stats
            variance = self.running_m2 / (self.num_permutations - 1)
            std_dev = np.sqrt(variance)                      # shape (M,)
            effective_range = range_multiplier * std_dev     # shape (M,)
        else:
            # No data yet — fall back to a unit range; caller should override
            effective_range = np.ones(self.M, dtype=np.float64)
            if verbose:
                print(
                    "\n ⚠️  No permutations collected yet. "
                    "Using unit range (b-a = 1.0) for the bound. "
                    "Re-run after a few permutations for a data-driven estimate."
                )

        # ------------------------------------------------------------------
        # 2. Compute the Hoeffding bound with a union correction over M features
        # ------------------------------------------------------------------
        #   n  ≥  (b-a)²  · ln(2M / δ)  / ( 2 ε² )
        worst_idx = int(np.argmax(effective_range))
        worst_range = float(effective_range[worst_idx])
        worst_feature = self.feature_names[worst_idx]

        log_term = np.log(2.0 * self.M / delta)   # ln(2M/δ)
        n_float = (worst_range ** 2) * log_term / (2.0 * epsilon ** 2)
        n_required = int(np.ceil(n_float))

        # Per-feature breakdown (useful for debugging / reporting)
        per_feature_n = np.ceil(
            (effective_range ** 2) * log_term / (2.0 * epsilon ** 2)
        ).astype(int)

        already_sufficient = (self.num_permutations >= n_required)

        # ------------------------------------------------------------------
        # 3. Verbose report
        # ------------------------------------------------------------------
        if verbose:
            print("\n" + "=" * 70)
            print("CHERNOFF / HOEFFDING BOUND — REQUIRED PERMUTATIONS")
            print("=" * 70)
            print(f"\n  Parameters")
            print(f"    ε  (desired accuracy)   : {epsilon}")
            print(f"    δ  (failure probability): {delta}   (confidence = {(1-delta)*100:.1f}%)")
            print(f"    M  (features)           : {self.M}")
            print(f"    Union-bound correction  : δ/M = {delta/self.M:.2e}")
            print(f"    Range estimator         : {range_multiplier}σ  "
                  f"({'empirical σ from ' + str(self.num_permutations) + ' perms' if self.num_permutations >= 2 else 'unit fallback'})")

            print(f"\n  Bound")
            print(f"    n  ≥  (b-a)² · ln(2M/δ) / (2ε²)")
            print(f"       =  {worst_range:.4f}² · {log_term:.4f} / (2 · {epsilon}²)")
            print(f"       =  {n_float:.1f}")
            print(f"    ⟹  n_required = {n_required:,} permutations")

            print(f"\n  Worst-case feature")
            print(f"    Name  : {worst_feature}")
            print(f"    Range : {worst_range:.6f}  (≈ {range_multiplier}σ)")

            print(f"\n  Current status")
            print(f"    Permutations so far : {self.num_permutations:,}")
            print(f"    Still needed        : {max(0, n_required - self.num_permutations):,}")

            if already_sufficient:
                print(f"\n  ✅ Already sufficient!  "
                      f"({self.num_permutations:,} ≥ {n_required:,})")
            else:
                pct = 100.0 * self.num_permutations / n_required if n_required > 0 else 0
                print(f"\n  ⏳ Progress: {pct:.1f}% complete")

            # Top-5 most demanding features
            top5_idx = np.argsort(per_feature_n)[::-1][:5]
            print(f"\n  Top-5 most demanding features:")
            print(f"    {'Feature':<50s} {'Range':>10s} {'n_req':>10s}")
            print(f"    {'-'*50} {'-'*10} {'-'*10}")
            for idx in top5_idx:
                print(
                    f"    {self.feature_names[idx]:<50s} "
                    f"{effective_range[idx]:>10.4f} "
                    f"{per_feature_n[idx]:>10,}"
                )
            print("=" * 70)

        return {
            "n_required": n_required,
            "n_required_float": n_float,
            "epsilon": epsilon,
            "delta": delta,
            "effective_range": effective_range,          # (M,) array
            "per_feature_n": per_feature_n,              # (M,) array
            "worst_feature": worst_feature,
            "worst_range": worst_range,
            "current_n": self.num_permutations,
            "already_sufficient": already_sufficient,
        }


# =============================================================================
# MODULE-LEVEL UTILITY — can be used without instantiating MonteCarloShapley
# =============================================================================

def compute_chernoff_n(
    epsilon: float,
    delta: float,
    range_val: float,
    M: int = 1,
) -> int:
    """
    Standalone Hoeffding / Chernoff sample-size calculator.

    Returns the minimum number of permutations  n  such that the Monte Carlo
    estimate of a Shapley value is within  epsilon  of the truth with
    probability  ≥  1 - delta, simultaneously for all M features
    (union-bound corrected).

    Formula:
        n  ≥  (b - a)²  · ln(2M / δ)  / ( 2 ε² )

    Args:
        epsilon:   Desired max error in Shapley units (e.g. 0.01).
        delta:     Failure probability (e.g. 0.05 for 95% confidence).
        range_val: Expected range  b - a  of a single marginal contribution.
                   A conservative default is  1.0  (contributions bounded in [0,1]).
                   Pass  6 * empirical_std  for a data-driven estimate.
        M:         Number of features (for union-bound correction); default 1
                   (no correction).

    Returns:
        n_required (int): minimum permutations required.

    Examples:
        >>> compute_chernoff_n(epsilon=0.01, delta=0.05, range_val=1.0, M=50)
        >>> compute_chernoff_n(epsilon=0.005, delta=0.01, range_val=0.3, M=100)
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    if range_val <= 0:
        raise ValueError("range_val must be positive")
    if M < 1:
        raise ValueError("M must be at least 1")

    log_term = np.log(2.0 * M / delta)
    n_float = (range_val ** 2) * log_term / (2.0 * epsilon ** 2)
    return int(np.ceil(n_float))


import copy
import torch
import numpy as np
from typing import Any
import torch_frame
from torch_frame import stype
from torch_frame.data import StatType
 
class MonteCarloShapleyRelGT(MonteCarloShapley):
    """
    Monte Carlo Shapley explainer for the RelGT (transformer) model.
 
    IMPORTANT ARCHITECTURE NOTE (read before touching this class again):
 
    ABLATION MECHANISM (v3 - backup/restore, matches FidelityCalculator):
 
    `create_coalition_data` is overridden to return the raw binary coalition
    vector unchanged (NOT a deep-copied/ablated HeteroData object). Actual
    masking happens in `f_model_all_predictions`, which mutates `self.data`
    - the single canonical graph object - directly in place, using
    `backup_column`/`restore_column` (from `Utils.Utils`, the exact same
    functions `FidelityCalculator` uses for its proven, working ablation:
    backup -> mutate self.data -> run inference -> restore in `finally`).
 
    Why not deep-copy per coalition (v2, an earlier version of this file)?
    `copy.deepcopy(self.data)` on every one of the 46+ evaluations per
    permutation is expensive, and was implicated in an intermittent
    `torch_frame` `AssertionError: offset[0] == 0` crash inside
    `MultiEmbeddingTensor.validate()` after many repeated deepcopy+mask
    cycles. Backup/restore never deep-copies anything - it mutates and then
    undoes the mutation on the one real object - which is both faster and
    sidesteps that failure mode entirely.
 
    Why not patch the already-tokenized `grouped_tfs` batch after the fact
    (v1, the original broken version)? `RelGTTokens(..., precompute=True)`
    derives its token representations from the raw HeteroData feature
    tensors at *construction* time (cached under `relgt-cache/*.h5`), so
    editing the collated batch afterward only touches a transient copy
    downstream of wherever the real signal was already baked in - masking
    edits were technically applied but had almost no real effect on model
    output, which is why every Shapley value came out ~0.0 regardless of
    the `node_type_to_id` mapping being correct.
 
    SAFETY: because this mutates the single shared `self.data` object in
    place (rather than an isolated copy), `f_model_all_predictions` MUST
    restore every backed-up column in a `finally` block - if an exception
    (e.g. a crash mid-batch) leaves a masked value in `self.data` un-restored,
    every subsequent evaluation for the rest of the run would silently use
    corrupted data with no error raised. Do not remove the try/finally.
    """
 
    def __init__(self, *args, relgt_tokens_kwargs: dict, **kwargs):
        """
        Args:
            relgt_tokens_kwargs: Dict passed directly to RelGTTokens.
                Must contain: RelGTTokens (class ref), data, task, K,
                precomputed_dir. Optional: undirected, num_workers.
            All other args/kwargs forwarded to MonteCarloShapley.
        """
        super().__init__(*args, **kwargs)
        self._relgt_tokens_kwargs = relgt_tokens_kwargs
 
    def create_coalition_data(self, z_prime: np.ndarray) -> np.ndarray:
        """
        Returns the coalition vector unchanged. Actual masking happens
        directly on `self.data` inside `f_model_all_predictions` via
        backup_column/restore_column - see class docstring.
        """
        return np.asarray(z_prime).copy()
 
 
    def _make_relgt_loader(self, split: str = "test", data_override: Any = None) -> Any:
        """
        Build a RelGTTokens DataLoader for the given split.
 
        Args:
            split:         Which split to load ("val", "test", etc.)
            data_override: If provided, use this HeteroData object instead of
                           the original data. Pass an ablated data_coalition
                           here so the ablation is reflected in the token
                           representations RelGTTokens derives at
                           construction time.
 
        Rebuilt fresh on every call by design - see class docstring.
        """
        from torch.utils.data import DataLoader
 
        RelGTTokens = self._relgt_tokens_kwargs.get("RelGTTokens")
        if RelGTTokens is None:
            raise ValueError(
                "RelGTTokens class reference missing from relgt_tokens_kwargs. "
                "Import it in run_monteCarlo_explainer.py and pass it through."
            )
 
        token_kwargs = {
            k: v for k, v in self._relgt_tokens_kwargs.items()
            if k not in ("RelGTTokens", "node_type_map")
        }
        if data_override is not None:
            token_kwargs["data"] = data_override
 
        # Force num_workers=0 for RelGTTokens's OWN internal precompute
        # parallelism (separate from the DataLoader's num_workers below).
        # This class rebuilds RelGTTokens from scratch on every single
        # coalition evaluation (46+ times per permutation) - after the very
        # first build, the ".h5" precompute cache already exists on disk
        # ("Found existing HDF5 at ..."), so spinning up a multiprocessing
        # pool on every subsequent rebuild is pure overhead. Worse, each
        # pool's worker processes hold shared-memory file descriptors that
        # can outlive the pool object longer than Python's GC gets around to
        # collecting it, and across dozens of rebuilds per permutation this
        # exhausts the process's open-file limit (EMFILE / "Too many open
        # files"). Reading from an existing cache single-process is cheap
        # regardless, so there's no real cost to forcing this down.
        token_kwargs["num_workers"] = 0

        dataset = RelGTTokens(split=split, precompute=True, **token_kwargs)

        batch_size = self.loader_config.get("batch_size", 64)
        # Force num_workers=0 for the DataLoader too (previously this
        # defaulted to loader_config["num_workers_loader"], which can be >0).
        # With num_workers>0, PyTorch spawns worker processes and `self.data`
        # (the mutated-then-restored HeteroData object driving RelGTTokens)
        # has to cross a process boundary (fork/spawn) into each one - on a
        # loader that's rebuilt from scratch 46+ times per permutation, that
        # is both expensive and was implicated in a `MultiEmbeddingTensor`
        # `offset[0]==0` AssertionError inside a DataLoader worker process
        # after many repeated rebuild cycles. Single-process loading is slower
        # per-batch but eliminates that entire class of failure, and the
        # model forward pass (not data loading) dominates the per-evaluation
        # cost anyway for these small (max_batches-limited) reads.
        num_workers = 0

        # RelGTTokens builds itself from the FULL split (it reads the task table
        # itself and pulls neighbors from the precomputed HDF5 cache by position),
        # so restrict it to the explanation subset here rather than shortening the
        # table — a shortened table would silently read the first N cached rows
        # instead of the subset's rows. Only for "test": this method is also
        # called with split="val" (threshold tuning), which stays full.
        # Subset has no .collate, hence capturing it first.
        collate_fn = dataset.collate
        if self.test_subset_idx is not None and split == "test":
            from torch.utils.data import Subset
            dataset = Subset(dataset, [int(i) for i in self.test_subset_idx])

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=False,
            pin_memory=(self.device.type == "cuda"),
        )
        return loader
 
    def f_model_all_predictions(
        self,
        loader: Any,
        data_coalition: Any,
        max_batches: int,
        split: str = "test",
    ) -> np.ndarray:
        """
        Get all predictions for a coalition using the RelGT forward pass.
 
        `loader` is ignored. `data_coalition` is the raw binary coalition
        vector (from the overridden `create_coalition_data`) - this method
        ablates `self.data` IN PLACE (backup -> mask -> infer -> restore),
        matching FidelityCalculator's proven pattern. See class docstring
        for why this replaced both the earlier per-call `deepcopy` approach
        and the original (broken) post-hoc `grouped_tfs` patching.
 
        `split` defaults to "test" (matching the rest of this pipeline's
        baseline predictions / checkpoint output), but can be overridden -
        e.g. threshold tuning needs "val" predictions specifically.
        """
        z_prime = np.asarray(data_coalition)
        backups: Dict[Tuple[str, str], torch.Tensor] = {}
        relgt_loader = None
 
        try:
            # Ablate: back up + mask every absent feature directly on
            # self.data (the single canonical graph object).
            for feat_idx, is_present in enumerate(z_prime):
                if float(is_present) == 1.0 or feat_idx not in self.feature_mapping:
                    continue
                feat = self.feature_mapping[feat_idx]
                key = (feat['node_type'], feat['col_name'])
                if key not in backups:
                    backups[key] = backup_column(self.data, key[0], key[1])
                self._apply_feature_mask(self.data, feat_idx, is_present)
 
            self.model.eval()
            all_predictions = []
            all_global_idx = []
 
            # data_override=None -> uses self.data as-is (already mutated above)
            relgt_loader = self._make_relgt_loader(split=split, data_override=None)
 
            with torch.no_grad():
                fix_randomness(self.loader_config.get("seed", 42))
                for batch_idx, batch in enumerate(relgt_loader):
                    if max_batches is not None and batch_idx >= max_batches:
                        break
 
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(self.device)
 
                    if "global_idx" in batch:
                        all_global_idx.append(batch["global_idx"].detach().cpu().numpy())
 
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
                        pred = apply_sigmoid(pred)
 
                    all_predictions.append(pred.detach().cpu().numpy())
 
            if not all_predictions:
                raise RuntimeError("MonteCarloShapleyRelGT.f_model_all_predictions: No predictions generated.")
 
            # Side-effect (doesn't change the return type/signature other
            # callers rely on): the real row index for each prediction just
            # produced, so diagnostics can align against externally-saved
            # predictions correctly instead of assuming matching positional
            # order.
            self._last_global_idx = np.concatenate(all_global_idx, axis=0) if all_global_idx else None
 
            result = np.concatenate(all_predictions, axis=0)
 
        finally:
            # CRITICAL: always undo the ablation, even on exception - self.data
            # is the single shared canonical object, so leaving it mutated
            # would silently corrupt every subsequent evaluation for the rest
            # of the run with no error raised. See class docstring (SAFETY).
            for (node_type, col_name), backup_tensor in backups.items():
                restore_column(self.data, node_type, col_name, backup_tensor)
 
        # Explicit cleanup: this method rebuilds the DataLoader from scratch
        # on every call (46+ times per permutation). Relying on Python's GC
        # to eventually collect it (and close out any lingering file handles)
        # risks accumulating open file descriptors faster than they're
        # reclaimed across dozens of rebuilds - this is what caused the
        # EMFILE ("Too many open files") crash previously. Force it
        # explicitly and immediately instead of waiting for GC's own timing.
        if relgt_loader is not None:
            del relgt_loader
        gc.collect()
        torch.cuda.empty_cache()
 
        return result
 
 
    def sanity_check_masking(self, loader: Any = None, max_batches: int = 1,
                              reference_predictions: Optional[np.ndarray] = None) -> None:
        """
        Fast smoke test: compares baseline predictions against predictions
        with EVERY feature masked out. If masking is actually wired up, these
        should differ substantially. If they are identical (or nearly so),
        masking is still a no-op and Shapley values will still come out as
        ~0.0 regardless of how many permutations you run.
 
        If `reference_predictions` is provided (e.g. the model's real
        `test_predictions_N.npy` loaded from disk), also compares the
        explainer's own re-inference path against it.
 
        NOTE: each call to f_model_all_predictions now rebuilds RelGTTokens
        from scratch (see class docstring), so this is no longer "cheap" in
        the sense of reusing a cached loader - but it's now actually correct.
        """
        print("\n" + "=" * 70)
        print("SANITY CHECK: does masking actually change model predictions?")
        print("=" * 70)
 
        z_full = np.ones(self.M, dtype=int)
        z_empty = np.zeros(self.M, dtype=int)
 
        # Noise floor: run the SAME (fully unmasked) input through twice.
        # GPU kernels can be mildly non-deterministic (algorithm selection,
        # reduction order), so two calls with identical input can differ by a
        # tiny amount even with nothing changed. If that noise floor is
        # comparable to the full-vs-empty diff below, the "masking changed
        # output" conclusion is not trustworthy - it could just be noise.
        pred_full_a = self.f_model_all_predictions(loader, self.create_coalition_data(z_full), max_batches)
        idx_full = getattr(self, '_last_global_idx', None)
        pred_full_b = self.f_model_all_predictions(loader, self.create_coalition_data(z_full), max_batches)
        noise_floor = float(np.max(np.abs(pred_full_a - pred_full_b)))
        pred_full = pred_full_a
 
        pred_empty = self.f_model_all_predictions(loader, self.create_coalition_data(z_empty), max_batches)
 
        max_abs_diff = float(np.max(np.abs(pred_full - pred_empty)))
        mean_abs_diff = float(np.mean(np.abs(pred_full - pred_empty)))
        identical = np.array_equal(pred_full, pred_empty)
 
        print(f" All-features vs no-features predictions ({len(pred_full)} samples, max_batches={max_batches}):")
        print(f"   identical arrays: {identical}")
        print(f"   max abs diff:  {max_abs_diff:.6f}")
        print(f"   mean abs diff: {mean_abs_diff:.6f}")
        print(f"   noise floor (same input, called twice): max abs diff = {noise_floor:.6f}")
        print(f"   pred_full  -> min={pred_full.min():.6f} max={pred_full.max():.6f} "
              f"mean={pred_full.mean():.6f} std={pred_full.std():.6f}")
        print(f"   pred_empty -> min={pred_empty.min():.6f} max={pred_empty.max():.6f} "
              f"mean={pred_empty.mean():.6f} std={pred_empty.std():.6f}")
 
        if self.task_type == 'classification':
            near_boundary = np.abs(pred_full - 0.5) < 0.05
            print(f"   samples within 0.05 of prob=0.5: {int(near_boundary.sum())} / {len(pred_full)}")
 
        if reference_predictions is not None:
            ref_full = np.asarray(reference_predictions)
            idx = idx_full
            if idx is not None and idx.max() < len(ref_full):
                ref = ref_full[idx]
                align_note = "aligned by global_idx (real row order)"
            else:
                ref = ref_full[:len(pred_full)]
                align_note = "WARNING: no global_idx available - aligned by naive position, may be MISLEADING if sample order differs"
            corr = float(np.corrcoef(ref, pred_full)[0, 1]) if len(ref) > 1 else float('nan')
            ref_vs_full_diff = float(np.max(np.abs(ref - pred_full)))
            print(f"\n Explainer re-inference vs real saved test predictions ({align_note}):")
            print(f"   reference  -> min={ref.min():.6f} max={ref.max():.6f} mean={ref.mean():.6f}")
            print(f"   correlation(reference, pred_full): {corr:.6f}")
            print(f"   max abs diff(reference, pred_full): {ref_vs_full_diff:.6f}")
            if corr < 0.9 or ref_vs_full_diff > 0.05:
                print("   ✗ Explainer's own re-inference path does NOT match real model "
                      "predictions.")
                # Check whether a simple affine transform explains the gap -
                # e.g. the original eval pipeline may de-normalize / invert a
                # log1p transform on regression output before saving
                # test_predictions_N.npy, which f_model_all_predictions (no
                # post-processing for regression) doesn't replicate.
                if len(ref) > 2 and self.task_type == 'regression':
                    slope, intercept = np.polyfit(pred_full, ref, deg=1)
                    fitted = slope * pred_full + intercept
                    residual = float(np.max(np.abs(fitted - ref)))
                    print(f"   Affine-fit check: ref ≈ {slope:.6f} * pred_full + {intercept:.6f} "
                          f"(max residual after fit: {residual:.6f})")
                    if residual < 0.005:
                        print("   → A simple affine transform explains almost all of the gap. "
                              "This points to a missing de-normalization/inverse-transform step "
                              "for regression predictions specifically (check how "
                              "test_predictions_N.npy was originally produced - e.g. scaling, "
                              "log1p/expm1, or similar applied to raw model output before "
                              "saving), NOT a masking or pipeline-ordering bug.")
                    else:
                        print("   → Not a simple affine relationship - a nonlinear transform "
                              "(e.g. log1p/expm1) or a genuinely different computation path "
                              "is more likely. Worth checking the original eval script for "
                              "exactly what happens to raw model output before it's saved.")
            else:
                print("   ✓ Explainer re-inference matches real predictions closely.")
 
        if identical or max_abs_diff < 1e-6:
            print(" ✗ Masking still appears to be a NO-OP. Shapley values will be ~0.0.")
        elif max_abs_diff < noise_floor * 3:
            print(f" ⚠ max abs diff ({max_abs_diff:.6f}) is within ~3x of the run-to-run "
                  f"noise floor ({noise_floor:.6f}) for IDENTICAL input. This is NOT reliable "
                  f"evidence that masking has a real effect - it may just be GPU "
                  f"non-determinism. Treat this task's Shapley values with suspicion "
                  f"until effect size clearly exceeds noise.")
        else:
            print(" ✓ Masking is changing model output well above the noise floor.")
 
    
    def sanity_check_per_feature_masking(self, loader: Any = None, max_batches: int = 1,
                                          disagreement_threshold: float = 0.5,
                                          feature_indices: Optional[List[int]] = None) -> None:
        """
        Removes ONE feature at a time from the full coalition (the exact
        perturbation a real permutation step measures) and reports the
        resulting prediction shift and importance score per feature.

        Args:
            feature_indices: subset of feature indices to test (default: all
                M features - can be slow for large M since each evaluation
                now rebuilds RelGTTokens from scratch; pass a subset to probe
                quickly first).
        """
        print("\n" + "=" * 70)
        print("SANITY CHECK: per-feature (single removal) masking effect")
        print("=" * 70)
 
        indices = feature_indices if feature_indices is not None else list(range(self.M))
 
        z_full = np.ones(self.M, dtype=int)
        pred_full = self.f_model_all_predictions(loader, self.create_coalition_data(z_full), max_batches)
 
        rows = []
        for idx in indices:
            z = np.ones(self.M, dtype=int)
            z[idx] = 0
            pred_ablated = self.f_model_all_predictions(loader, self.create_coalition_data(z), max_batches)
            max_diff = float(np.max(np.abs(pred_full - pred_ablated)))
            mean_diff = float(np.mean(np.abs(pred_full - pred_ablated)))
            importance_score = self.compute_importance_score(pred_ablated, pred_full, threshold=disagreement_threshold)
            rows.append((self.feature_names[idx], max_diff, mean_diff, importance_score))
 
        rows.sort(key=lambda r: -r[1])
 
        print(f"\n {'feature':45s} {'max_diff':>10s} {'mean_diff':>10s} {'importance_score':>17s}")
        for name, max_diff, mean_diff, importance_score in rows:
            # importance_score is now the same continuous mean-abs-diff metric
            # for both task types, so always print with full precision (a
            # nonzero value here can be much smaller than 0.05).
            print(f" {name:45s} {max_diff:10.6f} {mean_diff:10.6f} {importance_score:17.6f}")
 
        any_importance = any(r[3] > 0 for r in rows)
        any_meaningful_diff = any(r[1] > 1e-4 for r in rows)
        print()
        if any_importance:
            print(" ✓ At least one single-feature removal produced a real importance "
                  "score (nonzero prediction shift). Shapley values should be "
                  "non-trivial once enough permutations accumulate coverage of "
                  "that feature.")
        elif any_meaningful_diff:
            print(" ⚠ Single-feature removals shift raw predictions but the mean "
                  "importance score still rounds to ~0.0. Worth double-checking "
                  "the shift magnitude relative to your convergence threshold.")
        else:
            print(" ✗ No single-feature removal produces any meaningful prediction "
                  "shift at all. Worth re-checking global_masks/feature_mapping for "
                  "these specific features.")