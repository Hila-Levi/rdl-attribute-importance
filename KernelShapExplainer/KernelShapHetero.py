import torch
import numpy as np
import torch_frame
import os
import sys
import time
import pandas as pd
import copy
import itertools
import warnings
from contextlib import contextmanager
from collections import OrderedDict
from typing import Dict, Any, List, Tuple
from itertools import combinations
from tqdm import tqdm
from scipy.special import comb
from sklearn.linear_model import Lasso, LassoLarsIC, lars_path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_frame.data import StatType

# ============================================================================
# SETUP PATHS AND IMPORTS
# ============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from Utils.Utils import create_consistent_loader, fix_randomness

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

    Prints elapsed wall-clock time in seconds once the block exits
    (including if it raises), so every major stage of the pipeline is
    self-documenting about how much of the total runtime it consumed.
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"⏱  [{label}] completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")


class GlobalKernelSHAP:
    """
    Global Kernel SHAP implementation for heterogeneous graph neural networks.

    Computes global SHAP values for the mean absolute prediction shift
    between original and perturbed (ablated) data. Uses the SAME formula
    for classification and regression — classification no longer counts
    label flips against a threshold.
    """
        
    def __init__(self, 
        model: torch.nn.Module,
        data: Any,
        col_stats_dict: Dict[str, Dict[str, Dict]],
        entity_table: str,
        device: torch.device,
        loader_config: Dict,
        task_type: str = 'classification',
        task: Any = None,
        test_subset_idx=None):
        """
        Args:
            model: Trained GNN model
            data: HeteroData object containing the graph
            col_stats_dict: Dictionary containing column statistics
            entity_table: Name of the entity table (target nodes)
            device: torch device
            loader_config: Standardized loader configuration from Utils
            task_type: 'classification' for classification, 'regression' for regression tasks
            task: RelBench task object. Needed by create_loader() (which calls
                create_consistent_loader -> get_node_train_table_input, and that
                reads task.entity_col). Optional only for backward compatibility
                with older call sites that never used create_loader(); if
                create_loader() is called with task=None the underlying
                get_node_train_table_input() call will fail with
                "'NoneType' object has no attribute 'entity_col'".
        """
        self.model = model
        self.data = data
        self.col_stats_dict = col_stats_dict
        self.entity_table = entity_table
        self.device = device
        self.loader_config = loader_config
        self.task_type = task_type
        self.task = task
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
        
        # Initialize feature space and masks
        with timed_section("STEP 2: Construct feature space"):
            self._construct_feature_space()
        with timed_section("STEP 3: Compute global masks"):
            self._compute_global_masks()
    
    def create_loader(self, table, split_name: str = "test") -> "NeighborLoader":
        """
        Use the shared Utils to create a consistent loader.
        This ensures Kernel SHAP sees the exact same neighbors as other methods.
        """
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
        # draw — same bound as training, no RNG involved.
        deterministic_loader_config = dict(self.loader_config)
        deterministic_loader_config["temporal_strategy"] = "last"
        if self.task is None:
            raise ValueError(
                "GlobalKernelSHAP.create_loader() requires self.task to be set "
                "(pass task=... when constructing the explainer) — "
                "create_consistent_loader -> get_node_train_table_input needs "
                "task.entity_col."
            )
        return create_consistent_loader(
            data=self.data,
            loader_config=deterministic_loader_config,
            task=self.task,
            table=table,
            split_name=split_name
        )
        
    def _construct_feature_space(self):
        """
        Step 2: Construct M feature space.
        Format: table_name.column_name.type
        """
        print("\n" + "=" * 70)
        print("STEP 2: CONSTRUCTING FEATURE SPACE M")
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
        Step 3: Compute global masking values for each column.
        """
        print("\n" + "=" * 70)
        print("STEP 3: COMPUTING GLOBAL MASKING VALUES")
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
                
                # --- Determine Column Type and Masking Key ---
                col_type = None
                if 'numerical' in stype_str:
                    col_type = 'numerical'
                    stat_key, default_val = StatType.MEAN, 0.0
                elif 'timestamp' in stype_str:
                    col_type = 'timestamp'
                    # Use median time tensor as masking value for timestamps
                    stat_key, default_val = StatType.MEDIAN_TIME, torch.tensor([0]*7) 
                elif 'categorical' in stype_str:
                    col_type = 'categorical'
                    # Mode is derived from COUNT stats
                    stat_key, default_val = StatType.COUNT, 0 
                elif 'embedding' in stype_str or 'text' in stype_str:
                    col_type = 'text'
                
                if col_type is None:
                    continue


                # --- 1. Handle Numerical and Timestamp (Mean / Median Time) ---
                if col_type in ['numerical', 'timestamp']:
                    self.global_masks[node_type][col_type] = {}
                    
                    for col_name in col_names:
                        stats = self.col_stats_dict.get(node_type, {}).get(col_name, {})
                        mask_val = stats.get(stat_key, default_val)
                        
                        # Clean up NaN if the mean calculation failed previously
                        if isinstance(mask_val, float) and np.isnan(mask_val):
                            print("   Warning: NaN detected in stats, defaulting to 0.0")
                            mask_val = 0.0
                        
                        self.global_masks[node_type][col_type][col_name] = mask_val

                        if col_type == 'timestamp':
                            print(f"   Timestamp '{col_name}': Global Median Time Tensor (Mask)")
                        else:
                            print(f"   Numerical '{col_name}': Global Mean = {mask_val:.4f}")

                
                # --- 2. Handle Categorical (Most Frequent Value/Mode) ---
                elif col_type == 'categorical':
                    self.global_masks[node_type][col_type] = {}
                    
                    for col_name in col_names:
                        stats = self.col_stats_dict.get(node_type, {}).get(col_name, {})
                        count_data = stats.get(StatType.COUNT)
                        mode_val = 0 # Default: index 0 (often 'unknown' or the most common after encoding)
                        
                        if count_data is not None and isinstance(count_data, tuple) and len(count_data) == 2:
                            # count_data is (values_list, counts_list)
                            values, counts = count_data
                            if len(counts) > 0:
                                print(f"   Categorical '{col_name}': not default data")
                                # Find the index corresponding to the max count (Mode)
                                max_count_index = np.argmax(counts)
                                # The mode value in this context is usually the index itself
                                mode_val = max_count_index 
                        
                        self.global_masks[node_type][col_type][col_name] = int(mode_val)
                        print(f"   Categorical '{col_name}': Global MFV (Index) = {int(mode_val)}")
                        
                # --- 3. Handle Text/Embedding (Mean Embedding Computation) ---
                elif col_type == 'text':
                    self.global_masks[node_type]['text'] = {}

                    if hasattr(tf, 'feat_dict') and stype_key in tf.feat_dict:
                        stype_tensor = tf.feat_dict[stype_key]

                        # Compute the mean PER COLUMN (not once for the whole
                        # stype group) so each col_name gets its own
                        # correctly-shaped mean embedding, matching the
                        # tensor[:, col_idx] indexing create_coalition_data uses.
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

                                # Fallback to zero embedding based on stored dimension (EMB_DIM)
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
                                 
    def create_coalition_data(self, z_prime: np.ndarray) -> Any:
        """
        STEP 4 (Runtime): Creates data for coalition z' using precomputed values.
        """
        data_coalition = copy.deepcopy(self.data)

        # Loop over all M features, identified by index, to determine which column to mask
        for feat_idx, is_present in enumerate(z_prime):
            if feat_idx not in self.feature_mapping:
                continue
            
            feat_details = self.feature_mapping[feat_idx]
            node_type = feat_details['node_type']
            col_type = feat_details['col_type']
            col_name = feat_details['col_name']
            stype_key = feat_details['stype_key']
            col_idx_in_stype = feat_details['col_idx_in_stype']
            
            if node_type not in data_coalition.node_types or \
               not hasattr(data_coalition[node_type], 'tf') or \
               stype_key not in data_coalition[node_type].tf.feat_dict:
                continue
            
            tf = data_coalition[node_type].tf
            tensor = tf.feat_dict[stype_key] # The tensor block (e.g., N x 3600)
            
            # --- Get Mask Value and Original Tensor ---
            mask_dict = self.global_masks.get(node_type, {}).get(col_type, {})
            mask_value = mask_dict.get(col_name) # This is the full dimension mean for text!
            
            if mask_value is None:
                continue

            z_scalar = float(is_present)
            if z_scalar == 1.0:
                    continue  # Feature present, do nothing
                
            # --- 1. Text Feature Masking (Per-Column Blending/Replacement) ---
            if col_type == 'text':
                try:
                        # For MultiEmbeddingTensor stypes, do NOT slice a single
                        # column via tensor[:, col_idx_in_stype] and mutate its
                        # .values — torch_frame concatenates every column in
                        # this stype group into one shared `values` buffer, and
                        # a derived slice's own offset isn't guaranteed to
                        # satisfy the offset[0]==0 invariant the parent relies
                        # on. Write directly into the parent tensor's own
                        # values buffer using its own offset boundaries instead.
                        if hasattr(tensor, 'offset') and hasattr(tensor, 'values'):
                            start = int(tensor.offset[col_idx_in_stype])
                            end = int(tensor.offset[col_idx_in_stype + 1])

                            mean_embedding_slice = mask_value.to(self.device).to(tensor.values.dtype)
                            mask_expanded = mean_embedding_slice.unsqueeze(0).expand(
                                tensor.values.size(0), end - start
                            )
                            tensor.values[:, start:end] = mask_expanded
                        else:
                            # Fallback for plain (non-embedding) tensors, kept for
                            # safety though 'text' columns are expected to be
                            # MultiEmbeddingTensor in practice.
                            target_slice = tensor[:, col_idx_in_stype]
                            original_dtype = target_slice.dtype
                            mean_embedding_slice = mask_value.to(self.device).to(original_dtype)
                            target_slice.copy_(mean_embedding_slice.unsqueeze(0).expand_as(target_slice))

                except Exception as e:
                    # Catch the Mismatch error and skip (if the column index is somehow wrong)
                    warnings.warn(f"Failed to mask text column {col_name} (index {col_idx_in_stype}): {e}")

            # --- 2. Non-Text Feature Masking (Numerical, Categorical, Timestamp) ---
            elif col_type in ['numerical', 'categorical', 'timestamp']:

                # Same safeguard as above: some 'categorical'/'numerical'-
                # labeled columns turn out to be embedding-backed
                # (MultiEmbeddingTensor) rather than plain tensors — write
                # into the parent tensor's own values/offset rather than a
                # derived column slice whenever that's the case.
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
                    # Access the specific column slice
                    target_slice = tensor[:, col_idx_in_stype]
                    # Full mask: Apply mask_value

                    if isinstance(mask_value, torch.Tensor):
                        # Timestamp (N x 7) = (1 x 7) mask
                        mask_value = mask_value.to(target_slice.dtype).to(self.device)

                        # Apply to the target slice (which is the whole N x 7 block for a timestamp feature)
                        target_slice.copy_(mask_value.unsqueeze(0).expand_as(target_slice))
                    else:
                        # Numerical/Categorical (Scalar mask_value)
                        target_slice.fill_(mask_value)
        return data_coalition

    def f_model_all_predictions(self, loader: Any, data_coalition: Any, max_batches: int) -> np.ndarray:
        """
        Get ALL predictions (not averaged) for coalition.
        
        Args:
            loader: DataLoader
            data_coalition: Coalition data
            max_batches: Maximum batches to process
        
        Returns:
            predictions: Array of shape (total_nodes,) with ALL predictions
        """
        torch.use_deterministic_algorithms(True)
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            fix_randomness(self.loader_config.get('seed', 42))
            for batch_idx, batch in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                batch = batch.to(self.device)

                # Override batch features with coalition features
                for node_type in batch.node_types:
                    if node_type in data_coalition.node_types:
                        if hasattr(batch[node_type], 'tf') and hasattr(data_coalition[node_type], 'tf'):
                            # Get node IDs and ensure they're on CPU for indexing
                            batch_node_ids = batch[node_type].n_id
                            batch_node_ids_cpu = batch_node_ids.cpu()
                            
                            for stype_key in data_coalition[node_type].tf.feat_dict.keys():
                                if stype_key in batch[node_type].tf.feat_dict:
                                    coalition_feats = data_coalition[node_type].tf.feat_dict[stype_key]
                                    
                                    try:
                                        # Index with CPU indices (works for all tensor types)
                                        batch_feats = coalition_feats[batch_node_ids_cpu]
                                        
                                        # Move to device
                                        if isinstance(batch_feats, torch.Tensor):
                                            batch_feats = batch_feats.to(self.device)
                                        elif hasattr(batch_feats, 'to'):
                                            batch_feats = batch_feats.to(self.device)
                                        
                                        # Update batch
                                        batch[node_type].tf.feat_dict[stype_key] = batch_feats
                                        
                                    except Exception as e:
                                        # If indexing fails, keep original batch features
                                        print(f"   Warning: Could not update {node_type}.{stype_key}: {e}")
                                        pass
                
                # Get predictions for this batch
                pred = self.model(batch, self.entity_table)
                pred = pred.view(-1) if pred.size(1) == 1 else pred
                # Apply sigmoid for classification — consistent with train.py
                if self.task_type == "classification":
                    pred = apply_sigmoid(pred)
                all_predictions.append(pred.detach().cpu().numpy())
        
        # Concatenate ALL predictions
        return np.concatenate(all_predictions, axis=0)
    
    
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
            predictions_coalition: Model predictions from coalition data  (num_nodes,)
            predictions_original:  Model predictions from original data   (num_nodes,)
            threshold:             Unused. Kept only for backward-compatible call
                                    signatures (e.g. explain()); no longer affects
                                    the calculation since label-flip counting was
                                    replaced with mean absolute shift for both task types.

        Returns:
            importance_score: scalar float
        """
        # Mean absolute shift — continuous, unit-preserving, same formula
        # regardless of task_type (and regardless of ndim: for multi-class /
        # multi-output predictions this averages the per-output shift too).
        return float(np.abs(predictions_coalition - predictions_original).mean())

  
    def solve_wls(self, Z_matrix: np.ndarray, y_values: np.ndarray,
                  weights: np.ndarray, fx: float, fnull: float,
                  l1_reg: "str | float | bool | None" = None) -> np.ndarray:
        """
        Solve weighted least squares to get SHAP values.

        This matches the official SHAP implementation:
        - Eliminate last feature with constraint that all SHAP values sum to output
        - Solve: (X'WX)φ = X'Wy

        Args:
            Z_matrix: Coalition matrix (num_coalitions, M)
            y_values: Target values (num_coalitions,) 
            weights: Kernel weights (num_coalitions,)
            fx: Prediction with all features
            fnull: Prediction with no features
            l1_reg: Optional feature-selection step, mirroring the reference
                implementation's `l1_reg` (see ``_kernel.py``'s ``solve()``).
                The reference DEFAULTS to ``"num_features(10)"``, which would
                zero out all but 10 of your (up to 114) features — almost
                certainly not what you want for a full global feature
                ranking. So this stays OFF (``None``) unless you explicitly
                opt in:
                  - ``"num_features(k)"``: keep only the top-k features
                    (via ``lars_path``), zero out the rest.
                  - ``"aic"`` / ``"bic"``: adaptive sparsity via
                    ``LassoLarsIC``.
                  - a float: fixed L1 alpha via ``Lasso``.
                  - ``None`` / ``False`` / ``0`` (default): no feature
                    selection — solve for all M features directly, as
                    before.

        Returns:
            phi: SHAP values of shape (M,)
        """
        # Adjust y values relative to baseline
        eyAdj = y_values - fnull

        nonzero_inds = np.arange(self.M)
        if l1_reg not in (None, False, 0):
            # Mirrors _kernel.py's `solve()`: build the augmented
            # (mask, 1-mask) system and select a sparse support before the
            # final elimination-trick solve below.
            s = np.sum(Z_matrix, axis=1)
            w_aug = np.hstack((weights * (self.M - s), weights * s))
            w_sqrt_aug = np.sqrt(w_aug)
            eyAdj_aug = np.hstack((eyAdj, eyAdj - (fx - fnull)))
            eyAdj_aug = eyAdj_aug * w_sqrt_aug
            mask_aug = np.transpose(w_sqrt_aug * np.transpose(np.vstack((Z_matrix, Z_matrix - 1))))

            if isinstance(l1_reg, str) and l1_reg.startswith("num_features("):
                r = int(l1_reg[len("num_features("):-1])
                nonzero_inds = lars_path(mask_aug, eyAdj_aug, max_iter=r)[1]
            elif l1_reg in ("aic", "bic"):
                model = make_pipeline(StandardScaler(with_mean=False), LassoLarsIC(criterion=l1_reg))
                nonzero_inds = np.nonzero(model.fit(mask_aug, eyAdj_aug)[1].coef_)[0]
            else:
                nonzero_inds = np.nonzero(Lasso(alpha=l1_reg).fit(mask_aug, eyAdj_aug).coef_)[0]

            if len(nonzero_inds) == 0:
                return np.zeros(self.M)

        # Eliminate one selected feature with the constraint that all
        # (selected) SHAP values sum to the target output
        last = nonzero_inds[-1]
        eyAdj2 = eyAdj - Z_matrix[:, last] * (fx - fnull)
        etmp = Z_matrix[:, nonzero_inds[:-1]] - Z_matrix[:, last:last + 1]

        # Solve weighted least squares: (X'WX)φ = X'Wy
        WX = weights[:, None] * etmp

        try:
            w = np.linalg.solve(etmp.T @ WX, WX.T @ eyAdj2)
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix, using least squares")
            sqrt_W = np.sqrt(weights)
            w = np.linalg.lstsq(sqrt_W[:, None] * etmp, sqrt_W * eyAdj2, rcond=None)[0]

        # Reconstruct full phi (zeros for any feature dropped by l1_reg)
        phi = np.zeros(self.M)
        phi[nonzero_inds[:-1]] = w
        phi[last] = (fx - fnull) - np.sum(w)

        # Clean up rounding errors
        phi[np.abs(phi) < 1e-10] = 0

        return phi
    
    
    def explain_importance_score_aggregation(self,
                                             Z_matrix: np.ndarray,
                                             f_values_coalition: np.ndarray,
                                             f_values_original: np.ndarray,
                                             weights: np.ndarray,
                                             fx_importance: float,
                                             fnull_importance: float,
                                             threshold: float = 0.5,
                                             l1_reg: "str | float | bool | None" = None) -> Dict[str, float]:
        """
        Importance Score Aggregation.

        Computes f(z') = mean absolute prediction shift between coalition
        predictions and the original predictions, then solves weighted least
        squares to get global SHAP values. This is now the SAME calculation
        for both classification and regression tasks:

            f(z') = mean |f(coalition) - f(original)| across all nodes.
            SHAP values are in units of the model's output (predicted
            probability for classification, target variable for regression).

        (Previously, classification used a label-flip count against a
        decision threshold instead. That branch has been removed so both
        task types share one continuous, threshold-independent metric.)

        Args:
            Z_matrix:             Coalition matrix (num_coalitions, M) — the
                                   SAMPLED regression coalitions only. Unlike
                                   an earlier version of this code, this no
                                   longer includes the all-zeros/all-ones
                                   boundary masks: those are handled purely
                                   analytically via fx_importance/
                                   fnull_importance below, exactly like the
                                   reference SHAP implementation treats
                                   self.fx/self.fnull (fixed scalars used
                                   only in the efficiency constraint, never
                                   literal weighted rows in the regression).
            f_values_coalition:   Predictions from coalitions (num_coalitions, num_nodes)
            f_values_original: Predictions from original data (num_nodes,)
            weights:           SHAP kernel weights for each row of Z_matrix,
                               as returned by sample_coalitions() (NOT
                               recomputed here — see that method for how
                               enumerated vs. randomly-sampled rows get
                               their weight).
            fx_importance:     Importance score of the "all features present"
                               boundary coalition. Computed once in
                               explain() (it's exactly 0.0 analytically,
                               since that coalition IS the original data).
            fnull_importance:  Importance score of the "no features present"
                               boundary coalition. Computed once in
                               explain() from an explicit extra model pass.
            threshold: Unused — kept only for backward-compatible call signatures.
            l1_reg: Optional feature-selection passthrough to solve_wls()
                (default None = off, solves for all M features). See
                solve_wls() docstring.

        Returns:
            Dict[str, float] mapping feature names to SHAP values
        """
        print("\n" + "=" * 70)
        print("APPROACH 3: IMPORTANCE SCORE AGGREGATION")
        print("=" * 70)

        num_coalitions = f_values_coalition.shape[0]
        num_nodes = f_values_original.shape[0]
        importance_values = np.zeros(num_coalitions)

        print(f"\n Strategy: f(z') = mean |f(coalition) - f(original)| across all nodes")
        print("           (same formula for classification and regression)")
        print("           Measures which features cause prediction instability")

        with timed_section("Computing importance-score values for all coalitions"):
            print(f"\n Computing importance-score values for each coalition...")
            for i in tqdm(range(num_coalitions), desc="Computing importance scores"):
                importance_values[i] = self.compute_importance_score(
                    f_values_coalition[i],
                    f_values_original,
                    threshold=threshold,
                )

        print(f"\n Mean Absolute Prediction Shift:")
        print(f"   With all features:  {fx_importance:.4f} (avg |Δpred| vs original, over {num_nodes} nodes)")
        print(f"   With no features:   {fnull_importance:.4f} (avg |Δpred| vs original)")
        print(f"   Difference:         {fx_importance - fnull_importance:.4f}")
        print(f"   Mean across coalitions: {importance_values.mean():.4f}")
        print(f"   Std across coalitions:  {importance_values.std():.4f}")

        # Solve weighted least squares for the importance score
        with timed_section("Solving weighted least squares"):
            print(f"\n Solving weighted least squares for mean |Δpred|...")
            phi = self.solve_wls(
                Z_matrix, importance_values, weights,
                fx_importance, fnull_importance, l1_reg=l1_reg,
            )

        print(f"\n Results:")
        print(f"   Sum of φ: {np.sum(phi):.4f} (should equal {fx_importance - fnull_importance:.4f})")
        print(f"   Range: [{phi.min():.6f}, {phi.max():.6f}]")
        print(f"   Features with positive φ: removing them increases avg prediction shift")
        print(f"   Features with negative φ: removing them decreases avg prediction shift")

        # Create results dictionary
        shap_values = OrderedDict()
        for idx, feature_name in enumerate(self.feature_names):
            shap_values[feature_name] = phi[idx]

        return shap_values 
    
    def explain(self,
                loader: Any,
                num_coalitions: int = 3000,
                max_batches: int = 10,
                disagreement_threshold: float = 0.5,
                checkpoint_every: int = None,
                checkpoint_dir: Any = None,
                checkpoint_prefix: str = "kernelShap_checkpoint",
                l1_reg: "str | float | bool | None" = None) -> Dict[str, float]:
        """
        Compute global Kernel SHAP values using importance-score aggregation.

        Same calculation for BOTH task types:
            f(z') = mean absolute prediction shift vs the original data.

        For binary classification, predictions are sigmoid-applied
        probabilities, so SHAP values are in units of average probability
        shift. For regression, SHAP values are in units of the target
        variable (e.g. race positions). (Classification previously counted
        label flips against a decision threshold instead — that has been
        replaced so both task types are computed identically.)

        Args:
            loader:         DataLoader for evaluation
            num_coalitions: Number of REGRESSION coalitions to draw (via
                sample_coalitions() — see there). The all-zeros/all-ones
                boundary coalitions are evaluated separately, in addition
                to this count, and never counted against this budget or
                included as regression rows (see below).
            max_batches:    Maximum batches to process per coalition
            disagreement_threshold: Unused — retained only for backward-compatible
                call signatures (older scripts may still pass this).
            checkpoint_every: If set (e.g. 30), save an intermediate SHAP-values
                CSV to `checkpoint_dir` every `checkpoint_every` coalitions
                evaluated, so partial results are visible/recoverable while a
                long run is still in progress. The checkpoint at coalition i
                is computed via a WLS solve over just the first i sampled
                coalitions collected so far (fx_importance/fnull_importance
                are already fixed before sampling starts, so every
                checkpoint — even the first — has valid boundary values) —
                it is an approximation that becomes more accurate as more
                coalitions accumulate, not a partial view of the final
                result. Pass None (default) to disable checkpointing entirely.
            checkpoint_dir: Directory to write checkpoint CSVs to. Required if
                checkpoint_every is set.
            checkpoint_prefix: Filename prefix for checkpoint CSVs. Files are
                named f"{checkpoint_prefix}_{n}coalitions.csv".
            l1_reg: Optional feature-selection passthrough to solve_wls() /
                explain_importance_score_aggregation() (default None = off,
                solves for all M features — see solve_wls() docstring for
                why this differs from the reference implementation's
                default of "num_features(10)").

        Returns:
            Dict[str, float] mapping feature names to SHAP values
        """
        if checkpoint_every is not None and checkpoint_dir is None:
            raise ValueError(
                "checkpoint_every was set but checkpoint_dir is None — "
                "pass an output directory to write checkpoint CSVs to."
            )
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
        explain_start_time = time.time()

        print("\n" + "=" * 70)
        print("COMPUTING GLOBAL KERNEL SHAP VALUES")
        print("=" * 70)
        print(f"\n Task type: {self.task_type.upper()}")
        print(f" Aggregation: IMPORTANCE SCORE (mean |Δpred|, same formula for both task types)")

        # Get original predictions (all features present) as reference.
        # This ALSO doubles as the "all features present" boundary
        # coalition: since data_original IS the original data, its
        # importance score against itself is exactly 0.0 analytically —
        # no need to separately evaluate an all-ones coalition, and no
        # arbitrary weight is ever assigned to it (it's never a WLS row).
        with timed_section("Getting baseline predictions from original data"):
            fix_randomness(self.loader_config.get('seed', 42))
            print("\n Getting baseline predictions from ORIGINAL data (z_1)...")
            z_original = np.ones(self.M, dtype=int)
            data_original = self.create_coalition_data(z_original)
            f_values_original = self.f_model_all_predictions(loader, data_original, max_batches)
            print(f"   Original predictions shape: {f_values_original.shape}")
            print(f"   Original predictions — min: {f_values_original.min():.4f}, "
                  f"max: {f_values_original.max():.4f}, mean: {f_values_original.mean():.4f}")
        fx_importance = 0.0

        # Get the "no features present" boundary coalition — the other fixed
        # scalar used purely analytically in solve_wls()'s efficiency
        # constraint. It is NEVER added as a weighted row to Z_matrix/weights
        # below.
        with timed_section("Getting baseline predictions from the empty coalition"):
            fix_randomness(self.loader_config.get('seed', 42))
            print("\n Getting baseline predictions from the EMPTY coalition (z_0)...")
            z_empty = np.zeros(self.M, dtype=int)
            data_empty = self.create_coalition_data(z_empty)
            f_values_empty = self.f_model_all_predictions(loader, data_empty, max_batches)
            fnull_importance = self.compute_importance_score(f_values_empty, f_values_original)
            print(f"   fnull_importance (mean |Δpred| with no features): {fnull_importance:.4f}")

        # Sample REGRESSION coalitions (subset sizes 1..M-1 only) together
        # with their proper Shapley kernel weights — see sample_coalitions()
        # for how enumerated vs. randomly-sampled rows are weighted.
        with timed_section(f"Sampling {num_coalitions} coalitions"):
            fix_randomness(self.loader_config.get('seed', 42))
            print(f"\n Sampling {num_coalitions} coalitions...")
            coalitions, coalition_weights = self.sample_coalitions(num_coalitions)
            print(f"   Sampled {len(coalitions)} regression coalitions "
                  f"(requested {num_coalitions}; fewer only if the full "
                  f"subset space was exhaustively enumerated already)")

        Z_matrix = []
        f_values  = []
        weights   = []
        n_sampled = len(coalitions)

        with timed_section(f"Evaluating model on {n_sampled} coalitions"):
            print("\n Evaluating model on coalitions...")
            for coalition_idx, (z_prime, w) in enumerate(
                zip(tqdm(coalitions, desc="Coalitions"), coalition_weights), start=1
            ):
                fix_randomness(self.loader_config.get('seed', 42))
                data_coalition = self.create_coalition_data(z_prime)
                predictions    = self.f_model_all_predictions(loader, data_coalition, max_batches)
                Z_matrix.append(z_prime)
                f_values.append(predictions)
                weights.append(w)

                # --- Periodic checkpoint of intermediate SHAP values ---
                # fx_importance/fnull_importance are already fixed above, so
                # every checkpoint (even the very first) uses valid boundary
                # values — unlike before, there's no dependency on having
                # collected specific leading rows first.
                if (checkpoint_every is not None
                        and (coalition_idx % checkpoint_every == 0
                             or coalition_idx == n_sampled)):
                    with timed_section(
                        f"Checkpoint @ {coalition_idx}/{n_sampled} coalitions"
                    ):
                        Z_partial = np.array(Z_matrix)
                        f_partial = np.array(f_values)
                        w_partial = np.array(weights)

                        checkpoint_shap_values = self.explain_importance_score_aggregation(
                            Z_partial, f_partial, f_values_original, w_partial,
                            fx_importance, fnull_importance,
                            disagreement_threshold, l1_reg=l1_reg,
                        )

                        checkpoint_path = os.path.join(
                            checkpoint_dir,
                            f"{checkpoint_prefix}_{coalition_idx}coalitions.csv",
                        )
                        self.save_results(
                            checkpoint_shap_values, checkpoint_path,
                            disagreement_threshold=disagreement_threshold,
                        )
                        print(f"   ✓ Checkpoint saved: {checkpoint_path} "
                              f"({coalition_idx}/{n_sampled} coalitions)")

        Z_matrix = np.array(Z_matrix)  # (num_coalitions, M)
        f_values  = np.array(f_values)  # (num_coalitions, num_nodes)
        weights   = np.array(weights)   # (num_coalitions,)

        print(f"\n Data collected:")
        print(f"   Z_matrix: {Z_matrix.shape} (coalitions × features)")
        print(f"   f_values: {f_values.shape} (coalitions × nodes)")
        print(f"   weights:  {weights.shape}")

        fix_randomness(self.loader_config.get('seed', 42))
        with timed_section("Importance-score aggregation (WLS solve)"):
            result = self.explain_importance_score_aggregation(
                Z_matrix, f_values, f_values_original, weights,
                fx_importance, fnull_importance,
                disagreement_threshold, l1_reg=l1_reg,
            )

        total_elapsed = time.time() - explain_start_time
        print(f"\n⏱  [explain() TOTAL] completed in {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
        return result

    def sample_coalitions(self, num_samples: int) -> Tuple[List[np.ndarray], List[float]]:
        """
        Sample REGRESSION coalitions (subset sizes strictly between 0 and M)
        for Kernel SHAP approximation, together with their Shapley kernel
        weights — mirroring the mechanism used by the reference
        implementation, ``shap.explainers._kernel.KernelExplainer.explain``
        (see ``_kernel.py``), rather than either uniform random sampling or
        a single closed-form weight formula applied everywhere.

        The mechanism, matching the original in both what gets sampled and
        how it gets weighted:
          1. Subset sizes ``k = 1..M-1`` are weighted by the Shapley kernel
             subset-size weight ``w(k) = (M - 1) / (k * (M - k))``, sizes
             ``k`` and ``M - k`` are "paired" (drawing one gives you its
             complement for free) and their weight is doubled to reflect
             that pairing. ``weight_vector`` is normalized to sum to 1.
          2. Starting from the most heavily-weighted (smallest paired)
             sizes, subset sizes are FULLY enumerated (every ``C(M, k)``
             mask of that size) for as long as the coalition budget
             allows. Each enumerated mask gets the exact analytic weight
             ``weight_vector[k-1] / C(M, k)`` (halved for paired sizes,
             since the pair together carries ``weight_vector[k-1]``) —
             this is correct precisely because the FULL population of
             masks of that size is being used, so dividing the size's
             total weight evenly across all of them is exact.
          3. Whatever budget remains after full enumeration is filled with
             random samples, drawn with probability proportional to the
             weight remaining across the not-fully-enumerated sizes, in
             COMPLEMENT PAIRS (``mask``, ``1 - mask``) for paired sizes.
             Unlike the enumerated case, only a tiny fraction of the
             possible masks in this region are ever drawn, so each drawn
             mask does NOT get the analytic per-mask weight (which would
             be ~0 relative to the enumerated masks and effectively erase
             these coalitions' influence on the fit). Instead, exactly as
             the original does: each unique mask starts at weight 1.0, a
             repeated (duplicate) draw of the same mask INCREMENTS that
             existing mask's weight rather than being discarded, and once
             sampling is done the whole random batch's weights are
             rescaled so they sum to ``weight_left`` — the total kernel-
             weight mass belonging to the un-enumerated sizes combined.
             This is what makes the random draws a valid (Horvitz–
             Thompson-style) estimate of that stratum's contribution.

        Custom adaptation from this framework's earlier version: the
        all-zeros (size 0) and all-ones (size M) masks are NO LONGER
        returned here at all (previously they were two fixed leading
        coalitions). explain() now evaluates them separately and uses them
        purely analytically (as fx_importance/fnull_importance), exactly
        mirroring how the reference explainer treats self.fx/self.fnull —
        never as literal weighted rows in the regression. This avoids the
        previous version's issue of assigning them an arbitrary constant
        weight (10000.0) and effectively double-counting the boundary
        information (once as WLS rows, once via the elimination-trick
        constraint).

        Returns:
            coalitions:     list of M-length 0/1 arrays (subset sizes 1..M-1)
            sample_weights: parallel list of Shapley kernel weights
        """
        M = self.M
        coalitions: List[np.ndarray] = []
        sample_weights: List[float] = []

        if num_samples <= 0 or M <= 1:
            return coalitions, sample_weights

        # weight the different subset sizes (identical formula to _kernel.py)
        num_subset_sizes = int(np.ceil((M - 1) / 2.0))
        num_paired_subset_sizes = int(np.floor((M - 1) / 2.0))
        weight_vector = np.array(
            [(M - 1.0) / (k * (M - k)) for k in range(1, num_subset_sizes + 1)]
        )
        weight_vector[:num_paired_subset_sizes] *= 2
        weight_vector /= np.sum(weight_vector)

        # --- fully enumerate subset sizes while the budget allows ---
        num_full_subsets = 0
        num_samples_left = num_samples
        remaining_weight_vector = copy.copy(weight_vector)
        for subset_size in range(1, num_subset_sizes + 1):
            nsubsets = comb(M, subset_size)
            if subset_size <= num_paired_subset_sizes:
                nsubsets *= 2

            # see if we have enough budget to enumerate all subsets of this size
            if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                num_full_subsets += 1
                num_samples_left -= nsubsets

                # rescale what's left of the remaining weight vector to sum to 1
                if remaining_weight_vector[subset_size - 1] < 1.0:
                    remaining_weight_vector /= 1 - remaining_weight_vector[subset_size - 1]

                # exact analytic weight per mask of this size (see docstring)
                w = weight_vector[subset_size - 1] / comb(M, subset_size)
                if subset_size <= num_paired_subset_sizes:
                    w /= 2.0

                for inds in itertools.combinations(range(M), subset_size):
                    mask = np.zeros(M, dtype=int)
                    mask[list(inds)] = 1
                    coalitions.append(mask)
                    sample_weights.append(w)
                    if subset_size <= num_paired_subset_sizes:
                        coalitions.append(1 - mask)
                        sample_weights.append(w)
            else:
                break

        # --- random samples from what is left of the subset space ---
        nfixed = len(coalitions)
        samples_left = num_samples - nfixed
        if num_full_subsets != num_subset_sizes and samples_left > 0:
            remaining_weight_vector = copy.copy(weight_vector)
            remaining_weight_vector[:num_paired_subset_sizes] /= 2  # we draw a pair per pick below
            remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
            remaining_weight_vector /= np.sum(remaining_weight_vector)

            ind_set = np.random.choice(len(remaining_weight_vector), 4 * samples_left, p=remaining_weight_vector)
            ind_set_pos = 0
            used_masks: Dict[tuple, int] = {}  # mask tuple -> index into coalitions/sample_weights
            while samples_left > 0 and ind_set_pos < len(ind_set):
                ind = ind_set[ind_set_pos]
                ind_set_pos += 1
                subset_size = ind + num_full_subsets + 1

                mask = np.zeros(M, dtype=int)
                mask[np.random.permutation(M)[:subset_size]] = 1
                mask_tuple = tuple(mask)

                # only add a new row if we have not seen this mask before —
                # otherwise bump the EXISTING row's weight, exactly like the
                # original (never discard/redraw a duplicate)
                new_sample = False
                if mask_tuple not in used_masks:
                    new_sample = True
                    used_masks[mask_tuple] = len(coalitions)
                    coalitions.append(mask)
                    sample_weights.append(1.0)
                    samples_left -= 1
                else:
                    sample_weights[used_masks[mask_tuple]] += 1.0

                if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                    complement = 1 - mask
                    complement_tuple = tuple(complement)
                    if new_sample:
                        used_masks[complement_tuple] = len(coalitions)
                        coalitions.append(complement)
                        sample_weights.append(1.0)
                        samples_left -= 1
                    else:
                        # the complement is the row right after the
                        # original sample (mirrors the original's "+1")
                        sample_weights[used_masks[mask_tuple] + 1] += 1.0

            # safety fallback: in the unlikely event the 4x draw above still
            # left us short (e.g. many duplicate collisions), top up with
            # fresh weighted draws until the budget is exactly met.
            while samples_left > 0:
                ind = np.random.choice(len(remaining_weight_vector), p=remaining_weight_vector)
                subset_size = ind + num_full_subsets + 1
                mask = np.zeros(M, dtype=int)
                mask[np.random.permutation(M)[:subset_size]] = 1
                mask_tuple = tuple(mask)
                if mask_tuple not in used_masks:
                    used_masks[mask_tuple] = len(coalitions)
                    coalitions.append(mask)
                    sample_weights.append(1.0)
                    samples_left -= 1
                else:
                    sample_weights[used_masks[mask_tuple]] += 1.0

            # normalize the random batch's weights to sum to the kernel-weight
            # mass left over after full enumeration — exactly mirrors
            # `self.kernelWeights[nfixed_samples:] *= weight_left / ....sum()`
            # in the reference implementation. THIS is the step that was
            # missing before, and it's what makes the randomly-sampled
            # coalitions carry meaningful (rather than ~0) weight in the fit.
            weight_left = float(np.sum(weight_vector[num_full_subsets:]))
            random_weights = np.array(sample_weights[nfixed:])
            random_weights *= weight_left / random_weights.sum()
            sample_weights[nfixed:] = random_weights.tolist()

        return coalitions, sample_weights
     
    def save_results(self, results: Dict[str, Any], filepath: str,
                     disagreement_threshold: float = None):
        """
        Save SHAP values to CSV.

        Args:
            results: Dict[str, float] mapping feature names to SHAP values
            filepath: Output file path
            disagreement_threshold: No longer used to compute the importance
                score (classification and regression now share the same
                mean-|Δpred| calculation). Kept as an optional constant
                column, written for backward compatibility / record-keeping
                only. Pass None to omit the column.
        """
        # Handle both result formats
        if 'shap_values' in results:
            df = pd.DataFrame([
                {
                    'feature': name,
                    'shap_value': results['shap_values'][name],
                    'shap_std': results['shap_std'][name],
                    'shap_min': results['shap_min'][name],
                    'shap_max': results['shap_max'][name],
                    'abs_shap_value': abs(results['shap_values'][name])
                }
                for name in results['shap_values'].keys()
            ])
        else:
            df = pd.DataFrame([
                {
                    'feature': name,
                    'shap_value': value,
                    'abs_shap_value': abs(value)
                }
                for name, value in results.items()
            ])

        if disagreement_threshold is not None:
            df['disagreement_threshold'] = disagreement_threshold

        df = df.sort_values('abs_shap_value', ascending=False)
        df.to_csv(filepath, index=False)
        print(f"\n Results saved to: {filepath}")
        if disagreement_threshold is not None:
            print(f"   disagreement_threshold = {disagreement_threshold:.4f}")


# ============================================================================
# RELGT KERNEL SHAP EXPLAINER
# ============================================================================

class RelGTGlobalKernelSHAP(GlobalKernelSHAP):
    """
    KernelSHAP explainer for the RelGT (transformer) model.

    Overrides f_model_all_predictions() to use RelGTTokens + DataLoader
    and the RelGT forward pass interface instead of NeighborLoader.

    The coalition feature ablation (create_coalition_data) is inherited
    unchanged — it operates on the shared HeteroData object which RelGT
    reads node features from identically to HGS.
    """

    def __init__(self, *args, relgt_tokens_kwargs: dict, **kwargs):
        """
        Args:
            relgt_tokens_kwargs: Dict passed directly to RelGTTokens.
                Must contain: data, task, K, precomputed_dir.
                Optional: undirected, num_workers.
            All other args/kwargs forwarded to GlobalKernelSHAP.
        """
        super().__init__(*args, **kwargs)
        self._relgt_tokens_kwargs = relgt_tokens_kwargs

    def verify_relbench_data_integrity(self):
        """
        Proactively verify every table this run will need is actually
        readable from disk, BEFORE committing to what can be a many-hour
        (or multi-day) coalition loop.

        RelGTTokens.__init__ calls self.task.get_table(split) on EVERY
        single coalition rebuild — RelBench 3 removed all caching from
        get_db()/get_table() ("both are pure and uncached... calling
        dataset.get_db() in a loop now re-reads the data every time"). That
        means any file missing or corrupted in the local Hugging Face cache
        — as has happened repeatedly, across multiple different datasets
        and multiple different files within the same dataset — WILL
        eventually be hit by some coalition. Far better to find out in
        seconds, here, than 175 minutes or 5 days into an actual run.

        This deliberately checks both halves of what get_table() can touch:
        the underlying database tables (via task.dataset.get_db()) AND the
        task-specific label parquet files (via task.get_table(split) for
        every split) — a dataset's raw db tables can check out fine while
        its task-specific parquet is still missing, or vice versa, as seen
        in practice on rel-arxiv/author-publication.
        """
        task = self._relgt_tokens_kwargs.get("task")
        if task is None:
            raise ValueError(
                "relgt_tokens_kwargs['task'] is required for the data "
                "integrity check — pass it through from run_kernelShap_explainer.py."
            )

        print("\n" + "=" * 70)
        print("VERIFYING DATA INTEGRITY (before starting a potentially long run)")
        print("=" * 70)

        try:
            print(" Checking database tables...")
            db = task.dataset.get_db()
            for table in db.table_dict.values():
                _ = len(table.df)
            print(f" ✓ {len(db.table_dict)} database tables OK")
        except Exception as e:
            raise RuntimeError(
                f"Data integrity check FAILED reading a database table: {e}\n"
                f"The local RelBench/Hugging Face cache is missing or "
                f"corrupted for this dataset. Fix this BEFORE starting "
                f"explain() — clear and re-fetch the affected Hub repo's "
                f"local cache, then rerun this check."
            ) from e

        for split in ("train", "val", "test"):
            try:
                print(f" Checking task split '{split}'...")
                table = task.get_table(split)
                _ = len(table.df)
            except Exception as e:
                raise RuntimeError(
                    f"Data integrity check FAILED reading task split "
                    f"'{split}': {e}\nThe local RelBench/Hugging Face cache "
                    f"is missing or corrupted for this task. Fix this "
                    f"BEFORE starting explain() — otherwise this failure "
                    f"will surface unpredictably partway through a "
                    f"potentially many-hour coalition run instead of now."
                ) from e

        print(" ✓ Data integrity check passed — safe to proceed.\n")

    def explain(self, *args, **kwargs):
        """Runs the pre-flight data integrity check, then delegates to
        GlobalKernelSHAP.explain() unchanged. See
        verify_relbench_data_integrity() for why this check exists."""
        self.verify_relbench_data_integrity()
        return super().explain(*args, **kwargs)

    def _make_relgt_loader(self, split: str = "test", data_override=None):
        """Build a RelGTTokens DataLoader for the given split.

        Args:
            split:         Which split to load ("val", "test", etc.)
            data_override: If provided, use this HeteroData object instead of
                           the original data. Pass data_coalition here so that
                           ablated features are reflected in the token representations.

        RelGTTokens is imported in run_kernelShap_explainer.py where sys.path
        is already set up, and passed in via relgt_tokens_kwargs["RelGTTokens"].
        """
        from torch.utils.data import DataLoader

        RelGTTokens = self._relgt_tokens_kwargs.get("RelGTTokens")
        if RelGTTokens is None:
            raise ValueError(
                "RelGTTokens class must be passed in relgt_tokens_kwargs['RelGTTokens']. "
                "Import it in run_kernelShap_explainer.py and pass it through."
            )

        # Build kwargs without the class key, replacing data if override provided
        token_kwargs = {k: v for k, v in self._relgt_tokens_kwargs.items()
                        if k != "RelGTTokens"}
        if data_override is not None:
            token_kwargs["data"] = data_override

        # Force RelGTTokens's OWN internal precompute parallelism to a real
        # worker count. RelGTTokens.__init__ guards _precompute_sampling()
        # behind `if os.path.exists(precomputed_path)` internally, so
        # multiprocessing.Pool is only ever created on a genuine cache miss —
        # num_workers is unused (harmless) when the cache exists, and must be
        # >=1 when it doesn't (Pool(processes=0) raises ValueError).
        token_kwargs["num_workers"] = max(1, self._relgt_tokens_kwargs.get("num_workers", 4) or 4)
        
        ds = RelGTTokens(
            split=split,
            precompute=True,
            **token_kwargs,
        )
        batch_size = self.loader_config.get("batch_size", 64)

        # A nonzero worker count here is unsafe: this DataLoader is rebuilt
        # fresh for EVERY coalition while self.data is being mutated/restored
        # around it, so a forked worker can get a torn view of a
        # MultiEmbeddingTensor's values/offset pair. num_workers=0 removes
        # the worker-process boundary entirely (same fix applied in
        # MonteCarloShapleyRelGT and singleColumnAblation_explainer.py).
        num_workers = 0

        # RelGTTokens builds itself from the FULL split (it reads the task table
        # itself and pulls neighbors from the precomputed HDF5 cache by position),
        # so restrict it to the explanation subset here rather than shortening the
        # table — a shortened table would silently read the first N cached rows
        # instead of the subset's rows. Subset has no .collate, hence capturing it.
        collate_fn = ds.collate
        if self.test_subset_idx is not None and split == "test":
            from torch.utils.data import Subset
            ds = Subset(ds, [int(i) for i in self.test_subset_idx])

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=False,
            pin_memory=(self.device.type == "cuda"),
        )

    def f_model_all_predictions(self, loader: Any, data_coalition: Any,
                                max_batches: int) -> np.ndarray:
        """
        Get all predictions for a coalition using the RelGT forward pass.

        The NeighborLoader `loader` is ignored — RelGT uses RelGTTokens which
        reads features directly from the data object. We pass data_coalition
        (which has the ablated features) to RelGTTokens so the ablation is
        reflected in the token representations and thus in the predictions.
        """
        self.model.eval()
        all_predictions = []

        # Build a RelGT loader over the ABLATED data (data_coalition)
        relgt_loader = self._make_relgt_loader(split="test",
                                               data_override=data_coalition)

        with torch.no_grad():
            fix_randomness(self.loader_config.get("seed", 42))
            for batch_idx, batch in enumerate(relgt_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                # Move every tensor in the batch to device
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
                    pred = apply_sigmoid(pred)

                all_predictions.append(pred.detach().cpu().numpy())

        return np.concatenate(all_predictions, axis=0)