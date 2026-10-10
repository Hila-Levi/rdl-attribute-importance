"""
Configuration file for the Relational Graph Transformer (RelGT) model.

Mirrors the structure of train_config.py but carries Transformer-specific
hyperparameters (K, heads, channels, etc.) and points saved-model paths to
the task's  transformer/saved_model  sub-directory so results stay isolated
from the HGS pipeline.

Directory convention (per dataset / task):
    rel-<n>/
        data_<n>/                   ← shared raw/processed data (HGS cache lives here)
        <task-name>/
            HGS/
            │   saved_model/
            │   ...
            └── transformer/
                ├── saved_model/    ← this config writes here
                ├── relgt-cache/    ← token cache (written by RelGTTokens)
                └── ...
"""

from torch.nn import BCEWithLogitsLoss, L1Loss


# ============================================================================
# DATASET CONFIGURATIONS
# (identical to train_config.py — shared data roots, no transformer-specific
#  overrides needed at the dataset level)
# ============================================================================

DATASET_CONFIG = {
    "rel-arxiv": {
        "root_dir_name": "../rel-arxiv/data_arxiv",
        "cache_suffix": "_materialized_cache",
        "database_description": "rel-arxiv database",
    },
    "rel-trial": {
        "root_dir_name": "../rel-trial/data_trial",
        "cache_suffix": "_materialized_cache",
        "database_description": "rel-trial database",
    },
    "rel-f1": {
        "root_dir_name": "../rel-f1/data_f1",
        "cache_suffix": "_materialized_cache",
        "database_description": "rel-f1 database",
    },
    "rel-event": {
        "root_dir_name": "../rel-event/data_event",
        "cache_suffix": "_materialized_cache",
        "database_description": "rel-event database",
    },
}


# ============================================================================
# TASK CONFIGURATIONS
# saved_model_dir_name now points to  transformer/saved_model
# ============================================================================

TASK_CONFIG = {
    # -------------------------------------------------------------------------
    # rel-arxiv
    # -------------------------------------------------------------------------
    "paper-citation": {
        "saved_model_dir_name": "../rel-arxiv/paper-citation/transformer/saved_model",
        "out_channels": 1,
        "tune_metric": "roc_auc",
        "higher_is_better": True,
        "loss_fn": BCEWithLogitsLoss(),
        "target_col": "cited",
        "task_description": "paper-citation",
        "task_type": "classification",
        "label_names": {
            0: "Not Cited",
            1: "Cited",
        },
    },
    "author-publication": {
        "saved_model_dir_name": "../rel-arxiv/author-publication/transformer/saved_model",
        "out_channels": 1,
        "tune_metric": "mae",
        "higher_is_better": False,
        "loss_fn": L1Loss(),
        "target_col": "published",
        "task_description": "author-publication",
        "task_type": "regression",
        "label_names": None,
    },
    # -------------------------------------------------------------------------
    # rel-trial — classification
    # -------------------------------------------------------------------------
    "study-outcome": {
        "saved_model_dir_name": "../rel-trial/study-outcome/transformer/saved_model",
        "out_channels": 1,
        "tune_metric": "roc_auc",
        "higher_is_better": True,
        "loss_fn": BCEWithLogitsLoss(),
        "target_col": "outcome",
        "task_description": "predict the outcome of a clinical trial (success/failure)",
        "task_type": "classification",
        "label_names": {
            0: "Failure",
            1: "Success",
        },
    },
    # -------------------------------------------------------------------------
    # rel-trial — regression
    # -------------------------------------------------------------------------
    "study-adverse": {
        "saved_model_dir_name": "../rel-trial/study-adverse/transformer/saved_model",
        "out_channels": 1,
        "tune_metric": "mae",
        "higher_is_better": False,
        "loss_fn": L1Loss(),
        "target_col": "adverse_count",
        "task_description": "predict the number or severity of adverse events",
        "task_type": "regression",
        "label_names": None,
    },
    # -------------------------------------------------------------------------
    # rel-f1 — classification
    # -------------------------------------------------------------------------
    "driver-dnf": {
        "saved_model_dir_name": "../rel-f1/driver-dnf/transformer/saved_model",
        "out_channels": 1,
        "tune_metric": "roc_auc",
        "higher_is_better": True,
        "loss_fn": BCEWithLogitsLoss(),
        "target_col": "did_not_finish",
        "task_description": "predict if a driver will not finish the race (DNF)",
        "task_type": "classification",
        "label_names": {
            0: "Finished",
            1: "DNF",
        },
    },
    # -------------------------------------------------------------------------
    # rel-f1 — regression
    # -------------------------------------------------------------------------
    "driver-position": {
        "saved_model_dir_name": "../rel-f1/driver-position/transformer/saved_model",
        "out_channels": 1,
        "tune_metric": "mae",
        "higher_is_better": False,
        "loss_fn": L1Loss(),
        "target_col": "position",
        "task_description": "predict the finishing position of a driver",
        "task_type": "regression",
        "label_names": None,
    },
    # -------------------------------------------------------------------------
    # rel-event — classification
    # -------------------------------------------------------------------------
    "user-repeat": {
        "saved_model_dir_name": "../rel-event/user-repeat/transformer/saved_model",
        "out_channels": 1,
        "tune_metric": "roc_auc",
        "higher_is_better": True,
        "loss_fn": BCEWithLogitsLoss(),
        "target_col": "repeat",
        "task_description": "predict if a user will repeat/attend an event",
        "task_type": "classification",
        "label_names": {0: "No", 1: "Yes"},
    },
    # -------------------------------------------------------------------------
    # rel-event — regression
    # -------------------------------------------------------------------------
    "user-attendance": {
        "saved_model_dir_name": "../rel-event/user-attendance/transformer/saved_model",
        "out_channels": 1,
        "tune_metric": "mae",
        "higher_is_better": False,
        "loss_fn": L1Loss(),
        "target_col": "attendance",
        "task_description": "predict user attendance metric",
        "task_type": "regression",
        "label_names": None,
    },
}


# ============================================================================
# TRANSFORMER MODEL CONFIGURATIONS
# RelGT-specific hyperparameters.
#
# Key parameters:
#   K                  – number of neighbor tokens per seed node; must equal
#                        RelGTConfig.sample_node_len (both are set to K below)
#   local_num_layers   – number of local GNN layers inside the transformer block
#   channels           – hidden dimension for local GNN / node embeddings
#   global_dim         – hidden dimension for the global transformer
#   heads              – number of attention heads
#   ff_dropout         – feedforward dropout rate
#   attn_dropout       – attention dropout rate
#   conv_type          – "full" | "sparse"  (full = standard self-attention)
#   num_centroids      – centroids for sparse attention (ignored when conv_type="full")
#   max_steps_per_epoch – cap on gradient steps per epoch (None = no cap)
#   grad_clip_norm     – max gradient norm for clip_grad_norm_ (None = disabled)
# ============================================================================

TRANSFORMER_CONFIG = {
    # -------------------------------------------------------------------------
    # rel-arxiv - classification
    # -------------------------------------------------------------------------
    "paper-citation": {
        "K": 100,
        "local_num_layers": 1,
        "channels": 256,
        "global_dim": 128,
        "heads": 4,
        "ff_dropout": 0.1,
        "attn_dropout": 0.1,
        "conv_type": "full",
        "num_centroids": 4096,
    },
    # -------------------------------------------------------------------------
    # rel-arxiv - regression
    # -------------------------------------------------------------------------
    "author-publication": {
        "K": 100,
        "local_num_layers": 1,
        "channels": 128,
        "global_dim": 64,
        "heads": 4,
        "ff_dropout": 0.2,
        "attn_dropout": 0.1,
        "conv_type": "full",
        "num_centroids": 4096,
    },
    # -------------------------------------------------------------------------
    # rel-trial — classification
    # -------------------------------------------------------------------------
    "study-outcome": {
        "K": 300,
        "local_num_layers": 4,
        "channels": 128,
        "global_dim": 128,
        "heads": 4,
        "ff_dropout": 0.5,
        "attn_dropout": 0.3,
        "conv_type": "full",
        "num_centroids": 4096,
    },
    # -------------------------------------------------------------------------
    # rel-trial — regression
    # -------------------------------------------------------------------------
    "study-adverse": {
        "K": 100,
        "local_num_layers": 1,
        "channels": 128,
        "global_dim": 64,
        "heads": 4,
        "ff_dropout": 0.2,
        "attn_dropout": 0.1,
        "conv_type": "full",
        "num_centroids": 4096,
    },
    # -------------------------------------------------------------------------
    # rel-f1 — classification
    # -------------------------------------------------------------------------
    "driver-dnf": {
        "K": 100,
        "local_num_layers": 1,
        "channels": 128,
        "global_dim": 64,
        "heads": 4,
        "ff_dropout": 0.1,
        "attn_dropout": 0.1,
        "conv_type": "full",
        "num_centroids": 4096,
    },
    # -------------------------------------------------------------------------
    # rel-f1 — regression
    # -------------------------------------------------------------------------
    "driver-position": {
        "K": 100,
        "local_num_layers": 1,
        "channels": 128,
        "global_dim": 64,
        "heads": 4,
        "ff_dropout": 0.2,
        "attn_dropout": 0.1,
        "conv_type": "full",
        "num_centroids": 4096,
    },
    # -------------------------------------------------------------------------
    # rel-event — classification
    # -------------------------------------------------------------------------
    "user-repeat": {
        "K": 300,
        "local_num_layers": 4,
        "channels": 256,
        "global_dim": 128,
        "heads": 4,
        "ff_dropout": 0.3,
        "attn_dropout": 0.3,
        "conv_type": "full",
        "num_centroids": 4096,
    },
    # -------------------------------------------------------------------------
    # rel-event — regression
    # -------------------------------------------------------------------------
    "user-attendance": {
        "K": 100,
        "local_num_layers": 1,
        "channels": 128,
        "global_dim": 64,
        "heads": 4,
        "ff_dropout": 0.2,
        "attn_dropout": 0.1,
        "conv_type": "full",
        "num_centroids": 4096,
    },
}


# ============================================================================
# TRAINING CONFIGURATIONS
# ============================================================================

TRAINING_CONFIG = {
    # -------------------------------------------------------------------------
    # rel-arxiv
    # -------------------------------------------------------------------------
    "paper-citation": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "scheduler_factor": 0.5,
        "scheduler_patience": 4,
        "epochs": 92,
        "early_stopping_patience": 92,
        "batch_size": 64,
        "max_steps_per_epoch": 3000,
        "grad_clip_norm": 1.0,
        "scheduler_verbose": True,
    },
    # -------------------------------------------------------------------------
    # rel-arxiv — regression
    # -------------------------------------------------------------------------
    "author-publication": {
        "learning_rate": 5e-5,
        "weight_decay": 1e-3,
        "scheduler_factor": 0.5,
        "scheduler_patience": 3,
        "epochs": 100,
        "early_stopping_patience": 100,
        "batch_size": 64,
        "max_steps_per_epoch": 3000,
        "grad_clip_norm": 1.0,
        "scheduler_verbose": True,
    },
    # -------------------------------------------------------------------------
    # rel-trial — classification
    # -------------------------------------------------------------------------
    "study-outcome": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "scheduler_factor": 0.5,
        "scheduler_patience": 4,
        "epochs": 100,
        "early_stopping_patience": 100,
        "batch_size": 256,
        "max_steps_per_epoch": 3000,
        "grad_clip_norm": 1.0,
        "scheduler_verbose": True,
    },
    # -------------------------------------------------------------------------
    # rel-trial — regression
    # -------------------------------------------------------------------------
    "study-adverse": {
        "learning_rate": 5e-5,
        "weight_decay": 1e-3,
        "scheduler_factor": 0.5,
        "scheduler_patience": 3,
        "epochs": 100,
        "early_stopping_patience": 100,
        "batch_size": 64,
        "max_steps_per_epoch": 3000,
        "grad_clip_norm": 1.0,
        "scheduler_verbose": True,
    },
    # -------------------------------------------------------------------------
    # rel-f1 — classification
    # -------------------------------------------------------------------------
    "driver-dnf": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "scheduler_factor": 0.5,
        "scheduler_patience": 4,
        "epochs": 100,
        "early_stopping_patience": 100,
        "batch_size": 128,
        "max_steps_per_epoch": 3000,
        "grad_clip_norm": 1.0,
        "scheduler_verbose": True,
    },
    # -------------------------------------------------------------------------
    # rel-f1 — regression
    # -------------------------------------------------------------------------
    "driver-position": {
        "learning_rate": 5e-5,
        "weight_decay": 1e-3,
        "scheduler_factor": 0.5,
        "scheduler_patience": 3,
        "epochs": 100,
        "early_stopping_patience": 100,
        "batch_size": 64,
        "max_steps_per_epoch": 3000,
        "grad_clip_norm": 1.0,
        "scheduler_verbose": True,
    },
    # -------------------------------------------------------------------------
    # rel-event — classification
    # -------------------------------------------------------------------------
    "user-repeat": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "scheduler_factor": 0.5,
        "scheduler_patience": 5,
        "epochs": 100,
        "early_stopping_patience": 100,
        "batch_size": 1024,
        "max_steps_per_epoch": 3000,
        "grad_clip_norm": 1.0,
        "scheduler_verbose": True,
    },
    # -------------------------------------------------------------------------
    # rel-event — regression
    # -------------------------------------------------------------------------
    "user-attendance": {
        "learning_rate": 5e-5,
        "weight_decay": 1e-3,
        "scheduler_factor": 0.5,
        "scheduler_patience": 3,
        "epochs": 100,
        "early_stopping_patience": 100,
        "batch_size": 64,
        "max_steps_per_epoch": 3000,
        "grad_clip_norm": 1.0,
        "scheduler_verbose": True,
    },
}


# ============================================================================
# LOADER / TOKENISATION CONFIGURATIONS
# Unlike HGS (NeighborLoader), RelGT uses RelGTTokens + standard DataLoader.
# num_workers_precompute – workers for HDF5 token precomputation (one-time)
# num_workers_loader     – workers for DataLoader during training
# ============================================================================

LOADER_CONFIG = {
    "paper-citation": {
        "num_workers_precompute": 4,
        "num_workers_loader": 2,
        "persistent_workers": True,
    },
    "author-publication": {
        "num_workers_precompute": 4,
        "num_workers_loader": 2,
        "persistent_workers": True,
    },
    "study-outcome": {
        "num_workers_precompute": 4,
        "num_workers_loader": 2,
        "persistent_workers": True,
    },
    "study-adverse": {
        "num_workers_precompute": 4,
        "num_workers_loader": 2,
        "persistent_workers": True,
    },
    "driver-dnf": {
        "num_workers_precompute": 4,
        "num_workers_loader": 2,
        "persistent_workers": True,
    },
    "driver-position": {
        "num_workers_precompute": 4,
        "num_workers_loader": 2,
        "persistent_workers": True,
    },
    "user-repeat": {
        "num_workers_precompute": 4,
        "num_workers_loader": 2,
        "persistent_workers": True,
    },
    "user-attendance": {
        "num_workers_precompute": 4,
        "num_workers_loader": 2,
        "persistent_workers": True,
    },
}


# ============================================================================
# SANITY CHECK CONFIGURATIONS
# ============================================================================

SANITY_CHECK_CONFIG = {
    "paper-citation": {
        "sample_size": 100,
        "epochs": 60,
        "batch_size": 32,
        "learning_rate": 0.01,
        "K": 50,
        "channels": 64,
        "global_dim": 32,
        "heads": 2,
        "success_threshold": 0.95,
    },
    "author-publication": {
        "sample_size": 100,
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.01,
        "K": 50,
        "channels": 64,
        "global_dim": 32,
        "heads": 2,
        "success_threshold": 5.0,
    },
    "study-outcome": {
        "sample_size": 100,
        "epochs": 60,
        "batch_size": 32,
        "learning_rate": 0.01,
        "K": 50,
        "channels": 64,
        "global_dim": 32,
        "heads": 2,
        "success_threshold": 0.95,
    },
    "study-adverse": {
        "sample_size": 100,
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.01,
        "K": 50,
        "channels": 64,
        "global_dim": 32,
        "heads": 2,
        "success_threshold": 5.0,    # MAE — lower is better
    },
    "driver-dnf": {
        "sample_size": 100,
        "epochs": 60,
        "batch_size": 32,
        "learning_rate": 0.01,
        "K": 50,
        "channels": 64,
        "global_dim": 32,
        "heads": 2,
        "success_threshold": 0.95,
    },
    "driver-position": {
        "sample_size": 100,
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.01,
        "K": 50,
        "channels": 64,
        "global_dim": 32,
        "heads": 2,
        "success_threshold": 4.0,    # MAE — lower is better
    },
    "user-repeat": {
        "sample_size": 100,
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.01,
        "K": 50,
        "channels": 64,
        "global_dim": 32,
        "heads": 2,
        "success_threshold": 0.90,
    },
    "user-attendance": {
        "sample_size": 100,
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.01,
        "K": 50,
        "channels": 64,
        "global_dim": 32,
        "heads": 2,
        "success_threshold": 1.0,
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_dataset_config(dataset_name: str) -> dict:
    """Get configuration for a specific dataset."""
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in configuration. "
            f"Available datasets: {list(DATASET_CONFIG.keys())}"
        )
    return DATASET_CONFIG[dataset_name]


def get_task_config(task_name: str) -> dict:
    """Get task configuration (paths, loss, metric) for a specific task."""
    if task_name not in TASK_CONFIG:
        raise ValueError(
            f"Task '{task_name}' not found in configuration. "
            f"Available tasks: {list(TASK_CONFIG.keys())}"
        )
    return TASK_CONFIG[task_name]


def get_transformer_config(task_name: str) -> dict:
    """Get RelGT model hyperparameters for a specific task."""
    if task_name not in TRANSFORMER_CONFIG:
        raise ValueError(
            f"Transformer config for task '{task_name}' not found. "
            f"Available tasks: {list(TRANSFORMER_CONFIG.keys())}"
        )
    return TRANSFORMER_CONFIG[task_name]


def get_training_config(task_name: str) -> dict:
    """Get optimiser / scheduler / epoch settings for a specific task."""
    if task_name not in TRAINING_CONFIG:
        raise ValueError(
            f"Training config for task '{task_name}' not found. "
            f"Available tasks: {list(TRAINING_CONFIG.keys())}"
        )
    return TRAINING_CONFIG[task_name]


def get_loader_config(task_name: str) -> dict:
    """Get DataLoader / worker settings for a specific task."""
    if task_name not in LOADER_CONFIG:
        raise ValueError(
            f"Loader config for task '{task_name}' not found. "
            f"Available tasks: {list(LOADER_CONFIG.keys())}"
        )
    return LOADER_CONFIG[task_name]


def get_sanity_check_config(task_name: str) -> dict:
    """Get sanity-check (overfitting) configuration for a specific task."""
    if task_name not in SANITY_CHECK_CONFIG:
        raise ValueError(
            f"Sanity check config for task '{task_name}' not found. "
            f"Available tasks: {list(SANITY_CHECK_CONFIG.keys())}"
        )
    return SANITY_CHECK_CONFIG[task_name]