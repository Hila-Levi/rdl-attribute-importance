"""
Configuration file for dataset and task-specific settings.
Add new datasets/tasks by extending the DATASET_CONFIG and TASK_CONFIG dictionaries.

Directory structure per dataset:
    rel-<n>/
        data_<n>/          ← shared across all tasks (graph cache lives here)
        <task-name>/
            HGS/
                saved_model/      ← trained weights, predictions, metadata
                singleColumnAblation_explainer/
                kernel_shap/
                ...
            transformer/
                saved_model/
                relgt-cache/
                ...
"""

from torch.nn import BCEWithLogitsLoss, L1Loss


# ============================================================================
# DATASET CONFIGURATIONS
# Note: saved_model_dir_name has moved to TASK_CONFIG — it is task-specific.
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
# ============================================================================

TASK_CONFIG = {
    # -------------------------------------------------------------------------
    # rel-arxiv - classification
    # -------------------------------------------------------------------------
    "paper-citation": {
        "saved_model_dir_name": "../rel-arxiv/paper-citation/HGS/saved_model",
        "out_channels": 1,
        "tune_metric": "roc_auc",
        "higher_is_better": True,
        "loss_fn": BCEWithLogitsLoss(),
        "target_col": "cited",
        "task_description": "paper-citation",
        "task_type": "classification",
        "label_names": {
            0: "Not Cited",
            1: "Cited"
        },
    },
    # -------------------------------------------------------------------------
    # rel-arxiv - regression
    # -------------------------------------------------------------------------
    "author-publication": {
        "saved_model_dir_name": "../rel-arxiv/author-publication/HGS/saved_model",
        "out_channels": 1,
        "tune_metric": "mae",          
        "higher_is_better": False,     
        "loss_fn": L1Loss(),           
        "target_col": "published",
        "task_description": "predict whether an author will publish",
        "task_type": "regression",
        "label_names": None,           
    },
    # -------------------------------------------------------------------------
    # rel-trial - classification
    # -------------------------------------------------------------------------
    "study-outcome": {
        "saved_model_dir_name": "../rel-trial/study-outcome/HGS/saved_model",
        "out_channels": 1,
        "tune_metric": "roc_auc",
        "higher_is_better": True,
        "loss_fn": BCEWithLogitsLoss(),
        "target_col": "outcome",
        "task_description": "predict the outcome of a clinical trial (success/failure)",
        "task_type": "classification",
        "label_names": {
            0: "Failure",
            1: "Success"
        },
    },
    # -------------------------------------------------------------------------
    # rel-trial - regression
    # -------------------------------------------------------------------------
    "study-adverse": {
        "saved_model_dir_name": "../rel-trial/study-adverse/HGS/saved_model",
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
        "saved_model_dir_name": "../rel-f1/driver-dnf/HGS/saved_model",
        "out_channels": 1,
        "tune_metric": "roc_auc",
        "higher_is_better": True,
        "loss_fn": BCEWithLogitsLoss(),
        "target_col": "did_not_finish",
        "task_description": "predict if a driver will not finish the race (DNF)",
        "task_type": "classification",
        "label_names": {
            0: "Finished",
            1: "DNF"
        },
    },
    # -------------------------------------------------------------------------
    # rel-f1 — regression
    # -------------------------------------------------------------------------
    "driver-position": {
        "saved_model_dir_name": "../rel-f1/driver-position/HGS/saved_model",
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
        "saved_model_dir_name": "../rel-event/user-repeat/HGS/saved_model",
        "out_channels": 1,
        "tune_metric": "roc_auc",
        "higher_is_better": True,
        "loss_fn": BCEWithLogitsLoss(),
        "target_col": "repeat",
        "task_description": "predict if a user will repeat/attend an event",
        "task_type": "classification",
        "label_names": {
            0: "No",
            1: "Yes"
        },
    },
    # -------------------------------------------------------------------------
    # rel-event — regression
    # -------------------------------------------------------------------------
    "user-attendance": {
        "saved_model_dir_name": "../rel-event/user-attendance/HGS/saved_model",
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
# TRAINING CONFIGURATIONS (can be task-specific if needed)
# ============================================================================

TRAINING_CONFIG = {
    # -------------------------------------------------------------------------
    # rel-arxiv - classification
    # -------------------------------------------------------------------------
    "paper-citation": {
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "scheduler_factor": 0.5,
        "scheduler_patience": 4,
        "epochs": 100,
        "early_stopping_patience": 100,
        "model_channels": 256,
        "model_layers": 3,
        "model_aggr": "sum",
        "model_norm": "batch_norm",
        "model_dropout": 0.3,
        "scheduler_verbose": True,
    },
    # -------------------------------------------------------------------------
    # rel-arxiv - regression
    # -------------------------------------------------------------------------
    "author-publication": {
        "learning_rate": 0.0005,     
        "weight_decay": 1e-3,        
        "scheduler_factor": 0.5,
        "scheduler_patience": 3,   
        "epochs": 100,             
        "early_stopping_patience": 100,
        "model_channels": 128,      
        "model_layers": 2,          
        "model_aggr": "mean",        
        "model_norm": "batch_norm",
        "model_dropout": 0.5,       
        "scheduler_verbose": True,
    },
    # -------------------------------------------------------------------------
    # rel-trial - classification
    # -------------------------------------------------------------------------
    "study-outcome": {
        "learning_rate": 0.001,
        "weight_decay": 1e-3,
        "scheduler_factor": 0.5,
        "scheduler_patience": 4,
        "epochs": 100,
        "early_stopping_patience": 100,
        "model_channels": 256,
        "model_layers": 3,
        "model_aggr": "mean",
        "model_norm": "batch_norm",
        "model_dropout": 0.5,
        "scheduler_verbose": True,
    },
    # -------------------------------------------------------------------------
    # rel-trial - regression
    # -------------------------------------------------------------------------
    "study-adverse": {
        "learning_rate": 0.0005,     
        "weight_decay": 1e-3,        
        "scheduler_factor": 0.5,
        "scheduler_patience": 3,   
        "epochs": 100,             
        "early_stopping_patience": 100,
        "model_channels": 128,      
        "model_layers": 2,          
        "model_aggr": "mean",        
        "model_norm": "batch_norm",
        "model_dropout": 0.5,       
        "scheduler_verbose": True,
    },
    # -------------------------------------------------------------------------
    # rel-f1 — classification
    # -------------------------------------------------------------------------
    "driver-dnf": {
        "learning_rate": 0.01,
        "weight_decay": 1e-5,
        "scheduler_factor": 0.5,
        "scheduler_patience": 4,
        "epochs": 100,
        "early_stopping_patience": 100,
        "model_channels": 256,
        "model_layers": 3,
        "model_aggr": "sum",
        "model_norm": "batch_norm",
        "model_dropout": 0.3,
        "scheduler_verbose": True,
    },
    # -------------------------------------------------------------------------
    # rel-f1 — regression
    # -------------------------------------------------------------------------
    "driver-position": {
        "learning_rate": 0.0005,     
        "weight_decay": 1e-3,        
        "scheduler_factor": 0.5,
        "scheduler_patience": 3,   
        "epochs": 100,             
        "early_stopping_patience": 100,
        "model_channels": 128,      
        "model_layers": 2,          
        "model_aggr": "mean",        
        "model_norm": "batch_norm",
        "model_dropout": 0.5,       
        "scheduler_verbose": True,
    },
    # -------------------------------------------------------------------------
    # rel-event — classification
    # -------------------------------------------------------------------------
    "user-repeat": {
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "scheduler_factor": 0.5,
        "scheduler_patience": 4,
        "epochs": 100,
        "early_stopping_patience": 100,
        "model_channels": 256,
        "model_layers": 3,
        "model_aggr": "sum",
        "model_norm": "batch_norm",
        "model_dropout": 0.3,
        "scheduler_verbose": True,
    },
    # -------------------------------------------------------------------------
    # rel-event — regression
    # -------------------------------------------------------------------------
    "user-attendance": {
        "learning_rate": 0.0005,
        "weight_decay": 1e-3,
        "scheduler_factor": 0.5,
        "scheduler_patience": 3,
        "epochs": 100,
        "early_stopping_patience": 100,
        "model_channels": 128,
        "model_layers": 2,
        "model_aggr": "mean",
        "model_norm": "batch_norm",
        "model_dropout": 0.5,
        "scheduler_verbose": True,
    },
}


# ============================================================================
# LOADER CONFIGURATIONS (can be task-specific if needed)
# ============================================================================

LOADER_CONFIG = {
    # -------------------------------------------------------------------------
    # rel-arxiv - classification
    # -------------------------------------------------------------------------
    "paper-citation": {
        "num_neighbors": [25, 15, 10],
        "batch_size": 256,
        "temporal_strategy": "uniform",
    },
    # -------------------------------------------------------------------------
    # rel-arxiv - regression
    # -------------------------------------------------------------------------
    "author-publication": {
        "num_neighbors": [25, 15],
        "batch_size": 256,
        "temporal_strategy": "uniform",
    },
    # -------------------------------------------------------------------------
    # rel-trial - classification
    # -------------------------------------------------------------------------
    "study-outcome": {
        "num_neighbors": [25, 15, 10],
        "batch_size": 256,
        "temporal_strategy": "uniform",
    },
    # -------------------------------------------------------------------------
    # rel-trial - regression
    # -------------------------------------------------------------------------
    "study-adverse": {
        "num_neighbors": [25, 15],
        "batch_size": 256,
        "temporal_strategy": "uniform",
    },
    # -------------------------------------------------------------------------
    # rel-f1 — classification
    # -------------------------------------------------------------------------
    "driver-dnf": {
        "num_neighbors": [25, 15, 10],
        "batch_size": 256,
        "temporal_strategy": "uniform",
    },
    # -------------------------------------------------------------------------
    # rel-f1 — regression
    # -------------------------------------------------------------------------
    "driver-position": {
        "num_neighbors": [25, 15, 10],
        "batch_size": 256,
        "temporal_strategy": "uniform",
    },
    # -------------------------------------------------------------------------
    # rel-event — classification
    # -------------------------------------------------------------------------
    "user-repeat": {
        "num_neighbors": [20, 10, 5],
        "batch_size": 512,
        "temporal_strategy": "uniform",
    },
    # -------------------------------------------------------------------------
    # rel-event — regression
    # -------------------------------------------------------------------------
    "user-attendance": {
        "num_neighbors": [25, 15],
        "batch_size": 256,
        "temporal_strategy": "uniform",
    },
}


# ============================================================================
# SANITY CHECK CONFIGURATIONS (can be task-specific)
# ============================================================================

SANITY_CHECK_CONFIG = {
    # -------------------------------------------------------------------------
    # rel-arxiv - classification
    # -------------------------------------------------------------------------
    "paper-citation": {
        "sample_size": 100,
        "epochs": 100,
        "num_neighbors": [25, 15, 10],
        "batch_size": 32,
        "learning_rate": 0.01,
        "model_channels": 128,
        "model_layers": 3,
        "success_threshold": 0.95,  # AUROC — higher is better
    },
    # -------------------------------------------------------------------------
    # rel-arxiv - regression
    # -------------------------------------------------------------------------
    "author-publication": {
        "sample_size": 100,
        "epochs": 150,
        "num_neighbors": [25, 15, 10],
        "batch_size": 32,
        "learning_rate": 0.01,
        "model_channels": 128,
        "model_layers": 3,
        "success_threshold": 5.0,  
    },
    # -------------------------------------------------------------------------
    # rel-trial - classification
    # -------------------------------------------------------------------------
    "study-outcome": {
        "sample_size": 100,
        "epochs": 100,
        "num_neighbors": [25, 15, 10],
        "batch_size": 32,
        "learning_rate": 0.01,
        "model_channels": 128,
        "model_layers": 3,
        "success_threshold": 0.95,  # AUROC — higher is better
    },
    # -------------------------------------------------------------------------
    # rel-trial - regression
    # -------------------------------------------------------------------------
    "study-adverse": {
        "sample_size": 100,
        "epochs": 150,
        "num_neighbors": [25, 15, 10],
        "batch_size": 32,
        "learning_rate": 0.01,
        "model_channels": 128,
        "model_layers": 3,
        "success_threshold": 5.0,  
    },
    # -------------------------------------------------------------------------
    # rel-f1 — classification
    # -------------------------------------------------------------------------
    "driver-dnf": {
        "sample_size": 100,
        "epochs": 100,
        "num_neighbors": [25, 15, 10],
        "batch_size": 32,
        "learning_rate": 0.01,
        "model_channels": 128,
        "model_layers": 3,
        "success_threshold": 0.95,  # AUROC — higher is better
    },
    # -------------------------------------------------------------------------
    # rel-f1 — regression
    # -------------------------------------------------------------------------
    "driver-position": {
        "sample_size": 100,
        "epochs": 150,
        "num_neighbors": [25, 15, 10],
        "batch_size": 32,
        "learning_rate": 0.01,
        "model_channels": 128,
        "model_layers": 3,
        "success_threshold": 4.0,  # MAE — lower is better; overfitting should reach < 3 positions
    },
    # -------------------------------------------------------------------------
    # rel-event — classification
    # -------------------------------------------------------------------------
    "user-repeat": {
        "sample_size": 100,
        "epochs": 100,
        "num_neighbors": [20, 10, 5],
        "batch_size": 32,
        "learning_rate": 0.01,
        "model_channels": 128,
        "model_layers": 3,
        "success_threshold": 0.90,  # AUROC
    },
    # -------------------------------------------------------------------------
    # rel-event — regression
    # -------------------------------------------------------------------------
    "user-attendance": {
        "sample_size": 100,
        "epochs": 150,
        "num_neighbors": [25, 15],
        "batch_size": 32,
        "learning_rate": 0.01,
        "model_channels": 128,
        "model_layers": 3,
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
    """Get configuration for a specific task."""
    if task_name not in TASK_CONFIG:
        raise ValueError(
            f"Task '{task_name}' not found in configuration. "
            f"Available tasks: {list(TASK_CONFIG.keys())}"
        )
    return TASK_CONFIG[task_name]


def get_loader_config(task_name: str, seed: int) -> dict:
    """Get loader configuration for a specific task."""
    if task_name not in LOADER_CONFIG:
        raise ValueError(
            f"Loader config for task '{task_name}' not found. "
            f"Available tasks: {list(LOADER_CONFIG.keys())}"
        )
    config = LOADER_CONFIG[task_name].copy()
    config["seed"] = seed
    return config


def get_sanity_check_config(task_name: str) -> dict:
    """Get sanity check configuration for a specific task."""
    if task_name not in SANITY_CHECK_CONFIG:
        raise ValueError(
            f"Sanity check config for task '{task_name}' not found. "
            f"Available tasks: {list(SANITY_CHECK_CONFIG.keys())}"
        )
    return SANITY_CHECK_CONFIG[task_name]


def get_training_config(task_name: str) -> dict:
    """Get training configuration for a specific task."""
    if task_name not in TRAINING_CONFIG:
        raise ValueError(
            f"Training config for task '{task_name}' not found. "
            f"Available tasks: {list(TRAINING_CONFIG.keys())}"
        )
    return TRAINING_CONFIG[task_name]