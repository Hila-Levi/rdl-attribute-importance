"""
Configuration file for RDL explainer dataset and task-specific settings.
Add new datasets/tasks by extending the RDL_EXPLAINER_CONFIG dictionary.

Directory structure:
    rel-<n>/
        <task-name>/
            HGS/
                saved_model/          <- HGS trained weights
                rdl_explainer/        <- HGS RDL outputs
            transformer/
                saved_model/          <- RelGT trained weights
                rdl_explainer/        <- RelGT RDL outputs

Config structure per task:
    Shared fields (task_type, tune_metric) are defined ONCE.
    Model-specific fields live under 'model_paths':
        'model_paths': {
            'hgs':         { saved_model_base_dir, rdl_output_dir },
            'transformer': { saved_model_base_dir, rdl_output_dir },
        }

Note: RDL explanation learns masks on GNN layers so it is intrinsically
tied to the HGS model architecture. Transformer support runs RDL on the
local GNN sub-network inside RelGT.
"""

from pathlib import Path


# ============================================================================
# RDL EXPLAINER CONFIGURATIONS
# ============================================================================

RDL_EXPLAINER_CONFIG = {
    "rel-arxiv": {
        "paper-citation": {
            "task_type": "classification",
            "tune_metric": "roc_auc",
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-arxiv/paper-citation/HGS/saved_model",
                    "rdl_output_dir":       "../rel-arxiv/paper-citation/HGS/rdl_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-arxiv/paper-citation/transformer/saved_model",
                    "rdl_output_dir":       "../rel-arxiv/paper-citation/transformer/rdl_explainer",
                },
            },
        },
        "author-publication": {
            "task_type": "regression",
            "tune_metric": "mae",
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-arxiv/author-publication/HGS/saved_model",
                    "rdl_output_dir":       "../rel-arxiv/author-publication/HGS/rdl_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-arxiv/author-publication/transformer/saved_model",
                    "rdl_output_dir":       "../rel-arxiv/author-publication/transformer/rdl_explainer",
                },
            },
        },
    },
    "rel-trial": {
        "study-outcome": {
            "task_type": "classification",
            "tune_metric": "roc_auc",
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-trial/study-outcome/HGS/saved_model",
                    "rdl_output_dir":       "../rel-trial/study-outcome/HGS/rdl_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-trial/study-outcome/transformer/saved_model",
                    "rdl_output_dir":       "../rel-trial/study-outcome/transformer/rdl_explainer",
                },
            },
        },
        "study-adverse": {
            "task_type": "regression",
            "tune_metric": "mae",
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-trial/study-adverse/HGS/saved_model",
                    "rdl_output_dir":       "../rel-trial/study-adverse/HGS/rdl_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-trial/study-adverse/transformer/saved_model",
                    "rdl_output_dir":       "../rel-trial/study-adverse/transformer/rdl_explainer",
                },
            },
        },
    },
    "rel-f1": {
        "driver-dnf": {
            "task_type": "classification",
            "tune_metric": "roc_auc",
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-f1/driver-dnf/HGS/saved_model",
                    "rdl_output_dir":       "../rel-f1/driver-dnf/HGS/rdl_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-f1/driver-dnf/transformer/saved_model",
                    "rdl_output_dir":       "../rel-f1/driver-dnf/transformer/rdl_explainer",
                },
            },
        },
        "driver-position": {
            "task_type": "regression",
            "tune_metric": "mae",
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-f1/driver-position/HGS/saved_model",
                    "rdl_output_dir":       "../rel-f1/driver-position/HGS/rdl_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-f1/driver-position/transformer/saved_model",
                    "rdl_output_dir":       "../rel-f1/driver-position/transformer/rdl_explainer",
                },
            },
        },
    },
    "rel-event": {
        "user-repeat": {
            "task_type": "classification",
            "tune_metric": "roc_auc",
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-event/user-repeat/HGS/saved_model",
                    "rdl_output_dir":       "../rel-event/user-repeat/HGS/rdl_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-event/user-repeat/transformer/saved_model",
                    "rdl_output_dir":       "../rel-event/user-repeat/transformer/rdl_explainer",
                },
            },
        },
        "user-attendance": {
            "task_type": "regression",
            "tune_metric": "mae",
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-event/user-attendance/HGS/saved_model",
                    "rdl_output_dir":       "../rel-event/user-attendance/HGS/rdl_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-event/user-attendance/transformer/saved_model",
                    "rdl_output_dir":       "../rel-event/user-attendance/transformer/rdl_explainer",
                },
            },
        },
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_rdl_config(dataset_name: str, task_name: str,
                   model_type: str = "hgs") -> dict:
    """Return a flat config dict for a specific dataset, task, and model type.

    Merges shared task-level fields (task_type, tune_metric)
    with model-specific fields (saved_model_base_dir, rdl_output_dir).

    Args:
        dataset_name: e.g. 'rel-f1'
        task_name:    e.g. 'driver-position'
        model_type:   'hgs' (default) or 'transformer'
    """
    if dataset_name not in RDL_EXPLAINER_CONFIG:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in RDL explainer configuration. "
            f"Available datasets: {list(RDL_EXPLAINER_CONFIG.keys())}"
        )
    if task_name not in RDL_EXPLAINER_CONFIG[dataset_name]:
        raise ValueError(
            f"Task '{task_name}' not found for dataset '{dataset_name}'. "
            f"Available tasks: {list(RDL_EXPLAINER_CONFIG[dataset_name].keys())}"
        )

    task_cfg = RDL_EXPLAINER_CONFIG[dataset_name][task_name]

    if model_type not in task_cfg["model_paths"]:
        raise ValueError(
            f"model_type '{model_type}' not found for {dataset_name}/{task_name}. "
            f"Available model types: {list(task_cfg['model_paths'].keys())}"
        )

    flat_cfg = {k: v for k, v in task_cfg.items() if k != "model_paths"}
    flat_cfg.update(task_cfg["model_paths"][model_type])
    return flat_cfg


def get_rdl_saved_model_dir(dataset_name: str, task_name: str,
                            model_type: str = "hgs") -> Path:
    """Get the saved model directory path for a dataset/task/model_type."""
    config = get_rdl_config(dataset_name, task_name, model_type)
    return Path(config['saved_model_base_dir'])


def get_rdl_output_dir(dataset_name: str, task_name: str,
                       model_type: str = "hgs") -> Path:
    """Get the RDL output directory path for a dataset/task/model_type."""
    config = get_rdl_config(dataset_name, task_name, model_type)
    output_dir = Path(config['rdl_output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_rdl_inference_dir(dataset_name: str, task_name: str,
                          model_type: str = "hgs") -> Path:
    """Get the RDL inference directory path for a dataset/task/model_type."""
    output_dir = get_rdl_output_dir(dataset_name, task_name, model_type)
    inference_dir = output_dir / f"inference/{dataset_name}-{task_name}"
    inference_dir.mkdir(parents=True, exist_ok=True)
    return inference_dir