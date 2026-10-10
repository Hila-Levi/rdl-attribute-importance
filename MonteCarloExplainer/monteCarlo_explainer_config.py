"""
Configuration file for Monte Carlo Shapley explainer dataset and task-specific settings.
Add new datasets/tasks by extending the MONTE_CARLO_CONFIG dictionary.

Directory structure:
    rel-<n>/
        <task-name>/
            HGS/
                saved_model/              <- HGS trained weights
                monteCarlo_explainer/     <- HGS outputs
            transformer/
                saved_model/              <- RelGT trained weights
                monteCarlo_explainer/     <- RelGT outputs

Config structure per task:
    Shared fields (task_type, tune_metric, convergence_threshold, etc.)
    defined ONCE. Model-specific fields under 'model_paths':
        'model_paths': {
            'hgs':         { saved_model_base_dir, montecarlo_output_dir },
            'transformer': { saved_model_base_dir, montecarlo_output_dir },
        }
"""

from pathlib import Path


# ============================================================================
# MONTE CARLO SHAPLEY CONFIGURATIONS
# ============================================================================

MONTE_CARLO_CONFIG = {
    "rel-arxiv": {
        "paper-citation": {
            "task_type": "classification",
            "tune_metric": "roc_auc",
            "convergence_threshold": 0.001,
            "auto_checkpoint_interval": 1.0,
            "max_permutations": None,
            # None = no batch limit: rel-arxiv is explained on the saved subset
            # in rel-arxiv/<task>/test_subset.csv instead (see
            # Utils/create_test_subset.py), so the loader already holds only
            # those rows and capping batches here would cut the subset further.
            "max_batches": None,
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-arxiv/paper-citation/HGS/saved_model",
                    "montecarlo_output_dir": "../rel-arxiv/paper-citation/HGS/monteCarlo_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-arxiv/paper-citation/transformer/saved_model",
                    "montecarlo_output_dir": "../rel-arxiv/paper-citation/transformer/monteCarlo_explainer",
                },
            },
        },
        "author-publication": {
            "task_type": "regression",
            "tune_metric": "mae",
            "convergence_threshold": 0.001,
            "auto_checkpoint_interval": 1.0,
            "max_permutations": None,
            # None = no batch limit: rel-arxiv is explained on the saved subset
            # in rel-arxiv/<task>/test_subset.csv instead (see
            # Utils/create_test_subset.py), so the loader already holds only
            # those rows and capping batches here would cut the subset further.
            "max_batches": None,
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-arxiv/author-publication/HGS/saved_model",
                    "montecarlo_output_dir": "../rel-arxiv/author-publication/HGS/monteCarlo_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-arxiv/author-publication/transformer/saved_model",
                    "montecarlo_output_dir": "../rel-arxiv/author-publication/transformer/monteCarlo_explainer",
                },
            },
        },
    },
    "rel-trial": {
        "study-outcome": {
            "task_type": "classification",
            "tune_metric": "roc_auc",
            "convergence_threshold": 0.001,
            "auto_checkpoint_interval": 1.0,
            "max_permutations": None,
            "max_batches": None,
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-trial/study-outcome/HGS/saved_model",
                    "montecarlo_output_dir": "../rel-trial/study-outcome/HGS/monteCarlo_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-trial/study-outcome/transformer/saved_model",
                    "montecarlo_output_dir": "../rel-trial/study-outcome/transformer/monteCarlo_explainer",
                },
            },
        },
        "study-adverse": {
            "task_type": "regression",
            "tune_metric": "mae",
            "convergence_threshold": 0.001,
            "auto_checkpoint_interval": 1.0,
            "max_permutations": None,
            "max_batches": None,
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-trial/study-adverse/HGS/saved_model",
                    "montecarlo_output_dir": "../rel-trial/study-adverse/HGS/monteCarlo_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-trial/study-adverse/transformer/saved_model",
                    "montecarlo_output_dir": "../rel-trial/study-adverse/transformer/monteCarlo_explainer",
                },
            },
        },
    },
    "rel-f1": {
        "driver-dnf": {
            "task_type": "classification",
            "tune_metric": "roc_auc",
            "convergence_threshold": 0.001,
            "auto_checkpoint_interval": 1.0,
            "max_permutations": None,
            "max_batches": None,
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-f1/driver-dnf/HGS/saved_model",
                    "montecarlo_output_dir": "../rel-f1/driver-dnf/HGS/monteCarlo_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-f1/driver-dnf/transformer/saved_model",
                    "montecarlo_output_dir": "../rel-f1/driver-dnf/transformer/monteCarlo_explainer",
                },
            },
        },
        "driver-position": {
            "task_type": "regression",
            "tune_metric": "mae",
            "convergence_threshold": 0.001,
            "auto_checkpoint_interval": 1.0,
            "max_permutations": None,
            "max_batches": None,
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-f1/driver-position/HGS/saved_model",
                    "montecarlo_output_dir": "../rel-f1/driver-position/HGS/monteCarlo_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-f1/driver-position/transformer/saved_model",
                    "montecarlo_output_dir": "../rel-f1/driver-position/transformer/monteCarlo_explainer",
                },
            },
        },
    },
    "rel-event": {
        "user-repeat": {
            "task_type": "classification",
            "tune_metric": "roc_auc",
            "convergence_threshold": 0.001,
            "auto_checkpoint_interval": 1.0,
            "max_permutations": None,
            "max_batches": None,
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-event/user-repeat/HGS/saved_model",
                    "montecarlo_output_dir": "../rel-event/user-repeat/HGS/monteCarlo_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-event/user-repeat/transformer/saved_model",
                    "montecarlo_output_dir": "../rel-event/user-repeat/transformer/monteCarlo_explainer",
                },
            },
        },
        "user-attendance": {
            "task_type": "regression",
            "tune_metric": "mae",
            "convergence_threshold": 0.001,
            "auto_checkpoint_interval": 1.0,
            "max_permutations": None,
            "max_batches": None,
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-event/user-attendance/HGS/saved_model",
                    "montecarlo_output_dir": "../rel-event/user-attendance/HGS/monteCarlo_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-event/user-attendance/transformer/saved_model",
                    "montecarlo_output_dir": "../rel-event/user-attendance/transformer/monteCarlo_explainer",
                },
            },
        },
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_montecarlo_config(dataset_name: str, task_name: str,
                          model_type: str = "hgs") -> dict:
    """Return a flat config dict for a specific dataset, task, and model type.

    Args:
        dataset_name: e.g. 'rel-f1'
        task_name:    e.g. 'driver-position'
        model_type:   'hgs' (default) or 'transformer'
    """
    if dataset_name not in MONTE_CARLO_CONFIG:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in Monte Carlo configuration. "
            f"Available datasets: {list(MONTE_CARLO_CONFIG.keys())}"
        )
    if task_name not in MONTE_CARLO_CONFIG[dataset_name]:
        raise ValueError(
            f"Task '{task_name}' not found for dataset '{dataset_name}'. "
            f"Available tasks: {list(MONTE_CARLO_CONFIG[dataset_name].keys())}"
        )
    task_cfg = MONTE_CARLO_CONFIG[dataset_name][task_name]
    if model_type not in task_cfg["model_paths"]:
        raise ValueError(
            f"model_type '{model_type}' not found for {dataset_name}/{task_name}. "
            f"Available: {list(task_cfg['model_paths'].keys())}"
        )
    flat_cfg = {k: v for k, v in task_cfg.items() if k != "model_paths"}
    flat_cfg.update(task_cfg["model_paths"][model_type])
    return flat_cfg


def get_montecarlo_saved_model_dir(dataset_name: str, task_name: str,
                                   model_type: str = "hgs") -> Path:
    """Get the saved model directory path for a dataset/task/model_type."""
    config = get_montecarlo_config(dataset_name, task_name, model_type)
    return Path(config['saved_model_base_dir'])


def get_montecarlo_output_dir(dataset_name: str, task_name: str,
                              model_type: str = "hgs") -> Path:
    """Get the Monte Carlo output directory path for a dataset/task/model_type."""
    config = get_montecarlo_config(dataset_name, task_name, model_type)
    output_dir = Path(config['montecarlo_output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir