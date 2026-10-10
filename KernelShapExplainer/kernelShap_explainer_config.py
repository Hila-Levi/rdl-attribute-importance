"""
Configuration file for Kernel SHAP explainer dataset and task-specific settings.
Add new datasets/tasks by extending the KERNEL_SHAP_CONFIG dictionary.

Directory structure:
    rel-<n>/
        data_<n>/                        <- shared graph cache
        <task-name>/
            HGS/
                saved_model/             <- HGS trained weights
                kernelShap_explainer/    <- HGS KernelSHAP outputs
            transformer/
                saved_model/             <- RelGT trained weights
                kernelShap_explainer/    <- RelGT KernelSHAP outputs

Config structure per task:
    Shared fields (task_type, num_coalitions, max_batches,
    disagreement_threshold) are defined ONCE at the task level.

    Model-specific fields live under 'model_paths':
        'model_paths': {
            'hgs':         { saved_model_base_dir, kernelshap_output_dir },
            'transformer': { saved_model_base_dir, kernelshap_output_dir },
        }
"""

from pathlib import Path


# ============================================================================
# KERNEL SHAP EXPLAINER CONFIGURATIONS
# ============================================================================

KERNEL_SHAP_CONFIG = {
    "rel-arxiv": {
        "paper-citation": {
            "task_type": "classification",
            "num_coalitions": 3000,
            # None = no batch limit: rel-arxiv is explained on the saved subset
            # in rel-arxiv/<task>/test_subset.csv instead (see
            # Utils/create_test_subset.py), so the loader already holds only
            # those rows and capping batches here would cut the subset further.
            "max_batches": None,
            "disagreement_threshold": 0.5,
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-arxiv/paper-citation/HGS/saved_model",
                    "kernelshap_output_dir": "../rel-arxiv/paper-citation/HGS/kernelShap_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-arxiv/paper-citation/transformer/saved_model",
                    "kernelshap_output_dir": "../rel-arxiv/paper-citation/transformer/kernelShap_explainer",
                },
            },
        },
        "author-publication": {
            "task_type": "regression",
            "num_coalitions": 3000,
            # None = no batch limit: rel-arxiv is explained on the saved subset
            # in rel-arxiv/<task>/test_subset.csv instead (see
            # Utils/create_test_subset.py), so the loader already holds only
            # those rows and capping batches here would cut the subset further.
            "max_batches": None,
            "disagreement_threshold": 0.5,
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-arxiv/author-publication/HGS/saved_model",
                    "kernelshap_output_dir": "../rel-arxiv/author-publication/HGS/kernelShap_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-arxiv/author-publication/transformer/saved_model",
                    "kernelshap_output_dir": "../rel-arxiv/author-publication/transformer/kernelShap_explainer",
                },
            },
        },
    },
    "rel-trial": {
        "study-outcome": {
            "task_type": "classification",
            "num_coalitions": 3000,
            "max_batches": None,
            "disagreement_threshold": 0.5,
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-trial/study-outcome/HGS/saved_model",
                    "kernelshap_output_dir": "../rel-trial/study-outcome/HGS/kernelShap_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-trial/study-outcome/transformer/saved_model",
                    "kernelshap_output_dir": "../rel-trial/study-outcome/transformer/kernelShap_explainer",
                },
            },
        },
        "study-adverse": {
            "task_type": "regression",
            "num_coalitions": 3000,
            "max_batches": None,
            "disagreement_threshold": 0.5,
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-trial/study-adverse/HGS/saved_model",
                    "kernelshap_output_dir": "../rel-trial/study-adverse/HGS/kernelShap_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-trial/study-adverse/transformer/saved_model",
                    "kernelshap_output_dir": "../rel-trial/study-adverse/transformer/kernelShap_explainer",
                },
            },
        },
    },
    "rel-f1": {
        "driver-dnf": {
            "task_type": "classification",
            "num_coalitions": 3000,
            "max_batches": None,
            "disagreement_threshold": 0.5,
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-f1/driver-dnf/HGS/saved_model",
                    "kernelshap_output_dir": "../rel-f1/driver-dnf/HGS/kernelShap_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-f1/driver-dnf/transformer/saved_model",
                    "kernelshap_output_dir": "../rel-f1/driver-dnf/transformer/kernelShap_explainer",
                },
            },
        },
        "driver-position": {
            "task_type": "regression",
            "num_coalitions": 3000,
            "max_batches": None,
            "disagreement_threshold": 0.5,
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-f1/driver-position/HGS/saved_model",
                    "kernelshap_output_dir": "../rel-f1/driver-position/HGS/kernelShap_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-f1/driver-position/transformer/saved_model",
                    "kernelshap_output_dir": "../rel-f1/driver-position/transformer/kernelShap_explainer",
                },
            },
        },
    },
    "rel-event": {
        "user-repeat": {
            "task_type": "classification",
            "num_coalitions": 3000,
            "max_batches": None,
            "disagreement_threshold": 0.5,
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-event/user-repeat/HGS/saved_model",
                    "kernelshap_output_dir": "../rel-event/user-repeat/HGS/kernelShap_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-event/user-repeat/transformer/saved_model",
                    "kernelshap_output_dir": "../rel-event/user-repeat/transformer/kernelShap_explainer",
                },
            },
        },
        "user-attendance": {
            "task_type": "regression",
            "num_coalitions": 3000,
            "max_batches": None,
            "disagreement_threshold": 0.5,
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-event/user-attendance/HGS/saved_model",
                    "kernelshap_output_dir": "../rel-event/user-attendance/HGS/kernelShap_explainer",
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-event/user-attendance/transformer/saved_model",
                    "kernelshap_output_dir": "../rel-event/user-attendance/transformer/kernelShap_explainer",
                },
            },
        },
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_kernelshap_config(dataset_name: str, task_name: str,
                          model_type: str = "hgs") -> dict:
    """Return a flat config dict for a specific dataset, task, and model type.

    Merges shared task-level fields with model-specific fields (saved_model_base_dir,
    kernelshap_output_dir) into one flat dict — callers do not need to know about
    the nested model_paths structure.

    Args:
        dataset_name: e.g. 'rel-f1'
        task_name:    e.g. 'driver-position'
        model_type:   'hgs' (default) or 'transformer'
    """
    if dataset_name not in KERNEL_SHAP_CONFIG:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in Kernel SHAP configuration. "
            f"Available datasets: {list(KERNEL_SHAP_CONFIG.keys())}"
        )
    if task_name not in KERNEL_SHAP_CONFIG[dataset_name]:
        raise ValueError(
            f"Task '{task_name}' not found for dataset '{dataset_name}'. "
            f"Available tasks: {list(KERNEL_SHAP_CONFIG[dataset_name].keys())}"
        )

    task_cfg = KERNEL_SHAP_CONFIG[dataset_name][task_name]

    if model_type not in task_cfg["model_paths"]:
        raise ValueError(
            f"model_type '{model_type}' not found for {dataset_name}/{task_name}. "
            f"Available model types: {list(task_cfg['model_paths'].keys())}"
        )

    flat_cfg = {k: v for k, v in task_cfg.items() if k != "model_paths"}
    flat_cfg.update(task_cfg["model_paths"][model_type])
    return flat_cfg


def get_kernelshap_saved_model_dir(dataset_name: str, task_name: str,
                                   model_type: str = "hgs") -> Path:
    """Get the saved model directory path for a dataset/task/model_type."""
    config = get_kernelshap_config(dataset_name, task_name, model_type)
    return Path(config['saved_model_base_dir'])


def get_kernelshap_output_dir(dataset_name: str, task_name: str,
                              model_type: str = "hgs") -> Path:
    """Get the Kernel SHAP output directory path for a dataset/task/model_type."""
    config = get_kernelshap_config(dataset_name, task_name, model_type)
    output_dir = Path(config['kernelshap_output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir