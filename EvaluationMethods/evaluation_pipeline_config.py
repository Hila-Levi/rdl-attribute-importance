"""
Configuration file for the centralised XAI Evaluation Pipeline.
Add new datasets/tasks by extending the EVALUATION_CONFIG dictionary.

Directory structure:
    rel-<n>/
        <task-name>/
            HGS/
                saved_model/
                singleColumnAblation_explainer/
                kernelShap_explainer/
                rdl_explainer/
                monteCarlo_explainer/
            transformer/
                saved_model/
                singleColumnAblation_explainer/
                ...

    EvaluationMethods/
        <dataset_name>/
            <task_name>/
                hgs/
                    run_1/
                        profiles/
                        fidelity/
                        stability/
                        run_index.json
                    run_2/
                    ...
                transformer/
                    run_1/
                    ...

Config structure per task:
    Shared fields (task_type, tune_metric, fidelity_k_step_long) defined ONCE.
    Model-specific fields live under 'model_paths':
        'model_paths': {
            'hgs':         { task_dir, eval_output_dir },
            'transformer': { task_dir, eval_output_dir },
        }
"""

from pathlib import Path


# ============================================================================
# EVALUATION PIPELINE CONFIGURATIONS
# ============================================================================

EVALUATION_CONFIG = {
    "rel-arxiv": {
        # ---------------------------------------------------------------------
        # Binary classification
        # ---------------------------------------------------------------------
        "paper-citation": {
            "task_type": "classification",
            "tune_metric": "roc_auc",
            "fidelity_k_step_long": 10,
            "model_paths": {
                "hgs": {
                    "task_dir": "../rel-arxiv/paper-citation/HGS",
                    "eval_output_dir": "../rel-arxiv/paper-citation/HGS/evaluation_results",
                },
                "transformer": {
                    "task_dir": "../rel-arxiv/paper-citation/transformer",
                    "eval_output_dir": "../rel-arxiv/paper-citation/transformer/evaluation_results",
                },
            },
        },
        # ---------------------------------------------------------------------
        # Regression
        # ---------------------------------------------------------------------
        "author-publication": {
            "task_type": "regression",
            "tune_metric": "mae",
            "fidelity_k_step_long": 10,
            "model_paths": {
                "hgs": {
                    "task_dir": "../rel-arxiv/author-publication/HGS",
                    "eval_output_dir": "../rel-arxiv/author-publication/HGS/evaluation_results",
                },
                "transformer": {
                    "task_dir": "../rel-arxiv/author-publication/transformer",
                    "eval_output_dir": "../rel-arxiv/author-publication/transformer/evaluation_results",
                },
            },
        },
    },
    "rel-trial": {
        # ---------------------------------------------------------------------
        # Binary classification
        # ---------------------------------------------------------------------
        "study-outcome": {
            "task_type": "classification",
            "tune_metric": "roc_auc",
            "fidelity_k_step_long": 10,
            "model_paths": {
                "hgs": {
                    "task_dir": "../rel-trial/study-outcome/HGS",
                    "eval_output_dir": "../rel-trial/study-outcome/HGS/evaluation_results",
                },
                "transformer": {
                    "task_dir": "../rel-trial/study-outcome/transformer",
                    "eval_output_dir": "../rel-trial/study-outcome/transformer/evaluation_results",
                },
            },
        },
        # ---------------------------------------------------------------------
        # Regression
        # ---------------------------------------------------------------------
        "study-adverse": {
            "task_type": "regression",
            "tune_metric": "mae",
            "fidelity_k_step_long": 10,
            "model_paths": {
                "hgs": {
                    "task_dir": "../rel-trial/study-adverse/HGS",
                    "eval_output_dir": "../rel-trial/study-adverse/HGS/evaluation_results",
                },
                "transformer": {
                    "task_dir": "../rel-trial/study-adverse/transformer",
                    "eval_output_dir": "../rel-trial/study-adverse/transformer/evaluation_results",
                },
            },
        },
    },
    "rel-f1": {
        # ---------------------------------------------------------------------
        # Binary classification
        # ---------------------------------------------------------------------
        "driver-dnf": {
            "task_type": "classification",
            "tune_metric": "roc_auc",
            "fidelity_k_step_long": 10,
            "model_paths": {
                "hgs": {
                    "task_dir": "../rel-f1/driver-dnf/HGS",
                    "eval_output_dir": "../rel-f1/driver-dnf/HGS/evaluation_results",
                },
                "transformer": {
                    "task_dir": "../rel-f1/driver-dnf/transformer",
                    "eval_output_dir": "../rel-f1/driver-dnf/transformer/evaluation_results",
                },
            },
        },
        # ---------------------------------------------------------------------
        # Regression
        # ---------------------------------------------------------------------
        "driver-position": {
            "task_type": "regression",
            "tune_metric": "mae",
            "fidelity_k_step_long": 10,
            "model_paths": {
                "hgs": {
                    "task_dir": "../rel-f1/driver-position/HGS",
                    "eval_output_dir": "../rel-f1/driver-position/HGS/evaluation_results",
                },
                "transformer": {
                    "task_dir": "../rel-f1/driver-position/transformer",
                    "eval_output_dir": "../rel-f1/driver-position/transformer/evaluation_results",
                },
            },
        },
    },
    "rel-event": {
        # ---------------------------------------------------------------------
        # Binary classification
        # ---------------------------------------------------------------------
        "user-repeat": {
            "task_type": "classification",
            "tune_metric": "roc_auc",
            "fidelity_k_step_long": 10,
            "model_paths": {
                "hgs": {
                    "task_dir": "../rel-event/user-repeat/HGS",
                    "eval_output_dir": "../rel-event/user-repeat/HGS/evaluation_results",
                },
                "transformer": {
                    "task_dir": "../rel-event/user-repeat/transformer",
                    "eval_output_dir": "../rel-event/user-repeat/transformer/evaluation_results",
                },
            },
        },
        # ---------------------------------------------------------------------
        # Regression
        # ---------------------------------------------------------------------
        "user-attendance": {
            "task_type": "regression",
            "tune_metric": "mae",
            "fidelity_k_step_long": 10,
            "model_paths": {
                "hgs": {
                    "task_dir": "../rel-event/user-attendance/HGS",
                    "eval_output_dir": "../rel-event/user-attendance/HGS/evaluation_results",
                },
                "transformer": {
                    "task_dir": "../rel-event/user-attendance/transformer",
                    "eval_output_dir": "../rel-event/user-attendance/transformer/evaluation_results",
                },
            },
        },
    },
}


# ============================================================================
# K-VALUES  (fixed list, short-range step, and percentages of total features)
# ============================================================================

FIDELITY_K_FIXED    = [1, 3, 5, 10, 15, 20]
FIDELITY_K_STEP_SHORT = 1   # step for k = 1 → 25% of features
FIDELITY_K_PERCENT  = [5, 10, 15, 20, 25, 100]   # percentages of total_features to evaluate
                                                  # (100 => k = total_features, i.e. ALL features)


# ============================================================================
# COLUMN MAPPING  —  which columns carry feature name/score in each CSV
# ============================================================================

METHOD_COLUMNS = {
    "Kernel SHAP":         ("feature",      "abs_shap_value"),
    "RDL Explainer":       ("feature",      "Importance Score"),
    "SingleColumnAblation Strategy": ("feature_name", None),   # score_col resolved at runtime
    "Monte Carlo Shapley": ("feature",      "abs_shapley_value"),
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_eval_config(dataset_name: str, task_name: str,
                    model_type: str = "hgs") -> dict:
    """Return a flat config dict for a specific dataset, task, and model type.

    Merges shared task-level fields (task_type, tune_metric,
    fidelity_k_step_long) with model-specific fields (task_dir,
    eval_output_dir) into one flat dict.

    Args:
        dataset_name: e.g. 'rel-f1'
        task_name:    e.g. 'driver-position'
        model_type:   'hgs' (default) or 'transformer'
    """
    if dataset_name not in EVALUATION_CONFIG:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in evaluation configuration. "
            f"Available datasets: {list(EVALUATION_CONFIG.keys())}"
        )
    if task_name not in EVALUATION_CONFIG[dataset_name]:
        raise ValueError(
            f"Task '{task_name}' not found for dataset '{dataset_name}'. "
            f"Available tasks: {list(EVALUATION_CONFIG[dataset_name].keys())}"
        )

    task_cfg = EVALUATION_CONFIG[dataset_name][task_name]

    if model_type not in task_cfg["model_paths"]:
        raise ValueError(
            f"model_type '{model_type}' not found for {dataset_name}/{task_name}. "
            f"Available model types: {list(task_cfg['model_paths'].keys())}"
        )

    flat_cfg = {k: v for k, v in task_cfg.items() if k != "model_paths"}
    flat_cfg.update(task_cfg["model_paths"][model_type])
    return flat_cfg


def get_task_dir(dataset_name: str, task_name: str,
                 model_type: str = "hgs") -> Path:
    """Get the model's task directory (contains saved_model/, singleColumnAblation_explainer/, etc.)."""
    cfg = get_eval_config(dataset_name, task_name, model_type)
    config_dir = Path(__file__).resolve().parent
    return (config_dir / cfg["task_dir"]).resolve()


def get_eval_output_dir(dataset_name: str, task_name: str,
                        model_type: str = "hgs") -> Path:
    """Get the evaluation output directory for a dataset/task/model_type.

    Paths in the config are relative to the config file's own location,
    not the caller's working directory — so the notebook and the pipeline
    resolve to the same directory regardless of where they are run from.
    """
    cfg = get_eval_config(dataset_name, task_name, model_type)
    config_dir = Path(__file__).resolve().parent
    output_dir = (config_dir / cfg["eval_output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_latest_run_dir(dataset_name: str, task_name: str,
                       model_type: str = "hgs") -> Path:
    """Return the run_N/ directory with the highest index that contains a
    run_index.json, or raise if none exist.

    Any directory matching run_{int} that has a run_index.json is accepted —
    completed or in-progress. The notebook can still visualise whatever
    outputs are present. Use FORCE_RUN_INDEX to pin a specific run.
    """
    eval_dir = get_eval_output_dir(dataset_name, task_name, model_type)
    run_dirs = sorted(
        eval_dir.glob("run_*"),
        key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else -1
    )

    valid = [d for d in run_dirs if (d / "run_index.json").exists()]

    if not valid:
        raise FileNotFoundError(
            f"No pipeline runs found in {eval_dir}.\n"
            f"Run: python run_evaluation_pipeline.py "
            f"--dataset {dataset_name} --task {task_name} --model_type {model_type}"
        )
    return valid[-1]


def get_feature_score_cols(method_name: str, task_type: str = "classification",
                           classification_metric: str = "label_flip"):
    """Return (feature_col, score_col) for a given method name.

    Only "SingleColumnAblation Strategy" branches on task_type / classification_metric:
    its results CSV carries BOTH a label-flip count and a continuous
    importance-score column for every feature (see
    singleColumnAblation_explainer), so classification tasks must pick one to
    rank/build the profile with.

    Args:
        method_name: e.g. 'Kernel SHAP', 'SingleColumnAblation Strategy', ...
        task_type:   'classification' or 'regression'. Regression always
                     uses 'importance_score' (mean |Δpred|) regardless of
                     classification_metric.
        classification_metric: Only relevant when task_type == 'classification'
                     and method_name == 'SingleColumnAblation Strategy'.
                       - 'label_flip' (default): rank by 'num_label_flips'
                         (count of nodes whose predicted label flips).
                       - 'importance_score': rank by 'importance_score'
                         (mean |Δpred|), the same continuous metric used
                         for regression.
                     Kernel SHAP / RDL Explainer / Monte Carlo Shapley are
                     unaffected — their score column name doesn't change
                     based on how the underlying value was computed.
    """
    if method_name == "SingleColumnAblation Strategy":
        if task_type == "classification":
            score_col = "importance_score" if classification_metric == "importance_score" \
                else "num_label_flips"
        else:
            score_col = "importance_score"
        return "feature_name", score_col
    if method_name in METHOD_COLUMNS:
        return METHOD_COLUMNS[method_name]
    raise ValueError(f"Unknown method: {method_name}")