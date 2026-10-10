"""
Configuration file for explainer dataset and task-specific settings.

Directory structure:
    rel-<n>/
        data_<n>/                        <- shared graph cache
        <task-name>/
            HGS/
                saved_model/             <- HGS trained weights
                singleColumnAblation_explainer/       <- HGS explainer outputs
            transformer/
                saved_model/             <- RelGT trained weights
                singleColumnAblation_explainer/       <- RelGT explainer outputs

Config structure per task:
    Shared fields (task_type, tune_metric, key_columns_to_exclude,
    stype_mapping) are defined ONCE at the task level.

    Model-specific fields live under 'model_paths':
        'model_paths': {
            'hgs':         { saved_model_base_dir, explainer_output_dir, visualization },
            'transformer': { saved_model_base_dir, explainer_output_dir, visualization },
        }

    get_explainer_config() merges these into one flat dict for callers.
"""

from pathlib import Path


# ============================================================================
# EXPLAINER CONFIGURATIONS
# ============================================================================

EXPLAINER_CONFIG = {
    "rel-arxiv": {
        # -------------------------------------------------------------------
        # Binary classification
        # -------------------------------------------------------------------
        "paper-citation": {
            "task_type": "classification",
            "tune_metric": "roc_auc",
            "key_columns_to_exclude": [
                ('papers', 'Paper_ID'),
                ('authors', 'Author_ID'),
                ('categories', 'Category_ID'),
                ('citations', 'Paper_ID'),
                ('citations', 'References_Paper_ID'),
                ('paperCategories', 'Paper_ID'),
                ('paperCategories', 'Category_ID'),
                ('paperAuthors', 'Author_ID'),
                ('paperAuthors', 'Paper_ID'),
            ],
            # Primary_Category_ID is a foreign key into `categories`, but its
            # value is the paper's main arXiv category - genuine signal rather
            # than an arbitrary row identifier. RelBench 3 started declaring it
            # as a foreign key, which silently dropped it from this explainer
            # (11 features instead of 12) while every other explainer kept it.
            "keys_to_explain": [
                ('papers', 'Primary_Category_ID'),
            ],
            "stype_mapping": {
                "numerical": ["numerical"],
                "categorical": ["categorical"],
                "text": ["text", "embedding"],
                "timestamp": ["timestamp"],
            },
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-arxiv/paper-citation/HGS/saved_model",
                    "explainer_output_dir": "../rel-arxiv/paper-citation/HGS/singleColumnAblation_explainer",
                    "visualization": {
                        "figsize_per_row": (28, 8),
                        "bar_color": "#4682B4",
                        "bar_width": 0.6,
                        "top_n_features": 15,
                    },
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-arxiv/paper-citation/transformer/saved_model",
                    "explainer_output_dir": "../rel-arxiv/paper-citation/transformer/singleColumnAblation_explainer",
                    "visualization": {
                        "figsize_per_row": (28, 8),
                        "bar_color": "#FF9800",
                        "bar_width": 0.6,
                        "top_n_features": 15,
                    },
                },
            },
        },
        # -------------------------------------------------------------------
        # Regression
        # -------------------------------------------------------------------
        "author-publication": {
            "task_type": "regression",
            "tune_metric": "mae",
            "key_columns_to_exclude": [
                ('papers', 'Paper_ID'),
                ('authors', 'Author_ID'),
                ('categories', 'Category_ID'),
                ('citations', 'Paper_ID'),
                ('citations', 'References_Paper_ID'),
                ('paperCategories', 'Paper_ID'),
                ('paperCategories', 'Category_ID'),
                ('paperAuthors', 'Author_ID'),
                ('paperAuthors', 'Paper_ID'),
            ],
            # Primary_Category_ID is a foreign key into `categories`, but its
            # value is the paper's main arXiv category - genuine signal rather
            # than an arbitrary row identifier. RelBench 3 started declaring it
            # as a foreign key, which silently dropped it from this explainer
            # (11 features instead of 12) while every other explainer kept it.
            "keys_to_explain": [
                ('papers', 'Primary_Category_ID'),
            ],
            "stype_mapping": {
                "numerical": ["numerical"],
                "categorical": ["categorical"],
                "text": ["text", "embedding"],
                "timestamp": ["timestamp"],
            },
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-arxiv/author-publication/HGS/saved_model",
                    "explainer_output_dir": "../rel-arxiv/author-publication/HGS/singleColumnAblation_explainer",
                    "visualization": {
                        "figsize_per_row": (28, 8),
                        "bar_color": "#E07B39",
                        "bar_width": 0.6,
                        "top_n_features": 15,
                    },
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-arxiv/author-publication/transformer/saved_model",
                    "explainer_output_dir": "../rel-arxiv/author-publication/transformer/singleColumnAblation_explainer",
                    "visualization": {
                        "figsize_per_row": (28, 8),
                        "bar_color": "#FF9800",
                        "bar_width": 0.6,
                        "top_n_features": 15,
                    },
                },
            },
        },
    },
    "rel-trial": {
        # -------------------------------------------------------------------
        # Binary classification
        # -------------------------------------------------------------------
        "study-outcome": {
            "task_type": "classification",
            "tune_metric": "roc_auc",
            "key_columns_to_exclude": [
                ('studies', 'nct_id'),
                ('interventions', 'intervention_id'),
                ('interventions', 'nct_id'),
                ('conditions', 'condition_id'),
                ('conditions', 'nct_id'),
                ('sponsors', 'sponsor_id'),
                ('sponsors', 'nct_id'),
                ('eligibilities', 'eligibility_id'),
                ('eligibilities', 'nct_id'),
                ('outcomes', 'outcome_id'),
                ('outcomes', 'nct_id'),
                ('facilities', 'facility_id'),
                ('facilities', 'nct_id'),
            ],
            "stype_mapping": {
                "numerical": ["numerical"],
                "categorical": ["categorical"],
                "text": ["text", "embedding"],
                "timestamp": ["timestamp"],
            },
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-trial/study-outcome/HGS/saved_model",
                    "explainer_output_dir": "../rel-trial/study-outcome/HGS/singleColumnAblation_explainer",
                    "visualization": {
                        "figsize_per_row": (28, 32),
                        "bar_color": "#4682B4",
                        "bar_width": 0.6,
                        "top_n_features": 15,
                    },
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-trial/study-outcome/transformer/saved_model",
                    "explainer_output_dir": "../rel-trial/study-outcome/transformer/singleColumnAblation_explainer",
                    "visualization": {
                        "figsize_per_row": (28, 32),
                        "bar_color": "#FF9800",
                        "bar_width": 0.6,
                        "top_n_features": 15,
                    },
                },
            },
        },
        # -------------------------------------------------------------------
        # Regression
        # -------------------------------------------------------------------
        "study-adverse": {
            "task_type": "regression",
            "tune_metric": "mae",
            "key_columns_to_exclude": [
                ('studies', 'nct_id'),
                ('interventions', 'intervention_id'),
                ('interventions', 'nct_id'),
                ('conditions', 'condition_id'),
                ('conditions', 'nct_id'),
                ('sponsors', 'sponsor_id'),
                ('sponsors', 'nct_id'),
                ('eligibilities', 'eligibility_id'),
                ('eligibilities', 'nct_id'),
                ('outcomes', 'outcome_id'),
                ('outcomes', 'nct_id'),
                ('facilities', 'facility_id'),
                ('facilities', 'nct_id'),
            ],
            "stype_mapping": {
                "numerical": ["numerical"],
                "categorical": ["categorical"],
                "text": ["text", "embedding"],
                "timestamp": ["timestamp"],
            },
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-trial/study-adverse/HGS/saved_model",
                    "explainer_output_dir": "../rel-trial/study-adverse/HGS/singleColumnAblation_explainer",
                    "visualization": {
                        "figsize_per_row": (28, 32),
                        "bar_color": "#E07B39",
                        "bar_width": 0.6,
                        "top_n_features": 15,
                    },
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-trial/study-adverse/transformer/saved_model",
                    "explainer_output_dir": "../rel-trial/study-adverse/transformer/singleColumnAblation_explainer",
                    "visualization": {
                        "figsize_per_row": (28, 32),
                        "bar_color": "#FF9800",
                        "bar_width": 0.6,
                        "top_n_features": 15,
                    },
                },
            },
        },
    },
    "rel-f1": {
        # -------------------------------------------------------------------
        # Binary classification
        # -------------------------------------------------------------------
        "driver-dnf": {
            "task_type": "classification",
            "tune_metric": "roc_auc",
            "key_columns_to_exclude": [
                ('races', 'raceId'),
                ('drivers', 'driverId'),
                ('standings', 'driverStandingsId'),
                ('constructor_standings', 'constructorStandingsId'),
                ('circuits', 'circuitId'),
                ('qualifying', 'qualifyId'),
                ('constructors', 'constructorId'),
                ('results', 'resultId'),
                ('constructor_results', 'constructorResultsId'),
                ('races', 'circuitId'),
                ('standings', 'raceId'),
                ('standings', 'driverId'),
                ('constructor_standings', 'raceId'),
                ('constructor_standings', 'constructorId'),
                ('qualifying', 'raceId'),
                ('qualifying', 'driverId'),
                ('qualifying', 'constructorId'),
                ('results', 'raceId'),
                ('results', 'driverId'),
                ('results', 'constructorId'),
                ('constructor_results', 'raceId'),
                ('constructor_results', 'constructorId'),
            ],
            "stype_mapping": {
                "numerical": ["numerical"],
                "categorical": ["categorical"],
                "text": ["text", "embedding"],
                "timestamp": ["timestamp"],
            },
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-f1/driver-dnf/HGS/saved_model",
                    "explainer_output_dir": "../rel-f1/driver-dnf/HGS/singleColumnAblation_explainer",
                    "visualization": {
                        "figsize_per_row": (28, 20),
                        "bar_color": "#4682B4",
                        "bar_width": 0.6,
                        "top_n_features": 15,
                    },
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-f1/driver-dnf/transformer/saved_model",
                    "explainer_output_dir": "../rel-f1/driver-dnf/transformer/singleColumnAblation_explainer",
                    "visualization": {
                        "figsize_per_row": (28, 20),
                        "bar_color": "#FF9800",
                        "bar_width": 0.6,
                        "top_n_features": 15,
                    },
                },
            },
        },
        # -------------------------------------------------------------------
        # Regression
        # -------------------------------------------------------------------
        "driver-position": {
            "task_type": "regression",
            "tune_metric": "mae",
            "key_columns_to_exclude": [
                ('races', 'raceId'),
                ('drivers', 'driverId'),
                ('standings', 'driverStandingsId'),
                ('constructor_standings', 'constructorStandingsId'),
                ('circuits', 'circuitId'),
                ('qualifying', 'qualifyId'),
                ('constructors', 'constructorId'),
                ('results', 'resultId'),
                ('constructor_results', 'constructorResultsId'),
                ('races', 'circuitId'),
                ('standings', 'raceId'),
                ('standings', 'driverId'),
                ('constructor_standings', 'raceId'),
                ('constructor_standings', 'constructorId'),
                ('qualifying', 'raceId'),
                ('qualifying', 'driverId'),
                ('qualifying', 'constructorId'),
                ('results', 'raceId'),
                ('results', 'driverId'),
                ('results', 'constructorId'),
                ('constructor_results', 'raceId'),
                ('constructor_results', 'constructorId'),
            ],
            "stype_mapping": {
                "numerical": ["numerical"],
                "categorical": ["categorical"],
                "text": ["text", "embedding"],
                "timestamp": ["timestamp"],
            },
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-f1/driver-position/HGS/saved_model",
                    "explainer_output_dir": "../rel-f1/driver-position/HGS/singleColumnAblation_explainer",
                    "visualization": {
                        "figsize_per_row": (28, 20),
                        "bar_color": "#E07B39",
                        "bar_width": 0.6,
                        "top_n_features": 15,
                    },
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-f1/driver-position/transformer/saved_model",
                    "explainer_output_dir": "../rel-f1/driver-position/transformer/singleColumnAblation_explainer",
                    "visualization": {
                        "figsize_per_row": (28, 20),
                        "bar_color": "#FF9800",
                        "bar_width": 0.6,
                        "top_n_features": 15,
                    },
                },
            },
        },
    },
    "rel-event": {
        # -------------------------------------------------------------------
        # Binary classification
        # -------------------------------------------------------------------
        "user-repeat": {
            "task_type": "classification",
            "tune_metric": "roc_auc",
            "key_columns_to_exclude": [
                ('users', 'user_id'),
                ('events', 'event_id'),
                ('events', 'user_id'),
                ('event_attendees', 'event'),
                ('event_attendees', 'user_id'),
                ('user_friends', 'user'),
                ('user_friends', 'friend'),
                ('event_interest', 'event'),
                ('event_interest', 'user'),
            ],
            "stype_mapping": {
                "numerical": ["numerical"],
                "categorical": ["categorical"],
                "text": ["text", "embedding"],
                "timestamp": ["timestamp"],
            },
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-event/user-repeat/HGS/saved_model",
                    "explainer_output_dir": "../rel-event/user-repeat/HGS/singleColumnAblation_explainer",
                    "visualization": {
                        "figsize_per_row": (28, 20),
                        "bar_color": "#4682B4",
                        "bar_width": 0.6,
                        "top_n_features": 15,
                    },
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-event/user-repeat/transformer/saved_model",
                    "explainer_output_dir": "../rel-event/user-repeat/transformer/singleColumnAblation_explainer",
                    "visualization": {
                        "figsize_per_row": (28, 20),
                        "bar_color": "#FF9800",
                        "bar_width": 0.6,
                        "top_n_features": 15,
                    },
                },
            },
        },
        # -------------------------------------------------------------------
        # Regression
        # -------------------------------------------------------------------
        "user-attendance": {
            "task_type": "regression",
            "tune_metric": "mae",
            "key_columns_to_exclude": [
                ('users', 'user_id'),
                ('events', 'event_id'),
                ('events', 'user_id'),
                ('event_attendees', 'event'),
                ('event_attendees', 'user_id'),
                ('user_friends', 'user'),
                ('user_friends', 'friend'),
                ('event_interest', 'event'),
                ('event_interest', 'user'),
            ],
            "stype_mapping": {
                "numerical": ["numerical"],
                "categorical": ["categorical"],
                "text": ["text", "embedding"],
                "timestamp": ["timestamp"],
            },
            "model_paths": {
                "hgs": {
                    "saved_model_base_dir": "../rel-event/user-attendance/HGS/saved_model",
                    "explainer_output_dir": "../rel-event/user-attendance/HGS/singleColumnAblation_explainer",
                    "visualization": {
                        "figsize_per_row": (28, 20),
                        "bar_color": "#4682B4",
                        "bar_width": 0.6,
                        "top_n_features": 15,
                    },
                },
                "transformer": {
                    "saved_model_base_dir": "../rel-event/user-attendance/transformer/saved_model",
                    "explainer_output_dir": "../rel-event/user-attendance/transformer/singleColumnAblation_explainer",
                    "visualization": {
                        "figsize_per_row": (28, 20),
                        "bar_color": "#FF9800",
                        "bar_width": 0.6,
                        "top_n_features": 15,
                    },
                },
            },
        },
    },
}


# ============================================================================
# DEFAULT ABLATION STRATEGIES
# ============================================================================

DEFAULT_STRATEGIES = {
    'numerical': ['Zeroing', 'Mean', 'Permutation'],
    'text': ['Zeroing', 'Random Noise', 'Mean'],
    'timestamp': ['Max Time', 'Permutation', 'Mean'],
    'categorical': ['Zeroing', 'Permutation', 'MFV']
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_explainer_config(dataset_name: str, task_name: str,
                         model_type: str = "hgs") -> dict:
    """Return a flat config dict for a specific dataset, task, and model type.

    Merges shared task-level fields (task_type, tune_metric,
    key_columns_to_exclude, stype_mapping) with the
    model-specific fields (saved_model_base_dir, explainer_output_dir,
    visualization) into a single flat dict — callers do not need to know
    about the nested model_paths structure.

    Args:
        dataset_name: e.g. 'rel-f1'
        task_name:    e.g. 'driver-position'
        model_type:   'hgs' (default) or 'transformer'
    """
    if dataset_name not in EXPLAINER_CONFIG:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in explainer configuration. "
            f"Available datasets: {list(EXPLAINER_CONFIG.keys())}"
        )
    if task_name not in EXPLAINER_CONFIG[dataset_name]:
        raise ValueError(
            f"Task '{task_name}' not found for dataset '{dataset_name}'. "
            f"Available tasks: {list(EXPLAINER_CONFIG[dataset_name].keys())}"
        )

    task_cfg = EXPLAINER_CONFIG[dataset_name][task_name]

    if model_type not in task_cfg["model_paths"]:
        raise ValueError(
            f"model_type '{model_type}' not found for {dataset_name}/{task_name}. "
            f"Available model types: {list(task_cfg['model_paths'].keys())}"
        )

    # Merge shared fields with model-specific fields into one flat dict
    flat_cfg = {k: v for k, v in task_cfg.items() if k != "model_paths"}
    flat_cfg.update(task_cfg["model_paths"][model_type])
    return flat_cfg


def get_saved_model_dir(dataset_name: str, task_name: str,
                        model_type: str = "hgs") -> Path:
    """Get the saved model directory path for a dataset/task/model_type."""
    config = get_explainer_config(dataset_name, task_name, model_type)
    return Path(config['saved_model_base_dir'])


def get_explainer_output_dir(dataset_name: str, task_name: str,
                             model_type: str = "hgs") -> Path:
    """Get the explainer output directory path for a dataset/task/model_type."""
    config = get_explainer_config(dataset_name, task_name, model_type)
    output_dir = Path(config['explainer_output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir