# A Comparative Study of Importance Attribution to Columns in Relational Deep Learning

Code for the paper *A Comparative Study of Importance Attribution to Columns in Relational Deep Learning: [Experiments & Analysis]*.

The study compares explanation methods that assign an importance score to each data attribute (column) of a relational database, quantifying its contribution to the predictions of a Relational Deep Learning (RDL) model. Structural keys (primary and foreign keys) are excluded from the explanations, since masking them changes the relational structure rather than an attribute value.

Experiments use tasks from [RelBench](https://github.com/snap-stanford/relbench) and two RDL architectures:

- **HGS**: Heterogeneous GraphSAGE, following the RelBench methodology.
- **RelGT**: the Relational Graph Transformer ([snap-stanford/relgt](https://github.com/snap-stanford/relgt)).

## Explanation methods

| Method | Folder | Notion of importance |
|---|---|---|
| Single Column Ablation | `SingleColumnAblationExplainer/` | Ablates one attribute at a time and measures the mean absolute shift in the model's predictions |
| Shapley-MC (Monte Carlo Shapley) | `MonteCarloExplainer/` | Approximates the Shapley value by averaging marginal contributions over sampled permutations of the attributes |
| Shapley-KS (Kernel SHAP) | `KernelShapExplainer/` | Approximates the Shapley value with a weighted linear surrogate fitted over sampled attribute coalitions |
| Mask Learning | `MaskLearningExplainer/` | Learns a sparse mask over the attributes that preserves the model's predictions while retaining only the most important attributes |

The three removal-based methods (Single Column Ablation, Shapley-MC, Shapley-KS) share one deterministic ablation operator: mean imputation for numerical, temporal and text attributes (text is imputed over its pretrained embeddings), and the most frequent value (MFV) for categorical attributes. Mask Learning uses its own permutation-based masking.

## Comparing the explanations

The evaluation pipeline compares the methods along three dimensions:

1. **Attribute rankings.** Scores are grouped into buckets of near-ties (threshold `(w_max - w_min) / 100`), converted to top-k rankings with ties following Fagin et al., and compared pairwise with a normalized Spearman footrule similarity for a range of k.
2. **Importance-score distributions.** Scores are min-max normalized to [0, 1] and compared through their distribution over the attributes and the scores of the top-ranked attributes.
3. **Fidelity.** Fidelity+ (necessity) is the prediction shift when the top-k attributes are ablated. Fidelity- (sufficiency) is the prediction shift when only the top-k attributes are kept. Both use the same ablation operator as the removal-based methods.

## Datasets and tasks

| Dataset | Classification task | Regression task |
|---|---|---|
| `rel-trial` | `study-outcome` | `study-adverse` |
| `rel-arxiv` | `paper-citation` | `author-publication` |
| `rel-f1` | `driver-dnf` | `driver-position` |
| `rel-event` | `user-repeat` | `user-attendance` |

## Configuration

Every dataset/task pair has an entry in each of the config files below. Each entry holds the settings shared by both models plus a `model_paths` block with separate input and output directories for `hgs` and `transformer`. To add a dataset or task, add an entry to every file.

| File | What it controls |
|---|---|
| `Utils/train_config.py` | HGS training: dataset settings, task settings, loader settings and the `saved_model` directory |
| `Utils/train_config_trans.py` | The same for RelGT |
| `SingleColumnAblationExplainer/singleColumnAblation_explainer_config.py` | `key_columns_to_exclude` (structural keys left out of the explanation), `keys_to_explain` (key columns that carry real signal and are explained anyway, such as `papers.Primary_Category_ID` in rel-arxiv), `stype_mapping` (which attribute type each column counts as), plot settings, and `DEFAULT_STRATEGIES`, the ablation strategies tried for each attribute type |
| `KernelShapExplainer/kernelShap_explainer_config.py` | `num_coalitions` (3,000 in the paper) and `max_batches` (cap on evaluation batches; `None` = no cap) |
| `MonteCarloExplainer/monteCarlo_explainer_config.py` | `auto_checkpoint_interval` (hours between checkpoints), `convergence_threshold` (stop when the mean standard error drops below it), `max_permutations` and `max_batches` |
| `MaskLearningExplainer/rdl_explainer_config.py` | Model and output paths only. The mask-learning hyperparameters (learning rate 0.05, 50 epochs, automatic selection of the regularization weight) are set in `run_rdl_explainer.py` |
| `EvaluationMethods/evaluation_pipeline_config.py` | Task type, metric and the k values used for fidelity (`FIDELITY_K_FIXED`, `FIDELITY_K_PERCENT`, `fidelity_k_step_long`) |

For rel-arxiv, `max_batches` is `None` because the explainers already run on a fixed test subset (see step 1 below).

## Repository layout

```
Utils/
  train.py                  # train HGS
  train_trans.py            # train RelGT
  train_config*.py          # training settings
  create_test_subset.py     # fixed test subset shared by all explainers
  Utils.py                  # loading, seeding, run indexing helpers
SingleColumnAblationExplainer/
KernelShapExplainer/
MonteCarloExplainer/        # + monteCarlo_analyze.py for plotting results
MaskLearningExplainer/
  run_rdl_explainer.py
  src/rdl_explain_files/    # model definitions (HGS, RelGT) + mask-learning explainer
EvaluationMethods/
  run_evaluation_pipeline.py
  fidelity_calculator.py
  ComparingTopK.py          # ranking profiles and ranking similarity
unified_explainer_visualizations.ipynb   # the paper's cross-task, cross-model figures
kernel_shap_comparison.ipynb             # Shapley-KS convergence curves
monte_carlo_comparison.ipynb             # Shapley-MC convergence curves
paper_style.py                           # matplotlib style matching the ACM sigconf template
```

## Installation

Python 3.11. The versions in `requirements.txt` are pinned to a tested CUDA 12.1 / torch 2.4.0 environment.

```bash
conda create -n rdl-xai python=3.11
conda activate rdl-xai
pip install -r requirements.txt
```

The PyG extension wheels (`torch-scatter`, `torch-sparse`, `pyg-lib`, ...) come from the index in the first line of `requirements.txt`. If your machine uses a different CUDA version, check `nvidia-smi` and change `cu121` to the matching tag (see https://data.pyg.org/whl/).

RelBench downloads datasets automatically on first use.

## Usage

Every script resolves paths relative to its own location, so it can be run from any directory. All scripts take `--dataset` and `--task`. Most also take `--model_type hgs|transformer` (default `hgs`), `--seed` (default `42`) and `--resume <run_index>`.

### 1. Create a test subset (for cases where the test is too large, e.g., rel-arxiv)

The rel-arxiv test sets are too large to explain in full, so all explainers run on the same saved subset:

```bash
python Utils/create_test_subset.py --dataset rel-arxiv --task paper-citation --size 1000
python Utils/create_test_subset.py --dataset rel-arxiv --task author-publication --size 1000
```

This writes `rel-arxiv/<task>/test_subset.csv`, plus a metadata file with its checksum. The script refuses to overwrite an existing subset unless you pass `--force`, because changing it invalidates every explainer run that used it.

### 2. Train the models

```bash
python Utils/train.py       --dataset rel-f1 --task driver-dnf --seed 42   # HGS
python Utils/train_trans.py --dataset rel-f1 --task driver-dnf --seed 42   # RelGT
```

`train_trans.py --resume <N> --finalize` skips training and saves the artifacts from the best checkpoint of run `N`.

### 3. Run the explainers

```bash
python SingleColumnAblationExplainer/run_singleColumnAblation_explainer.py --dataset rel-f1 --task driver-dnf --model_type hgs --strategies fast
python KernelShapExplainer/run_kernelShap_explainer.py                     --dataset rel-f1 --task driver-dnf --model_type hgs
python MonteCarloExplainer/run_monteCarlo_explainer.py                     --dataset rel-f1 --task driver-dnf --model_type hgs
python MaskLearningExplainer/run_rdl_explainer.py                          --dataset rel-f1 --task driver-dnf --model_type hgs
```

Notes:

- `--strategies fast` runs Single Column Ablation with only the shared ablation operator (mean / MFV), which is the setting used in the paper. The default, `all`, also runs every other strategy in `DEFAULT_STRATEGIES` (zeroing, permutation, random noise, max time).
- Shapley-MC can run for hours or days. It writes a checkpoint every `auto_checkpoint_interval` hours and can be continued with `--resume auto` or `--resume <checkpoint.csv>`. Pass `--chernoff` to print how many permutations the Chernoff-Hoeffding bound requires. To run it in the background:
  ```bash
  nohup python MonteCarloExplainer/run_monteCarlo_explainer.py --dataset rel-f1 --task driver-dnf > monte_carlo.log 2>&1 &
  ```
- In the paper, Shapley-MC is stopped once its convergence curve flattens: the mean and maximum absolute change in the scores between consecutive checkpoints. `monte_carlo_comparison.ipynb` and `kernel_shap_comparison.ipynb` plot these curves for Shapley-MC and Shapley-KS. They can be re-run while a job is still writing checkpoints.

### 4. Evaluate

```bash
python EvaluationMethods/run_evaluation_pipeline.py --dataset rel-f1 --task driver-dnf --model_type hgs
```

The pipeline finds the latest results of each explainer, builds a full ranking profile for each one, computes the pairwise ranking similarity for every pair of explainers, and then computes Fidelity+ and Fidelity- over a range of k. The ranking comparison runs first because it needs no model inference. Useful flags:

- `--skip_monte_carlo`: leave Shapley-MC out (for example, while it is still running).
- `--monte_carlo_file <csv>`: use a specific Shapley-MC checkpoint instead of the latest one.
- `--classification_metric label_flip|importance_score`: for classification tasks, how Single Column Ablation ranks attributes and how fidelity is scored. The default, `importance_score`, matches the paper.

### 5. Visualize

`unified_explainer_visualizations.ipynb` scans every completed evaluation run and draws the paper's figures across all tasks and both models: pairwise similarity heatmaps, attribute-weight distributions, normalized-score scatter plots, consensus score grids and fidelity bar charts.

The figures use `paper_style.py`, which sizes them for the ACM sigconf column and text widths and uses the template's body font, Linux Libertine. To install the font:

```bash
mkdir -p ~/.fonts && unzip LinLibertineTTF_5.3.0_2012_07_02.zip -d ~/.fonts
rm -rf ~/.cache/matplotlib   # rebuild matplotlib's font cache
```

If the font is missing, matplotlib falls back to DejaVu Serif and `use_paper_style()` prints a warning.

## Outputs

Results are written next to the code, under one folder per dataset and task:

```
rel-<dataset>/<task>/
  test_subset.csv                      # rel-arxiv only
  HGS/  or  transformer/
    saved_model/                       # weights, predictions, metadata, checkpoints/run_N
    singleColumnAblation_explainer/
    kernelShap_explainer/
    rdl_explainer/
    monteCarlo_explainer/
EvaluationMethods/<dataset>/<task>/<hgs|transformer>/run_N/
  profiles/  stability/  fidelity/  run_index.json
```

Each new run gets the next free index (`run_1`, `run_2`, ...), so earlier results are never overwritten.

## Example: pretrained models for rel-f1 / driver-position

To make it possible to try the explainers without training a model first, the repository includes one example task folder, `rel-f1/`, with a trained HGS model and a trained RelGT model for the `driver-position` regression task (seed 42):

```
rel-f1/
  data_f1/rel-f1_materialized_cache/   # materialized rel-f1 tables (the root_dir_name in train_config.py)
  driver-position/
    HGS/saved_model/                   # trained HGS model
    transformer/saved_model/           # trained RelGT model
```

Each `saved_model/` folder holds the files written by the training script for run 1:

| File | Contents |
|---|---|
| `model_weights_1.pt` | Weights of the best epoch (by validation MAE) |
| `graph_data_1.pt` | The heterogeneous graph built from the database |
| `dataset_1.pkl`, `task_1.pkl`, `col_stats_dict_1.pkl` | The RelBench dataset and task objects, and the column statistics used by the encoders |
| `loader_config_1.json` | Neighbor sampling / RelGT loader settings |
| `metadata_1.json` | Model hyperparameters and train / validation / test MAE |
| `epoch_history_1.json` | Per-epoch training history |
| `{train,val,test}_predictions_1.npy` | The model's predictions on each split |

| Model | Best epoch | Val MAE | Test MAE |
|---|---|---|---|
| HGS (2 layers, 128 channels) | 19 | 2.90 | 3.87 |
| RelGT (K = 100, 128 channels, 4 heads) | 33 | 3.00 | 6.24 |

The paths in the config files already point to these folders, so the explainers and the evaluation pipeline can be run on them directly, skipping step 2 of [Usage](#usage). In each explainer's config file, the `rel-f1` / `driver-position` entry points to the trained model through `saved_model_base_dir`. For example, in `SingleColumnAblationExplainer/singleColumnAblation_explainer_config.py`:

```python
"model_paths": {
    "hgs": {
        "saved_model_base_dir": "../rel-f1/driver-position/HGS/saved_model",
        ...
    },
    "transformer": {
        "saved_model_base_dir": "../rel-f1/driver-position/transformer/saved_model",
        ...
    },
},
```

To explain a different trained model, change `saved_model_base_dir` in the config of every explainer you run, and `task_dir` in `EvaluationMethods/evaluation_pipeline_config.py`.

```bash
python SingleColumnAblationExplainer/run_singleColumnAblation_explainer.py --dataset rel-f1 --task driver-position --model_type hgs --strategies fast
python EvaluationMethods/run_evaluation_pipeline.py                        --dataset rel-f1 --task driver-position --model_type hgs
```

Use `--model_type transformer` for the RelGT model. Its neighbor-token cache (`transformer/relgt-cache/`) is not included in the repository; it is rebuilt automatically the first time the model is loaded.

## Reproducibility

All scripts set `CUBLAS_WORKSPACE_CONFIG` and fix every random seed (`Utils.fix_randomness`). The data loaders are built deterministically (`Utils.create_consistent_loader`), so the same seed explains the same rows in the same order.

## Acknowledgements

The Mask Learning explainer follows *Database Views as Explanations for Relational Deep Learning* (PVLDB 19(7), 2026). The model code in `MaskLearningExplainer/src/rdl_explain_files/model/` is adapted from [RelBench](https://github.com/snap-stanford/relbench), [RelGT](https://github.com/snap-stanford/relgt) and [PyTorch Frame](https://github.com/pyg-team/pytorch-frame) (MIT License). Each file's header lists its source and the changes made.
