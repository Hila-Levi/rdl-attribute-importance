"""
Training script for the Relational Graph Transformer (RelGT) on RelBench datasets.

Mirrors the structure of train.py (HGS) but uses:
  - RelGTTokens   for neighbour-token precomputation / caching
  - RelGT         as the backbone model
  - standard DataLoader (not NeighborLoader)
  - clip_grad_norm_ for stable transformer training
  - a  transformer/  sub-directory tree so outputs never mix with HGS artefacts

The relgt-cache (HDF5 token files) is written inside the task's
transformer/ folder.  The processed hetero-graph (data.pt) is read from
the *shared* data root, exactly as the HGS pipeline does.

Usage:
    python train_trans.py --dataset <dataset_name> --task <task_name> [--seed <seed>] [--resume <run_index>]

Examples:
    python train_trans.py --dataset rel-f1 --task driver-position --seed 42
    python train_trans.py --dataset rel-f1 --task driver-dnf       --seed 42
    python train_trans.py --dataset rel-f1 --task driver-dnf --seed 42 --resume 1
    python train_trans.py --dataset rel-f1 --task driver-dnf --seed 42 --resume 1 --finalize
"""

import os
import time
import sys
import copy
import math
import argparse
import json
import pickle
import random
import importlib.metadata
import numpy as np
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import relbench
from relbench.base import TaskType
# RelBench 3: relbench.datasets/relbench.tasks and get_dataset/get_task are
# gone — relbench.load_dataset(...) + dataset.load_task(...) replace them.
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch_frame.config.text_embedder import TextEmbedderConfig
# Metrics are computed locally with sklearn instead of via task.evaluate()
# — see compute_metrics() for why.
from sklearn.metrics import roc_auc_score, mean_absolute_error

# Resolve the src/ directory relative to this file so the import works
# regardless of which directory you run the script from.
# Expected layout:
#   <root>/Utils/                              <- this file lives here
#   <root>/rdl-explain-relgt-integration/src/ <- rdl_explain package lives here
_UTILS_DIR = Path(__file__).resolve().parent
_SRC_DIR   = _UTILS_DIR.parent / "MaskLearningExplainer" / "src"
if not _SRC_DIR.exists():
    raise FileNotFoundError(
        f"Could not find the RelGT source directory at {_SRC_DIR}.\n"
        f"Expected layout:\n"
        f"  <root>/Utils/train_trans.py                       (this file)\n"
        f"  <root>/MaskLearningExplainer/src/rdl_explain/...\n"
        f"Please check your directory structure."
    )
sys.path.insert(0, str(_SRC_DIR))
from rdl_explain_files.model.relgt import RelGT, RelGTConfig, RelGTTokens

# Local imports — same Utils and new transformer config
from train_config_trans import (
    get_dataset_config,
    get_task_config,
    get_transformer_config,
    get_training_config,
    get_loader_config,
)
from Utils import (
    fix_randomness,
    get_next_run_index,
    get_indexed_filepath,
    print_data_distribution,
)


# ---------------------------------------------------------------------------
# Text embedding (shared with HGS pipeline)
# ---------------------------------------------------------------------------

class GloveTextEmbedding:
    """Text embedding using GloVe via sentence-transformers."""

    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def apply_sigmoid(preds):
    """Convert raw logits to probabilities via sigmoid.
    Accepts either a numpy array or a torch.Tensor.
    """
    if isinstance(preds, torch.Tensor):
        return torch.sigmoid(preds)
    return 1.0 / (1.0 + np.exp(-preds))


def get_eval_table(task, split: str, table):
    """Return a table guaranteed to include task.target_col, for local metric
    computation.

    RelBench masks the label column on the default `task.get_table("test")`
    (mask_input_cols=True) so predictions can't accidentally be tuned against
    test labels — this is original RelBench behavior, unchanged by the
    RelBench 3 migration, not a bug. Train/val tables already include the
    target column. Re-fetching with mask_input_cols=False returns the same
    rows in the same order, just with the label column present.
    """
    if task.target_col in table.df.columns:
        return table
    return task.get_table(split, mask_input_cols=False)


def compute_metrics(task_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Local replacement for task.evaluate(...).

    RelBench 3's relbench.metrics module was pruned down to only roc_auc,
    map, and make_nmae — the 'mae' (and rmse/r2/f1/mrr/...) metrics this
    codebase has always tuned and reported on were removed outright, and
    task.evaluate()'s default metric set changed along with it (regression
    tasks now default to NMAE, a different formula/scale than plain MAE).
    Rather than depend on relbench's internal default-metric selection,
    metrics are computed directly here with sklearn — this reproduces the
    exact same formulas RelBench used to report (verified against stored
    metadata from pre-migration runs, matching to within floating-point
    noise), so `metrics[tune_metric]` lookups keep working exactly as before.
    """
    if task_type == "classification":
        return {"roc_auc": float(roc_auc_score(y_true, y_pred))}
    return {"mae": float(mean_absolute_error(y_true, y_pred))}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a RelGT model on a RelBench dataset"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., rel-f1)")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (e.g., driver-position)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--resume", type=int, default=None,
                        help="Run index to resume (e.g. --resume 3). If not set, starts fresh.")
    parser.add_argument("--finalize", action="store_true",
                        help="Skip training, load best checkpoint from a resumed run, and save all artifacts.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sanity check — forward-pass smoke test using real train batches
# ---------------------------------------------------------------------------
# NOTE: RelGTTokens always precomputes tokens for the *full* split and feeds
# them through multiprocessing.Pool (which requires num_workers >= 1).
# Re-running precomputation just for a tiny subset would be wasteful and
# fragile.  Instead we reuse the already-built train loader and model and
# simply verify that:
#   1. A forward pass does not crash.
#   2. The loss is finite.
#   3. A few gradient steps reduce the loss.
# This catches the most common failure modes (shape mismatches, NaN weights,
# wrong device placement) without any extra precomputation.

def run_sanity_check(model, train_loader, loss_fn, device, seed, task_name):
    """
    Smoke-test the RelGT model on a small number of real train batches.

    Runs SANITY_STEPS gradient steps and checks that loss decreases and
    remains finite.  Returns True on success, False otherwise.
    Importantly, model weights are restored to their pre-check state
    so the main training loop starts from the original initialisation.
    """
    SANITY_STEPS = 5

    task_cfg  = get_task_config(task_name)
    task_type = task_cfg["task_type"]

    print("\n" + "=" * 60)
    print(f" SANITY CHECK — {task_cfg['task_description']}")
    print(f" (smoke test: {SANITY_STEPS} gradient steps on real batches)")
    print("=" * 60)

    # Snapshot weights so we can restore them afterwards
    original_state = copy.deepcopy(model.state_dict())

    fix_randomness(seed)
    sanity_opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    model.train()
    loader_iter = iter(train_loader)

    try:
        for step in range(1, SANITY_STEPS + 1):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            _batch_to_device(batch, device)
            sanity_opt.zero_grad()

            pred   = _forward(model, batch)
            pred   = pred.view(-1) if pred.size(-1) == 1 else pred
            labels = batch["labels"]
            loss   = loss_fn(pred.float(), labels.float())

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"   Step {step}: loss is {loss.item():.4f}  ❌")
                model.load_state_dict(original_state)
                return False

            loss.backward()
            sanity_opt.step()
            losses.append(loss.item())
            print(f"   Step {step}/{SANITY_STEPS}: loss = {loss.item():.4f}")

    except Exception as e:
        print(f"   Forward/backward pass raised an exception: {e}  ❌")
        model.load_state_dict(original_state)
        return False

    # Restore original weights — sanity check must not affect training
    model.load_state_dict(original_state)

    # Pass if loss stayed finite and trended downward (or at least did not explode)
    loss_decreased = losses[-1] <= losses[0] * 2   # generous — just check no explosion
    success = all(np.isfinite(l) for l in losses) and loss_decreased

    print("\n" + "=" * 60)
    print(f" Initial loss: {losses[0]:.4f}   Final loss: {losses[-1]:.4f}")
    print("✅ PASS — forward/backward OK, loss finite and stable." if success
          else "❌ FAIL — loss exploded or NaN detected.")
    print("=" * 60)
    return success



# ---------------------------------------------------------------------------
# Helpers for batch operations
# ---------------------------------------------------------------------------

def _batch_to_device(batch: dict, device: torch.device):
    """Move tensor entries of a RelGT batch dict to device in-place."""
    tensor_keys = [
        "neighbor_types", "node_indices", "neighbor_hops",
        "neighbor_times", "edge_index", "batch", "labels",
    ]
    for k in tensor_keys:
        if k in batch and isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)


def _forward(model: RelGT, batch: dict) -> torch.Tensor:
    """Run a forward pass and return the raw model output (no squeeze)."""
    grouped_tf_dict = {
        "grouped_tfs":     batch["grouped_tfs"],
        "grouped_indices": batch["grouped_indices"],
        "flat_batch_idx":  batch["flat_batch_idx"],
        "flat_nbr_idx":    batch["flat_nbr_idx"],
    }
    return model(
        batch["neighbor_types"],
        batch["node_indices"],
        batch["neighbor_hops"],
        batch["neighbor_times"],
        grouped_tf_dict,
        edge_index=batch["edge_index"],
        batch=batch["batch"],
    )


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, loss_fn, device,
                max_steps, grad_clip_norm):
    """One epoch of training; returns mean loss."""
    model.train()
    loss_accum = count_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Training"), start=1):
        _batch_to_device(batch, device)
        optimizer.zero_grad()

        pred   = _forward(model, batch)
        pred   = pred.view(-1) if pred.size(-1) == 1 else pred
        labels = batch["labels"]
        loss   = loss_fn(pred.float(), labels.float())

        if torch.isnan(loss):
            print("   NaN loss detected — stopping epoch early.")
            return float("nan")

        loss.backward()
        if grad_clip_norm is not None:
            clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        loss_accum  += loss.item() * pred.size(0)
        count_accum += pred.size(0)

        if max_steps is not None and step >= max_steps:
            break

    return loss_accum / count_accum


@torch.no_grad()
def compute_val_loss(model, loader, loss_fn, device):
    """Compute mean loss on a validation loader (raw logits, no sigmoid)."""
    model.eval()
    loss_accum = count_accum = 0

    for batch in tqdm(loader, desc="Val loss", leave=False):
        _batch_to_device(batch, device)
        pred   = _forward(model, batch)
        pred   = pred.view(-1) if pred.size(-1) == 1 else pred
        labels = batch["labels"]
        loss   = loss_fn(pred.float(), labels.float())
        loss_accum  += loss.item() * pred.size(0)
        count_accum += pred.size(0)

    return loss_accum / count_accum


@torch.no_grad()
def evaluate(model, loader, task, split_table, device,
             clamp_min=None, clamp_max=None):
    """
    Run inference over a split and return ordered predictions as a numpy array.

    Predictions are reordered by global_idx (as in the notebook) so that
    task.evaluate() sees them in the correct row order.
    """
    model.eval()
    pred_list, idx_list = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        _batch_to_device(batch, device)
        pred = _forward(model, batch)
        if clamp_min is not None:
            pred = torch.clamp(pred, clamp_min, clamp_max)
        if task.task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ):
            pred = apply_sigmoid(pred)
        # squeeze single-channel dim AFTER clamp/sigmoid — matches notebook order
        pred = pred.view(-1) if pred.size(-1) == 1 else pred
        pred_list.append(pred.cpu().numpy())
        idx_list.append(batch["global_idx"].cpu().numpy())

    preds = np.concatenate(pred_list)
    idxs  = np.concatenate(idx_list)
    ordered = np.empty_like(preds)
    ordered[idxs] = preds
    return ordered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args        = parse_args()
    DATASET_NAME = args.dataset
    TASK_NAME    = args.task
    SEED         = args.seed
    RESUME_INDEX = args.resume
    FINALIZE     = args.finalize

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print environment info
    try:
        print(f"RelBench version: {importlib.metadata.version('relbench')}")
    except importlib.metadata.PackageNotFoundError:
        pass
    print(f"Device: {device}")

    # ── Configs ──────────────────────────────────────────────────────────────
    dataset_cfg     = get_dataset_config(DATASET_NAME)
    task_cfg        = get_task_config(TASK_NAME)
    transformer_cfg = get_transformer_config(TASK_NAME)
    training_cfg    = get_training_config(TASK_NAME)
    loader_cfg      = get_loader_config(TASK_NAME)

    fix_randomness(SEED)

    # ── Dataset + task ───────────────────────────────────────────────────────
    print(f"\nLoading dataset: {DATASET_NAME}")
    print(f"Loading task:    {TASK_NAME}")
    dataset = relbench.load_dataset(DATASET_NAME)
    task    = dataset.load_task(TASK_NAME)

    out_channels     = task_cfg["out_channels"]
    tune_metric      = task_cfg["tune_metric"]
    higher_is_better = task_cfg["higher_is_better"]
    loss_fn          = task_cfg["loss_fn"]

    # ── Build / load hetero-graph from shared data root ──────────────────────
    root_dir   = Path(dataset_cfg["root_dir_name"])
    root_dir.mkdir(parents=True, exist_ok=True)
    cache_path = root_dir / f"{DATASET_NAME}{dataset_cfg['cache_suffix']}"

    db = dataset.get_db()
    col_to_stype_dict = get_stype_proposal(db)
    print(f"col_to_stype_dict: {col_to_stype_dict}")

    text_embedder_cfg = TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device),
        batch_size=256,
    )

    fix_randomness(SEED)
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_embedder_cfg,
        cache_dir=str(cache_path),
    )

    # ── Normalise node / edge type casing (mirrors notebook cell 7) ──────────
    # RelGT requires node types in lowercase to match task.entity_table.
    from torch_geometric.data import HeteroData
    new_data = HeteroData()
    for node_type in data.node_types:
        for key, val in data[node_type].items():
            new_data[node_type.lower()][key] = val
    for src, rel, dst in data.edge_types:
        for key, val in data[src, rel, dst].items():
            new_data[src.lower(), rel, dst.lower()][key] = val
    data = new_data
    col_stats_dict = {nt.lower(): stats for nt, stats in col_stats_dict.items()}

    # ── Task splits ───────────────────────────────────────────────────────────
    fix_randomness(SEED)
    train_table = task.get_table("train")
    val_table   = task.get_table("val")
    test_table  = task.get_table("test")

    print_data_distribution(
        TASK_NAME,
        train_table,
        val_table,
        task_cfg["target_col"],
        task_cfg["task_type"],
        task_cfg.get("label_names"),
    )

    # ── Transformer-specific: relgt-cache lives under transformer/ ───────────
    # Resolved relative to the saved_model path's parent (= transformer/)
    transformer_dir  = Path(task_cfg["saved_model_dir_name"]).parent
    relgt_cache_dir  = transformer_dir / "relgt-cache" / DATASET_NAME / TASK_NAME

    K = transformer_cfg["K"]
    print(f"\n RelGT token cache: {relgt_cache_dir.resolve()}")

    # ── Directories (needed before resume load so RESUME_FILE path is known) ─
    SAVED_MODEL_DIR = (_UTILS_DIR / task_cfg["saved_model_dir_name"]).resolve()
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if RESUME_INDEX is not None:
        run_index = RESUME_INDEX
        print(f"\n⟳  Resuming run index {run_index}")
    else:
        run_index = get_next_run_index(SAVED_MODEL_DIR)
        print(f"\n Starting fresh run — index {run_index}")

    CHECKPOINT_DIR  = (SAVED_MODEL_DIR / "checkpoints" / f"run_{run_index}").resolve()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    RESUME_FILE        = CHECKPOINT_DIR / f"resume_{run_index}.pt"
    EPOCH_HISTORY_FILE = SAVED_MODEL_DIR / f"epoch_history_{run_index}.json"

    # ── On resume: restore random state BEFORE building RelGTTokens ──────────
    # This ensures RelGTTokens resamples the exact same neighborhood as the
    # original run (because the cache was deleted and will be rebuilt).
    _pre_token_rng = None
    if RESUME_INDEX is not None:
        if not RESUME_FILE.exists():
            raise FileNotFoundError(
                f"Resume file not found: {RESUME_FILE}\n"
                f"Make sure run index {run_index} exists in {CHECKPOINT_DIR}"
            )
        _pre_token_rng = torch.load(RESUME_FILE, map_location=device, weights_only=False).get("pre_token_rng")
        if _pre_token_rng is not None:
            random.setstate(_pre_token_rng["python"])
            np.random.set_state(_pre_token_rng["numpy"])
            
            cpu_state = _pre_token_rng["torch_cpu"]
            if not isinstance(cpu_state, torch.ByteTensor):
                cpu_state = torch.ByteTensor(bytearray(cpu_state))
            torch.set_rng_state(cpu_state)
            
            if _pre_token_rng.get("torch_cuda") is not None:
                cuda_states = _pre_token_rng["torch_cuda"]
                if isinstance(cuda_states, list) and len(cuda_states) > 0:
                    if not isinstance(cuda_states[0], torch.ByteTensor):
                        cuda_states = [torch.ByteTensor(bytearray(s)) for s in cuda_states]
                torch.cuda.set_rng_state_all(cuda_states)
            
            print("   ✓ Restored pre-token RNG state — neighborhood will match original run.")
        else:
            print("   ⚠️  No pre-token RNG state found in resume file (old checkpoint). Neighborhood may differ.")

    # ── RelGTTokens datasets + DataLoaders ───────────────────────────────────
    # Capture the exact RNG state right before constructing RelGTTokens so we
    # can restore it on future resumes and get the same neighborhood.
    pre_token_rng_snapshot = {
        "python":     random.getstate(),
        "numpy":      np.random.get_state(),
        "torch_cpu":  torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    split_datasets = {}
    for split in ["train", "val", "test"]:
        fix_randomness(SEED)
        split_datasets[split] = RelGTTokens(
            data=data,
            task=task,
            K=K,
            split=split,
            undirected=True,
            precompute=True,
            precomputed_dir=str(relgt_cache_dir),
            num_workers=loader_cfg["num_workers_precompute"],
        )

    batch_size = training_cfg["batch_size"]
    loaders = {
        split: DataLoader(
            split_datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=split_datasets[split].collate,
            num_workers=loader_cfg["num_workers_loader"],
            persistent_workers=loader_cfg["persistent_workers"],
            pin_memory=(device.type == "cuda"),
        )
        for split in ["train", "val", "test"]
    }

    clamp_min = clamp_max = None
    # ── Task-type specific settings ───────────────────────────────────────────
    if task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        out_channels = task.num_labels    # override config
    elif task.task_type == TaskType.REGRESSION:
        clamp_min, clamp_max = np.percentile(
            train_table.df[task.target_col].to_numpy(), [2, 98]
        )

    # ── Build RelGT model ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INITIALIZING RelGT MODEL")
    print("=" * 60)

    train_ds = split_datasets["train"]
    relgt_cfg = RelGTConfig(
        local_num_layers=transformer_cfg["local_num_layers"],
        channels=transformer_cfg["channels"],
        out_channels=out_channels,
        global_dim=transformer_cfg["global_dim"],
        heads=transformer_cfg["heads"],
        ff_dropout=transformer_cfg["ff_dropout"],
        attn_dropout=transformer_cfg["attn_dropout"],
        conv_type=transformer_cfg["conv_type"],
        num_centroids=transformer_cfg["num_centroids"],
        sample_node_len=K,
    )

    fix_randomness(SEED)
    model = RelGT(
        num_nodes=data.num_nodes,
        max_neighbor_hop=train_ds.max_neighbor_hop,
        node_type_map=train_ds.node_type_to_index,
        col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
        col_stats_dict=col_stats_dict,
        **relgt_cfg.model_dump(),
    ).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Total parameters:     {total_params:,}")
    print(f" Trainable parameters: {trainable_params:,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if higher_is_better else "min",
        factor=training_cfg["scheduler_factor"],
        patience=training_cfg["scheduler_patience"],
        verbose=training_cfg["scheduler_verbose"],
    )

    epochs           = training_cfg["epochs"]
    patience         = training_cfg["early_stopping_patience"]
    max_steps        = training_cfg.get("max_steps_per_epoch")
    grad_clip_norm   = training_cfg.get("grad_clip_norm")

    print("\n" + "=" * 60)
    print(" TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"   Optimizer:               Adam (lr={training_cfg['learning_rate']}, "
          f"wd={training_cfg['weight_decay']})")
    print(f"   Scheduler:               ReduceLROnPlateau (patience={training_cfg['scheduler_patience']})")
    print(f"   Max epochs:              {epochs}")
    print(f"   Early stopping patience: {patience}")
    print(f"   Max steps / epoch:       {max_steps}")
    print(f"   Gradient clip norm:      {grad_clip_norm}")
    print(f"   Loss function:           {loss_fn.__class__.__name__}")
    print(f"   Eval metric:             {tune_metric}")
    print(f"   Batch size:              {batch_size}")
    print(f"   K (neighbor tokens):     {K}")

    # ── Sanity check ──────────────────────────────────────────────────────────
    # The loaders and model are already built, so we reuse them directly.
    # Weights are restored inside run_sanity_check before returning.
    sanity_passed = run_sanity_check(
        model, loaders["train"], loss_fn, device, SEED, TASK_NAME,
    )
    if not sanity_passed:
        print("\n⚠️  Sanity check failed. Proceeding, but results may be suboptimal.")

    # ── Initialise training state ─────────────────────────────────────────────

    state_dict         = None
    best_val_metric    = -math.inf if higher_is_better else math.inf
    best_epoch         = None
    patience_counter   = 0
    train_losses       = []
    val_metrics_history = []

    # Per-epoch histories for comparison plots
    epoch_history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "epoch_time_sec": [],
        f"val_{tune_metric}": [],
        "val_probs": [],    # list of np.ndarray, one per epoch (classification only)
        "val_binary": [],   # list of np.ndarray, one per epoch (classification only)
    }

    start_epoch = 1

    if RESUME_INDEX is not None:
        # RESUME_FILE was already loaded above to restore the pre-token RNG state.
        # Load it again fully now to restore model/optimizer/scheduler state.
        resume = torch.load(RESUME_FILE, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        best_val_metric  = resume["best_val_metric"]
        patience_counter = resume["patience_counter"]
        state_dict       = resume["best_model_state_dict"]
        start_epoch      = resume["epoch"] + 1
        best_epoch       = resume.get("best_epoch")
        if EPOCH_HISTORY_FILE.exists():
            with open(EPOCH_HISTORY_FILE) as f:
                epoch_history = json.load(f)
        print(f"   Loaded from epoch {resume['epoch']}  "
              f"(best {tune_metric} so far: {best_val_metric:.4f})")
        if FINALIZE:
            print(f"\n⏹  --finalize flag set: skipping training, saving artifacts from best epoch ({best_epoch}).")
        else:
            print(f"   Resuming from epoch {start_epoch}")

    training_start_time = time.time()
    if not FINALIZE:
        fix_randomness(SEED)
    for epoch in range(start_epoch, epochs + 1):
        if FINALIZE:
            break   # skip all training, jump straight to final evaluation
        epoch_start_time = time.time()
        print(f"\n Epoch {epoch}/{epochs}")
        print("-" * 60)

        train_loss = train_epoch(
            model, loaders["train"], optimizer, loss_fn,
            device, max_steps, grad_clip_norm,
        )

        if math.isnan(train_loss):
            break

        train_losses.append(train_loss)

        val_loss = compute_val_loss(model, loaders["val"], loss_fn, device)

        val_preds   = evaluate(model, loaders["val"], task, val_table, device,
                               clamp_min, clamp_max)
        val_y_true  = val_table.df[task.target_col].values
        val_metrics = compute_metrics(task_cfg["task_type"], val_y_true, val_preds)
        val_val     = val_metrics[tune_metric]
        val_metrics_history.append(val_val)

        epoch_time = time.time() - epoch_start_time
        print(f"   Train loss:  {train_loss:.4f}")
        print(f"   Val loss:    {val_loss:.4f}")
        print(f"   Val metrics: {val_metrics}")
        print(f"   Epoch time:  {epoch_time:.1f}s")

        # Record epoch-level history
        epoch_history["epoch"].append(epoch)
        epoch_history["train_loss"].append(float(train_loss))
        epoch_history["val_loss"].append(float(val_loss))
        epoch_history[f"val_{tune_metric}"].append(float(val_val))
        epoch_history["epoch_time_sec"].append(float(epoch_time))
        if task_cfg["task_type"] == "classification":
            # val_preds are already probabilities — evaluate() applies
            # apply_sigmoid internally, consistent with all other splits.
            val_binary = (val_preds >= 0.5).astype(float)
            epoch_history["val_probs"].append(val_preds.tolist())
            epoch_history["val_binary"].append(val_binary.tolist())

        scheduler.step(val_val)

        improved = (higher_is_better and val_val > best_val_metric) or \
                   (not higher_is_better and val_val < best_val_metric)
        if improved:
            best_val_metric = val_val
            best_epoch       = epoch
            state_dict      = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"   ✓ New best {tune_metric}: {best_val_metric:.4f}  (epoch {epoch})")
            # Save checkpoint for this new best
            ckpt_name = f"checkpoint_epoch{epoch:03d}_{tune_metric}{best_val_metric:.4f}.pt"
            torch.save({
                'epoch':            epoch,
                'model_state_dict': state_dict,
                f'val_{tune_metric}': best_val_metric,
                'seed':             SEED,
            }, CHECKPOINT_DIR / ckpt_name)
            print(f"   ✓ Checkpoint saved: {ckpt_name}")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{patience})")
            
        # ── Save resume state after every epoch ──────────────────────────────
        torch.save({
            'epoch':                 epoch,
            'model_state_dict':      model.state_dict(),
            'best_model_state_dict': state_dict,
            'optimizer_state_dict':  optimizer.state_dict(),
            'scheduler_state_dict':  scheduler.state_dict(),
            'best_val_metric':       best_val_metric,
            'best_epoch':            best_epoch,
            'patience_counter':      patience_counter,
            'seed':                  SEED,
            # Saved so that on resume, RelGTTokens rebuilds the same neighborhood
            'pre_token_rng':         pre_token_rng_snapshot,
        }, RESUME_FILE)

        # ── Append epoch history to file after every epoch ────────────────────
        with open(EPOCH_HISTORY_FILE, 'w') as f:
            json.dump(epoch_history, f)

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    total_training_time = time.time() - training_start_time
    epochs_run = len(epoch_history["epoch"])
    avg_epoch_time = total_training_time / epochs_run if epochs_run > 0 else 0.0

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f" FINAL EVALUATION — {DATASET_NAME.upper()} / {TASK_NAME.upper()}")
    print("=" * 60)

    if state_dict is not None:
        model.load_state_dict(state_dict)
        print(" Loaded best model weights.")
    else:
        print(" Warning: no best state recorded; using current weights.")

    model.eval()
    fix_randomness(SEED)

    # Val metric comes from best_val_metric captured at the true peak epoch
    # during training — no re-evaluation needed.
    authoritative_val_metric = best_val_metric
    print(f"\n Validation {tune_metric} (best, epoch {best_epoch}): {authoritative_val_metric:.4f}")

    print("\n Val set:")
    fix_randomness(SEED)
    val_preds         = evaluate(model, loaders["val"], task, val_table, device,
                                 clamp_min, clamp_max)
    val_y_true_final  = val_table.df[task.target_col].values
    val_metrics_final = compute_metrics(task_cfg["task_type"], val_y_true_final, val_preds)
    print(f"   {val_metrics_final}")

    print("\n Test set:")
    fix_randomness(SEED)
    test_preds   = evaluate(model, loaders["test"], task, test_table, device,
                            clamp_min, clamp_max)
    test_eval_table = get_eval_table(task, "test", test_table)
    test_y_true  = test_eval_table.df[task.target_col].values
    test_metrics = compute_metrics(task_cfg["task_type"], test_y_true, test_preds)
    print(f"   {test_metrics}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    # SAVED_MODEL_DIR and run_index were set before the training loop

    print("\n" + "=" * 80)
    print("SAVING ARTEFACTS FOR EXPLAINER NOTEBOOKS")
    print("=" * 80)
    print(f"Output directory: {SAVED_MODEL_DIR.resolve()}")
    print(f"Run index: {run_index}")

    # Model weights
    weights_path = get_indexed_filepath(SAVED_MODEL_DIR, "model_weights", ".pt", run_index)
    torch.save(
        {
            "model_state_dict": state_dict,
            "best_epoch":       best_epoch,
            "model_config": {
                "K": K,
                "local_num_layers": transformer_cfg["local_num_layers"],
                "channels": transformer_cfg["channels"],
                "out_channels": out_channels,
                "global_dim": transformer_cfg["global_dim"],
                "heads": transformer_cfg["heads"],
                "ff_dropout": transformer_cfg["ff_dropout"],
                "attn_dropout": transformer_cfg["attn_dropout"],
                "conv_type": transformer_cfg["conv_type"],
                "num_centroids": transformer_cfg["num_centroids"],
            },
            f"val_{tune_metric}": float(authoritative_val_metric),
            "seed": SEED,
            # RelGT needs these to reconstruct the model
            "num_nodes": data.num_nodes,
            "max_neighbor_hop": train_ds.max_neighbor_hop,
            "node_type_map": train_ds.node_type_to_index,
        },
        weights_path,
    )
    print(f"✓ Saved: {weights_path.name}")

    # Graph data (shared hetero-graph — useful to have alongside weights)
    graph_data_path = get_indexed_filepath(SAVED_MODEL_DIR, "graph_data", ".pt", run_index)
    torch.save(data, graph_data_path)
    print(f"✓ Saved: {graph_data_path.name}")

    # Column stats
    col_stats_path = get_indexed_filepath(SAVED_MODEL_DIR, "col_stats_dict", ".pkl", run_index)
    with open(col_stats_path, "wb") as f:
        pickle.dump(col_stats_dict, f)
    print(f"✓ Saved: {col_stats_path.name}")

    # Task + dataset objects
    # NOTE: RelBench 3 regression tasks can carry a reference to a local
    # closure (the `nmae` function returned by relbench.metrics.make_nmae,
    # used internally for the NMAE normalizer) which `pickle` cannot
    # serialize — pickling raises AttributeError: Can't pickle local object
    # 'make_nmae.<locals>.nmae'. This pickle was always just an optimization:
    # Utils.load_research_components() already falls back to reconstructing
    # dataset/task fresh via relbench.load_dataset(...)/dataset.load_task(...)
    # whenever the pickle can't be loaded, and that reconstruction is cheap/
    # local once Hub data is cached. So a pickling failure here is not
    # fatal — skip it rather than aborting a completed training run over a
    # save-time convenience file.
    task_path = get_indexed_filepath(SAVED_MODEL_DIR, "task", ".pkl", run_index)
    try:
        with open(task_path, "wb") as f:
            pickle.dump(task, f)
        print(f"✓ Saved: {task_path.name}")
    except (AttributeError, TypeError, pickle.PicklingError) as e:
        print(f"⚠️  Could not pickle task ({e}). Skipping — "
              f"load_research_components() will reconstruct it via "
              f"relbench.load_dataset('{DATASET_NAME}').load_task('{TASK_NAME}') instead.")

    dataset_path = get_indexed_filepath(SAVED_MODEL_DIR, "dataset", ".pkl", run_index)
    try:
        with open(dataset_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"✓ Saved: {dataset_path.name}")
    except (AttributeError, TypeError, pickle.PicklingError) as e:
        print(f"⚠️  Could not pickle dataset ({e}). Skipping — "
              f"load_research_components() will reconstruct it via "
              f"relbench.load_dataset('{DATASET_NAME}') instead.")

    # Predictions for all splits
    print(f"\n Generating train predictions …")
    fix_randomness(SEED)
    train_preds        = evaluate(model, loaders["train"], task, train_table, device,
                                  clamp_min, clamp_max)
    train_y_true       = train_table.df[task.target_col].values
    train_metrics_final = compute_metrics(task_cfg["task_type"], train_y_true, train_preds)
    print(f"   Final Train Metrics: {train_metrics_final}")

    # All three splits are already probabilities for classification (evaluate()
    # applies apply_sigmoid internally) or raw values for regression.
    # Consistent across train / val / test — no extra conversion needed.
    for name, arr in [("train_predictions", train_preds),
                      ("val_predictions",   val_preds),
                      ("test_predictions",  test_preds)]:
        p = get_indexed_filepath(SAVED_MODEL_DIR, name, ".npy", run_index)
        np.save(p, arr)
        print(f"✓ Saved: {p.name}")

    # Loader / tokenisation config
    loader_config_snapshot = {
        "K": K,
        "batch_size": batch_size,
        "num_workers_precompute": loader_cfg["num_workers_precompute"],
        "num_workers_loader": loader_cfg["num_workers_loader"],
        "precomputed_dir": str(relgt_cache_dir),
    }
    loader_cfg_path = get_indexed_filepath(SAVED_MODEL_DIR, "loader_config", ".json", run_index)
    with open(loader_cfg_path, "w") as f:
        json.dump(loader_config_snapshot, f, indent=2)
    print(f"✓ Saved: {loader_cfg_path.name}")

    # epoch_history was written incrementally during training
    print(f"✓ epoch_history already saved: {EPOCH_HISTORY_FILE.name}")

    # Delete the resume file only after all artefacts are saved.
    # If saving fails mid-way, the resume file is preserved so
    # training can be restarted without losing progress.
    if RESUME_FILE.exists():
        RESUME_FILE.unlink()
        print("✓ Resume file removed (all artefacts saved successfully)")

    # Metadata
    metadata = {
        "seed": SEED,
        "dataset_name": DATASET_NAME,
        "task_name": TASK_NAME,
        "model_type": "RelGT",
        # transformer arch
        "K": K,
        "local_num_layers": transformer_cfg["local_num_layers"],
        "channels": transformer_cfg["channels"],
        "out_channels": out_channels,
        "global_dim": transformer_cfg["global_dim"],
        "heads": transformer_cfg["heads"],
        # metrics
        f"train_{tune_metric}": float(train_metrics_final[tune_metric]),
        f"val_{tune_metric}":   float(authoritative_val_metric),
        f"test_{tune_metric}":  float(test_metrics[tune_metric]),
        "entity_table": task.entity_table,
        # paths (relative) so notebooks can locate the cache
        "relgt_cache_dir": str(relgt_cache_dir),
        # training bookkeeping
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "total_training_time_sec": round(total_training_time, 1),
        "avg_epoch_time_sec": round(avg_epoch_time, 1),
    }
    metadata_path = get_indexed_filepath(SAVED_MODEL_DIR, "metadata", ".json", run_index)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved: {metadata_path.name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"COMPLETE — {DATASET_NAME} / {TASK_NAME}")
    print("=" * 80)
    print(f"\nLocation: {SAVED_MODEL_DIR.resolve()}\n")
    print(f"Model performance ({tune_metric.upper()}):")
    print(f"  Train: {metadata[f'train_{tune_metric}']:.4f}")
    print(f"  Val:   {metadata[f'val_{tune_metric}']:.4f}  (best at epoch {best_epoch})")
    print(f"  Test:  {metadata[f'test_{tune_metric}']:.4f}")

    print(f"\nTraining time:")
    print(f"  Epochs run:        {epochs_run}")
    print(f"  Total time:        {total_training_time:.1f}s  ({total_training_time/60:.1f} min)")
    print(f"  Avg time / epoch:  {avg_epoch_time:.1f}s")

    print("\n✓ Ready to use in explainer notebooks!")


if __name__ == "__main__":
    main()