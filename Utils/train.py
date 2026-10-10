"""
Main training script for RelBench datasets.

Usage:
    python train.py --dataset <dataset_name> --task <task_name> [--seed <seed>] [--resume <run_index>]

Example:
    python train.py --dataset rel-f1 --task driver-position --seed 42
    python train.py --dataset rel-f1 --task driver-dnf --seed 42
    python train.py --dataset rel-f1 --task driver-dnf --seed 42 --resume 1
"""

import os
import time
import sys
import copy
import math
import argparse
import torch
import numpy as np
import json
import pickle
import importlib.metadata
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List

# Third-party imports
import relbench
import torch_geometric
import torch_frame
from torch import Tensor
from sentence_transformers import SentenceTransformer
# RelBench 3: relbench.datasets/relbench.tasks and get_dataset/get_task are
# gone — relbench.load_dataset(...) + dataset.load_task(...) replace them.
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph
from torch_frame.config.text_embedder import TextEmbedderConfig
# Metrics are computed locally with sklearn instead of via task.evaluate()
# — see compute_metrics() for why.
from sklearn.metrics import roc_auc_score, mean_absolute_error

# Force local src to front of sys.path so it takes priority over
# any installed (namespace) package version of rdl_explain.
_THIS_DIR     = Path(os.path.dirname(os.path.abspath(__file__)))
_RESEARCH_DIR = _THIS_DIR.parent   # research-clean/

if str(_RESEARCH_DIR) not in sys.path:
    sys.path.append(str(_RESEARCH_DIR))

# Insert at position 0 to override any installed rdl_explain namespace package
model_src_path = str(_RESEARCH_DIR / "MaskLearningExplainer" / "src")
sys.path.insert(0, model_src_path)

# Local imports
from train_config import (
    get_dataset_config, 
    get_task_config, 
    get_loader_config,
    get_sanity_check_config,
    get_training_config
)
from Utils import (
    fix_randomness, 
    create_consistent_loader,
    get_next_run_index,
    get_indexed_filepath,
    print_data_distribution,
    print_loader_info
)

from rdl_explain_files.model.gnn.model import HGSModel
from rdl_explain_files.model.gnn.config import ModelConfig


class GloveTextEmbedding:
    """Text embedding using GloVe."""
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))


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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a model on RelBench dataset')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Dataset name (e.g., rel-f1)')
    parser.add_argument('--task', type=str, required=True,
                       help='Task name (e.g., driver-position)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--resume', type=int, default=None,
                       help='Run index to resume (e.g. --resume 3). If not set, starts fresh.')
    return parser.parse_args()


def run_sanity_check(data, col_stats_dict, task, train_table, device, seed,
                     task_name, entity_table):
    """
    Run sanity check to verify model capacity by overfitting on a small subset.

    For classification tasks  → tracks AUROC; passes when AUROC > success_threshold.
    For regression    → tracks MAE;   passes when MAE   < success_threshold.

    Returns:
        bool: True if sanity check passed, False otherwise.
    """
    from relbench.modeling.graph import get_node_train_table_input

    sanity_config = get_sanity_check_config(task_name)
    task_config = get_task_config(task_name)
    task_type = task_config['task_type']

    print("\n" + "=" * 60)
    print(f" SANITY CHECK: Testing Model Capacity on {task_config['task_description']}")
    print("=" * 60)

    fix_randomness(seed)

    small_train_df = train_table.df.sample(
        n=sanity_config['sample_size'],
        random_state=seed
    ).copy()
    print(f"\n Small training subset: {len(small_train_df)} samples")

    class SmallTable:
        def __init__(self, df, original_table):
            self.df = df
            self.time_col = getattr(original_table, 'time_col', None)
            self.target_col = getattr(original_table, 'target_col', None)
            for attr in dir(original_table):
                if not attr.startswith('_') and attr not in ['df', 'time_col', 'target_col']:
                    try:
                        setattr(self, attr, getattr(original_table, attr))
                    except:
                        pass

    small_table_obj = SmallTable(small_train_df, train_table)

    # Remove existing sampler if present
    if hasattr(data, '_sampler'):
        delattr(data, '_sampler')

    small_loader = create_consistent_loader(
        data=data,
        loader_config={
            'num_neighbors': sanity_config['num_neighbors'],
            'batch_size': sanity_config['batch_size']
        },
        task=task,
        table=small_table_obj,
        split_name="train"
    )

    # Initialize fresh model (no dropout so it can overfit freely)
    fix_randomness(seed)
    sanity_model = HGSModel(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=sanity_config['model_layers'],
        channels=sanity_config['model_channels'],
        out_channels=task_config['out_channels'],
        aggr="sum",
        norm="batch_norm",
        dropout=0.0,
    ).to(device)

    sanity_optimizer = torch.optim.Adam(
        sanity_model.parameters(),
        lr=sanity_config['learning_rate']
    )
    sanity_loss_fn = task_config['loss_fn']

    print(f"Training for {sanity_config['epochs']} epochs on {len(small_train_df)} samples...")

    # Initialise metric to worst-case value for the task type
    current_metric = float('inf') if task_type == "regression" else 0.0

    for epoch in range(1, sanity_config['epochs'] + 1):
        sanity_model.train()
        epoch_loss = epoch_count = 0

        for batch in small_loader:
            batch = batch.to(device)
            sanity_optimizer.zero_grad()

            pred = sanity_model(batch, entity_table).view(-1)
            y_true = batch[entity_table].y

            loss = sanity_loss_fn(pred.float(), y_true.float())
            loss.backward()
            sanity_optimizer.step()

            epoch_loss += loss.item() * pred.size(0)
            epoch_count += pred.size(0)

        # Log metric every 10 epochs and on the first epoch
        if epoch % 10 == 0 or epoch == 1:
            sanity_model.eval()
            all_preds, all_labels = [], []

            with torch.no_grad():
                for batch in small_loader:
                    batch = batch.to(device)
                    pred = sanity_model(batch, entity_table).view(-1)
                    all_preds.append(pred.cpu())
                    all_labels.append(batch[entity_table].y.cpu())

            preds_cat = torch.cat(all_preds)
            labels_cat = torch.cat(all_labels)

            if task_type == "regression":
                current_metric = torch.mean(torch.abs(preds_cat - labels_cat)).item()
                print(f"   Epoch {epoch:3d}: Loss={epoch_loss/epoch_count:.4f}, "
                      f"MAE={current_metric:.4f}")
            else:
                # Binary classification — apply sigmoid then compute AUROC
                from sklearn.metrics import roc_auc_score
                y_scores = apply_sigmoid(preds_cat.numpy())
                y_true_np = labels_cat.numpy()
                try:
                    current_metric = roc_auc_score(y_true_np, y_scores)
                except ValueError:
                    current_metric = 0.5
                print(f"   Epoch {epoch:3d}: Loss={epoch_loss/epoch_count:.4f}, "
                      f"AUROC={current_metric:.4f}")

    print("\n" + "=" * 60)
    metric_label = "MAE" if task_type == "regression" else "AUROC"
    print(f" FINAL SANITY {metric_label}: {current_metric:.4f}")

    threshold = sanity_config['success_threshold']
    if task_type == "regression":
        success = current_metric < threshold          # lower MAE is better
    else:
        success = current_metric > threshold          # higher AUROC is better

    if success:
        print("✅ PASS: Model successfully overfitted. Capacity is good.")
    else:
        print("❌ FAIL: Check if features are present or if the model has enough capacity.")
    print("=" * 60)

    return success


def train_epoch(model, loader, optimizer, loss_fn, entity_table, device):
    """Train for one epoch."""
    model.train()
    loss_accum = count_accum = 0
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred = model(batch, entity_table)
        pred = pred.view(-1) if pred.dim() > 1 and pred.size(1) == 1 else pred
        
        y_true = batch[entity_table].y
        loss = loss_fn(pred.float(), y_true.float())
        
        if torch.isnan(loss):
            print(f"   NaN loss detected! Stopping training.")
            return float('nan')
        
        loss.backward()
        optimizer.step()
        
        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)
    
    return loss_accum / count_accum


@torch.no_grad()
def compute_val_loss(model, loader, loss_fn, entity_table, device):
    """Compute mean loss on a validation loader (raw logits, no sigmoid)."""
    model.eval()
    loss_accum = count_accum = 0

    for batch in tqdm(loader, desc="Val loss", leave=False):
        batch = batch.to(device)
        pred    = model(batch, entity_table)
        pred    = pred.view(-1) if pred.dim() > 1 and pred.size(1) == 1 else pred
        y_true  = batch[entity_table].y
        loss    = loss_fn(pred.float(), y_true.float())
        loss_accum  += loss.item() * pred.size(0)
        count_accum += pred.size(0)

    return loss_accum / count_accum


@torch.no_grad()
def test(model, loader, entity_table, device, task_type="regression"):
    """Test the model and return predictions.
    For classification, returns probabilities (sigmoid applied).
    For regression, returns raw values.
    """
    model.eval()
    pred_list = []
    
    for batch in tqdm(loader, desc="Testing", leave=False):
        batch = batch.to(device)
        pred = model(batch, entity_table)
        pred = pred.view(-1) if pred.dim() > 1 and pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    
    result = torch.cat(pred_list, dim=0).numpy()
    
    if task_type == "classification":
        result = apply_sigmoid(result)
    
    if np.isnan(result).any():
        print(f"   WARNING: {np.isnan(result).sum()} NaN values detected in predictions!")
    
    return result


def main():
    """Main training pipeline."""
    args = parse_args()
    DATASET_NAME = args.dataset
    TASK_NAME = args.task
    SEED = args.seed
    RESUME_INDEX = args.resume

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        version = importlib.metadata.version("relbench")
        print(f"RelBench version: {version}")
    except importlib.metadata.PackageNotFoundError:
        print("RelBench version not found via metadata")
    print(f"Device: {device}")

    dataset_config = get_dataset_config(DATASET_NAME)
    task_config = get_task_config(TASK_NAME)

    fix_randomness(SEED)

    print(f"\nLoading dataset: {DATASET_NAME}")
    print(f"Loading task: {TASK_NAME}")
    dataset = relbench.load_dataset(DATASET_NAME)
    task    = dataset.load_task(TASK_NAME)

    out_channels = task_config['out_channels']
    tune_metric = task_config['tune_metric']
    higher_is_better = task_config['higher_is_better']
    loss_fn = task_config['loss_fn']

    root_dir = Path(f"./{dataset_config['root_dir_name']}")
    root_dir.mkdir(parents=True, exist_ok=True)

    db = dataset.get_db()
    col_to_stype_dict = get_stype_proposal(db)
    print(f"col_to_stype_dict: {col_to_stype_dict}")

    text_embedder_cfg = TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), 
        batch_size=256
    )
    
    cache_path = root_dir / f"{DATASET_NAME}{dataset_config['cache_suffix']}"
    
    fix_randomness(SEED)
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_embedder_cfg,
        cache_dir=str(cache_path),
    )

    fix_randomness(SEED)
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    print_data_distribution(
        TASK_NAME, 
        train_table, 
        val_table, 
        task_config['target_col'],
        task_config['task_type'],
        task_config.get('label_names')
    )

    loader_config = get_loader_config(TASK_NAME, SEED)
    loader_dict = {}
    
    for split, table in [
        ("train", train_table),
        ("val", val_table),
        ("test", test_table),
    ]:
        fix_randomness(SEED)
        loader_dict[split] = create_consistent_loader(
            data=data,
            loader_config=loader_config,
            task=task,
            table=table,
            split_name=split
        )
        print_loader_info(split, table, loader_config['batch_size'])

    print("\n" + "=" * 60)
    print("INITIALIZING MODEL")
    print("=" * 60)

    training_config = get_training_config(TASK_NAME)
    
    model_channels = training_config['model_channels']
    model_layers = training_config['model_layers']
    model_aggr = training_config['model_aggr']
    model_norm = training_config['model_norm']
    model_dropout = training_config['model_dropout']
    
    model_config = ModelConfig(
        gnn_layers=model_layers,
        channels=model_channels,
        out_channels=out_channels,
        aggr=model_aggr,
        norm=model_norm,
    )
    
    fix_randomness(SEED)
    model = HGSModel(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=model_config.gnn_layers,
        channels=model_config.channels,
        out_channels=model_config.out_channels,
        aggr=model_config.aggr,
        norm=model_config.norm,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f" Model initialized")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Layers: {model_config.gnn_layers}")
    print(f"   Hidden channels: {model_config.channels}")

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=training_config['learning_rate'], 
        weight_decay=training_config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max' if higher_is_better else 'min', 
        factor=training_config['scheduler_factor'], 
        patience=training_config['scheduler_patience'],
        verbose=training_config['scheduler_verbose']
    )
    
    epochs = training_config['epochs']
    patience = training_config['early_stopping_patience']
    entity_table = task.entity_table
    
    print("\n" + "=" * 60)
    print(" TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"   Optimizer: Adam (lr={training_config['learning_rate']}, weight_decay={training_config['weight_decay']})")
    print(f"   Scheduler: ReduceLROnPlateau (patience={training_config['scheduler_patience']})")
    print(f"   Max epochs: {epochs}")
    print(f"   Early stopping patience: {patience}")
    print(f"   Loss function: {loss_fn.__class__.__name__}")
    print(f"   Evaluation metric: {tune_metric}")
    print(f"   Batch size: {loader_config['batch_size']}")
    print(f"   Entity table: {entity_table}")

    sanity_passed = run_sanity_check(
        data, col_stats_dict, task, train_table, 
        device, SEED, TASK_NAME, entity_table
    )
    
    if not sanity_passed:
        print("\n⚠️  Sanity check failed. Proceeding anyway, but results may be suboptimal.")

    # ── Directories ───────────────────────────────────────────────────────────
    SAVED_MODEL_DIR = (_THIS_DIR / task_config['saved_model_dir_name']).resolve()
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if RESUME_INDEX is not None:
        run_index = RESUME_INDEX
        print(f"\n⟳  Resuming run index {run_index}")
    else:
        run_index = get_next_run_index(SAVED_MODEL_DIR)
        print(f"\n Starting fresh run — index {run_index}")

    # Each run gets its own checkpoint subdirectory so checkpoints from
    # different runs never mix together.
    CHECKPOINT_DIR  = (SAVED_MODEL_DIR / "checkpoints" / f"run_{run_index}").resolve()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    RESUME_FILE        = CHECKPOINT_DIR / f"resume_{run_index}.pt"
    EPOCH_HISTORY_FILE = SAVED_MODEL_DIR / f"epoch_history_{run_index}.json"

    # ── Initialise training state ──────────────────────────────────────────────────────────
    state_dict = None
    best_val_metric = -math.inf if higher_is_better else math.inf
    best_epoch       = None
    patience_counter = 0
    train_losses = []
    val_metrics_history = []
    # Per-epoch histories for comparison plots
    epoch_history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        f"val_{tune_metric}": [],
        "epoch_time_sec": [],
        # val probabilities and binary predictions at each epoch (classification only)
        "val_probs": [],       # list of np.ndarray, one per epoch
        "val_binary": [],      # list of np.ndarray, one per epoch
    }
    
    start_epoch = 1

    if RESUME_INDEX is not None:
        if not RESUME_FILE.exists():
            raise FileNotFoundError(
                f"Resume file not found: {RESUME_FILE}\n"
                f"Make sure run index {run_index} exists in {CHECKPOINT_DIR}"
            )
        resume = torch.load(RESUME_FILE, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        best_val_metric  = resume["best_val_metric"]
        patience_counter = resume["patience_counter"]
        state_dict       = resume["best_model_state_dict"]
        start_epoch      = resume["epoch"] + 1
        if EPOCH_HISTORY_FILE.exists():
            with open(EPOCH_HISTORY_FILE) as f:
                epoch_history = json.load(f)
        print(f"   Loaded from epoch {resume['epoch']}  "
              f"(best {tune_metric} so far: {best_val_metric:.4f})")
        print(f"   Resuming from epoch {start_epoch}")

    training_start_time = time.time()
    fix_randomness(SEED)
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        print(f"\n Epoch {epoch}/{epochs}")
        print("-" * 60)
        
        train_loss = train_epoch(model, loader_dict["train"], optimizer, 
                                loss_fn, entity_table, device)
        
        if math.isnan(train_loss):
            break
        
        train_losses.append(train_loss)
        
        val_loss = compute_val_loss(model, loader_dict["val"], loss_fn,
                                     entity_table, device)

        val_pred = test(model, loader_dict["val"], entity_table, device,
                         task_type=task_config["task_type"])
        val_y_true = val_table.df[task.target_col].values
        val_metrics = compute_metrics(task_config["task_type"], val_y_true, val_pred)
        
        val_metric_value = val_metrics[tune_metric]
        val_metrics_history.append(val_metric_value)
        
        epoch_time = time.time() - epoch_start_time
        print(f"   Train loss:  {train_loss:.4f}")
        print(f"   Val loss:    {val_loss:.4f}")
        print(f"   Val metrics: {val_metrics}")
        print(f"   Epoch time:  {epoch_time:.1f}s")
        
        # Record epoch-level history
        epoch_history["epoch"].append(epoch)
        epoch_history["train_loss"].append(float(train_loss))
        epoch_history["val_loss"].append(float(val_loss))
        epoch_history[f"val_{tune_metric}"].append(float(val_metric_value))
        epoch_history["epoch_time_sec"].append(float(epoch_time))
        if task_config["task_type"] == "classification":
            # val_pred is already probabilities — test() applied sigmoid internally
            val_binary = (val_pred >= 0.5).astype(float)
            epoch_history["val_probs"].append(val_pred.tolist())
            epoch_history["val_binary"].append(val_binary.tolist())

        scheduler.step(val_metric_value)
        
        if (higher_is_better and val_metric_value > best_val_metric) or \
           (not higher_is_better and val_metric_value < best_val_metric):
            best_val_metric = val_metric_value
            best_epoch      = epoch
            state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"   New best {tune_metric}: {best_val_metric:.4f}  (epoch {epoch})")
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
            'patience_counter':      patience_counter,
            'seed':                  SEED,
        }, RESUME_FILE)

        # ── Append epoch history to file after every epoch ────────────────────
        with open(EPOCH_HISTORY_FILE, 'w') as f:
            json.dump(epoch_history, f)

        if patience_counter >= patience:
            print(f"\n  Early stopping triggered at epoch {epoch}")
            break
    
    total_training_time = time.time() - training_start_time
    epochs_run = len(epoch_history["epoch"])
    avg_epoch_time = total_training_time / epochs_run if epochs_run > 0 else 0.0

    print("\n" + "=" * 60)
    print(f" FINAL EVALUATION ON {DATASET_NAME.upper()}")
    print("=" * 60)

    if state_dict is not None:
        model.load_state_dict(state_dict)
        print(" Successfully loaded best model weights.")
    else:
        print(" Warning: state_dict is None. Using current model weights.")

    model.eval()
    fix_randomness(SEED)
    
    # Val metric comes from best_val_metric captured at the true peak epoch
    # during training — no re-evaluation needed.
    authoritative_val_metric = best_val_metric
    print(f"\n Validation {tune_metric} (best, epoch {best_epoch}): {authoritative_val_metric:.4f}")
    
    print("\n Test set (Blind Evaluation):")
    fix_randomness(SEED)
    test_pred = test(model, loader_dict["test"], entity_table, device,
                     task_type=task_config["task_type"])
    test_eval_table = get_eval_table(task, "test", test_table)
    test_y_true = test_eval_table.df[task.target_col].values
    test_metrics = compute_metrics(task_config["task_type"], test_y_true, test_pred)
    print(f"   Final Test Metrics: {test_metrics}")
    
    print(f"\nModel is set to evaluation mode for {DATASET_NAME} - {TASK_NAME}.")

    # SAVED_MODEL_DIR and run_index were set before the training loop

    print("\n" + "=" * 80)
    print("SAVING EVERYTHING FOR EXPLAINER NOTEBOOKS")
    print("=" * 80)
    print(f"Output directory: {SAVED_MODEL_DIR.absolute()}")
    print(f"Run index: {run_index}")

    model_weights_path = get_indexed_filepath(SAVED_MODEL_DIR, 'model_weights', '.pt', run_index)
    torch.save({
        'model_state_dict': state_dict,
        'best_epoch':       best_epoch,
        'model_config': {
            'num_layers': model_layers,
            'channels': model_channels,
            'out_channels': out_channels,
            'aggr': model_aggr,
            'norm': model_norm,
        },
        f'val_{tune_metric}': float(authoritative_val_metric),
        'seed': SEED,
    }, model_weights_path)
    print(f"✓ Saved: {model_weights_path.name}")

    graph_data_path = get_indexed_filepath(SAVED_MODEL_DIR, 'graph_data', '.pt', run_index)
    torch.save(data, graph_data_path)
    print(f"✓ Saved: {graph_data_path.name}")
    
    col_stats_path = get_indexed_filepath(SAVED_MODEL_DIR, 'col_stats_dict', '.pkl', run_index)
    with open(col_stats_path, 'wb') as f:
        pickle.dump(col_stats_dict, f)
    print(f"✓ Saved: {col_stats_path.name}")
    
    # NOTE: RelBench 3 regression tasks can carry a reference to a local
    # closure (the `nmae` function returned by relbench.metrics.make_nmae,
    # used internally for the NMAE normalizer) which `pickle` cannot
    # serialize — pickling raises AttributeError: Can't pickle local object
    # 'make_nmae.<locals>.nmae'. This pickle was always just an optimization:
    # Utils.load_research_components() already falls back to reconstructing
    # dataset/task fresh via relbench.load_dataset(...)/dataset.load_task(...)
    # whenever the pickle can't be loaded (e.g. after a relbench version
    # change), and that reconstruction is cheap/local once Hub data is
    # cached. So a pickling failure here is not fatal — skip it rather than
    # aborting a completed training run over a save-time convenience file.
    task_path = get_indexed_filepath(SAVED_MODEL_DIR, 'task', '.pkl', run_index)
    try:
        with open(task_path, 'wb') as f:
            pickle.dump(task, f)
        print(f"✓ Saved: {task_path.name}")
    except (AttributeError, TypeError, pickle.PicklingError) as e:
        print(f"⚠️  Could not pickle task ({e}). Skipping — "
              f"load_research_components() will reconstruct it via "
              f"relbench.load_dataset('{DATASET_NAME}').load_task('{TASK_NAME}') instead.")

    dataset_path = get_indexed_filepath(SAVED_MODEL_DIR, 'dataset', '.pkl', run_index)
    try:
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"✓ Saved: {dataset_path.name}")
    except (AttributeError, TypeError, pickle.PicklingError) as e:
        print(f"⚠️  Could not pickle dataset ({e}). Skipping — "
              f"load_research_components() will reconstruct it via "
              f"relbench.load_dataset('{DATASET_NAME}') instead.")

    print(f"\nGenerating baseline predictions for {DATASET_NAME}...")
    
    fix_randomness(SEED)
    train_pred = test(model, loader_dict["train"], entity_table, device,
                      task_type=task_config["task_type"])
    train_y_true = train_table.df[task.target_col].values
    train_metrics_final = compute_metrics(task_config["task_type"], train_y_true, train_pred)
    print(f"   Final Train Metrics: {train_metrics_final}")
    # All three splits are already probabilities for classification (test() applies
    # sigmoid internally) or raw values for regression — consistent, no extra step needed.

    train_pred_path = get_indexed_filepath(SAVED_MODEL_DIR, 'train_predictions', '.npy', run_index)
    val_pred_path   = get_indexed_filepath(SAVED_MODEL_DIR, 'val_predictions',   '.npy', run_index)
    test_pred_path  = get_indexed_filepath(SAVED_MODEL_DIR, 'test_predictions',  '.npy', run_index)
    
    np.save(train_pred_path, train_pred)
    np.save(val_pred_path,   val_pred)
    np.save(test_pred_path,  test_pred)
    
    print(f"✓ Saved: {train_pred_path.name}")
    print(f"✓ Saved: {val_pred_path.name}")
    print(f"✓ Saved: {test_pred_path.name}")

    loader_config_path = get_indexed_filepath(SAVED_MODEL_DIR, 'loader_config', '.json', run_index)
    with open(loader_config_path, 'w') as f:
        json.dump(loader_config, f, indent=2)
    print(f"✓ Saved: {loader_config_path.name}")
    
    # epoch_history was written incrementally during training
    print(f"✓ epoch_history already saved: {EPOCH_HISTORY_FILE.name}")

    # Delete the resume file only after all artefacts are saved.
    # If saving fails mid-way, the resume file is preserved so
    # training can be restarted without losing progress.
    if RESUME_FILE.exists():
        RESUME_FILE.unlink()
        print("✓ Resume file removed (all artefacts saved successfully)")

    metadata = {
        'seed': SEED,
        'model_layers': model_layers,
        'model_channels': model_channels,
        'model_aggr': model_aggr,
        'model_norm': model_norm,
        f'train_{tune_metric}': float(train_metrics_final[tune_metric]),
        f'val_{tune_metric}': float(authoritative_val_metric),
        f'test_{tune_metric}': float(test_metrics[tune_metric]),
        'entity_table': entity_table,
        'dataset_name': DATASET_NAME,
        'task_name': TASK_NAME,
        'best_epoch': best_epoch,
        'epochs_run': epochs_run,
        'total_training_time_sec': round(total_training_time, 1),
        'avg_epoch_time_sec': round(avg_epoch_time, 1),
    }
    
    metadata_path = get_indexed_filepath(SAVED_MODEL_DIR, 'metadata', '.json', run_index)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved: {metadata_path.name}")

    print("\n" + "=" * 80)
    print(f"COMPLETE! All files saved for {DATASET_NAME} / {TASK_NAME}")
    print("=" * 80)
    print(f"\nLocation: {SAVED_MODEL_DIR.absolute()}\n")
    
    print("Saved files:")
    for filepath in [model_weights_path, graph_data_path, col_stats_path, task_path, dataset_path,
                     train_pred_path, val_pred_path, test_pred_path,
                     loader_config_path, metadata_path]:
        print(f"  - {filepath.name}")
    
    print(f"\nModel performance ({tune_metric.upper()}):")
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