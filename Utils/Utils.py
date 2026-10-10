import os
import re
import torch
import pickle
import json
import copy
import random
import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path
from torch_geometric.seed import seed_everything
from torch_geometric.loader import NeighborLoader
from relbench.modeling.graph import get_node_train_table_input

# RelBench 3 API import — relbench.datasets/relbench.tasks and
# get_dataset/get_task/get_task_names/get_dataset_names are gone.
# relbench.load_dataset(...) is now the single entry point, and
# load_task lives on the dataset object: dataset.load_task(task_name).
import relbench

# Repository root (the directory holding rel-<name>/, Utils/, KernelShapExplainer/, ...).
REPO_ROOT = Path(__file__).resolve().parent.parent

def load_research_components(saved_model_dir, device='cpu'):  
    """
    Loads model metadata, graph data, and reconstructs RelBench task/dataset 
    objects to ensure compatibility with the current library version.
    
    Automatically finds and loads files with the latest index.
    Note: graph_data.pt is expected to be created by the user's notebook, 
    not by the training script, so it won't have an index.
    """
    path = Path(saved_model_dir)
    
    run_index = get_next_run_index(path) - 1
    
    # Find the latest index by checking for metadata files
    latest_metadata = get_latest_indexed_file(path, 'metadata', '.json')
    
    if latest_metadata is None:
        raise FileNotFoundError(
            f"No metadata files found in {saved_model_dir}. "
            f"Please ensure training outputs exist in this directory."
        )
    
    # Extract the index from the filename (e.g., metadata_3.json -> 3)
    match = re.search(r'_(\d+)\.json$', latest_metadata.name)
    if match:
        file_index = int(match.group(1))
        print(f"Loading research components with index: {file_index}")
    else:
        raise ValueError(f"Could not extract index from filename: {latest_metadata.name}")
    
    # 1. Load Metadata (JSON)
    metadata_path = get_indexed_filepath(path, 'metadata', '.json', file_index)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✓ Loaded: {metadata_path.name}")
        
    # 2. Load Graph Data (PyG Data object)
    # Try indexed version first (new format), then fall back to non-indexed (old format)
    graph_data_path = get_indexed_filepath(path, 'graph_data', '.pt', file_index)
    
    if not graph_data_path.exists():
        # Fall back to non-indexed version (backward compatibility)
        graph_data_path = path / 'graph_data.pt'
        if not graph_data_path.exists():
            raise FileNotFoundError(
                f"Graph data file not found.\n"
                f"Tried: graph_data_{file_index}.pt and graph_data.pt\n"
                f"Please ensure graph_data is saved in {saved_model_dir}.\n"
                f"If using the new train.py, it will be saved as graph_data_N.pt automatically."
            )
        print(f"✓ Loaded: {graph_data_path.name} (backward compatibility mode)")
    else:
        print(f"✓ Loaded: {graph_data_path.name}")
    
    data = torch.load(graph_data_path, map_location='cpu')
    
    # 3. Load Column Statistics
    col_stats_path = get_indexed_filepath(path, 'col_stats_dict', '.pkl', file_index)
    with open(col_stats_path, 'rb') as f:
        col_stats_dict = pickle.load(f)
    print(f"✓ Loaded: {col_stats_path.name}")
        
    # 4. Reconstruct Dataset and Task via API
    print(f"Re-initializing RelBench components...")
    
    # Get dataset and task names from metadata (no hardcoded defaults)
    dataset_name = metadata.get('dataset_name')
    task_name = metadata.get('task_name')
    
    if dataset_name is None:
        raise ValueError(
            "Dataset name not found in metadata.json. "
            "Please ensure the metadata file contains 'dataset_name' field."
        )
    
    if task_name is None:
        raise ValueError(
            "Task name not found in metadata.json. "
            "Please ensure the metadata file contains 'task_name' field."
        )
    
    print(f"  Dataset: {dataset_name}")
    print(f"  Task: {task_name}")

    # NOTE: dataset/task are ALWAYS reconstructed fresh here via
    # relbench.load_dataset(...)/dataset.load_task(...) rather than unpickled
    # from task_N.pkl/dataset_N.pkl (even when those files exist on disk).
    #
    # Two separate problems with trusting the pickles surfaced during the
    # RelBench 3 migration:
    #   1. Regression tasks can hold a reference to a local closure (the
    #      `nmae` function from relbench.metrics.make_nmae) that pickle
    #      can't even serialize in the first place.
    #   2. More subtly: a pickled Dataset object freezes whatever local
    #      Hugging Face Hub cache snapshot path it resolved to *at pickle
    #      time* as a plain attribute. If that snapshot later loses a file
    #      (cache eviction, disk cleanup, partial download, etc.),
    #      unpickling succeeds structurally — no exception, so the old
    #      try/except-around-pickle.load() fallback never triggers — and the
    #      failure only surfaces much later and deeper, e.g.
    #      `task.get_table("train")` erroring with a FileNotFoundError deep
    #      in pandas/parquet. A freshly-called relbench.load_dataset(...),
    #      by contrast, goes back through huggingface_hub's own cache
    #      verification and re-downloads whatever's missing.
    # Since RelBench 3's Hub-backed loading is designed to be cheap and
    # local once cached (no re-download needed for unchanged data), there's
    # no real performance cost to always reconstructing fresh instead of
    # reading task_N.pkl/dataset_N.pkl.
    dataset = relbench.load_dataset(dataset_name)
    task = dataset.load_task(task_name)
        
    # 5. Load Loader Configuration
    loader_config_path = get_indexed_filepath(path, 'loader_config', '.json', file_index)
    with open(loader_config_path, 'r') as f:
        loader_config = json.load(f)
    print(f"✓ Loaded: {loader_config_path.name}")
        
    # 6. Load Baseline Predictions
    test_predictions_path = get_indexed_filepath(path, 'test_predictions', '.npy', file_index)
    test_predictions = np.load(test_predictions_path)
    print(f"✓ Loaded: {test_predictions_path.name}")
    
    return data, col_stats_dict, task, dataset, metadata, loader_config, test_predictions
  
def get_trained_model(model_class, data, col_stats_dict, metadata, weights_path, device):
    """
    Initializes the GNN model architecture and loads trained weights.
    """
    # Initialize architecture on CPU first
    model = model_class(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=metadata['model_layers'],
        channels=metadata['model_channels'],
        out_channels=1,
        aggr=metadata.get('model_aggr', 'sum'),
        norm=metadata.get('model_norm', 'layer_norm'),
    )
    
    # Load state dict
    # weights_only=False is used because GNN checkpoints often contain custom dictionaries
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    # Move model to target device (CUDA/CPU) and set to evaluation mode
    model.to(device)
    model.eval()
    return model

def fix_randomness(seed: int):
    """
    Set random seeds for reproducibility across different libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_consistent_loader(data, loader_config, task, table, split_name):
    """
    Create a consistent data loader with proper configuration.
    
    Args:
        data: Graph data object
        loader_config: Dictionary with loader configuration
        task: Task object
        table: Table object for the split
        split_name: Name of the split ('train', 'val', 'test')
    
    Returns:
        Configured data loader
    """
    # Get node indices and timestamps for the specific task/table
    table_input = get_node_train_table_input(table=table, task=task)
    
    # Clear cached sampler to ensure the loader respects the current 'data' state
    if hasattr(data, '_sampler'):
        delattr(data, '_sampler')
    
    # num_neighbors may be absent when a transformer loader_config is passed
    # (those configs store K/batch_size instead). Fall back to a sensible default.
    num_neighbors = loader_config.get('num_neighbors', [15, 10, 5])

    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=loader_config['batch_size'],
        temporal_strategy=loader_config.get('temporal_strategy', 'uniform'),
        shuffle=(split_name == "train"),
        num_workers=0,
        persistent_workers=False,
    )
    
    return loader

def get_next_run_index(directory: Path) -> int:
    """
    Find the next available run index by checking all indexed files.
    This ensures all files from the same run share the same index.
    
    Args:
        directory: Directory to search in
    
    Returns:
        Next available run index (starting from 1)
    """
    if not directory.exists():
        return 1
    
    # Pattern to match any file with index pattern (e.g., "anything_1.ext")
    pattern = re.compile(r".*_(\d+)\.\w+$")
    
    max_index = 0
    for file in directory.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                index = int(match.group(1))
                max_index = max(max_index, index)
    
    return max_index + 1


def get_indexed_filepath(directory: Path, base_filename: str, extension: str, run_index: int) -> Path:
    """
    Get an indexed filepath for saving a file with a specific run index.
    
    Args:
        directory: Directory where the file will be saved
        base_filename: Base name without extension
        extension: File extension (e.g., '.pt', '.pkl', '.json')
        run_index: The run index to use for this file
    
    Returns:
        Path object with indexed filename
    """
    filename = f"{base_filename}_{run_index}{extension}"
    return directory / filename


def get_latest_indexed_file(directory: Path, base_filename: str, extension: str) -> Optional[Path]:
    """
    Find the file with the highest index for a given base filename.
    
    Args:
        directory: Directory to search in
        base_filename: Base name without extension and index
        extension: File extension (e.g., '.pt', '.pkl', '.json')
    
    Returns:
        Path to the latest indexed file, or None if no files found
    
    Examples:
        >>> get_latest_indexed_file(Path('./models'), 'model_weights', '.pt')
        Path('./models/model_weights_3.pt')  # if files _1, _2, _3 exist
    """
    if not directory.exists():
        return None
    
    # Pattern to match files like "base_filename_1.ext", "base_filename_2.ext", etc.
    pattern = re.compile(rf"^{re.escape(base_filename)}_(\d+){re.escape(extension)}$")
    
    max_index = -1
    latest_file = None
    
    for file in directory.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                index = int(match.group(1))
                if index > max_index:
                    max_index = index
                    latest_file = file
    
    return latest_file


def print_data_distribution(task_name: str, train_table, val_table, target_col: str, 
                           task_type: str, label_names: Optional[dict] = None):
    """
    Print the distribution of labels in train and validation sets.
    
    Args:
        task_name: Name of the task
        train_table: Training table
        val_table: Validation table
        target_col: Name of the target column
        task_type: Type of task ('binary', 'multiclass', 'regression')
        label_names: Optional dictionary mapping label values to names
    """
    print(f"\n--- Data Distribution Check ({task_name}) ---")
    
    for name, table in [("Train", train_table), ("Val", val_table)]:
        df = table.df
        
        if target_col in df.columns:
            y_values = df[target_col]
            total = len(y_values)
            
            print(f"{name} Split:")
            print(f"  Total samples: {total}")
            
            if task_type == "binary":
                # Binary classification
                pos = int(y_values.sum())
                neg = total - pos
                
                if label_names:
                    print(f"  {label_names.get(1, 'Class 1')} (1): {pos} ({pos/total:.2%})")
                    print(f"  {label_names.get(0, 'Class 0')} (0): {neg} ({neg/total:.2%})")
                else:
                    print(f"  Positive (1): {pos} ({pos/total:.2%})")
                    print(f"  Negative (0): {neg} ({neg/total:.2%})")
            
            elif task_type == "multiclass":
                # Multiclass classification
                unique_values = y_values.value_counts().sort_index()
                for class_idx, count in unique_values.items():
                    class_name = label_names.get(class_idx, f"Class {class_idx}") if label_names else f"Class {class_idx}"
                    print(f"  {class_name}: {count} ({count/total:.2%})")
            
            elif task_type == "regression":
                # Regression task
                print(f"  Mean: {y_values.mean():.4f}")
                print(f"  Std: {y_values.std():.4f}")
                print(f"  Min: {y_values.min():.4f}")
                print(f"  Max: {y_values.max():.4f}")
            
            else:
                print(f"  Unknown task type: {task_type}")
        else:
            print(f"{name} Split: Target column '{target_col}' not found.")
            print(f"  Available columns: {df.columns.tolist()}")
        
        print("-" * 40)


def print_loader_info(split_name: str, table, batch_size: int):
    """
    Print information about a created data loader.
    
    Args:
        split_name: Name of the split ('train', 'val', 'test')
        table: Table object
        batch_size: Batch size used
    """
    num_samples = len(table.df)
    num_batches = num_samples // batch_size + (1 if num_samples % batch_size else 0)
    is_shuffled = (split_name == 'train')
    
    print(f"\n {split_name.capitalize()} loader created:")
    print(f"   Samples: {num_samples:,}")
    print(f"   Batches: ~{num_batches}")
    print(f"   Shuffle: {is_shuffled}")
    
def backup_column(data_obj, node_type: str, col_name: str) -> torch.Tensor:
    """Create a backup of a specific column's tensor data."""
    
    if not hasattr(data_obj[node_type], 'tf'):
        raise ValueError(f"Node type {node_type} has no TensorFrame")
        
    tf = data_obj[node_type].tf
    stype_key = None

    for s, cols in tf.col_names_dict.items():
        if col_name in cols:
            stype_key = s
            break
            
    if stype_key is None:
        raise ValueError(f"Column {col_name} not found in TensorFrame of {node_type}")

    tensor = tf.feat_dict[stype_key]
    idx = tf.col_names_dict[stype_key].index(col_name)
    target_slice = tensor[:, idx]
    
    if hasattr(target_slice, 'values') and hasattr(target_slice, 'offset'):
        return target_slice.values.clone()
    else:
        return target_slice.clone()


def restore_column(data_obj, node_type: str, col_name: str, backup_tensor: torch.Tensor):
    """Restore a column from backup."""
    
    tf = data_obj[node_type].tf
    stype_key = None
    for s, cols in tf.col_names_dict.items():
        if col_name in cols:
            stype_key = s
            break

    if stype_key is None:
        raise ValueError(f"Column {col_name} not found in TensorFrame of {node_type}")

    tensor = tf.feat_dict[stype_key]
    idx = tf.col_names_dict[stype_key].index(col_name)
    target_slice = tensor[:, idx]
    
    if hasattr(target_slice, 'values') and hasattr(target_slice, 'offset'):
        target_slice.values.copy_(backup_tensor)
    else:
        target_slice.copy_(backup_tensor)


def load_test_subset(dataset_name, task_name, task, verbose=True):
    """Load the saved explanation subset for a task, if one exists.

    Some test sets (rel-arxiv) are far too large to explain in full, so every
    explainer (KernelSHAP, MonteCarlo, SingleColumnAblation, rdl) explains the SAME saved
    subset of rows instead — created once by Utils/create_test_subset.py.

    Returns (row_idx, meta): row_idx is a sorted numpy array of positions into
    the test table, or (None, None) when the task has no subset file — then the
    caller explains the full test set, exactly as before.

    Verifies that the saved entity ids and timestamps still match the rows at
    those positions, so a change in the underlying data (e.g. a refreshed
    Hugging Face cache) fails loudly here instead of silently explaining
    different rows than a previous run did.
    """
    subset_path = REPO_ROOT / dataset_name / task_name / "test_subset.csv"
    if not subset_path.exists():
        if verbose:
            print(f"✓ No test subset file for {dataset_name}/{task_name} — "
                  f"explaining the FULL test set")
        return None, None

    saved = pd.read_csv(subset_path)
    row_idx = saved["test_row_idx"].to_numpy()
    test_table = task.get_table("test")
    test_df = test_table.df
    entity_col = task.entity_col

    picked = test_df.iloc[row_idx]
    if not np.array_equal(picked[entity_col].to_numpy(), saved[entity_col].to_numpy()):
        raise ValueError(
            f"Test subset mismatch for {dataset_name}/{task_name}: entity ids at the "
            f"saved row positions differ from {subset_path.name}. The underlying test "
            f"table changed — rerun create_test_subset.py and rerun every explainer."
        )

    time_col = test_table.time_col
    if time_col is not None and time_col in saved.columns:
        if not (pd.to_datetime(picked[time_col]).to_numpy()
                == pd.to_datetime(saved[time_col]).to_numpy()).all():
            raise ValueError(
                f"Test subset mismatch for {dataset_name}/{task_name}: timestamps at the "
                f"saved row positions differ from {subset_path.name}. See the entity-id "
                f"error in this function for what to do."
            )

    meta_path = subset_path.with_name("test_subset_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if verbose:
        print(f"✓ Test subset: {subset_path} | {len(row_idx)} of {len(test_df)} rows "
              f"| checksum {str(meta.get('checksum_sha256', 'n/a'))[:16]}")
    return row_idx, meta


def subset_table(table, row_idx):
    """Return a copy of `table` keeping only the rows at `row_idx`, in order.

    Shallow-copies the Table so its schema (fkey/pkey/time columns) is kept and
    the original table object is left untouched.
    """
    subset = copy.copy(table)
    subset.df = table.df.iloc[row_idx].reset_index(drop=True)
    return subset
