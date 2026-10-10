"""
Create a fixed, reusable test-set subset for explanation runs.

rel-arxiv's test set is too large to explain in full, so every explainer
(KernelSHAP, MonteCarlo, SingleColumnAblation, rdl) runs on the SAME saved subset instead.
Run this once per task; the resulting CSV is the single source of truth for
which rows get explained, and its checksum lets every run prove it used it.

Usage:
    python create_test_subset.py --dataset rel-arxiv --task author-publication --size 1000
    python create_test_subset.py --dataset rel-arxiv --task paper-citation --size 1000
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import relbench
from relbench.base import TaskType

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Save a fixed test-set subset for explanation runs")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g. rel-arxiv)')
    parser.add_argument('--task', type=str, required=True, help='Task name (e.g. author-publication)')
    parser.add_argument('--size', type=int, default=1000, help='Number of test rows to keep (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for the sample (default: 42)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite an existing subset file (DANGEROUS: invalidates runs that used it)')
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = REPO_ROOT / args.dataset / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    subset_path = out_dir / "test_subset.csv"
    meta_path = out_dir / "test_subset_meta.json"

    if subset_path.exists() and not args.force:
        raise SystemExit(
            f"{subset_path} already exists. Explanation runs may already rely on it — "
            f"re-creating it would silently change which rows they explain. "
            f"Pass --force only if you intend to replace it and rerun every explainer."
        )

    # Dataset/task are always loaded fresh (never unpickled) — same reasoning as
    # load_research_components() in Utils.py.
    print(f"Loading {args.dataset} / {args.task} ...")
    dataset = relbench.load_dataset(args.dataset)
    task = dataset.load_task(args.task)

    # get_table("test") hides the target column (RelBench masks labels on the
    # test split), so read the unmasked table too — same rows, same order.
    test_table = task.get_table("test")
    labeled_df = task.get_table("test", mask_input_cols=False).df
    df = test_table.df

    n_total = len(df)
    n_keep = min(args.size, n_total)
    if n_keep < args.size:
        print(f"  Test set has only {n_total} rows — keeping all of them.")

    # Sorted, so the subset keeps the test table's own row order. Every explainer
    # then walks the same rows in the same order.
    rng = np.random.default_rng(args.seed)
    row_idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))

    entity_col = task.entity_col
    time_col = test_table.time_col

    subset = df.iloc[row_idx][[entity_col] + ([time_col] if time_col else [])].copy()
    subset.insert(0, "test_row_idx", row_idx)
    subset.to_csv(subset_path, index=False)

    checksum = hashlib.sha256(subset_path.read_bytes()).hexdigest()

    # Class balance — only meaningful for binary classification
    class_stats = None
    is_binary = getattr(task, "task_type", None) == TaskType.BINARY_CLASSIFICATION
    if is_binary and task.target_col in labeled_df.columns:
        subset_y = labeled_df.iloc[row_idx][task.target_col]
        full_y = labeled_df[task.target_col]
        n_pos, n_neg = int((subset_y == 1).sum()), int((subset_y == 0).sum())
        class_stats = {
            "n_positive": n_pos,
            "n_negative": n_neg,
            "positive_rate_subset": round(n_pos / len(subset_y), 4),
            "positive_rate_full_test": round(float((full_y == 1).mean()), 4),
        }
        print(f"\n  Class balance (target = '{task.target_col}'):")
        print(f"    Positive: {n_pos:,} ({100 * n_pos / len(subset_y):.2f}%)")
        print(f"    Negative: {n_neg:,} ({100 * n_neg / len(subset_y):.2f}%)")
        print(f"    Full test set positive rate: {100 * (full_y == 1).mean():.2f}%  "
              f"(subset should be close to this)")
        if subset_y.isna().any():
            print(f"    WARNING: {int(subset_y.isna().sum())} rows have no label")
    elif is_binary:
        print(f"\n  WARNING: target column '{task.target_col}' not found — "
              f"skipping class balance")
    else:
        print(f"\n  Regression task — no class balance to report "
              f"(target '{task.target_col}': mean {labeled_df[task.target_col].mean():.4f})")

    meta = {
        "dataset": args.dataset,
        "task": args.task,
        "n_subset": int(n_keep),
        "n_test_total": int(n_total),
        "seed": args.seed,
        "selection": "random_without_replacement_sorted",
        "entity_col": entity_col,
        "time_col": time_col,
        "class_balance": class_stats,
        "checksum_sha256": checksum,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n✓ Saved {n_keep} of {n_total} test rows")
    print(f"  Subset:   {subset_path}")
    print(f"  Metadata: {meta_path}")
    print(f"  Checksum: {checksum[:16]}...")


if __name__ == "__main__":
    main()
