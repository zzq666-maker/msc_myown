from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_dataset import (
    ADHD_GROUP_DIR,
    CONTROL_GROUP_DIR,
    FOLDS_PATH,
    PROCESSED_DIR,
    balanced_subject_records,
    ensure_local_ieee_layout,
)


def build_subject_folds(seed: int = 42, n_splits: int = 5):
    control_records, adhd_records, dropped_subject = balanced_subject_records(seed=seed)

    subject_ids = [subject_id for subject_id, _ in control_records] + [subject_id for subject_id, _ in adhd_records]
    labels = [0] * len(control_records) + [1] * len(adhd_records)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[tuple[list[str], list[str]]] = []
    metadata = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(subject_ids, labels), start=1):
        train_subjects = [subject_ids[i] for i in train_idx]
        test_subjects = [subject_ids[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        metadata.append(
            {
                "fold": fold_idx,
                "train_subjects": len(train_subjects),
                "test_subjects": len(test_subjects),
                "test_control": sum(label == 0 for label in test_labels),
                "test_adhd": sum(label == 1 for label in test_labels),
            }
        )
        folds.append((train_subjects, test_subjects))

    return folds, metadata, dropped_subject, len(control_records), len(adhd_records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic subject folds for local GRENet reproduction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for balancing and fold generation.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of subject-wise stratified folds.")
    parser.add_argument(
        "--setup-layout",
        action="store_true",
        help="Copy raw .mat files into a combined local IEEE-style directory before generating folds.",
    )
    args = parser.parse_args()

    copied = {"adhd": 0, "control": 0}
    if args.setup_layout:
        copied = ensure_local_ieee_layout(copy_files=True)

    folds, metadata, dropped_subject, control_count, adhd_count = build_subject_folds(
        seed=args.seed,
        n_splits=args.n_splits,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with FOLDS_PATH.open("wb") as handle:
        pickle.dump(folds, handle)

    metadata_path = PROCESSED_DIR / "folds_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n_splits": args.n_splits,
                "control_count": control_count,
                "adhd_count": adhd_count,
                "dropped_subject_for_balance": dropped_subject,
                "copied_raw_files": copied,
                "local_adhd_group_dir": str(ADHD_GROUP_DIR),
                "local_control_group_dir": str(CONTROL_GROUP_DIR),
                "folds": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved folds to: {FOLDS_PATH}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Balanced subjects -> control: {control_count}, adhd: {adhd_count}, dropped: {dropped_subject}")


if __name__ == "__main__":
    main()
