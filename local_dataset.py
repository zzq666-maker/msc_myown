from __future__ import annotations

import os
import random
import re
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_ROOT / "processed"
LOCAL_IEEE_ROOT = DATA_ROOT / "ieee"
FOLDS_PATH = PROCESSED_DIR / "folds.pkl"

DEFAULT_RAW_ROOT = DATA_ROOT / "raw"
RAW_ROOT = Path(os.environ.get("GRENET_RAWDATA_ROOT", str(DEFAULT_RAW_ROOT))).expanduser()

ADHD_PART_DIRS = [RAW_ROOT / "ADHD_part1", RAW_ROOT / "ADHD_part2"]
CONTROL_PART_DIRS = [RAW_ROOT / "Control_part1", RAW_ROOT / "Control_part2"]

ADHD_GROUP_DIR = LOCAL_IEEE_ROOT / "ADHD_group"
CONTROL_GROUP_DIR = LOCAL_IEEE_ROOT / "Control_group"


def natural_subject_key(subject_id: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"v(\d+)(p?)", subject_id)
    if match is None:
        return (10**9, 1, subject_id)
    number = int(match.group(1))
    suffix = 1 if match.group(2) == "p" else 0
    return (number, suffix, subject_id)


def expected_raw_layout() -> dict[str, Path]:
    return {
        "adhd_part1": ADHD_PART_DIRS[0],
        "adhd_part2": ADHD_PART_DIRS[1],
        "control_part1": CONTROL_PART_DIRS[0],
        "control_part2": CONTROL_PART_DIRS[1],
    }


def _format_expected_layout() -> str:
    return "\n".join(f"  - {name}: {path}" for name, path in expected_raw_layout().items())


def _subject_records_from_dir(directory: Path) -> list[tuple[str, Path]]:
    if not directory.exists():
        return []
    return sorted(
        ((path.stem, path) for path in directory.glob("*.mat")),
        key=lambda item: natural_subject_key(item[0]),
    )


def _subject_records_from_dirs(part_dirs: list[Path]) -> list[tuple[str, Path]]:
    records: dict[str, Path] = {}
    for directory in part_dirs:
        for subject_id, path in _subject_records_from_dir(directory):
            records[subject_id] = path
    return sorted(records.items(), key=lambda item: natural_subject_key(item[0]))


def _subject_records_from_group_dirs() -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    control_records = _subject_records_from_dir(CONTROL_GROUP_DIR)
    adhd_records = _subject_records_from_dir(ADHD_GROUP_DIR)
    return control_records, adhd_records


def _missing_raw_part_dirs() -> list[Path]:
    return [path for path in (*ADHD_PART_DIRS, *CONTROL_PART_DIRS) if not path.exists()]


def _dataset_setup_error() -> FileNotFoundError:
    return FileNotFoundError(
        "Could not find the EEG .mat files. Place the raw dataset under the repository at "
        f"'{DEFAULT_RAW_ROOT}' or set GRENET_RAWDATA_ROOT to the folder that contains "
        "ADHD_part1, ADHD_part2, Control_part1, and Control_part2.\n"
        "Expected layout:\n"
        f"{_format_expected_layout()}"
    )


def ensure_local_ieee_layout(copy_files: bool = True) -> dict[str, int]:
    LOCAL_IEEE_ROOT.mkdir(parents=True, exist_ok=True)
    ADHD_GROUP_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_GROUP_DIR.mkdir(parents=True, exist_ok=True)

    copied = {"adhd": 0, "control": 0}
    if not copy_files:
        return copied

    missing_raw_dirs = _missing_raw_part_dirs()
    if missing_raw_dirs:
        raise _dataset_setup_error()

    for subject_id, source in _subject_records_from_dirs(ADHD_PART_DIRS):
        target = ADHD_GROUP_DIR / f"{subject_id}.mat"
        if not target.exists():
            shutil.copy2(source, target)
            copied["adhd"] += 1

    for subject_id, source in _subject_records_from_dirs(CONTROL_PART_DIRS):
        target = CONTROL_GROUP_DIR / f"{subject_id}.mat"
        if not target.exists():
            shutil.copy2(source, target)
            copied["control"] += 1

    if not _subject_records_from_dir(ADHD_GROUP_DIR) or not _subject_records_from_dir(CONTROL_GROUP_DIR):
        raise _dataset_setup_error()

    return copied


def balanced_subject_records(seed: int = 42) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]], str | None]:
    control_records, adhd_records = _subject_records_from_group_dirs()
    if not control_records or not adhd_records:
        control_records = _subject_records_from_dirs(CONTROL_PART_DIRS)
        adhd_records = _subject_records_from_dirs(ADHD_PART_DIRS)

    if not control_records or not adhd_records:
        raise _dataset_setup_error()

    dropped_subject = None
    if len(adhd_records) > len(control_records):
        rng = random.Random(seed)
        dropped_subject = rng.choice([subject_id for subject_id, _ in adhd_records])
        adhd_records = [record for record in adhd_records if record[0] != dropped_subject]
    elif len(control_records) > len(adhd_records):
        rng = random.Random(seed)
        dropped_subject = rng.choice([subject_id for subject_id, _ in control_records])
        control_records = [record for record in control_records if record[0] != dropped_subject]

    return control_records, adhd_records, dropped_subject
