from __future__ import annotations

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "gmrrnet_adhd" / "L24SO" / "results"
TARGET_DIR = PROJECT_ROOT / "data" / "processed"
IGNORED_PARTS = {".ipynb_checkpoints", ".venv", ".venv312", "__pycache__"}
IGNORED_PREFIXES = [PROJECT_ROOT / "TGARNet" / "SGKF" / "Msc_thesis"]

BOOTSTRAP_CELL_SOURCE = [
    "from pathlib import Path\n",
    "import sys\n",
    "\n",
    "for candidate in (Path.cwd(), *Path.cwd().parents):\n",
    "    if (candidate / 'local_dataset.py').exists():\n",
    "        REPO_ROOT = candidate\n",
    "        break\n",
    "else:\n",
    "    raise FileNotFoundError('Could not locate the repository root from the current notebook working directory.')\n",
    "\n",
    "if str(REPO_ROOT) not in sys.path:\n",
    "    sys.path.insert(0, str(REPO_ROOT))\n",
    "\n",
    "from local_dataset import ADHD_GROUP_DIR, CONTROL_GROUP_DIR, FOLDS_PATH, PROCESSED_DIR\n",
]

PREPROCESSED_CELL_SOURCE = [
    "import pickle\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "\n",
    "from local_dataset import PROCESSED_DIR\n",
    "\n",
    "with open(PROCESSED_DIR / 'X_preprocessed.pkl', 'rb') as f:\n",
    "    X = pickle.load(f)\n",
    "\n",
    "with open(PROCESSED_DIR / 'y.pkl', 'rb') as f:\n",
    "    y = pickle.load(f)\n",
    "\n",
    "_, _, sbjs = get_segmented_data()\n",
    "\n",
    "X.shape, y.shape, len(sbjs)\n",
]

FOLDS_OLD = 'with open("/kaggle/input/ieee-tdah-control-database/folds.pkl", "rb") as f:'
FOLDS_NEW = 'with open(FOLDS_PATH, "rb") as f:'
ADHD_PATH_OLD = "ruta_carpeta_TDAH = '/kaggle/input/ieee-tdah-control-database/ieee/ADHD_group'  \n"
ADHD_PATH_NEW = "ruta_carpeta_TDAH = str(ADHD_GROUP_DIR)\n"
CONTROL_PATH_OLD = "ruta_carpeta_control = '/kaggle/input/ieee-tdah-control-database/ieee/Control_group'  \n"
CONTROL_PATH_NEW = "ruta_carpeta_control = str(CONTROL_GROUP_DIR)\n"
PREPROCESSED_URL = "raw.githubusercontent.com/dannasalazar11/Msc_thesis/main/gmrrnet_adhd/L24SO/results/X_preprocessed.pkl"
BOOTSTRAP_MARKER = "from local_dataset import ADHD_GROUP_DIR, CONTROL_GROUP_DIR, FOLDS_PATH, PROCESSED_DIR"


def should_patch(path: Path) -> bool:
    if any(part in IGNORED_PARTS for part in path.parts):
        return False
    return not any(path.is_relative_to(prefix) for prefix in IGNORED_PREFIXES)


def sync_preprocessed() -> list[str]:
    copied_names: list[str] = []
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("X_preprocessed.pkl", "y.pkl"):
        source = SOURCE_DIR / name
        if source.exists():
            shutil.copy2(source, TARGET_DIR / name)
            copied_names.append(name)
    return copied_names


def patch_notebook(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source_lines = cell.get("source", [])
        source = "".join(source_lines)

        if "sys.path.append('/kaggle/working/Msc_thesis')" in source or BOOTSTRAP_MARKER in source:
            if source_lines != BOOTSTRAP_CELL_SOURCE:
                cell["source"] = BOOTSTRAP_CELL_SOURCE.copy()
                changed = True
            continue

        if PREPROCESSED_URL in source:
            if source_lines != PREPROCESSED_CELL_SOURCE:
                cell["source"] = PREPROCESSED_CELL_SOURCE.copy()
                changed = True
            continue

        updated = source.replace(FOLDS_OLD, FOLDS_NEW)
        updated = updated.replace(ADHD_PATH_OLD, ADHD_PATH_NEW)
        updated = updated.replace(CONTROL_PATH_OLD, CONTROL_PATH_NEW)

        if updated != source:
            cell["source"] = updated.splitlines(keepends=True)
            changed = True

    if changed:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return changed


def iter_notebooks() -> list[Path]:
    return sorted(path for path in PROJECT_ROOT.rglob("*.ipynb") if should_patch(path))


def main() -> None:
    copied_names = sync_preprocessed()
    changed_paths = [str(path.relative_to(PROJECT_ROOT)) for path in iter_notebooks() if patch_notebook(path)]

    if copied_names:
        print("Synced preprocessed files to", TARGET_DIR)
        for name in copied_names:
            print(f"  - {name}")
    else:
        print("Preprocessed files were not found under", SOURCE_DIR)

    if changed_paths:
        print("Patched notebooks:")
        for item in changed_paths:
            print(f"  - {item}")
    else:
        print("No notebook changes were necessary.")


if __name__ == "__main__":
    main()
