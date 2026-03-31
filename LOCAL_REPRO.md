# Local Reproduction Notes

This repository was originally developed in Kaggle notebooks, but the local workflow is now clone-friendly for Windows hosts.

## Recommended workflow

1. Clone the repository onto the remote Windows machine.
2. Create and activate `.venv312`.
3. Install the dependencies from `gmrrnet_adhd\requirements.txt` plus the scientific packages your notebooks use.
4. Place the raw dataset in one of these ways:
   - Preferred: copy the four folders into `data\raw\ADHD_part1`, `data\raw\ADHD_part2`, `data\raw\Control_part1`, and `data\raw\Control_part2`.
   - Alternative: keep the raw dataset elsewhere and set `GRENET_RAWDATA_ROOT` to the parent folder that contains those four directories.
5. Run `powershell -ExecutionPolicy Bypass -File .\scripts\prepare_windows_clone.ps1`.
6. Open Jupyter with the `GRENet (py312)` kernel.

## What the preparation script does

- Copies the raw `.mat` files into a combined IEEE-style layout under `data\ieee`.
- Generates a deterministic `data\processed\folds.pkl` using subject-wise 5-fold stratification with seed `42`.
- Synchronizes `X_preprocessed.pkl` and `y.pkl` into `data\processed` when those files are present in `gmrrnet_adhd\L24SO\results`.
- Patches the tracked notebooks so they import the repository with dynamic relative paths instead of `/kaggle/working/Msc_thesis` and `/kaggle/input/...`.

## Important

- The paper states that one ADHD subject was randomly removed to balance the dataset to 120 subjects. Because the original removed subject is not documented in the repository, the local setup uses a deterministic seed (`42`) and records the dropped subject in `data\processed\folds_metadata.json`.
- This makes the workflow reproducible locally, but it may not exactly match the authors' unpublished random draw.
- The visualization notebooks under `TGARNet\results` still reference some extra experiment artifacts that are not part of the standard training setup. Those may need manual file placement if you want to run them too.
