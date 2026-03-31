# Msc_thesis

This repository can be cloned onto a Windows host and prepared locally without recreating the original Kaggle folder layout.

## Windows clone workflow

1. Clone the repository.
2. Create a Python 3.12 virtual environment at `.venv312` and install `gmrrnet_adhd\requirements.txt` plus the scientific packages used by the notebooks.
3. Put the raw EEG folders under `data\raw\ADHD_part1`, `data\raw\ADHD_part2`, `data\raw\Control_part1`, and `data\raw\Control_part2`.
   You can also keep the raw data somewhere else and set `GRENET_RAWDATA_ROOT` to that folder before running the setup script.
4. Run `powershell -ExecutionPolicy Bypass -File .\scripts\prepare_windows_clone.ps1`.
5. Start Jupyter and open the notebooks with the prepared environment.

The preparation script copies the raw `.mat` files into `data\ieee`, generates `data\processed\folds.pkl`, and patches the notebooks so they use repository-relative paths instead of `/kaggle/...` paths.
