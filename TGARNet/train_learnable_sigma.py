import argparse
import json
import os
import pickle
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TGARNet.models.TGARNet import (  # noqa: E402
    NormalizedBinaryCrossentropy,
    RenyiMutualInformation,
    TGARNet,
    inspect_gaussian_sigmas,
)
from TGARNet.utils import get_segmented_data  # noqa: E402


class DynamicSchedule(Callback):
    def __init__(self, total_epochs, eta_0=1e-3, alpha=10.0, beta=0.75, delta=10.0):
        super().__init__()
        self.total_epochs = total_epochs
        self.eta_0 = eta_0
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.lambda_val = 0.0

    def get_eta(self, epoch):
        p = epoch / max(float(self.total_epochs), 1.0)
        return self.eta_0 * (1.0 + self.alpha * p) ** (-self.beta)

    def get_lambda(self, epoch):
        p = epoch / max(float(self.total_epochs), 1.0)
        return 2.0 * (1.0 - np.exp(-self.delta * p)) / (1.0 + np.exp(-self.delta * p))

    def on_epoch_begin(self, epoch, logs=None):
        new_lr = self.get_eta(epoch)
        self.lambda_val = self.get_lambda(epoch)

        optimizer = self.model.optimizer
        if hasattr(optimizer.learning_rate, "assign"):
            optimizer.learning_rate.assign(new_lr)
        else:
            tf.keras.backend.set_value(optimizer.learning_rate, new_lr)

        if hasattr(self.model, "loss_weights") and isinstance(self.model.loss_weights, dict):
            self.model.loss_weights["out_activation"] = 1.0
            self.model.loss_weights["entropies_out"] = self.lambda_val

        print(f"[Epoch {epoch + 1}] LR={float(new_lr):.6f} | lambda={self.lambda_val:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train learnable-sigma TGARNet using the cleaned TGARNet.py implementation.")
    parser.add_argument("--folds-path", default=str(REPO_ROOT / "data" / "processed" / "folds.pkl"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "TGARNet" / "results" / "learnable_sigma_runs"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta", type=float, default=20.0)
    parser.add_argument("--num-kernels", type=int, default=3)
    parser.add_argument("--kernel-sigmas", type=float, nargs="+", default=[5.0, 2.5, 1.25])
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--intermediate-dim", type=int, default=64)
    parser.add_argument("--norm-rate", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--disable-dynamic-schedule", action="store_true")
    return parser.parse_args()


def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def load_folds(folds_path):
    with open(folds_path, "rb") as f:
        folds = pickle.load(f)
    return folds


def make_compile_args(chans, learning_rate):
    return {
        "optimizer": Adam(learning_rate=learning_rate),
        "loss": {
            "out_activation": NormalizedBinaryCrossentropy(name="NormalizedBinaryCrossentropy"),
            "entropies_out": RenyiMutualInformation(C=float(chans), name="MutualInfo"),
            "kernel_weights_out": "mean_squared_error",
        },
        "loss_weights": {
            "out_activation": 0.5,
            "entropies_out": 0.5,
            "kernel_weights_out": 0.0,
        },
        "metrics": {
            "out_activation": [
                "binary_accuracy",
                tf.keras.metrics.AUC(name="AUC"),
            ]
        },
    }


def to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def train_cv(model_builder, X, y, sbjs, model_args, compile_args, folds, args):
    all_fold_metrics = []
    sigma_reports = {}
    history_by_fold = {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_folds = folds[: args.max_folds] if args.max_folds else folds

    for fold_idx, (train_subjects, test_subjects) in enumerate(selected_folds):
        print(f"\n{'-' * 70}")
        print(f"Fold {fold_idx + 1}/{len(selected_folds)} | Test subjects: {test_subjects}")
        print(f"{'-' * 70}")

        train_idx = [i for i, sbj in enumerate(sbjs) if sbj in train_subjects]
        test_idx = [i for i, sbj in enumerate(sbjs) if sbj in test_subjects]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        sbjs_test = [sbjs[i] for i in test_idx]

        tf.keras.backend.clear_session()
        set_global_seed(args.seed + fold_idx)

        model = model_builder(**model_args)
        compile_args_local = deepcopy(compile_args)
        compile_args_local["optimizer"] = Adam(learning_rate=args.learning_rate)
        model.compile(**compile_args_local)

        print("Initial sigma values:")
        sigma_reports[f"fold_{fold_idx + 1}_before"] = inspect_gaussian_sigmas(model)

        num_kernels = model_args["num_kernels"]
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                min_delta=args.min_delta,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(5, args.patience // 2),
                min_lr=1e-6,
                verbose=1,
            ),
        ]
        if not args.disable_dynamic_schedule:
            callbacks.append(
                DynamicSchedule(
                    total_epochs=args.epochs,
                    eta_0=args.learning_rate,
                    delta=args.delta,
                )
            )

        history = model.fit(
            X_train,
            {
                "out_activation": y_train,
                "entropies_out": np.zeros((len(y_train), 4), dtype=np.float32),
                "kernel_weights_out": np.zeros((len(y_train), num_kernels), dtype=np.float32),
            },
            validation_data=(
                X_test,
                {
                    "out_activation": y_test,
                    "entropies_out": np.zeros((len(y_test), 4), dtype=np.float32),
                    "kernel_weights_out": np.zeros((len(y_test), num_kernels), dtype=np.float32),
                },
            ),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=2,
        )

        history_by_fold[f"fold_{fold_idx + 1}"] = {
            key: [to_builtin(v) for v in values]
            for key, values in history.history.items()
        }

        print("Trained sigma values:")
        sigma_reports[f"fold_{fold_idx + 1}_after"] = inspect_gaussian_sigmas(model)

        alpha_weights = model.get_layer("convex_combination").get_weights()[0]
        convex_weights = tf.nn.softmax(alpha_weights).numpy().tolist()
        print("Convex kernel weights (softmax):", convex_weights)

        preds = model.predict(X_test, verbose=0)
        y_pred_probs = preds["out_activation"]
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)

        fold_metrics = {
            "fold": fold_idx + 1,
            "test_subjects": list(test_subjects),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(y_true, y_pred)),
            "auc": float(roc_auc_score(y_true, y_pred_probs[:, 1])),
            "convex_kernel_weights": convex_weights,
        }
        all_fold_metrics.append(fold_metrics)
        print("Fold metrics:", fold_metrics)

        per_subject = {}
        for yt, yp, sbj in zip(y_true, y_pred, sbjs_test):
            per_subject.setdefault(sbj, []).append(int(yt == yp))
        fold_metrics["subject_accuracy"] = {
            sbj: float(np.mean(vals)) for sbj, vals in per_subject.items()
        }

        if args.save_models:
            model_path = output_dir / f"tgarnet_learnable_sigma_fold_{fold_idx + 1}.keras"
            model.save(model_path, include_optimizer=False)
            print(f"Saved fold model to {model_path}")

    summary = {}
    if all_fold_metrics:
        metric_keys = ["accuracy", "recall", "precision", "kappa", "auc"]
        for key in metric_keys:
            vals = [fold[key] for fold in all_fold_metrics]
            summary[f"mean_{key}"] = float(np.mean(vals))
            summary[f"std_{key}"] = float(np.std(vals))

    return all_fold_metrics, summary, history_by_fold, sigma_reports


def main():
    args = parse_args()
    set_global_seed(args.seed)

    if len(args.kernel_sigmas) != args.num_kernels:
        raise ValueError(
            f"--kernel-sigmas length ({len(args.kernel_sigmas)}) must match --num-kernels ({args.num_kernels})"
        )

    folds_path = Path(args.folds_path)
    if not folds_path.exists():
        raise FileNotFoundError(f"Folds file not found: {folds_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading segmented EEG data...")
    X, y, sbjs = get_segmented_data()
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Number of subject labels: {len(sbjs)}")

    folds = load_folds(folds_path)
    print(f"Loaded {len(folds)} folds from {folds_path}")

    model_args = {
        "num_kernels": args.num_kernels,
        "nb_classes": y.shape[1],
        "Chans": X.shape[1],
        "Samples": X.shape[2],
        "norm_rate": args.norm_rate,
        "alpha": args.alpha,
        "num_heads": args.num_heads,
        "intermediate_dim": args.intermediate_dim,
        "kernel_sigmas": args.kernel_sigmas,
    }
    compile_args = make_compile_args(chans=X.shape[1], learning_rate=args.learning_rate)

    config_payload = {
        "model_args": model_args,
        "training_args": vars(args),
    }
    save_json(output_dir / "run_config.json", config_payload)

    fold_metrics, summary_metrics, history_by_fold, sigma_reports = train_cv(
        TGARNet,
        X,
        y,
        sbjs,
        model_args,
        compile_args,
        folds,
        args,
    )

    save_json(output_dir / "fold_metrics.json", fold_metrics)
    save_json(output_dir / "summary_metrics.json", summary_metrics)
    save_json(output_dir / "sigma_reports.json", sigma_reports)
    save_json(output_dir / "history.json", history_by_fold)

    print("\n=== Summary metrics ===")
    for key, value in summary_metrics.items():
        print(f"{key}: {value:.6f}")

    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
