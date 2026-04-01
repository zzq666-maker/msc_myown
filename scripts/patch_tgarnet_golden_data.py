from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(r"C:\GRENet\Msc_thesis\TGARNet\SGKF\TGARNet.ipynb")

CELL_4_SOURCE = [
    "import numpy as np\n",
    "\n",
    "# ===== 原始数据加载（弃用）=====\n",
    "# X, y, sbjs = get_segmented_data()\n",
    "# X.shape, y.shape, len(sbjs)\n",
    "\n",
    "# ===== Golden 数据注入 =====\n",
    "X = np.load(\"X_golden.npy\").astype(np.float32)           # (10023, 19, 256)\n",
    "y = np.load(\"Y_golden.npy\").astype(np.float32)           # (10023, 2), one-hot\n",
    "groups = np.load(\"Groups_golden.npy\")                    # (10023,)\n",
    "sample_weights = np.load(\"TBR_weights_golden.npy\").astype(np.float32)\n",
    "\n",
    "# 为了无缝接入后续 train_L24O_cv(...)，继续沿用 notebook 原变量名 sbjs\n",
    "sbjs = groups\n",
    "\n",
    "print(\"X.shape =\", X.shape)\n",
    "print(\"y.shape =\", y.shape)\n",
    "print(\"groups.shape =\", groups.shape)\n",
    "print(\"sample_weights.shape =\", sample_weights.shape)\n",
    "\n",
    "assert X.ndim == 3 and X.shape[1:] == (19, 256), f\"Unexpected X shape: {X.shape}\"\n",
    "assert y.ndim == 2 and y.shape[1] == 2, f\"Unexpected y shape: {y.shape}\"\n",
    "assert len(X) == len(y) == len(groups) == len(sample_weights), \"X/y/groups/sample_weights 长度不一致\"\n",
]

CELL_6_SOURCE = [
    "from sklearn.model_selection import StratifiedGroupKFold\n",
    "import numpy as np\n",
    "\n",
    "# ===== 原始 folds.pkl（弃用）=====\n",
    "# import os\n",
    "# import pickle\n",
    "# with open(\"/kaggle/input/ieee-tdah-control-database/folds.pkl\", \"rb\") as f:\n",
    "#     folds = pickle.load(f)\n",
    "\n",
    "# ===== 基于 golden 数据动态生成 SGKF folds =====\n",
    "y_classes = np.argmax(y, axis=1)\n",
    "\n",
    "sgkf = StratifiedGroupKFold(\n",
    "    n_splits=5,\n",
    "    shuffle=True,\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "folds = []\n",
    "for fold_id, (train_idx, test_idx) in enumerate(sgkf.split(X, y_classes, groups=groups), start=1):\n",
    "    train_subjects = np.unique(groups[train_idx]).tolist()\n",
    "    test_subjects = np.unique(groups[test_idx]).tolist()\n",
    "\n",
    "    folds.append((train_subjects, test_subjects))\n",
    "\n",
    "    print(f\"Fold {fold_id}\")\n",
    "    print(f\"  train samples: {len(train_idx)}\")\n",
    "    print(f\"  test samples : {len(test_idx)}\")\n",
    "    print(f\"  train groups : {len(train_subjects)}\")\n",
    "    print(f\"  test groups  : {len(test_subjects)}\")\n",
    "    print(f\"  test class balance: {np.bincount(y_classes[test_idx], minlength=2)}\")\n",
    "\n",
    "# 额外一致性检查：folds 里存的是 subject/group id，能直接喂给后面的 train_L24O_cv(...)\n",
    "print(f\"Total folds: {len(folds)}\")\n",
    "print(\"Example fold subject split:\")\n",
    "print(\"  train_subjects[:5] =\", folds[0][0][:5])\n",
    "print(\"  test_subjects[:5]  =\", folds[0][1][:5])\n",
]

CELL_15_SOURCE = [
    "from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError\n",
    "from tensorflow.keras.utils import get_custom_objects\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "\n",
    "get_custom_objects().update({\n",
    "    \"RenyiMutualInformation\": RenyiMutualInformation\n",
    "})\n",
    "\n",
    "model_name = 'GMRRNet'\n",
    "\n",
    "# 关键改动：显式把 Samples 改成 256\n",
    "# 其余核心超参数保持 notebook 原设定不变\n",
    "model_args = {\n",
    "    'num_kernels': 3,\n",
    "    'nb_classes': 2,\n",
    "    'Chans': 19,\n",
    "    'Samples': 256,\n",
    "    'norm_rate': 0.1,\n",
    "    'alpha': 2,\n",
    "    'num_heads': 2,\n",
    "    'intermediate_dim': 64\n",
    "}\n",
    "\n",
    "compile_args = {\n",
    "    'optimizer': Adam(learning_rate=1e-3),\n",
    "    'loss': {\n",
    "        'out_activation': NormalizedBinaryCrossentropy(name=\"NormalizedBinaryCrossentropy\"),\n",
    "        'entropies_out': RenyiMutualInformation(C=19.0, name='MutualInfo'),\n",
    "        'kernel_weights_out': 'mean_squared_error'\n",
    "    },\n",
    "    'loss_weights': {\n",
    "        'out_activation': 0.5,  # Inicial\n",
    "        'entropies_out': 0.5,\n",
    "        'kernel_weights_out': 0.0\n",
    "    },\n",
    "    'metrics': {'out_activation': ['binary_accuracy', tf.keras.metrics.AUC(name=\"AUC\")]}\n",
    "    }\n",
    "\n",
    "class DynamicSchedule(tf.keras.callbacks.Callback):\n",
    "    def __init__(self, total_epochs, optimizer, eta_0=1e-3, alpha=10, beta=0.75, delta=10):\n",
    "        super().__init__()\n",
    "        self.total_epochs = total_epochs\n",
    "        self.eta_0 = eta_0\n",
    "        self.alpha = alpha\n",
    "        self.beta = beta\n",
    "        self.delta = delta\n",
    "        self.optimizer = optimizer\n",
    "        self.lambda_val = 0.0\n",
    "\n",
    "    def get_eta(self, epoch):\n",
    "        p = epoch / self.total_epochs\n",
    "        return self.eta_0 * (1 + self.alpha * p) ** (-self.beta)\n",
    "\n",
    "    def get_lambda(self, epoch):\n",
    "        p = epoch / self.total_epochs\n",
    "        return 2 * (1 - np.exp(-self.delta * p)) / (1 + np.exp(-self.delta * p))\n",
    "\n",
    "    def on_epoch_begin(self, epoch, logs=None):\n",
    "        new_lr = self.get_eta(epoch)\n",
    "        self.lambda_val = self.get_lambda(epoch)\n",
    "\n",
    "        # ✅ Ajustar learning rate correctamente\n",
    "        if hasattr(self.optimizer.learning_rate, \"assign\"):\n",
    "            self.optimizer.learning_rate.assign(new_lr)\n",
    "        else:\n",
    "            tf.keras.backend.set_value(self.optimizer.learning_rate, new_lr)\n",
    "\n",
    "        # ✅ Actualizar dinámicamente los pesos de pérdida SIN recompilar\n",
    "        if hasattr(self.model, \"loss_weights\") and isinstance(self.model.loss_weights, dict):\n",
    "            self.model.loss_weights[\"out_activation\"] = 1.0\n",
    "            self.model.loss_weights[\"entropies_out\"] = self.lambda_val\n",
    "\n",
    "        print(f\"[Epoch {epoch+1}] LR={float(new_lr):.6f} | λ={self.lambda_val:.3f}\")\n",
]


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    notebook["cells"][4]["source"] = CELL_4_SOURCE
    notebook["cells"][4]["outputs"] = []
    notebook["cells"][4]["execution_count"] = None

    notebook["cells"][6]["source"] = CELL_6_SOURCE
    notebook["cells"][6]["outputs"] = []
    notebook["cells"][6]["execution_count"] = None

    notebook["cells"][15]["source"] = CELL_15_SOURCE
    notebook["cells"][15]["outputs"] = []
    notebook["cells"][15]["execution_count"] = None

    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
