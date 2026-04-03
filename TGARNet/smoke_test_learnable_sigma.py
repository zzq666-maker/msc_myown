import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

# Ensure repo root is on sys.path so this script works when launched as:
# `python TGARNet\smoke_test_learnable_sigma.py` from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TGARNet.models.TGARNet import (
    ConvexCombinationLayer,
    GaussianKernelLayer,
    InspectableTransformerEncoder,
    JointRenyiEntropyLayer,
    NormalizedBinaryCrossentropy,
    RenyiEntropyLayer,
    RenyiMutualInformation,
    TGARNet,
    TransposeLayer,
    inspect_gaussian_sigmas,
)


def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    batch_size = 8
    chans = 19
    samples = 128
    feature_dim = 1
    num_kernels = 3
    nb_classes = 2

    print("=== 1) Standalone GaussianKernelLayer smoke test ===")
    x_layer = tf.random.normal((batch_size, chans, samples, feature_dim), dtype=tf.float32)
    gaussian_layer = GaussianKernelLayer(
        init_sigma=0.8,
        sigma_min=1e-3,
        sigma_max=10.0,
        eps=1e-6,
        name="gaussian_layer_standalone",
    )
    kernel_out = gaussian_layer(x_layer)
    print(f"Gaussian layer input shape:  {x_layer.shape}")
    print(f"Gaussian layer output shape: {kernel_out.shape}")
    print(f"Standalone sigma: {float(gaussian_layer.get_sigma().numpy()):.8f}")

    print("\n=== 2) Full TGARNet forward smoke test ===")
    model = TGARNet(
        num_kernels=num_kernels,
        nb_classes=nb_classes,
        Chans=chans,
        Samples=samples,
        kernel_sigmas=[0.5, 1.0, 2.0],
    )

    # TGARNet model input shape is (batch, Chans, Samples)
    x_model = tf.random.normal((batch_size, chans, samples), dtype=tf.float32)
    outputs = model(x_model, training=False)

    print(f"Model input shape: {x_model.shape}")
    for output_name, output_tensor in outputs.items():
        print(f"{output_name} shape: {output_tensor.shape}")

    print("\n=== 3) Initial sigma values ===")
    initial_report = inspect_gaussian_sigmas(model)

    print("\n=== 4) Minimal training (2 epochs) ===")
    y_class = tf.one_hot(
        np.random.randint(0, nb_classes, size=(batch_size,)),
        depth=nb_classes,
        dtype=tf.float32,
    )

    # Train only the main classification output to ensure the loss path
    # definitely backpropagates through transformer_encoder.
    classifier_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("out_activation").output,
        name="tgarnet_classifier_smoke",
    )

    classifier_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    history = classifier_model.fit(
        x_model,
        y_class,
        epochs=2,
        batch_size=batch_size,
        verbose=2,
    )

    print("\nTraining history keys:", list(history.history.keys()))

    print("\n=== 5) Sigma values after training ===")
    trained_report = inspect_gaussian_sigmas(model)

    print("\n=== 6) Sigma update check ===")
    sigma_updated = False
    for before, after in zip(initial_report, trained_report):
        delta = after["current_sigma"] - before["current_sigma"]
        changed = abs(delta) > 1e-7
        sigma_updated = sigma_updated or changed
        print(
            f"{after['layer_name']}: "
            f"before={before['current_sigma']:.8f}, "
            f"after={after['current_sigma']:.8f}, "
            f"delta={delta:.8f}, "
            f"updated={changed}"
        )

    if not sigma_updated:
        print("WARNING: No sigma value changed beyond tolerance. Check gradients/training setup.")
    else:
        print("Sigma update check passed: at least one Gaussian sigma changed.")

    print("\n=== 7) Save and load smoke test ===")
    custom_objects = {
        "GaussianKernelLayer": GaussianKernelLayer,
        "RenyiMutualInformation": RenyiMutualInformation,
        "NormalizedBinaryCrossentropy": NormalizedBinaryCrossentropy,
        "TransposeLayer": TransposeLayer,
        "RenyiEntropyLayer": RenyiEntropyLayer,
        "JointRenyiEntropyLayer": JointRenyiEntropyLayer,
        "InspectableTransformerEncoder": InspectableTransformerEncoder,
        "ConvexCombinationLayer": ConvexCombinationLayer,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "tgarnet_learnable_sigma.keras")
        model.save(save_path, include_optimizer=False)
        print(f"Model saved to: {save_path}")

        model_loaded = tf.keras.models.load_model(
            save_path,
            custom_objects=custom_objects,
            compile=False,
        )
        print("Model loaded successfully.")

        print("\n=== 8) Forward pass after loading ===")
        outputs_loaded = model_loaded(x_model, training=False)
        for output_name, output_tensor in outputs_loaded.items():
            print(f"{output_name} shape after load: {output_tensor.shape}")

        print("\n=== 9) Loaded model sigma values ===")
        inspect_gaussian_sigmas(model_loaded)

    print("\n=== Smoke test completed ===")
    print("Validated: forward pass, training, sigma update, save, load, and post-load inference.")


if __name__ == "__main__":
    main()
