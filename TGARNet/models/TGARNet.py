import math
from collections import defaultdict

import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras import activations, initializers, layers, ops
from keras_nlp.layers import TransformerEncoder
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Layer,
    LayerNormalization,
    Reshape,
)
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model


@tf.keras.utils.register_keras_serializable()
class GaussianKernelLayer(Layer):
    def __init__(
        self,
        init_sigma=1.0,
        sigma_min=1e-3,
        sigma_max=10.0,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.init_sigma = float(init_sigma)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.eps = float(eps)
        self.raw_sigma = None

    def build(self, input_shape):
        clipped_init_sigma = min(max(self.init_sigma, self.sigma_min), self.sigma_max)
        safe_init_sigma = max(clipped_init_sigma - self.eps, self.eps)
        raw_sigma_init = float(np.log(np.expm1(safe_init_sigma)))

        self.raw_sigma = self.add_weight(
            name="raw_sigma",
            shape=(),
            initializer=tf.keras.initializers.Constant(raw_sigma_init),
            trainable=True,
        )
        super().build(input_shape)

    def get_sigma(self):
        sigma = tf.nn.softplus(self.raw_sigma) + self.eps
        sigma = tf.clip_by_value(sigma, self.sigma_min, self.sigma_max)
        return sigma

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)

        # x shape: (N, C, T, F)
        input_shape = tf.shape(x)
        n = input_shape[0]
        c = input_shape[1]
        t = input_shape[2]
        f = input_shape[3]

        # Reshape to (N*F, C, T)
        x = tf.transpose(x, perm=(0, 3, 1, 2))  # (N, F, C, T)
        x = tf.reshape(x, (n * f, c, t))  # (N*F, C, T)

        # Pairwise squared Euclidean distance:
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2ab
        x_sq = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)  # (N*F, C, 1)
        x_xt = tf.matmul(x, x, transpose_b=True)  # (N*F, C, C)
        pairwise_distances_squared = x_sq - 2.0 * x_xt + tf.transpose(x_sq, perm=(0, 2, 1))
        pairwise_distances_squared = tf.maximum(pairwise_distances_squared, 0.0)

        # Restore to (N, C, C, F)
        pairwise_distances_squared = tf.reshape(pairwise_distances_squared, (n, f, c, c))
        pairwise_distances_squared = tf.transpose(pairwise_distances_squared, perm=(0, 2, 3, 1))

        sigma = tf.cast(self.get_sigma(), tf.float32)
        sigma_sq = tf.maximum(tf.square(sigma), tf.constant(self.eps, dtype=tf.float32))
        denom = 2.0 * sigma_sq + tf.constant(self.eps, dtype=tf.float32)

        gaussian_kernel = tf.exp(-pairwise_distances_squared / denom)
        return gaussian_kernel

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_sigma": self.init_sigma,
                "sigma_min": self.sigma_min,
                "sigma_max": self.sigma_max,
                "eps": self.eps,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def renyi_entropy(K, alpha=2):
    """
    input: K tensor, (N,F,C,C)
    output: NxF
    """
    C = K.shape[1]

    diag = tf.expand_dims(tf.linalg.diag_part(K), -1)
    denominator = tf.math.sqrt(tf.linalg.matmul(diag, diag, transpose_b=True))
    X = (1 / C) * tf.math.divide(K, denominator)

    if alpha == 2:
        X_matmul = tf.linalg.matmul(X, X)
        return -tf.math.log(tf.linalg.trace(X_matmul))
    else:
        e, _ = tf.linalg.eigh(X)
        return tf.math.log(tf.reduce_sum(tf.math.real(tf.math.pow(e, alpha)), axis=-1)) / (1 - alpha)


def joint_renyi_entropy(K, alpha):
    """
    input: K, (N,F,C,C)
    output: Nx1
    """
    C = K.shape[-1]
    product = tf.reduce_prod(K, axis=1)  # (N,C,C)

    trace = tf.linalg.trace(product)
    trace = tf.expand_dims(tf.expand_dims(trace, axis=-1), axis=-1)
    trace = tf.tile(trace, [1, C, C])

    argument = product / trace
    argument = tf.expand_dims(argument, axis=1)  # es necesario porque renyi_entropy recibe 4 dimensiones (1,C,C)
    joint_entropy = renyi_entropy(argument, alpha=alpha)

    return joint_entropy


@tf.keras.utils.register_keras_serializable()
class RenyiMutualInformation(Loss):
    def __init__(self, C, **kwargs):
        self.C = C
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        F = y_pred.shape[1] - 1
        entropy, joint_entropy = tf.split(y_pred, [F, 1], axis=-1)

        entropy = tf.cast(entropy, tf.float64)
        joint_entropy = tf.cast(joint_entropy, tf.float64)
        log_C = tf.math.log(tf.cast(self.C, tf.float64))

        mutual_information = tf.math.abs(
            tf.expand_dims(tf.reduce_sum(entropy, axis=-1), axis=-1) - joint_entropy
        ) / (F * log_C)

        return mutual_information

    def get_config(self):
        config = super().get_config()
        config.update({"C": self.C})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class NormalizedBinaryCrossentropy(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """
        y_true: N x 2
        y_pred: N x 2
        """
        batch_size = tf.shape(y_pred)[0]

        cce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        left = tf.tile(tf.expand_dims([1.0, 0.0], axis=0), [batch_size, 1])
        right = tf.tile(tf.expand_dims([0.0, 1.0], axis=0), [batch_size, 1])

        cce_left = tf.keras.losses.binary_crossentropy(left, y_pred)
        cce_right = tf.keras.losses.binary_crossentropy(right, y_pred)

        cce_norm = tf.divide(cce, (cce_left + cce_right))
        return cce_norm


@tf.keras.utils.register_keras_serializable()
class TransposeLayer(Layer):
    def call(self, x):
        return tf.transpose(x, perm=(0, 3, 1, 2))


@tf.keras.utils.register_keras_serializable()
class RenyiEntropyLayer(tf.keras.layers.Layer):
    def __init__(self, alpha=2, **kwargs):
        super(RenyiEntropyLayer, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, K):
        """
        input: K tensor, (N, F, C, C)
        output: NxF
        """
        C = tf.shape(K)[-1]

        diag = tf.linalg.diag_part(K)
        denominator = tf.math.sqrt(
            tf.linalg.matmul(tf.expand_dims(diag, -1), tf.expand_dims(diag, -1), transpose_b=True)
        )
        X = tf.cast((1 / C), tf.float32) * tf.math.divide(K, denominator)

        if self.alpha == 2:
            X_matmul = tf.linalg.matmul(X, X)
            return -tf.math.log(tf.linalg.trace(X_matmul))
        else:
            e, _ = tf.linalg.eigh(X)
            return tf.math.log(tf.reduce_sum(tf.math.real(tf.math.pow(e, self.alpha)), axis=-1)) / (1 - self.alpha)


@tf.keras.utils.register_keras_serializable()
class JointRenyiEntropyLayer(tf.keras.layers.Layer):
    def __init__(self, alpha, **kwargs):
        super(JointRenyiEntropyLayer, self).__init__(**kwargs)
        self.alpha = alpha
        self.renyi_entropy_layer = RenyiEntropyLayer(alpha)

    def build(self, input_shape):
        renyi_input_shape = tf.TensorShape(
            [input_shape[0], 1, input_shape[-2], input_shape[-1]]
        )
        self.renyi_entropy_layer.build(renyi_input_shape)
        super(JointRenyiEntropyLayer, self).build(input_shape)

    def call(self, K):
        """
        input: K tensor, (N, F, C, C)
        output: Nx1
        """
        C = tf.shape(K)[-1]
        product = tf.reduce_prod(K, axis=1)  # (N, C, C)

        trace = tf.linalg.trace(product)
        trace = tf.expand_dims(tf.expand_dims(trace, axis=-1), axis=-1)
        trace = tf.tile(trace, [1, C, C])

        argument = product / trace
        argument = tf.expand_dims(argument, axis=1)  # Necesario porque renyi_entropy recibe 4 dimensiones (1, C, C)

        joint_entropy = self.renyi_entropy_layer(argument)
        return joint_entropy

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class InspectableMultiHeadAttention(layers.MultiHeadAttention):
    """
    Extiende la MultiHeadAttention original de Keras para añadir
    un método de inspección de pesos de proyección.
    """

    def get_projection_weights(self):
        if not self.built:
            raise ValueError("La capa no ha sido construida.")

        weights_dict = {
            "query": self._query_dense.kernel.numpy(),
            "key": self._key_dense.kernel.numpy(),
            "value": self._value_dense.kernel.numpy(),
            "output": self._output_dense.kernel.numpy(),
        }
        return weights_dict


@tf.keras.utils.register_keras_serializable()
class InspectableTransformerEncoder(Layer):
    """
    TransformerEncoder que permite inspeccionar pesos de proyección y
    los últimos mapas de atención generados (C x C).
    """

    def __init__(
        self,
        num_heads,
        intermediate_dim,
        dropout=0.0,
        activation="relu",
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self._last_attention_scores = None

    def build(self, input_shape):
        hidden_dim = int(input_shape[-1])
        key_dim = int(hidden_dim // self.num_heads)

        self._self_attention_layer = InspectableMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            value_dim=key_dim,
            dropout=self.dropout,
            name="self_attention_inspectable",
        )
        self._self_attention_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self._self_attention_dropout = layers.Dropout(rate=self.dropout)
        self._feedforward_intermediate_dense = layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
        )
        self._feedforward_output_dense = layers.Dense(hidden_dim)
        self._feedforward_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self._feedforward_dropout = layers.Dropout(rate=self.dropout)

        self._self_attention_layer.build(input_shape, input_shape, input_shape)
        self._self_attention_layer_norm.build(input_shape)
        self._feedforward_intermediate_dense.build(input_shape)
        intermediate_shape = tf.TensorShape(input_shape[:-1]).concatenate(self.intermediate_dim)
        self._feedforward_output_dense.build(intermediate_shape)
        self._feedforward_layer_norm.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, padding_mask=None, training=False):
        attention_output, attention_scores = self._self_attention_layer(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=padding_mask,
            training=training,
            return_attention_scores=True,
        )

        self._last_attention_scores = attention_scores

        attention_output = self._self_attention_dropout(attention_output, training=training)
        attention_output = self._self_attention_layer_norm(inputs + attention_output)

        ff_output = self._feedforward_intermediate_dense(attention_output)
        ff_output = self._feedforward_output_dense(ff_output)
        ff_output = self._feedforward_dropout(ff_output, training=training)
        output = self._feedforward_layer_norm(attention_output + ff_output)

        return output

    def get_attention_scores(self):
        if self._last_attention_scores is None:
            raise ValueError("No se han calculado aún los attention scores. Haz un forward pass primero.")
        return self._last_attention_scores

    def get_attention_weights(self):
        if not self.built:
            raise ValueError("La capa Encoder no ha sido construida.")
        return self._self_attention_layer.get_projection_weights()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
                "activation": self.activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ConvexCombinationLayer(Layer):
    def __init__(self, num_kernels, **kwargs):
        super().__init__(**kwargs)
        self.num_kernels = num_kernels

    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=(self.num_kernels,),
            initializer=tf.keras.initializers.Constant(1.0 / self.num_kernels),
            trainable=True,
            name="kernel_weights",
        )

    def call(self, inputs):
        weights = tf.nn.softmax(self.alpha)
        weights_reshaped = tf.reshape(weights, (self.num_kernels, 1, 1, 1))
        combined = tf.add_n([w * k for w, k in zip(tf.unstack(weights_reshaped), inputs)])
        weights_broadcast = tf.reshape(weights, (1, self.num_kernels))
        return combined, weights_broadcast


def inception_block(x, F, num_kernels, kernel_sigmas, sigma_min=1e-3, sigma_max=10.0, eps=1e-6):
    kernels = []

    for i in range(0, num_kernels):
        name = f"gaussian_layer_{i+1}"
        init_sigma = kernel_sigmas[i]
        branch_k = GaussianKernelLayer(
            init_sigma=init_sigma,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            eps=eps,
            name=name,
        )(x)
        kernels.append(branch_k)

    combined_input, kernel_weights_out = ConvexCombinationLayer(
        num_kernels, name="convex_combination"
    )(kernels)

    inception = Conv2D(F, (3, 3), padding="same", activation="relu", name="conv_after_inception")(
        combined_input
    )

    concatenated_kernels = Concatenate(axis=-1, name="concatenated_kernels")(kernels)

    return concatenated_kernels, inception, kernel_weights_out


def inspect_gaussian_sigmas(model, atol=1e-6, boundary_tol=1e-4):
    report = []

    for layer in model.layers:
        if isinstance(layer, GaussianKernelLayer):
            sigma = float(layer.get_sigma().numpy())
            init_sigma = float(layer.init_sigma)

            item = {
                "layer_name": layer.name,
                "init_sigma": init_sigma,
                "current_sigma": sigma,
                "delta": sigma - init_sigma,
                "updated": abs(sigma - init_sigma) > atol,
                "near_sigma_min": abs(sigma - layer.sigma_min) <= boundary_tol,
                "near_sigma_max": abs(sigma - layer.sigma_max) <= boundary_tol,
            }
            report.append(item)

    if not report:
        print("No GaussianKernelLayer found in model.")
        return report

    print("=== Gaussian sigma inspection ===")
    for item in report:
        flags = []
        flags.append("UPDATED" if item["updated"] else "UNCHANGED")
        if item["near_sigma_min"]:
            flags.append("NEAR_MIN")
        if item["near_sigma_max"]:
            flags.append("NEAR_MAX")

        print(
            f"{item['layer_name']}: "
            f"init={item['init_sigma']:.6f}, "
            f"current={item['current_sigma']:.6f}, "
            f"delta={item['delta']:.6f} "
            f"[{' | '.join(flags)}]"
        )

    return report


def TGARNet(
    num_kernels=3,
    nb_classes=2,
    Chans=19,
    Samples=512,
    norm_rate=0.25,
    alpha=2,
    num_heads=3,
    intermediate_dim=128,
    kernel_sigmas=None,
):
    if kernel_sigmas is None:
        kernel_sigmas = [5.0, 2.5, 1.25]

    input1 = Input(shape=(Chans, Samples))

    # 1 Reorganize data for Transformer (Samples, Chans)
    x = Reshape((Samples, Chans))(input1)

    # 2 Normalización antes del Transformer
    x = LayerNormalization()(x)

    # 3 Apply TransformerEncoder
    transformer_encoder = InspectableTransformerEncoder(
        num_heads=num_heads,
        intermediate_dim=intermediate_dim,
        name="transformer_encoder",
    )
    x = transformer_encoder(x)

    # 4 Normalización después del Transformer
    x = LayerNormalization()(x)

    # 5 Restore original shape (Chans, Samples, 1)
    x = Reshape((Chans, Samples, 1))(x)

    # 6 Inception with KernelConv
    concatenated_kernels, inception, kernel_weights = inception_block(x, 5, num_kernels, kernel_sigmas)

    # 7 Renyi entropies
    concatenated_kernels = TransposeLayer()(concatenated_kernels)
    layer_entropy = RenyiEntropyLayer(alpha=alpha)(concatenated_kernels)
    layer_joint_entropy = JointRenyiEntropyLayer(alpha=alpha)(concatenated_kernels)
    entropies_out = Concatenate(axis=-1, name="concatenated_entropies")(
        [layer_entropy, layer_joint_entropy]
    )

    # 8 Extra convolutional stack
    final_conv = Conv2D(3, kernel_size=3, padding="same", activation="relu", name="Conv2D_2")(inception)
    final_conv = BatchNormalization()(final_conv)

    final_conv = Conv2D(3, kernel_size=3, padding="same", activation="relu", name="Conv2D_3")(final_conv)
    final_conv = BatchNormalization()(final_conv)

    flat = Flatten()(final_conv)
    drop = Dropout(0.3)(flat)
    dense = Dense(nb_classes, name="output", kernel_constraint=max_norm(norm_rate))(drop)
    softmax = Activation("softmax", name="out_activation")(dense)

    model = Model(
        inputs=input1,
        outputs={
            "out_activation": softmax,
            "entropies_out": entropies_out,
            "kernel_weights_out": kernel_weights,
        },
    )

    return model
