"""Grad-CAM utilities for compatible Keras image-classification models."""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf


def _last_convolutional_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if isinstance(layer, tf.keras.Model):
            try:
                return _last_convolutional_layer(layer)
            except ValueError:
                continue
    raise ValueError("No Conv2D layer was found; Grad-CAM is unavailable for this model.")


def make_gradcam_heatmap(
    image: np.ndarray,
    model: tf.keras.Model,
    class_index: int | None = None,
    layer_name: str | None = None,
) -> np.ndarray:
    """Create a normalized Grad-CAM heatmap for one preprocessed image batch."""
    layer_name = layer_name or _last_convolutional_layer(model)
    target_layer = model.get_layer(layer_name)
    grad_model = tf.keras.Model(model.inputs, [target_layer.output, model.output])
    batch = tf.convert_to_tensor(image, dtype=tf.float32)
    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(batch, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        score = predictions[:, class_index]
    gradients = tape.gradient(score, feature_maps)
    if gradients is None:
        raise ValueError(f"Gradients are unavailable for layer '{layer_name}'.")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * feature_maps, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, maximum)
    return heatmap.numpy()


def overlay_heatmap(original: Any, heatmap: np.ndarray, alpha: float = 0.4) -> Any:
    """Blend a colorized heatmap with a PIL image, resizing to image dimensions."""
    from PIL import Image

    image = original.convert("RGB")
    resized = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(image.size)
    color = resized.convert("RGB")
    # Use a simple red/yellow heatmap without an additional plotting dependency.
    rgb = np.asarray(color, dtype=np.float32) / 255.0
    colored = np.stack((rgb[..., 0], np.minimum(rgb[..., 0] * 2, 1), 1 - rgb[..., 0]), axis=-1)
    base = np.asarray(image, dtype=np.float32) / 255.0
    blended = np.clip((1 - alpha) * base + alpha * colored, 0, 1)
    return Image.fromarray(np.uint8(blended * 255))
