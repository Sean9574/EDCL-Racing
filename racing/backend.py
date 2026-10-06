"""Keras access with GPUs hidden: the networks are small enough that the CPU is faster."""


def keras_cpu():
    """Return the keras module after hiding GPUs from TensorFlow. Safe to call repeatedly."""
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass  # TensorFlow is already initialized in this process; visibility can no longer change
    import keras
    return keras
