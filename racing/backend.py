"""Keras access with GPUs hidden: the networks are small enough that the CPU is faster."""
from racing import config


def keras_cpu():
    """Return the keras module after hiding GPUs from TensorFlow and setting its threads (config.KERAS_THREADS and
    config.KERAS_INTER_THREADS, spec §3.2). Safe to call repeatedly."""
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass  # TensorFlow is already initialized in this process; visibility can no longer change
    try:
        tf.config.threading.set_intra_op_parallelism_threads(config.KERAS_THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(config.KERAS_INTER_THREADS)
    except RuntimeError:
        pass  # likewise: the thread pools exist already
    import keras
    return keras
