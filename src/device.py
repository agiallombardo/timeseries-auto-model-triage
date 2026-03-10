"""
TensorFlow device and threading setup for MacBook Pro.
- Prefer Metal GPU on Apple Silicon when available.
- Apply TF_NUM_INTRAOP_THREADS / TF_NUM_INTEROP_THREADS from environment.
Called at startup so TF uses the right device and thread counts before any training.
"""
import logging
import os

logger = logging.getLogger(__name__)


def init_tensorflow_device_and_threading():
    """Configure TensorFlow device (prefer Metal on Apple Silicon) and thread counts from env."""
    try:
        import tensorflow as tf
    except ImportError:
        return

    # Threading: respect standard TF env vars (or set from .env)
    intra = os.environ.get("TF_NUM_INTRAOP_THREADS")
    inter = os.environ.get("TF_NUM_INTEROP_THREADS")
    if intra is not None:
        try:
            n = int(intra)
            tf.config.threading.set_intra_op_parallelism_threads(n)
            logger.debug("TF_NUM_INTRAOP_THREADS=%s", n)
        except ValueError:
            pass
    if inter is not None:
        try:
            n = int(inter)
            tf.config.threading.set_inter_op_parallelism_threads(n)
            logger.debug("TF_NUM_INTEROP_THREADS=%s", n)
        except ValueError:
            pass

    # Prefer Metal GPU when available (Apple Silicon)
    try:
        devices = tf.config.list_physical_devices()
        gpu_devices = [d for d in devices if d.device_type in ("GPU", "METAL")]
        if gpu_devices:
            tf.config.set_visible_devices(gpu_devices[0], "GPU")
            logger.info("TensorFlow using device: %s", gpu_devices[0].name)
    except Exception as e:  # pylint: disable=broad-except
        logger.debug("Could not set TensorFlow visible devices: %s", e)
