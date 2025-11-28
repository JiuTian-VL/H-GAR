"""Utils for evaluating OpenVLA or fine-tuned OpenVLA policies."""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import json_numpy
import numpy as np
import requests
import tensorflow as tf
import torch
from PIL import Image

from real_experiments.robot.rotation_transformer import RotationTransformer

# Apply JSON numpy patch for serialization
json_numpy.patch()

# Initialize important constants
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
OPENVLA_IMAGE_SIZE = 224  # Standard image size expected by OpenVLA

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    return img.numpy()


def get_action_from_server(
    observation: Dict[str, Any], server_endpoint: str = "http://0.0.0.0:8777/act"
) -> Dict[str, Any]:
    """
    Get VLA action from remote inference server.

    Args:
        observation: Observation data to send to server
        server_endpoint: URL of the inference server

    Returns:
        Dict[str, Any]: Action response from server
    """
    import os
    os.environ['all_proxy'] = ''
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''

    response = requests.post(
        server_endpoint,
        json=observation,
    )
    return response.json()
    

def undo_transform_action(action, abs_action = False):
    rotation_transformer = None
    assert abs_action, '[ERR] Set `abs_action = True` to convert actions to aloha!'
    rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

    raw_shape = action.shape
    if raw_shape[-1] == 20:
        # dual arm
        action = action.reshape(-1, 2, 10)

    d_rot = action.shape[-1] - 4
    pos = action[..., :3]
    rot = action[..., 3 : 3 + d_rot]
    gripper = action[..., [-1]]
    rot = rotation_transformer.inverse(rot)
    uaction = np.concatenate([pos, rot, gripper], axis=-1)

    if raw_shape[-1] == 20:
        # dual arm
        uaction = uaction.reshape(*raw_shape[:-1], 14)

    return uaction
