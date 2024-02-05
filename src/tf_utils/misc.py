import tensorflow
import pickle
import json
import skimage
from pathlib import Path
import datetime
import os
import numpy as np
import tensorflow as tf
import itertools
import random
import matplotlib.pyplot as plt
import math


def get_base_path(training, prefix=""):
    base_path = str(Path(__file__).parent.parent.parent) + "/"

    if training:
        checkpoint_path = base_path + "checkpoints/" + str(prefix) + "ckp_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
        os.mkdir(checkpoint_path)
        return base_path, checkpoint_path
    else:
        return base_path


def create_TF_records_folder(data_path, data_purposes):
    TF_records_path = data_path + "TF_records_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "/"

    os.mkdir(TF_records_path)
    for purpose in data_purposes:
        os.mkdir(TF_records_path + purpose)
    return TF_records_path


def save_pickle(path, array):
    with open(path, 'wb') as handle:
        pickle.dump(array, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def save_json(path, array):
    with open(path, 'w') as f:
        json.dump(array, f)


def load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def split_into_chunks(array, size):
    """
        Yield successive n-sized chunks from list.
        i.e. : list(split_into_chunks(list(range(10, 75)), 10))
    """
    for i in range(0, len(array), size):
        yield array[i:i + size]


def make_patches(volume, padding, patch_size):
    if padding is not None:
        volume = np.pad(volume, padding, 'constant')
    blocks = skimage.util.shape.view_as_blocks(volume, (patch_size, patch_size, patch_size))

    patches = []
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            for k in range(blocks.shape[2]):
                patches.append(blocks[i, j, k, :, :, :])

    return np.array(patches)


def flip_data(data):
    dims = list(range(len(data.shape)))
    flips = []
    for L in range(0, len(dims) + 1):
        for subset in itertools.combinations(dims, L):
            flips.append(np.flip(data, axis=subset))

    return np.array(flips)


def get_argmax_prediction(logits):
    probs = tf.nn.softmax(logits)
    predictions = tf.math.argmax(probs, axis=-1)

    return tf.cast(predictions[..., tf.newaxis], tf.float32)


def get_cut_indices(volume, desired_shape):
    """
    :param volume: Given a volume, find indices where we could cut removing some useless background
    :param desired_shape: Chosen indices
    :return:
    """
    #assert (len(volume.shape) == 3 and len(desired_shape) == 3)
    result = (volume.shape[0] - desired_shape[0], volume.shape[1] - desired_shape[1], volume.shape[2] - desired_shape[2])

    volume_non_zeros = np.count_nonzero(volume)
    range_x, range_y, range_z = list(range(result[0])), list(range(result[1])), list(range(result[2]))

    combinations = list(itertools.product(range_x, range_y, range_z))
    random.shuffle(combinations)

    checked = 0
    for triplet in combinations:
        i, j, k = triplet[0], triplet[1], triplet[2]
        if np.count_nonzero(volume[i:-(result[0] - i), j:-(result[1] - j), k:-(result[2] - k)]) == volume_non_zeros:
            return (i, result[0] - i), (j, result[1] - j), (k, result[2] - k)

        checked += 1
        if checked > 1000:
            break

    return (result[0] // 2, result[0] // 2), (result[1] // 2, result[1] // 2), (result[2] // 2, result[2] // 2)


def get_cut_volume(volume, x_cut, y_cut, z_cut):
    cut_volume = volume[x_cut[0]:-x_cut[1], y_cut[0]:-y_cut[1], z_cut[0]:-z_cut[1]]
    return cut_volume


def add_padding(volumes, pad_size):
    assert (len(volumes.shape) == 5 and len(pad_size) == 3)
    padded_volumes = []
    shape = volumes.shape[1:]
    for volume in volumes:
        # Add one if shape is not EVEN
        padded = np.pad(volume[:, :, :, 0], [(int(shape[0] % 2 != 0), 0), (int(shape[1] % 2 != 0), 0), (int(shape[2] % 2 != 0), 0)], 'constant', constant_values=0.0)

        # Calculate how much padding to give
        val_x = (pad_size[0] - padded.shape[0]) // 2
        val_y = (pad_size[1] - padded.shape[1]) // 2
        val_z = (pad_size[2] - padded.shape[2]) // 2

        # Append padded volume
        padded_volumes.append(np.pad(padded, [(val_x,), (val_y,), (val_z,)], 'constant', constant_values=0.0))

    padded_volumes = np.array(padded_volumes)
    assert (padded_volumes.shape[1] == pad_size[0] and padded_volumes.shape[2] == pad_size[1] and padded_volumes.shape[3] == pad_size[2])

    return np.expand_dims(padded_volumes, -1), np.array(shape[:-1]), np.array([val_x, val_y, val_z])


def remove_padding(volumes, orig_shape, values):
    assert (len(volumes.shape) == 5 and len(orig_shape) == 3 and len(values) == 3)
    # Remove padding
    if values[0] != 0:
        volumes = volumes[:, values[0]:-values[0], :, :]
    if values[1] != 0:
        volumes = volumes[:, :, values[1]:-values[1], :]
    if values[2] != 0:
        volumes = volumes[:, :, :, values[2]:-values[2]]

    volumes = volumes[:, int(orig_shape[0] % 2 != 0):, int(orig_shape[1] % 2 != 0):, int(orig_shape[2] % 2 != 0):]
    assert (volumes.shape[1] == orig_shape[0] and volumes.shape[2] == orig_shape[1] and volumes.shape[3] == orig_shape[2])

    return volumes


def plot_figures(ante, **kwargs):
    """ misc.plot_figures('Prepend text', x=f_batch, y=y_batch) """

    for key, value in kwargs.items():
        for i in range(len(value)):
            if len(value.shape) == 4:
                value = np.expand_dims(np.array(value), -1)

            plt.figure()
            plt.imshow(value[i][:, :, value[i].shape[-2] // 2, 0], cmap="gray")
            plt.colorbar()
            plt.title(str(ante) + " : " + key + "_" + str(i))
    plt.show()


def get_how_much_to_pad(shape, multiple):
    pad = []
    is_same_shape = True
    for val in shape:
        pad.append(math.ceil(val/multiple) * multiple)
        if pad[-1] != val:
            is_same_shape = False

    return pad, is_same_shape


def merge_dicts(dict1, dict2):
    return {**dict1, **dict2}


def get_act_function(label):
    if label == "relu":
        return tf.keras.layers.ReLU
    elif label == "leaky_relu":
        return tf.keras.layers.LeakyReLU
    else:
        raise Exception(label)
