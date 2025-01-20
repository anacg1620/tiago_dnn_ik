#!/usr/bin/env python

import yaml
import math
import tensorflow as tf
import numpy as np


with open('mlp_config.yaml') as f:
    config = yaml.safe_load(f)


def position_error(y_true, y_pred):
    pos_true = y_true[:, :3]
    pos_pred = y_pred[:, :3]

    return tf.math.squared_difference(pos_true, pos_pred)


########### QUATERNIONS ###########

def quaternion_error_1(y_true, y_pred):
    orient_true = y_true[:, 3:].numpy()
    orient_pred = y_pred[:, 3:].numpy()

    dif1 = np.linalg.norm(orient_true - orient_pred, axis=1)
    dif2 = np.linalg.norm(orient_true + orient_pred, axis=1)

    return np.minimum(dif1, dif2)


def quaternion_error_2(y_true, y_pred):
    orient_true = y_true[:, 3:].numpy()
    orient_pred = y_pred[:, 3:].numpy()

    prod = np.sum(orient_true * orient_pred, axis=1)

    return np.arccos(np.abs(prod))


def quaternion_error_3(y_true, y_pred):
    orient_true = y_true[:, 3:]
    orient_pred = y_pred[:, 3:]

    prod = np.sum(orient_true * orient_pred, axis=1)

    return 1 - np.abs(prod)
