#!/usr/bin/env python

import yaml
import tensorflow as tf
import numpy as np
from pykin.robots.single_arm import SingleArm


file_path = 'urdf/tiago/tiago.urdf'
robot = SingleArm(file_path)

with open('mlp_config.yaml') as f:
    config = yaml.safe_load(f)


########### POSITION ###########

def position_error(y_true, y_pred):
    fk_true = [robot.forward_kin(y)['tiago_link_ee'].pos for y in y_true]
    fk_pred = [robot.forward_kin(y)['tiago_link_ee'].pos for y in y_pred]

    return tf.math.squared_difference(fk_true, fk_pred)


########### QUATERNIONS ###########

def quaternion_error_1(y_true, y_pred):
    orient_true = np.array([robot.forward_kin(y)['tiago_link_ee'].rot for y in y_true])
    orient_pred = np.array([robot.forward_kin(y)['tiago_link_ee'].rot for y in y_pred])

    dif1 = np.linalg.norm(orient_true - orient_pred, axis=1)
    dif2 = np.linalg.norm(orient_true + orient_pred, axis=1)

    return np.minimum(dif1, dif2)


def quaternion_error_2(y_true, y_pred):
    orient_true = np.array([robot.forward_kin(y)['tiago_link_ee'].rot for y in y_true])
    orient_pred = np.array([robot.forward_kin(y)['tiago_link_ee'].rot for y in y_pred])

    prod = np.sum(orient_true * orient_pred, axis=1)

    return np.arccos(np.abs(prod))


def quaternion_error_3(y_true, y_pred):
    orient_true = np.array([robot.forward_kin(y)['tiago_link_ee'].rot for y in y_true])
    orient_pred = np.array([robot.forward_kin(y)['tiago_link_ee'].rot for y in y_pred])

    prod = np.sum(orient_true * orient_pred, axis=1)

    return 1 - np.abs(prod)
