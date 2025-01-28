#!/usr/bin/env python

import yaml
import tensorflow as tf
import numpy as np
from pykin.robots.single_arm import SingleArm


file_path = 'urdf/tiago/tiago.urdf'
robot = SingleArm(file_path)

with open('mlp_config.yaml') as f:
    config = yaml.safe_load(f)

orient_true = []
orient_pred = []


########### POSITION ###########

def position_error(y_true, y_pred):
    fk_true = np.array([robot.forward_kin(y)['tiago_link_ee'].pos for y in y_true])
    fk_pred = np.array([robot.forward_kin(y)['tiago_link_ee'].pos for y in y_pred])

    return np.linalg.norm(fk_true - fk_pred, axis=1)


########### QUATERNIONS ###########

def quaternion_error_1(y_true, y_pred):
    global orient_true
    global orient_pred
    orient_true = np.array([robot.forward_kin(y)['tiago_link_ee'].rot for y in y_true])
    orient_pred = np.array([robot.forward_kin(y)['tiago_link_ee'].rot for y in y_pred])

    dif1 = np.linalg.norm(orient_true - orient_pred, axis=1)
    dif2 = np.linalg.norm(orient_true + orient_pred, axis=1)

    return np.minimum(dif1, dif2)


def quaternion_error_2(y_true, y_pred):
    global orient_true
    global orient_pred

    prod = np.sum(orient_true * orient_pred, axis=1)

    return np.arccos(np.abs(prod))


def quaternion_error_3(y_true, y_pred):
    global orient_true
    global orient_pred

    prod = np.sum(orient_true * orient_pred, axis=1)

    return 1 - np.abs(prod)


########### ROTATION MATRIX ###########

def rotmatrix_error_1(y_true, y_pred):
    global orient_true
    global orient_pred
    orient_true = np.array([np.matrix(robot.forward_kin(y)['tiago_link_ee'].rotation_matrix) for y in y_true])
    orient_pred = np.array([np.matrix(robot.forward_kin(y)['tiago_link_ee'].rotation_matrix) for y in y_pred])

    rot = np.einsum('fij,fjk->fik', orient_true, np.transpose(orient_pred, axes=(0, 2, 1)))
    ret = np.arccos((rot.trace(axis1=1, axis2=2) - 1) / 2)
    ret[np.isnan(ret)] = 0

    return ret


def rotmatrix_error_2(y_true, y_pred):
    global orient_true
    global orient_pred

    return np.linalg.norm(orient_true - orient_pred, ord='fro', axis=(1, 2))


def rotmatrix_error_3(y_true, y_pred):
    global orient_true
    global orient_pred

    rot = np.einsum('fij,fjk->fik', np.transpose(orient_true, axes=(0, 2, 1)), orient_pred)
    angle = np.arccos((rot.trace(axis1=1, axis2=2) - 1) / 2)
    angle[np.isnan(angle)] = 0
    # log as in https://lcvmwww.epfl.ch/publications/data/articles/63/Means_and_Averaging_in_the_Group_of_Rotations.pdf page 3
    rotlog = (angle / (2 * np.sin(angle)))[:, None, None] * (rot - np.transpose(rot, axes=(0, 2, 1)))
    rotlog[np.isnan(rotlog)] = 0

    return 1 / np.sqrt(2) * np.linalg.norm(rotlog, ord='fro', axis=(1, 2))


def rotmatrix_error_4(y_true, y_pred):
    global orient_true
    global orient_pred
    
    rot = np.einsum('fij,fjk->fik', orient_true, np.transpose(orient_pred, axes=(0, 2, 1)))
    dif = np.identity(3) - rot

    return np.linalg.norm(dif, ord='fro', axis=(1, 2))
