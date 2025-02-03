#!/usr/bin/env python

import wandb
import tensorflow as tf
import numpy as np
from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform


file_path = 'urdf/tiago/tiago.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

orient_true = []
orient_pred = []


########### POSITION ###########

def position_error(y_true, y_pred):
    fk_true = np.array([robot.forward_kin(y)['tiago_link_ee'].pos for y in y_true])
    fk_pred = np.array([robot.forward_kin(y)['tiago_link_ee'].pos for y in y_pred])

    return np.linalg.norm(fk_true - fk_pred, axis=1)


class PositionError(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.pos_error = []

    def eval_map(self):
        x_true, y_true = self.validation_data
        y_pred = self.model.predict(x_true)
        x_pred = np.array([robot.forward_kin(y)['tiago_link_ee'].pos for y in y_pred])
        x_true = np.array(x_true)[:, :3]

        return np.mean(np.linalg.norm(x_true - x_pred, axis=1))

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_map()
        print ("Position error for epoch %d is %f"%(epoch, score))
        wandb.log({'mean-position-error': score})
        self.pos_error.append(score)


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


class QuaternionError1(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.quat_error1 = []

    def eval_map(self):
        global orient_true
        global orient_pred
        x_true, y_true = self.validation_data
        y_pred = self.model.predict(x_true)
        orient_pred = np.array([robot.forward_kin(y)['tiago_link_ee'].rot for y in y_pred])
        orient_true = np.array(x_true)[:, 3:]

        dif1 = np.linalg.norm(orient_true - orient_pred, axis=1)
        dif2 = np.linalg.norm(orient_true + orient_pred, axis=1)

        return np.mean(np.minimum(dif1, dif2))

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_map()
        print ("Quaternion error 1 for epoch %d is %f"%(epoch, score))
        wandb.log({'quaternion-error-1': score})
        self.quat_error1.append(score)


class QuaternionError2(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.quat_error2 = []

    def eval_map(self):
        global orient_true
        global orient_pred

        prod = np.sum(orient_true * orient_pred, axis=1)

        return np.mean(np.arccos(np.abs(prod)))

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_map()
        print ("Quaternion error 2 for epoch %d is %f"%(epoch, score))
        wandb.log({'quaternion-error-2': score})
        self.quat_error2.append(score)


class QuaternionError3(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.quat_error3 = []

    def eval_map(self):
        global orient_true
        global orient_pred

        prod = np.sum(orient_true * orient_pred, axis=1)

        return np.mean(1 - np.abs(prod))

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_map()
        print ("Quaternion error 3 for epoch %d is %f"%(epoch, score))
        wandb.log({'quaternion-error-3': score})
        self.quat_error3.append(score)


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


class RotMatrixError1(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.rotmatrix_error1 = []

    def eval_map(self):
        global orient_true
        global orient_pred
        x_true, y_true = self.validation_data
        y_pred = self.model.predict(x_true)
        orient_pred = np.array([np.matrix(robot.forward_kin(y)['tiago_link_ee'].rotation_matrix) for y in y_pred])
        orient_true = np.array(x_true)[:, 3:].reshape((3000, 3, 3))

        rot = np.einsum('fij,fjk->fik', orient_true, np.transpose(orient_pred, axes=(0, 2, 1)))
        ret = np.arccos((rot.trace(axis1=1, axis2=2) - 1) / 2)
        ret[np.isnan(ret)] = 0

        return np.mean(ret)

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_map()
        print ("Rotation Matrix error 1 for epoch %d is %f"%(epoch, score))
        wandb.log({'rotmatrix-error-1': score})
        self.rotmatrix_error1.append(score)


class RotMatrixError2(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.rotmatrix_error2 = []

    def eval_map(self):
        global orient_true
        global orient_pred

        ret = np.linalg.norm(orient_true - orient_pred, ord='fro', axis=(1, 2))

        return np.mean(ret)

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_map()
        print ("Rotation Matrix error 2 for epoch %d is %f"%(epoch, score))
        wandb.log({'rotmatrix-error-2': score})
        self.rotmatrix_error2.append(score)


class RotMatrixError3(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.rotmatrix_error3 = []

    def eval_map(self):
        global orient_true
        global orient_pred

        rot = np.einsum('fij,fjk->fik', np.transpose(orient_true, axes=(0, 2, 1)), orient_pred)
        angle = np.arccos((rot.trace(axis1=1, axis2=2) - 1) / 2)
        angle[np.isnan(angle)] = 0
        # log as in https://lcvmwww.epfl.ch/publications/data/articles/63/Means_and_Averaging_in_the_Group_of_Rotations.pdf page 3
        rotlog = (angle / (2 * np.sin(angle)))[:, None, None] * (rot - np.transpose(rot, axes=(0, 2, 1)))
        rotlog[np.isnan(rotlog)] = 0

        return np.mean(1 / np.sqrt(2) * np.linalg.norm(rotlog, ord='fro', axis=(1, 2)))

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_map()
        print ("Rotation Matrix error 3 for epoch %d is %f"%(epoch, score))
        wandb.log({'rotmatrix-error-3': score})
        self.rotmatrix_error3.append(score)


class RotMatrixError4(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.rotmatrix_error4 = []

    def eval_map(self):
        global orient_true
        global orient_pred

        rot = np.einsum('fij,fjk->fik', orient_true, np.transpose(orient_pred, axes=(0, 2, 1)))
        dif = np.identity(3) - rot

        return np.mean(np.linalg.norm(dif, ord='fro', axis=(1, 2)))

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_map()
        print ("Rotation Matrix error 4 for epoch %d is %f"%(epoch, score))
        wandb.log({'rotmatrix-error-4': score})
        self.rotmatrix_error4.append(score)
