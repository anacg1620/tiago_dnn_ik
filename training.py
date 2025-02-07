#!/usr/bin/env python

import yaml
import wandb
import math
import argparse
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

import custom_metrics
from tiago_dnn_mlp.simple_mlp_train import SimpleMlp
from tiago_dnn_cnn.cnn_train import Cnn
from tiago_dnn_rnn.simple_rnn_train import SimpleRnn
from tiago_dnn_rnn.lstm_train import Lstm


current_curr = 1

def train_generator(curriculums, batch_size):
    global current_curr

    for curr in range(curriculums):
        current_curr = curr
        x_train = np.load(f"data/{dnn.config['data_dir']}/x_train_curr{curr+1}.npy")
        y_train = np.load(f"data/{dnn.config['data_dir']}/y_train_curr{curr+1}.npy")

        batch_x = np.zeros((batch_size, x_train.shape[1]))
        batch_y = np.zeros((batch_size, y_train.shape[1]))

        for times in range(math.floor(x_train.shape[0] / batch_size)):
            for i in range(batch_size):
                # choose random index in features
                index = random.randint(0, x_train.shape[0]-1)
                batch_x[i] = x_train[index]
                batch_y[i] = y_train[index]

            yield batch_x, batch_y


def val_generator(batch_size):
    global current_curr
    x_test = np.load(f"data/{dnn.config['data_dir']}/x_test_curr{current_curr+1}.npy")
    y_test = np.load(f"data/{dnn.config['data_dir']}/y_test_curr{current_curr+1}.npy")

    batch_x = np.zeros((batch_size, x_test.shape[1]))
    batch_y = np.zeros((batch_size, y_test.shape[1]))

    for times in range(math.floor(x_test.shape[0] / batch_size)):
        for i in range(batch_size):
            # choose random index in features
            index = random.randint(0, x_test.shape[0]-1)
            batch_x[i] = x_test[index]
            batch_y[i] = y_test[index]

        yield batch_x, batch_y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="choose name of the DNN model to train", type=str)
    args = parser.parse_args()

    if args.name == 'SimpleMlp':
        dnn = SimpleMlp()
    elif args.name == 'Cnn':
        dnn = Cnn()
    elif args.name == 'SimpleRnn':
        dnn = SimpleRnn()
    elif args.name == 'Lstm':
        dnn = Lstm()
    else:
        raise Exception('model name not recognized')
    
    with open(f"data/{dnn.config['data_dir']}/data_stats.yaml") as f:
        stats = yaml.safe_load(f)

    # wandb.init(project='tiago_ik', name=args.name, tensorboard=True, config=dnn.config)

    # Specify the loss fuction, optimizer, metrics
    dnn.model.compile(
        loss = 'mean_squared_error',
        optimizer = tf.keras.optimizers.Adam(learning_rate=dnn.config['lr']),
        metrics = ['accuracy', 'mean_squared_error'],
        run_eagerly=True # to access individual elements in loss funct 
    )

    # Train the model
    callbacks_list = [
        keras.callbacks.TensorBoard (
            log_dir="logs/"      
        ),
        keras.callbacks.EarlyStopping (
                monitor='val_loss',
                patience=3,
        ),
        custom_metrics.PositionError(validation_data=val_generator(dnn.config['batch_size']), stats=stats)
    ]

    if dnn.input_size == 7:
        callbacks_list.append([custom_metrics.QuaternionError1(validation_data=val_generator(dnn.config['batch_size']), stats=stats),
                                custom_metrics.QuaternionError2(validation_data=val_generator(dnn.config['batch_size'])),
                                custom_metrics.QuaternionError3(validation_data=val_generator(dnn.config['batch_size']))])
        
    elif dnn.input_size == 12:
        callbacks_list.append([custom_metrics.RotMatrixError1(validation_data=val_generator(dnn.config['batch_size']), stats=stats),
                                custom_metrics.RotMatrixError2(validation_data=val_generator(dnn.config['batch_size'])),
                                custom_metrics.RotMatrixError3(validation_data=val_generator(dnn.config['batch_size'])),
                                custom_metrics.RotMatrixError4(validation_data=val_generator(dnn.config['batch_size']))])

    history = dnn.model.fit_generator(
        generator=train_generator(stats['curriculums'], dnn.config['batch_size']),
        steps_per_epoch=dnn.config['steps_per_epoch'],
        epochs=dnn.config['epochs'],
        verbose=dnn.config['verbose'],
        callbacks=callbacks_list,
        validation_data=val_generator(dnn.config['batch_size']),
        validation_steps=dnn.config['validation_steps'],
    )

    dnn.save()
