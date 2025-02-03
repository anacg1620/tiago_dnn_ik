#!/usr/bin/env python

import yaml
import wandb
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

import custom_metrics
from tiago_dnn_mlp.simple_mlp_train import SimpleMlp
from tiago_dnn_cnn.cnn_train import Cnn
from tiago_dnn_rnn.simple_rnn_train import SimpleRnn
from tiago_dnn_rnn.lstm_train import Lstm


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

    wandb.init(project='tiago_ik', name=args.name, tensorboard=True, config=dnn.config)

    # Specify the loss fuction, optimizer, metrics
    if dnn.input_size == 3:
        dnn.model.compile(
            loss = 'mean_squared_error',
            optimizer = tf.keras.optimizers.Adam(learning_rate=dnn.config['lr']),
            metrics = ['accuracy', 'mean_squared_error'],
            run_eagerly=False # to access individual elements in loss funct 
        )
    elif dnn.input_size == 7:
        dnn.model.compile(
            loss = 'mean_squared_error',
            optimizer = tf.keras.optimizers.Adam(learning_rate=dnn.config['lr']),
            metrics = ['accuracy', 'mean_squared_error', custom_metrics.position_error, 
                    custom_metrics.quaternion_error_1, custom_metrics.quaternion_error_2, custom_metrics.quaternion_error_3],
            run_eagerly=True # to access individual elements in loss funct 
        )
    elif dnn.input_size == 12:
        dnn.model.compile(
            loss = 'mean_squared_error',
            optimizer = tf.keras.optimizers.Adam(learning_rate=dnn.config['lr']),
            metrics = ['accuracy', 'mean_squared_error', custom_metrics.position_error,
                    custom_metrics.rotmatrix_error_1, custom_metrics.rotmatrix_error_2, custom_metrics.rotmatrix_error_3],
            run_eagerly=True # to access individual elements in loss funct 
        )
    else:
        raise Exception('Data format not recognized')

    history = []
    for i in range(stats['curriculums']):
        x_train = np.load(f"data/{dnn.config['data_dir']}/x_train_curr{i+1}.npy")
        y_train = np.load(f"data/{dnn.config['data_dir']}/y_train_curr{i+1}.npy")
        x_test = np.load(f"data/{dnn.config['data_dir']}/x_test_curr{i+1}.npy")
        y_test = np.load(f"data/{dnn.config['data_dir']}/y_test_curr{i+1}.npy")

        # Train the model
        callbacks_list = [
            keras.callbacks.TensorBoard (
                log_dir="logs/"      
            ),
            keras.callbacks.EarlyStopping (
                    monitor='val_loss',
                    patience=3,
            ),
            custom_metrics.PositionError(validation_data=(x_test, y_test))
        ]

        history.append(dnn.model.fit(
            x=x_train,
            y=y_train,
            batch_size=dnn.config['batch_size'],
            epochs=dnn.config['epochs'],
            verbose=dnn.config['verbose'],
            callbacks=callbacks_list,
            validation_data=(x_test, y_test)
        ))

    dnn.save()
