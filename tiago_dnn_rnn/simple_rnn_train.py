#!/usr/bin/env python
# See https://ieeexplore.ieee.org/document/10185778

import yaml
import numpy as np
import tensorflow as tf
import custom_metrics


class SimpleRnn():
    def __init__(self): 
        with open('tiago_dnn_rnn/rnn_config.yaml') as f:
            self.config = yaml.safe_load(f)['simple_rnn']

        x_train = np.load(f"data/{self.config['data_dir']}/x_train_curr1.npy")
        y_train = np.load(f"data/{self.config['data_dir']}/y_train_curr1.npy")

        # Specify the model's architecture
        input_size = x_train.shape[1]
        output_size = y_train.shape[1]

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.SimpleRNN(self.config['units'], input_shape=(input_size, 1)),
            tf.keras.layers.Dense(output_size, activation='softmax'),
        ])

        # Specify the loss fuction, optimizer, metrics
        if input_size == 3:
            self.model.compile(
                loss = 'mean_squared_error',
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
                metrics = ['accuracy', 'mean_squared_error', custom_metrics.position_error],
                run_eagerly=True # to access individual elements in loss funct 
            )
        elif input_size == 7:
            self.model.compile(
                loss = 'mean_squared_error',
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
                metrics = ['accuracy', 'mean_squared_error', custom_metrics.position_error, 
                        custom_metrics.quaternion_error_1, custom_metrics.quaternion_error_2, custom_metrics.quaternion_error_3],
                run_eagerly=True # to access individual elements in loss funct 
            )
        elif input_size == 12:
            self.model.compile(
                loss = 'mean_squared_error',
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
                metrics = ['accuracy', 'mean_squared_error', custom_metrics.position_error,
                        custom_metrics.rotmatrix_error_1, custom_metrics.rotmatrix_error_2, custom_metrics.rotmatrix_error_3],
                run_eagerly=True # to access individual elements in loss funct 
            )
        else:
            raise Exception('Data format not recognized')

        self.model.summary()
        tf.keras.utils.plot_model(self.model, 'tiago_dnn_rnn/simple_rnn_model.png', show_shapes=True)

    def save(self):
        # Save model
        if self.config['simple_rnn']['save']:
            self.model.save('tiago_dnn_cnn/simple_rnn.keras')
            print('Trained model saved to .keras file')
