#!/usr/bin/env python

import yaml
import wandb
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


def generator(input, output, batch_size):
    batch_x = np.zeros((batch_size, input.shape[1]))
    batch_y = np.zeros((batch_size, output.shape[1]))

    for i in range(batch_size):
        pos = []
        for joint in joint_names:
            pos.append(random.uniform(limits[joint]['lower'], limits[joint]['upper']))

        fk = robot.forward_kin(pos)
        shoulder_pos = fk['arm_1_link']
        fk = fk['tiago_link_ee']

        if fk.pos[2] > 0.3 and np.linalg.norm(shoulder_pos.pos - fk.pos) < 0.8:
            # choose random index in features
            index = random.randint(0, input.shape[0])
            batch_x[i] = input[index]
            batch_y[i] = output[index]

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
            custom_metrics.PositionError(validation_data=(x_test, y_test), stats=stats)
        ]

        if dnn.input_size == 7:
            callbacks_list.append([custom_metrics.QuaternionError1(validation_data=(x_test, y_test), stats=stats),
                                   custom_metrics.QuaternionError2(validation_data=(x_test, y_test)),
                                   custom_metrics.QuaternionError3(validation_data=(x_test, y_test))])
            
        elif dnn.input_size == 12:
            callbacks_list.append([custom_metrics.RotMatrixError1(validation_data=(x_test, y_test), stats=stats),
                                   custom_metrics.RotMatrixError2(validation_data=(x_test, y_test)),
                                   custom_metrics.RotMatrixError3(validation_data=(x_test, y_test)),
                                   custom_metrics.RotMatrixError4(validation_data=(x_test, y_test))])

        history.append(dnn.model.fit_generator(
            generator(x_train, y_train, dnn.config['batch_size']),
            epochs=dnn.config['epochs'],
            verbose=dnn.config['verbose'],
            callbacks=callbacks_list,
            validation_data=(x_test, y_test)
        ))

    dnn.save()
