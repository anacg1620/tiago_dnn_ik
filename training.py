#!/usr/bin/env python

import yaml
import wandb
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tiago_dnn_mlp.simple_mlp_train import SimpleMlp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="choose name of the DNN model to train", type=str)
    args = parser.parse_args()

    if args.name == 'SimpleMlp':
        dnn = SimpleMlp()
    else:
        raise Exception('model name not recognized')
    
    with open(f"data/{dnn.config['data_dir']}/data_stats.yaml") as f:
        stats = yaml.safe_load(f)

    wandb.init(project='tiago_ik', name=args.name, tensorboard=True, config=dnn.config)

    # Train the model
    callbacks_list = [
        keras.callbacks.TensorBoard (
            log_dir="logs/simple_mlp"      
        ),
        keras.callbacks.EarlyStopping (
                monitor='val_loss',
                patience=3,
        )
    ]

    history = []
    for i in range(stats['curriculums']):
        x_train = np.load(f"data/{dnn.config['data_dir']}/x_train_curr{i+1}.npy")
        y_train = np.load(f"data/{dnn.config['data_dir']}/y_train_curr{i+1}.npy")
        x_test = np.load(f"data/{dnn.config['data_dir']}/x_test_curr{i+1}.npy")
        y_test = np.load(f"data/{dnn.config['data_dir']}/y_test_curr{i+1}.npy")

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
