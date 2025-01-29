#!/usr/bin/env python

import yaml
import wandb
import numpy as np
import tensorflow as tf
from tensorflow import keras
import custom_metrics


with open('mlp_config.yaml') as f:
    config = yaml.safe_load(f)

with open(f"../data/{config['data_dir']}/data_stats.yaml") as f:
    stats = yaml.safe_load(f)

wandb.init(project='tiago_ik', name='pykin_simple_mlp', tensorboard=True, config=config['simple_mlp'])

x_train = np.load(f"../data/{config['data_dir']}/x_train_curr1.npy")
y_train = np.load(f"../data/{config['data_dir']}/y_train_curr1.npy")

print("x_train shape :", x_train.shape)
print("y_train shape :", y_train.shape)

input_size = x_train.shape[1]
output_size = y_train.shape[1]

# Specify the model's architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[input_size]),
    tf.keras.layers.Dense(config['simple_mlp']['units'], activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax'),
])

# Specify the loss fuction, optimizer, metrics
if input_size == 3:
    model.compile(
        loss = 'mean_squared_error',
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['simple_mlp']['lr']),
        metrics = ['accuracy', 'mean_squared_error', custom_metrics.position_error],
        run_eagerly=True # to access individual elements in loss funct 
    )
elif input_size == 7:
    model.compile(
        loss = 'mean_squared_error',
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['simple_mlp']['lr']),
        metrics = ['accuracy', 'mean_squared_error', custom_metrics.position_error, 
                   custom_metrics.quaternion_error_1, custom_metrics.quaternion_error_2, custom_metrics.quaternion_error_3],
        run_eagerly=True # to access individual elements in loss funct 
    )
elif input_size == 12:
    model.compile(
        loss = 'mean_squared_error',
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['simple_mlp']['lr']),
        metrics = ['accuracy', 'mean_squared_error', custom_metrics.position_error,
                   custom_metrics.rotmatrix_error_1, custom_metrics.rotmatrix_error_2, custom_metrics.rotmatrix_error_3],
        run_eagerly=True # to access individual elements in loss funct 
    )
else:
    raise Exception('Data format not recognized')

model.summary()
keras.utils.plot_model(model, 'simple_mlp_model.png', show_shapes=True)

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
    x_train = np.load(f"../data/{config['data_dir']}/x_train_curr{i+1}.npy")
    y_train = np.load(f"../data/{config['data_dir']}/y_train_curr{i+1}.npy")
    x_test = np.load(f"../data/{config['data_dir']}/x_test_curr{i+1}.npy")
    y_test = np.load(f"../data/{config['data_dir']}/y_test_curr{i+1}.npy")

    history.append(model.fit(
        x=x_train,
        y=y_train,
        batch_size=config['simple_mlp']['batch_size'],
        epochs=config['simple_mlp']['epochs'],
        verbose=config['simple_mlp']['verbose'],
        callbacks=callbacks_list,
        validation_data=(x_test, y_test)
    ))

# Save model
if config['simple_mlp']['save']:
    model.save('simple_mlp.keras')
    print('Trained model saved to .keras file')
