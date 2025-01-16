#!/usr/bin/env python

import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
import wandb


def orientation_error(y_true, y_pred):
    orient_true = y_true[:, 3:]
    orient_pred = y_pred[:, 3:]

    return 0


def position_error(y_true, y_pred):
    pos_true = y_true[:, :3]
    pos_pred = y_pred[:, :3]

    return tf.math.squared_difference(pos_true, pos_pred)


def custom_loss(y_true, y_pred):
    orient_error = orientation_error(y_true, y_pred)
    pos_error = position_error(y_true, y_pred)

    return config['simple_mlp']['orient_factor'] * orient_error + \
           config['simple_mlp']['pos_factor'] * pos_error


with open('mlp_config.yaml') as f:
    config = yaml.safe_load(f)

# wandb.init(project='tiago_ik', name='simple_mlp', tensorboard=True, config=config['simple_mlp'])

x_train = np.load(f"../data/{config['x_train_file']}")
y_train = np.load(f"../data/{config['y_train_file']}")
x_test = np.load(f"../data/{config['x_test_file']}")
y_test = np.load(f"../data/{config['y_test_file']}")

print("x_train shape :", x_train.shape)
print("x_test shape :", x_test.shape)
print("y_train shape :", y_train.shape)
print("y_test shape :", y_test.shape)

input_size = x_train.shape[1]
output_size = y_train.shape[1]

# Specify the model's architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[input_size]),
    tf.keras.layers.Dense(config['simple_mlp']['units'], activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax'),
])

# Specify the loss fuction, optimizer, metrics
model.compile(
    loss = custom_loss,
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['simple_mlp']['lr']),
    metrics = ['accuracy', 'mean_squared_error', orientation_error, position_error],
    run_eagerly=True # to access individual elements in loss funct 
)

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

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=config['simple_mlp']['batch_size'],
    epochs=config['simple_mlp']['epochs'],
    verbose=config['simple_mlp']['verbose'],
    callbacks=callbacks_list,
    validation_data=(x_test, y_test)
)

# Save model
if config['simple_mlp']['save']:
    model.save('simple_mlp.keras')
    print('Trained model saved to .keras file')

# Predict
x_pred = x_test[:1]
y_exp = y_test[:1]
y_pred = model.predict(x_pred, batch_size=None, verbose="auto", steps=None, callbacks=None)
print(f'Input {x_pred}, expected output {y_exp}, predicted output {y_pred}')
