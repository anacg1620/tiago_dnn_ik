#!/usr/bin/env python

import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras

with open('mlp_config.yaml') as f:
    config = yaml.safe_load(f)

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
    tf.keras.layers.Flatten(input_shape = [input_size]),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax'),
])

# Specify the loss fuction, optimizer, metrics
model.compile(
    loss = 'mean_squared_error',
    optimizer = 'sgd',
    metrics = ['mean_squared_error']
)

model.summary()
keras.utils.plot_model(model, 'simple_mlp_model.png', show_shapes=True)

# Train the model
callbacks_list = [
    keras.callbacks.TensorBoard (
        log_dir="logs/simple_mlp"      
    )
]

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=config['batch_size'],
    epochs=config['epochs'],
    verbose=config['verbose'],
    callbacks=callbacks_list,
    validation_split=config['validation_split'],
    validation_data=None,
    shuffle=config['shuffle'],
    class_weight=None,
    sample_weight=None,
    initial_epoch=config['initial_epoch'],
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=config['validation_freq'],
)

# Test
loss, acc = model.evaluate(x_test, y_test)

# Save model
if config['save']:
    model.save('simple_mlp.keras')
    print('Trained model saved to .keras file')

# Predict
x_pred = x_test[:1]
y_pred = model.predict(x_pred, batch_size=None, verbose="auto", steps=None, callbacks=None)
print(f'Input {x_pred}, predicted output {y_pred}')
