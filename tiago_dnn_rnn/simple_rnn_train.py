#!/usr/bin/env python
# See https://ieeexplore.ieee.org/document/10185778

import yaml
import numpy as np
import tensorflow as tf
import wandb

wandb.init(project='tiago_ik', name='simple_rnn', tensorboard=True)

with open('rnn_config.yaml') as f:
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

model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(config['simple_rnn']['units'], input_shape=(input_size, 1)),
    tf.keras.layers.Dense(output_size, activation='softmax'),
])

# Specify the loss fuction, optimizer, metrics
model.compile(
    loss = 'mean_squared_error',
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['simple_rnn']['lr']),
    metrics = ['mean_squared_error']
)

model.summary()
tf.keras.utils.plot_model(model, 'simple_rnn_model.png', show_shapes=True)

# Train the model
callbacks_list = [
    tf.keras.callbacks.TensorBoard (
        log_dir="logs/simple_rnn"      
    ),
    tf.keras.callbacks.EarlyStopping (
            monitor='val_loss',
            patience=3,
    )
]

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=config['simple_rnn']['batch_size'],
    epochs=config['simple_rnn']['epochs'],
    verbose=config['simple_rnn']['verbose'],
    callbacks=callbacks_list,
    validation_data=(x_test, y_test)
)

# Save model
if config['simple_rnn']['save']:
    model.save('simple_rnn.keras')
    print('Trained model saved to .keras file')

# Predict
x_pred = x_test[:1]
print(x_pred)
y_exp = y_test[:1]
y_pred = model.predict(x_pred, batch_size=None, verbose="auto", steps=None, callbacks=None)
print(f'Input {x_pred}, expected output {y_exp}, predicted output {y_pred}')
