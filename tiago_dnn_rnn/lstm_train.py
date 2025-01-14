#!/usr/bin/env python
# From https://www.sciencedirect.com/science/article/pii/S0952197623004852 Fig. 14. Beware, description and image do not correspond
# See also https://easychair.org/publications/paper/J1qk/open, https://ieeexplore.ieee.org/document/10217721

import yaml
import numpy as np
import tensorflow as tf
import wandb

wandb.init(project='tiago_ik', name='lstm', tensorboard=True)

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
    tf.keras.layers.LSTM(1000, return_sequences=True, input_shape=(input_size, 1)),
    tf.keras.layers.Dense(1000, activation='swish'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(800, return_sequences=True),
    tf.keras.layers.Dense(800, activation='swish'),
    tf.keras.layers.LSTM(600, return_sequences=True),
    tf.keras.layers.Dense(600, activation='swish'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(400, return_sequences=True),
    tf.keras.layers.Dense(400, activation='swish'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation='swish'),
    tf.keras.layers.LSTM(100, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(output_size)
])

# Specify the loss fuction, optimizer, metrics
model.compile(
    loss = 'mean_squared_error',
    optimizer = tf.keras.optimizers.Nadam(learning_rate=config['lstm']['lr']),
    metrics = ['mean_squared_error']
)

model.summary()
tf.keras.utils.plot_model(model, 'lstm_model.png', show_shapes=True)

# Train the model
callbacks_list = [
    tf.keras.callbacks.TensorBoard (
        log_dir="logs/lstm"      
    ),
    tf.keras.callbacks.EarlyStopping (
            monitor='val_loss',
            patience=3,
    )
]

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=config['lstm']['batch_size'],
    epochs=config['lstm']['epochs'],
    verbose=config['lstm']['verbose'],
    callbacks=callbacks_list,
    validation_data=(x_test, y_test)
)

# Save model
if config['lstm']['save']:
    model.save('lstm_rnn.keras')
    print('Trained model saved to .keras file')

# Predict
x_pred = x_test[:1]
y_exp = y_test[:1]
y_pred = model.predict(x_pred, batch_size=None, verbose="auto", steps=None, callbacks=None)
print(f'Input {x_pred}, expected output {y_exp}, predicted output {y_pred}')
