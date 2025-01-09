#!/usr/bin/env python
# After https://www.sciencedirect.com/science/article/pii/S0952197623004852

import yaml
import numpy as np
import tensorflow as tf
import wandb

wandb.init(project='tiago_ik', name='cnn', tensorboard=True)

with open('cnn_config.yaml') as f:
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
    tf.keras.layers.Conv1D(filters=config['filters1'], kernel_size=1, input_shape=(input_size, 1)),
    tf.keras.layers.Conv1D(filters=config['filters2'], kernel_size=3, activation='elu'),
    # tf.keras.layers.MaxPooling1D(pool_size=3), # MaxPooling1D downsizes the model by 2, but the output of the previous one without orient is size 1
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(config['units1'], activation='elu'),
    tf.keras.layers.Dense(config['units2'], activation='elu'),
    tf.keras.layers.Dense(config['units3'], activation='elu'),
    tf.keras.layers.Dense(config['units4'], activation='elu'),
    tf.keras.layers.Dense(config['units5'], activation='elu'),
    tf.keras.layers.Dense(output_size, activation='elu'),
])

# Specify the loss fuction, optimizer, metrics
model.compile(
    loss = 'mean_squared_error',
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['lr']),
    metrics = ['mean_squared_error']
)

model.summary()
tf.keras.utils.plot_model(model, 'cnn_model.png', show_shapes=True)

# Train the model
callbacks_list = [
    tf.keras.callbacks.TensorBoard(log_dir="logs/cnn"),
    tf.keras.callbacks.EarlyStopping(patience=3)
]

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=config['batch_size'],
    epochs=config['epochs'],
    verbose=config['verbose'],
    callbacks=callbacks_list,
    validation_split=config['validation_split'],
    validation_data=[x_test, y_test],
    shuffle=config['shuffle'],
    class_weight=None,
    sample_weight=None,
    initial_epoch=config['initial_epoch'],
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=config['validation_freq'],
)

# Save model
if config['save']:
    model.save('cnn.keras')
    print('Trained model saved to .keras file')

# Predict
x_pred = x_test[:1]
y_pred = model.predict(x_pred, batch_size=None, verbose="auto", steps=None, callbacks=None)
print(f'Input {x_pred}, predicted output {y_pred}')
