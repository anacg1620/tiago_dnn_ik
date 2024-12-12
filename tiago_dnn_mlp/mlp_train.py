#!/usr/bin/env python
# See https://github.com/dcagigas/Robotics/blob/main/Scorbot_ER_VII/inverse_kinematics_ANN/Scorbot_ANN_bootstrapping_96/scorbot_make_models_inverse_kinematics.py

import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

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

inputX = keras.Input(shape=(1,), name="Xcoor")
inputY = keras.Input(shape=(1,), name="Ycoor")
inputZ = keras.Input(shape=(1,), name="Zcoor")

# subnet 1:
base_input = layers.concatenate([inputX, inputY, inputZ], name="base_input")
base_layer = layers.Dense(16, activation="sigmoid", name="base_layer")(base_input)
output1 = layers.Dense(1, activation = "linear", name="q1")(base_layer)

# subnet 2:
input2 = layers.concatenate([inputX, inputY, inputZ], name="input2")
shoulder_layer = layers.Dense(272, activation="relu", name="shoulder_layer") (input2)
output2 = layers.Dense(1, activation = "linear", name="q2")(shoulder_layer)

# subnet 3:
input3 = layers.concatenate([inputX, inputY, inputZ, output2], name="input3")    
elbow_layer = layers.Dense(496, activation="relu", name="elbow_layer") (input3)
output3 = layers.Dense(1, activation = "linear", name="q3")(elbow_layer)

# subnet 4:
input4 = layers.concatenate([inputX, inputY, inputZ, output3], name="input4")
pitch_layer = layers.Dense(496, activation="relu", name="pitch_layer") (input4)
output4 = layers.Dense(1, activation = "linear", name="q4")(pitch_layer)

# subnet 5:
input5 = layers.concatenate([inputX, inputY, inputZ, output4], name="input5")
shoulder_layer = layers.Dense(272, activation="relu", name="shoulder_layer") (input5)
output5 = layers.Dense(1, activation = "linear", name="q2")(shoulder_layer)

# subnet 6:
input6 = layers.concatenate([inputX, inputY, inputZ, output5], name="input6")    
elbow_layer = layers.Dense(496, activation="relu", name="elbow_layer") (input6)
output6 = layers.Dense(1, activation = "linear", name="q3")(elbow_layer)

# subnet 7:
input7 = layers.concatenate([inputX, inputY, inputZ, output6], name="input7")
pitch_layer = layers.Dense(496, activation="relu", name="pitch_layer") (input7)
output7 = layers.Dense(1, activation = "linear", name="q4")(pitch_layer)

model = keras.Model(inputs=[inputX, inputY, inputZ], outputs=[output1, output2, output3, output4, output5, output6, output7], name="ConcatMLP")

model.summary()
keras.utils.plot_model(model, 'concat_mlp_model.png', show_shapes=True)

optimizer = tf.keras.optimizers.Adam(0.01)
model.compile(optimizer=optimizer, loss=keras.losses.mean_squared_error, metrics=['mse'])

x = x_train[:,[0]]
y = x_train[:,[1]]
z = x_train[:,[2]]

q1_ = y_train[:,[0]]
q2_ = y_train[:,[1]]
q3_ = y_train[:,[2]]
q4_ = y_train[:,[3]]
q5_ = y_train[:,[4]]
q6_ = y_train[:,[5]]
q7_ = y_train[:,[6]]

x_t = x_test[:,[0]]
y_t = x_test[:,[1]]
z_t = x_test[:,[2]]

q1_t = y_test[:,[0]]
q2_t = y_test[:,[1]]
q3_t = y_test[:,[2]]
q4_t = y_test[:,[3]]
q5_t = y_test[:,[4]]
q6_t = y_test[:,[5]]
q7_t = y_test[:,[6]]

callbacks_list = [
    keras.callbacks.EarlyStopping (
            monitor='val_loss',
            patience=4,
    ),
    keras.callbacks.ReduceLROnPlateau (
        factor=0.1,  
        patience=3
    ),
    keras.callbacks.TensorBoard (
        log_dir="logs/concat_mlp"     
    )
]

model.fit({'Xcoor':x, 'Ycoor':y, 'Zcoor':z}, {'q1': q1_, 'q2':q2_, 'q3':q3_, 'q4':q4_, 'q5':q5_, 'q6':q6_, 'q7':q7_}, 
           epochs=config['epochs'], 
           callbacks=callbacks_list, 
           validation_data=([x_t, y_t, z_t], [q1_t, q2_t, q3_t, q4_t, q5_t, q6_t, q7_t]) )

# Save model
if config['save']:
    model.save('simple_mlp.keras')
    print('Trained model saved to .keras file')
