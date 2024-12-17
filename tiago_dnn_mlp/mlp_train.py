#!/usr/bin/env python
# See https://github.com/dcagigas/Robotics/blob/main/Scorbot_ER_VII/inverse_kinematics_ANN/Scorbot_ANN_bootstrapping_96/scorbot_make_models_inverse_kinematics.py

import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import wandb

with open('mlp_config.yaml') as f:
    config = yaml.safe_load(f)

wandb.init(project='tiago_ik', name='seven_subnet_mlp', tensorboard=True, config=config['seven_subnet_mlp'])

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
layer1 = layers.Dense(config['seven_subnet_mlp']['units1'], activation="sigmoid", name="layer1")(base_input)
output1 = layers.Dense(1, activation = "linear", name="q1")(layer1)

# subnet 2:
input2 = layers.concatenate([inputX, inputY, inputZ], name="input2")
layer2 = layers.Dense(config['seven_subnet_mlp']['units2'], activation="relu", name="layer2") (input2)
output2 = layers.Dense(1, activation = "linear", name="q2")(layer2)

# subnet 3:
input3 = layers.concatenate([inputX, inputY, inputZ, output2], name="input3")    
layer3 = layers.Dense(config['seven_subnet_mlp']['units3'], activation="relu", name="layer3") (input3)
output3 = layers.Dense(1, activation = "linear", name="q3")(layer3)

# subnet 4:
input4 = layers.concatenate([inputX, inputY, inputZ, output3], name="input4")
layer4 = layers.Dense(config['seven_subnet_mlp']['units4'], activation="relu", name="layer4") (input4)
output4 = layers.Dense(1, activation = "linear", name="q4")(layer4)

# subnet 5:
input5 = layers.concatenate([inputX, inputY, inputZ, output4], name="input5")
layer5 = layers.Dense(config['seven_subnet_mlp']['units5'], activation="relu", name="layer5") (input5)
output5 = layers.Dense(1, activation = "linear", name="q5")(layer5)

# subnet 6:
input6 = layers.concatenate([inputX, inputY, inputZ, output5], name="input6")    
layer6 = layers.Dense(config['seven_subnet_mlp']['units6'], activation="relu", name="layer6") (input6)
output6 = layers.Dense(1, activation = "linear", name="q6")(layer6)

# subnet 7:
input7 = layers.concatenate([inputX, inputY, inputZ, output6], name="input7")
layer7 = layers.Dense(config['seven_subnet_mlp']['units7'], activation="relu", name="layer7") (input7)
output7 = layers.Dense(1, activation = "linear", name="q7")(layer7)

model = keras.Model(inputs=[inputX, inputY, inputZ], outputs=[output1, output2, output3, output4, output5, output6, output7], name="ConcatMLP")

model.summary()
keras.utils.plot_model(model, 'concat_mlp_model.png', show_shapes=True)

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=config['seven_subnet_mlp']['lr']), 
    loss = 'mean_squared_error',
    metrics = ['mse']
)

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
        log_dir="logs/seven_subnet_mlp"     
    )
]

model.fit({'Xcoor':x, 'Ycoor':y, 'Zcoor':z}, {'q1': q1_, 'q2':q2_, 'q3':q3_, 'q4':q4_, 'q5':q5_, 'q6':q6_, 'q7':q7_}, 
           batch_size=config['seven_subnet_mlp']['batch_size'],
           epochs=config['seven_subnet_mlp']['epochs'],
           verbose=config['seven_subnet_mlp']['verbose'],
           callbacks=callbacks_list, 
           validation_data=([x_t, y_t, z_t], [q1_t, q2_t, q3_t, q4_t, q5_t, q6_t, q7_t]) )

# Save model
if config['seven_subnet_mlp']['save']:
    model.save('7subnet_mlp.keras')
    print('Trained model saved to .keras file')
