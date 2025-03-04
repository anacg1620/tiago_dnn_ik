#!/usr/bin/env python
# See https://github.com/dcagigas/Robotics/blob/main/Scorbot_ER_VII/inverse_kinematics_ANN/Scorbot_ANN_bootstrapping_96/scorbot_make_models_inverse_kinematics.py

import yaml
import wandb
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


class SevenSubnetMlp():
    def __init__(self):
        with open('mlp_config.yaml') as f:
            self.config = yaml.safe_load(f)['seven_subnet_mlp']

        x_train = np.load(f"../data/{self.config['data_dir']}/x_train_curr1.npy")
        y_train = np.load(f"../data/{self.config['data_dir']}/y_train_curr1.npy")

        self.input_size = x_train.shape[1]

        inputX = keras.Input(shape=(1,), name="Xcoor")
        inputY = keras.Input(shape=(1,), name="Ycoor")
        inputZ = keras.Input(shape=(1,), name="Zcoor")

        # subnet 1:
        base_input = layers.concatenate([inputX, inputY, inputZ], name="base_input")
        layer1 = layers.Dense(self.config['units1'], activation="sigmoid", name="layer1")(base_input)
        output1 = layers.Dense(1, activation = "linear", name="q1")(layer1)

        # subnet 2:
        input2 = layers.concatenate([inputX, inputY, inputZ], name="input2")
        layer2 = layers.Dense(self.config['units2'], activation="relu", name="layer2") (input2)
        output2 = layers.Dense(1, activation = "linear", name="q2")(layer2)

        # subnet 3:
        input3 = layers.concatenate([inputX, inputY, inputZ, output2], name="input3")    
        layer3 = layers.Dense(self.config['units3'], activation="relu", name="layer3") (input3)
        output3 = layers.Dense(1, activation = "linear", name="q3")(layer3)

        # subnet 4:
        input4 = layers.concatenate([inputX, inputY, inputZ, output3], name="input4")
        layer4 = layers.Dense(self.config['units4'], activation="relu", name="layer4") (input4)
        output4 = layers.Dense(1, activation = "linear", name="q4")(layer4)

        # subnet 5:
        input5 = layers.concatenate([inputX, inputY, inputZ, output4], name="input5")
        layer5 = layers.Dense(self.config['units5'], activation="relu", name="layer5") (input5)
        output5 = layers.Dense(1, activation = "linear", name="q5")(layer5)

        # subnet 6:
        input6 = layers.concatenate([inputX, inputY, inputZ, output5], name="input6")    
        layer6 = layers.Dense(self.config['units6'], activation="relu", name="layer6") (input6)
        output6 = layers.Dense(1, activation = "linear", name="q6")(layer6)

        # subnet 7:
        input7 = layers.concatenate([inputX, inputY, inputZ, output6], name="input7")
        layer7 = layers.Dense(self.config['units7'], activation="relu", name="layer7") (input7)
        output7 = layers.Dense(1, activation = "linear", name="q7")(layer7)

        self.model = keras.Model(inputs=[inputX, inputY, inputZ], outputs=[output1, output2, output3, output4, output5, output6, output7], name="ConcatMLP")
        
        self.model.summary()
        # keras.utils.plot_model(self.model, 'tiago_dnn_mlp/concat_mlp_model.png', show_shapes=True)

    def save(self):
        # Save model
        if self.config['save']:
            self.model.save('tiago_dnn_mlp/7subnet_mlp.keras')
            print('Trained model saved to .keras file')


if __name__ == '__main__':
    dnn = SevenSubnetMlp()

    x_train = np.load(f"../data/{dnn.config['data_dir']}/x_train_curr1.npy")
    y_train = np.load(f"../data/{dnn.config['data_dir']}/y_train_curr1.npy")
    x_test = np.load(f"../data/{dnn.config['data_dir']}/x_test_curr1.npy")
    y_test = np.load(f"../data/{dnn.config['data_dir']}/y_test_curr1.npy")
    
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

    with open(f"../data/{dnn.config['data_dir']}/data_stats.yaml") as f:
        stats = yaml.safe_load(f)

    wandb.init(project='tiago_dnn_ik_tests', name="SevenSubnetMlp", tensorboard=True, config=dnn.config)

    # Specify the loss fuction, optimizer, metrics
    dnn.model.compile(
        loss = 'mean_squared_error',
        optimizer = tf.keras.optimizers.Adam(learning_rate=dnn.config['lr']),
        metrics = ['mean_squared_error', 'accuracy'],
        run_eagerly=True # to access individual elements in loss funct 
    )

    callbacks_list = [
        keras.callbacks.TensorBoard(log_dir="logs/"),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=dnn.config['verbose'], 
                                    min_lr=0.0001, max_lr=dnn.config['lr'], cooldown=5)
    ]

    history = dnn.model.fit({'Xcoor':x, 'Ycoor':y, 'Zcoor':z}, {'q1': q1_, 'q2':q2_, 'q3':q3_, 'q4':q4_, 'q5':q5_, 'q6':q6_, 'q7':q7_}, 
                            epochs=200, validation_split=0.3, batch_size=dnn.config['batch_size'], callbacks=callbacks_list, verbose=2)
