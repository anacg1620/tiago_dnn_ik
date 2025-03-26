#!/usr/bin/env python
# See https://github.com/dcagigas/Robotics/blob/main/Scorbot_ER_VII/inverse_kinematics_ANN/Scorbot_ANN_bootstrapping_96/scorbot_make_models_inverse_kinematics.py

import yaml
import wandb
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform


current_curr = 1
file_path = 'urdf/tiago/tiago.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))


class UpdateLRCurriculumOnPlateau(tf.keras.callbacks.Callback):
    def __init__(self, monitor="val_loss", factor=0.1, patience=10, verbose=0, mode="auto", 
                 min_delta=1e-4, cooldown=0, min_lr=0.0, max_lr=0.1, total_currs=1, **kwargs):
        super().__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError(f"ReduceLROnPlateau does not support a factor >= 1.0. Received factor={factor}")

        self.total_currs = total_currs
        self.factor = factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter."""
        if self.mode not in {"auto", "min", "max"}:
            print(f"Learning rate reduction mode {self.mode} is unknown, fallback to auto mode.", stacklevel=2,)
            self.mode = "auto"

        if self.mode == "min" or (
            self.mode == "auto" and "acc" not in self.monitor
        ):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.inf

        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.inf

        self.cooldown_counter = self.cooldown
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        global current_curr

        logs = logs or {}
        logs["learning_rate"] = float(np.array(self.model.optimizer.learning_rate))
        current = logs.get(self.monitor)

        if current is None:
            print(
                "Learning rate reduction is conditioned on metric "
                f"`{self.monitor}` which is not available. Available metrics "
                f"are: {','.join(list(logs.keys()))}.",
                stacklevel=2,
            )
        else:
            # Update LR
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0
            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(np.array(self.model.optimizer.learning_rate))
                    
                    if old_lr > np.float32(self.min_lr):
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        self.model.optimizer.learning_rate = new_lr

                        if self.verbose > 0:
                            print(f"\nEpoch {epoch + 1}:  Reducing learning rate to {new_lr}.")

                        self.cooldown_counter = self.cooldown
                        self.wait = 0

                    else:
                        if current_curr < self.total_currs:
                            current_curr += 1
                            print(f'\n\n--- Change to curriculum {current_curr} ---\n')
                            self.model.optimizer.learning_rate = self.max_lr
                            self._reset()
                        elif self.model.optimizer.learning_rate == self.min_lr:
                            print('All curriculums and LRs tested. Stopping...')
                            self.model.stop_training = True

    def in_cooldown(self):
        return self.cooldown_counter > 0


def position_error(stats):
    def pos_error(y_true, y_pred):
        # De-standardization or de-normalization of inputs
        if 'std' == stats['norm']:
            y_true = y_true * np.array(stats['df_std_out']) + np.array(stats['df_mean_out'])
            y_pred = y_pred * np.array(stats['df_std_out']) + np.array(stats['df_mean_out'])
        elif 'norm' == stats['norm']:
            y_true = y_true * (np.array(stats['df_max_out']) - np.array(stats['df_min_out'])) + np.array(stats['df_min_out'])
            y_pred = y_pred * (np.array(stats['df_max_out']) - np.array(stats['df_min_out'])) + np.array(stats['df_min_out'])
        elif 'max-abs' == stats['norm']:
            y_true = y_true * np.array(stats['df_maxabs_out'])
            y_pred = y_pred * np.array(stats['df_maxabs_out'])
        elif 'iqr' == stats['norm']:
            y_true = y_true * (np.array(stats['df_quantile75_out']) - np.array(stats['df_quantile25_out'])) + np.array(stats['df_median_out'])
            y_pred = y_pred * (np.array(stats['df_quantile75_out']) - np.array(stats['df_quantile25_out'])) + np.array(stats['df_median_out'])

        fk_true = np.array([robot.forward_kin(y)['tiago_link_ee'].pos for y in y_true])
        fk_pred = np.array([robot.forward_kin(y)['tiago_link_ee'].pos for y in y_pred])

        return np.linalg.norm(fk_true - fk_pred, axis=1)
    
    return pos_error


def orientation_error(stats):
    def orient_error(y_true, y_pred):
        # De-standardization or de-normalization of inputs
        if 'std' == stats['norm']:
            y_true = y_true * np.array(stats['df_std_out']) + np.array(stats['df_mean_out'])
            y_pred = y_pred * np.array(stats['df_std_out']) + np.array(stats['df_mean_out'])
        elif 'norm' == stats['norm']:
            y_true = y_true * (np.array(stats['df_max_out']) - np.array(stats['df_min_out'])) + np.array(stats['df_min_out'])
            y_pred = y_pred * (np.array(stats['df_max_out']) - np.array(stats['df_min_out'])) + np.array(stats['df_min_out'])
        elif 'max-abs' == stats['norm']:
            y_true = y_true * np.array(stats['df_maxabs_out'])
            y_pred = y_pred * np.array(stats['df_maxabs_out'])
        elif 'iqr' == stats['norm']:
            y_true = y_true * (np.array(stats['df_quantile75_out']) - np.array(stats['df_quantile25_out'])) + np.array(stats['df_median_out'])
            y_pred = y_pred * (np.array(stats['df_quantile75_out']) - np.array(stats['df_quantile25_out'])) + np.array(stats['df_median_out'])

        fk_true = np.array([robot.forward_kin(y)['tiago_link_ee'].rot for y in y_true])
        fk_pred = np.array([robot.forward_kin(y)['tiago_link_ee'].rot for y in y_pred])
        dif1 = np.linalg.norm(fk_true - fk_pred, axis=1)
        dif2 = np.linalg.norm(fk_true + fk_pred, axis=1)

        return np.minimum(dif1, dif2)
    
    return orient_error


class PositionError(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, stats):
        self.validation_data = validation_data
        self.stats = stats
        self.pos_error = []

    def eval_map(self):
        x_true, y_true = next(self.validation_data)
        y_pred = self.model.predict(x_true)[7]
        y_true = y_true['combined_output']
        x_true = np.array([x_true['Xcoor'], x_true['Ycoor'], x_true['Zcoor']])

        # De-standardization or de-normalization of inputs
        if 'std' == self.stats['norm']:
            x_true = x_true * np.array(self.stats['df_std_in'])[:3] + np.array(self.stats['df_mean_in'])[:3]
            y_pred = y_pred * np.array(self.stats['df_std_out']) + np.array(self.stats['df_mean_out'])
        elif 'norm' == self.stats['norm']:
            x_true = x_true * (np.array(self.stats['df_max_in'])[:3] - np.array(self.stats['df_min_in'])[:3]) + np.array(self.stats['df_min_in'])[:3]
            y_pred = y_pred * (np.array(self.stats['df_max_out']) - np.array(self.stats['df_min_out'])) + np.array(self.stats['df_min_out'])
        elif 'max-abs' == self.stats['norm']:
            x_true = x_true * np.array(self.stats['df_maxabs_in'])[:3]
            y_pred = y_pred * np.array(self.stats['df_maxabs_out'])
        elif 'iqr' == self.stats['norm']:
            x_true = x_true * (np.array(self.stats['df_quantile75_in'])[:3] - np.array(self.stats['df_quantile25_in'])[:3]) + np.array(self.stats['df_median_in'])[:3]
            y_pred = y_pred * (np.array(self.stats['df_quantile75_out']) - np.array(self.stats['df_quantile25_out'])) + np.array(self.stats['df_median_out'])

        x_pred = np.array([robot.forward_kin(y)['tiago_link_ee'].pos for y in y_pred])

        return np.mean(np.linalg.norm(x_true - x_pred, axis=1))

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_map()
        # print ("Position error for epoch %d is %f"%(epoch, score))
        wandb.log({'mean-position-error': score})
        self.pos_error.append(score)


class OrientationError(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, stats):
        self.validation_data = validation_data
        self.stats = stats
        self.orient_error = []

    def eval_map(self):
        x_true, y_true = next(self.validation_data)
        y_pred = self.model.predict(x_true)[7]
        y_true = y_true['combined_output']

        # De-standardization or de-normalization of inputs
        # x -> pos, (orient), y -> joint angles
        if 'std' == self.stats['norm']:
            y_true = y_true * np.array(self.stats['df_std_out']) + np.array(self.stats['df_mean_out'])
            y_pred = y_pred * np.array(self.stats['df_std_out']) + np.array(self.stats['df_mean_out'])
        elif 'norm' == self.stats['norm']:
            y_true = y_true * (np.array(self.stats['df_max_out']) - np.array(self.stats['df_min_out'])) + np.array(self.stats['df_min_out'])
            y_pred = y_pred * (np.array(self.stats['df_max_out']) - np.array(self.stats['df_min_out'])) + np.array(self.stats['df_min_out'])
        elif 'max-abs' == self.stats['norm']:
            y_true = y_true * np.array(self.stats['df_maxabs_out'])
            y_pred = y_pred * np.array(self.stats['df_maxabs_out'])
        elif 'iqr' == self.stats['norm']:
            y_true = y_true * (np.array(self.stats['df_quantile75_out']) - np.array(self.stats['df_quantile25_out'])) + np.array(self.stats['df_median_out'])
            y_pred = y_pred * (np.array(self.stats['df_quantile75_out']) - np.array(self.stats['df_quantile25_out'])) + np.array(self.stats['df_median_out'])

        orient_pred = np.array([robot.forward_kin(y)['tiago_link_ee'].rot for y in y_pred])
        orient_true = np.array([robot.forward_kin(y)['tiago_link_ee'].rot for y in y_true])
        dif1 = np.linalg.norm(orient_true - orient_pred, axis=1)
        dif2 = np.linalg.norm(orient_true + orient_pred, axis=1)

        return np.mean(np.minimum(dif1, dif2))

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_map()
        # print ("Position error for epoch %d is %f"%(epoch, score))
        wandb.log({'mean-orientation-error': score})
        self.orient_error.append(score)


def train_generator(curriculums, batch_size):
    global current_curr

    while True:
        x_train = np.load(f"data/{dnn.config['data_dir']}/x_train_curr{current_curr}.npy")
        y_train = np.load(f"data/{dnn.config['data_dir']}/y_train_curr{current_curr}.npy")

        batch_x = np.zeros((batch_size, x_train.shape[1]))
        batch_y = np.zeros((batch_size, y_train.shape[1]))

        for i in range(batch_size):
            # choose random index in features
            index = random.randint(0, x_train.shape[0]-1)
            batch_x[i] = x_train[index]
            batch_y[i] = y_train[index]

        x = batch_x[:,[0]]
        y = batch_x[:,[1]]
        z = batch_x[:,[2]]

        q1_ = batch_y[:,[0]]
        q2_ = batch_y[:,[1]]
        q3_ = batch_y[:,[2]]
        q4_ = batch_y[:,[3]]
        q5_ = batch_y[:,[4]]
        q6_ = batch_y[:,[5]]
        q7_ = batch_y[:,[6]]
        combined_ = batch_y

        yield {'Xcoor':x, 'Ycoor':y, 'Zcoor':z}, {'q1': q1_, 'q2':q2_, 'q3':q3_, 'q4':q4_, 'q5':q5_, 'q6':q6_, 'q7':q7_, 'combined_output': combined_}


def val_generator(batch_size):
    global current_curr

    while True:
        x_test = np.load(f"data/{dnn.config['data_dir']}/x_test.npy")
        y_test = np.load(f"data/{dnn.config['data_dir']}/y_test.npy")

        batch_x = np.zeros((batch_size, x_test.shape[1]))
        batch_y = np.zeros((batch_size, y_test.shape[1]))

        for i in range(batch_size):
            # choose random index in features
            index = random.randint(0, x_test.shape[0]-1)
            batch_x[i] = x_test[index]
            batch_y[i] = y_test[index]

        x = batch_x[:,[0]]
        y = batch_x[:,[1]]
        z = batch_x[:,[2]]

        q1_ = batch_y[:,[0]]
        q2_ = batch_y[:,[1]]
        q3_ = batch_y[:,[2]]
        q4_ = batch_y[:,[3]]
        q5_ = batch_y[:,[4]]
        q6_ = batch_y[:,[5]]
        q7_ = batch_y[:,[6]]
        combined_ = batch_y

        yield {'Xcoor':x, 'Ycoor':y, 'Zcoor':z}, {'q1': q1_, 'q2':q2_, 'q3':q3_, 'q4':q4_, 'q5':q5_, 'q6':q6_, 'q7':q7_, 'combined_output': combined_}


class SevenSubnetMlp():
    def __init__(self):
        with open('tiago_dnn_mlp/mlp_config.yaml') as f:
            self.config = yaml.safe_load(f)['seven_subnet_mlp']

        x_train = np.load(f"data/{self.config['data_dir']}/x_train_curr1.npy")
        y_train = np.load(f"data/{self.config['data_dir']}/y_train_curr1.npy")

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

        # combined outputs
        combined_outputs = layers.concatenate([output1, output2, output3, output4, output5, output6, output7], name="combined_output")

        self.model = keras.Model(inputs=[inputX, inputY, inputZ], outputs=[output1, output2, output3, output4, output5, output6, output7, combined_outputs], name="ConcatMLP")
        
        self.model.summary()
        # keras.utils.plot_model(self.model, 'tiago_dnn_mlp/concat_mlp_model.png', show_shapes=True)

    def save(self):
        # Save model
        if self.config['save']:
            self.model.save('tiago_dnn_mlp/7subnet_mlp.keras')
            print('Trained model saved to .keras file')


if __name__ == '__main__':
    dnn = SevenSubnetMlp()

    x_train = np.load(f"data/{dnn.config['data_dir']}/x_train_curr1.npy")
    y_train = np.load(f"data/{dnn.config['data_dir']}/y_train_curr1.npy")
    x_test = np.load(f"data/{dnn.config['data_dir']}/x_test.npy")
    y_test = np.load(f"data/{dnn.config['data_dir']}/y_test.npy")

    with open(f"data/{dnn.config['data_dir']}/data_stats.yaml") as f:
        stats = yaml.safe_load(f)

    wandb.init(project='tiago_dnn_ik_results', name="SevenSubnetMlp", tensorboard=True, config=dnn.config)

    # Specify the loss fuction, optimizer, metrics
    dnn.model.compile(
        loss = 'mean_squared_error',
        optimizer = tf.keras.optimizers.Adam(learning_rate=dnn.config['lr']),
        metrics = ['mean_squared_error'],
        run_eagerly=True # to access individual elements in loss funct 
    )

    callbacks_list = [
        keras.callbacks.TensorBoard(log_dir="logs/"),
        UpdateLRCurriculumOnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=dnn.config['verbose'], 
                                    min_lr=0.0001, max_lr=dnn.config['lr'], total_currs=stats['curriculums'], 
                                    cooldown=5),
        PositionError(validation_data=val_generator(dnn.config['batch_size']), stats=stats),
        OrientationError(validation_data=val_generator(dnn.config['batch_size']), stats=stats)
    ]
 
    # Train
    history = dnn.model.fit(train_generator(stats['curriculums'], dnn.config['batch_size']), 
                            steps_per_epoch=((stats['curriculum_sizes'][0] * (1-stats['test_size'])) // dnn.config['batch_size']),
                            epochs=dnn.config['epochs'],
                            verbose=dnn.config['verbose'],
                            callbacks=callbacks_list,
                            validation_data=val_generator(dnn.config['batch_size']),
                            validation_steps=((stats['curriculum_sizes'][0] * stats['test_size']) // dnn.config['batch_size']))
    
    # Validation
    x_val = np.load(f"data/{dnn.config['data_dir']}/x_val.npy")
    y_val = np.load(f"data/{dnn.config['data_dir']}/y_val.npy")

    x_v = x_val[:,[0]]
    y_v = x_val[:,[1]]
    z_v = x_val[:,[2]]

    q1_v = y_val[:,[0]]
    q2_v = y_val[:,[1]]
    q3_v = y_val[:,[2]]
    q4_v = y_val[:,[3]]
    q5_v = y_val[:,[4]]
    q6_v = y_val[:,[5]]
    q7_v = y_val[:,[6]]
    combined_v = y_val

    dnn.model.compile(
        loss = 'mean_squared_error',
        optimizer = tf.keras.optimizers.Adam(learning_rate=dnn.config['lr']),
        metrics = ['mean_squared_error', 'mean_squared_error', 'mean_squared_error', 'mean_squared_error', 'mean_squared_error', 'mean_squared_error', 'mean_squared_error',
                    ['mean_squared_error', 'accuracy', position_error(stats), orientation_error(stats)]],
        run_eagerly=True # to access individual elements in loss funct 
    )
    
    eval = dnn.model.evaluate({'Xcoor':x_v, 'Ycoor':y_v, 'Zcoor':z_v}, 
                              {'q1': q1_v, 'q2':q2_v, 'q3':q3_v, 'q4':q4_v, 'q5':q5_v, 'q6':q6_v, 'q7':q7_v, 'combined_output': combined_v}, 
                              batch_size=dnn.config['batch_size'], 
                              callbacks=callbacks_list)
    
    wandb.log({'evaluation': {
                'mse': eval[0], 
                'accuracy': eval[1], 
                'pos_error': eval[3], 
                'orient_error': eval[4]
                }})
