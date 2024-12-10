#!/usr/bin/env python

import os
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

with open(os.path.join('/tiago_public_ws/src/tiago_data_generation/', 'config/datagen_controller.yaml')) as f:
    config = yaml.safe_load(f)

df = pd.read_csv(os.path.join('/home/anacg/.ros', config['datagen_controller']['filename']))

x_cols = ['ee_x', 'ee_y', 'ee_z']
y_cols = ['arm_1', 'arm_2', 'arm_3', 'arm_4', 'arm_5', 'arm_6', 'arm_7']

# Standardization
# df = (df - df.mean()) / df.std()

# Normalization
# df = (df - df.min()) / (df.max() - df.min())

x = df[x_cols].to_numpy()
y = df[y_cols].to_numpy()

# Split and save
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

with open(os.path.join('/home/anacg/repos/tiago/tiago_dnn_ik/data/', 'x_train.npy'), 'w') as f:
    np.save(f, x_train)

with open(os.path.join('/home/anacg/repos/tiago/tiago_dnn_ik/data/','y_train.npy'), 'w') as f:
    np.save(f, y_train)

with open(os.path.join('/home/anacg/repos/tiago/tiago_dnn_ik/data/','x_test.npy'), 'w') as f:
    np.save(f, x_test)

with open(os.path.join('/home/anacg/repos/tiago/tiago_dnn_ik/data/','y_test.npy'), 'w') as f:
    np.save(f, y_test)
