#!/usr/bin/env python

import yaml
import pandas as pd
import numpy as np
import os

with open(os.path.join('/tiago_public_ws/src/tiago_data_generation/', 'config/datagen_controller.yaml')) as f:
    config = yaml.safe_load(f)

df = pd.read_csv(os.path.join('/home/anacg/.ros', config['datagen_controller']['filename']))
df['distance'] = np.sqrt(np.square(df['ee_x']) + np.square(df['ee_y']) + np.square(df['ee_z']))

print(df[['ee_x', 'ee_y', 'ee_z', 'distance']].describe())
