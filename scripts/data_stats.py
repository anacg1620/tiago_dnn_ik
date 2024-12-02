#!/usr/bin/env python

import yaml
import pandas as pd
import os

with open(os.path.join('/tiago_public_ws/src/tiago_data_generation/', 'config/datagen_controller.yaml')) as f:
    config = yaml.safe_load(f)

df = pd.read_csv(os.path.join('/home/anacg/.ros', config['datagen_controller']['filename']))

print('max x value: ')
print(df['ee_x'].max())
print('min x value: ')
print(df['ee_x'].min())

print('max y value: ')
print(df['ee_y'].max())
print('min y value: ')
print(df['ee_y'].min())

print('max z value: ')
print(df['ee_z'].max())
print('min z value: ')
print(df['ee_z'].min())
