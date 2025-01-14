#!/usr/bin/env python

import yaml
import tensorflow as tf
import pandas as pd
from random import random


model = tf.keras.saving.load_model('tiago_dnn_mlp/simple_mlp.keras')
with open('tiago_dnn_mlp/mlp_config.yaml') as f:
    config = yaml.safe_load(f)

name = config['x_train_file'].replace('x_train_', '')
name = name.replace('.npy', '')

with open('data/data_' + name + '.yaml', 'r') as f:
    data = yaml.safe_load(f)

x = random() * (data['df_max_in'][0] - data['df_min_in'][0]) + data['df_min_in'][0]
y = random() * (data['df_max_in'][1] - data['df_min_in'][1]) + data['df_min_in'][1]
z = random() * (data['df_max_in'][2] - data['df_min_in'][2]) + data['df_min_in'][2]
input_og = [x, y, z]
input = pd.DataFrame(input_og)

# Standardization or normalization of input
if 'std' in name:
    input = (input - pd.DataFrame(data['df_mean_in'])) / pd.DataFrame(data['df_std_in'])
elif 'norm' in name:
    input = (input - pd.DataFrame(data['df_min_in'])) / (pd.DataFrame(data['df_max_in']) - pd.DataFrame(data['df_min_in']))

input = input.to_numpy().transpose()
output = model.predict(input)
output = pd.DataFrame(output)

# De-standardization or de-normalization of output
if 'std' in name:
    output = output * pd.DataFrame(data['df_std_out']).transpose() + pd.DataFrame(data['df_mean_out']).transpose()
elif 'norm' in name:
    output = output * (pd.DataFrame(data['df_max_out']).transpose() - pd.DataFrame(data['df_min_out']).transpose()) + pd.DataFrame(data['df_min_out']).transpose()

output = output.to_numpy().tolist()
print(f'For input {input_og}, predicted output is {output}')
