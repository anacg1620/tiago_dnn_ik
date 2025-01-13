#!/usr/bin/env python

import os
import yaml
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("norm", help="choose normalization type: 0 for standardization, 1 for min-max normalization, other for none")
    args = parser.parse_args()

    with open(os.path.join('/tiago_public_ws/src/tiago_data_generation/', 'config/datagen_controller.yaml')) as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(os.path.join('/home/anacg/.ros', config['datagen_controller']['filename']))

    x_cols = ['ee_x', 'ee_y', 'ee_z']
    y_cols = ['arm_1', 'arm_2', 'arm_3', 'arm_4', 'arm_5', 'arm_6', 'arm_7']

    data_stats = {'df_mean_in': df[x_cols].mean().tolist(), 
                  'df_mean_out': df[y_cols].mean().tolist(),
                  'df_std_in': df[x_cols].std().tolist(), 
                  'df_std_out': df[y_cols].std().tolist(),
                  'df_max_in': df[x_cols].max().tolist(), 
                  'df_max_out': df[y_cols].max().tolist(),
                  'df_min_in': df[x_cols].min().tolist(), 
                  'df_min_out': df[y_cols].min().tolist()}

    # Standardization
    if args.norm == '0':
        df = (df - df.mean()) / df.std()
        data_stats['norm'] = 'std'

    # Normalization
    elif args.norm == '1':
        df = (df - df.min()) / (df.max() - df.min())
        data_stats['norm'] = 'norm'

    # None
    else:
        data_stats['norm'] = 'none'

    x = df[x_cols].to_numpy()
    y = df[y_cols].to_numpy()

    # Split and save
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    name = 'mil_norm'

    with open(os.path.join('/home/anacg/repos/tiago/tiago_dnn_ik/data/', 'x_train_' + name + '.npy'), 'w') as f:
        np.save(f, x_train)

    with open(os.path.join('/home/anacg/repos/tiago/tiago_dnn_ik/data/','y_train_' + name + '.npy'), 'w') as f:
        np.save(f, y_train)

    with open(os.path.join('/home/anacg/repos/tiago/tiago_dnn_ik/data/','x_test_' + name + '.npy'), 'w') as f:
        np.save(f, x_test)

    with open(os.path.join('/home/anacg/repos/tiago/tiago_dnn_ik/data/','y_test_' + name + '.npy'), 'w') as f:
        np.save(f, y_test)

    with open(os.path.join('/home/anacg/repos/tiago/tiago_dnn_ik/data/','data_' + name + '.yaml'), 'w') as f:
        yaml.dump(data_stats, f)
