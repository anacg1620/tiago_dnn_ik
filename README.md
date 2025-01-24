# tiago_dnn_ik

Different methods for learning IK with TIAGo.

## Step 1: Data generation
To generate the data using gazebo, see instructions [here](tiago_data_generation/datagen_info.md).

To use pykin, run
```bash
bash docker/build.sh
bash docker/run.sh
python3 tiago_data_generation/pykin_generation.py
```
Urdf and meshes must be in `data/urdf` folder before building docker container. Current urdf comes from [here](https://gitioc.upc.edu/rostutorials/ktmpb/-/blob/d9caa6ef59ffe14d557de47f60590e62d4cf4efa/demos/models/robots/tiago/tiago_with_gripper.urdf)

## Step 2: Data preprocessing
Run `data_preprocessing.py` from tiago_data_generation scripts. This creates four files `x_train`, `x_test`, `y_train` and `y_test` in the data folder. This scripts needs 3 parameters, namely:
- `--norm`: chooses normalization type: 0 for standardization, 1 for min-max normalization, other for none
- `--orient`: chooses whether orientation data is used: 1 for pose, 0 for just position
- `--name`: choose name for split data files.

For example, running
```bash
rosrun tiago_data_generation data_preprocessing.py --norm=1 --orient=1 --name='name'
```
will produce files named "x_train_name.npy", "x_test_name.npy", etc, which will contain normalized pose inputs (position+orientation) and normalized joint outputs.

## Step 3: Model building and training
To build and train the DNN models, tensorflow is needed. The docker image can be built from the `tiago_dnn_ik` folder with:
```bash
bash docker/build.sh
```
Then, run the image with visualization using:
```bash
bash docker/run.sh 1
```
or without it
```bash
bash docker/run.sh
```
Move to the desired DNN model folder and run the training with python. Parameters can be modified by changing the corresponding yaml.
Hyperparameter tuning can be performed by running the corresponding script with python. Additional arguments may be neccesary if more than one model is available in the selected folder.
