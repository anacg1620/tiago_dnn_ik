# tiago_dnn_ik

Different methods for learning IK with TIAGo.

## Step 1: Data generation
To generate the data using gazebo, see instructions [here](tiago_data_generation/datagen_info.md).

To use pykin, build first the docker image from the `tiago_dnn_ik` folder with:
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
Urdf and meshes for TIAGo must be in `data/urdf` folder before building the container. Current urdf comes from [here](https://gitioc.upc.edu/rostutorials/ktmpb/-/blob/d9caa6ef59ffe14d557de47f60590e62d4cf4efa/demos/models/robots/tiago/)

To generate the data, run `pykin_generation.py` from tiago_data_generation scripts. This creates a csv file with the information needed for training.
This scripts needs 3 parameters, namely:
- `--num`: chooses number of samples to generate
- `--file`: chooses resulting csv filename
- `--orient`: chooses orientation (quaternion, matrix or none)

For example, running
```bash
python3 tiago_data_generation/pykin_generation.py --num=100 --file='name' --orient='quaternion'
```
will produce a file named "name.csv" in the `data/pykin` folder that contains 100 samples of the form (joint position, ee position, ee quaternion)

## Step 2: Data preprocessing
To preprocess data generated using gazebo, see instructions [here](tiago_data_generation/datagen_info.md).

To use pykin, run `pykin_preprocessing` from tiago_data_generation scripts in the docker container. This creates four files `x_train`, `x_test`, `y_train` and `y_test` in the data pykin folder. This scripts needs 3 parameters, namely:
- `--file`: chooses input csv filename
- `--norm`: chooses normalization type: 0 for standardization, 1 for min-max normalization, other for none
- `--name`: chooses name for split data files

For example, running
```bash
python3 tiago_data_generation/pykin_preprocessing.py --file='file' --norm=1 --name='name'
```
will read "data/pykin/file.csv" and produce files named "x_train_name.npy", "x_test_name.npy", etc, which will contain normalized pose or position inputs (depending on what the csv contains) and normalized joint outputs.

## Step 3: Model building and training
To build and train the DNN models, tensorflow is needed, so build and run the docker container. 

Move to the desired DNN model folder and run the training with python. Parameters can be modified by changing the corresponding yaml.
Hyperparameter tuning can be performed by running the corresponding script with python. Additional arguments may be neccesary if more than one model is available in the selected folder.
