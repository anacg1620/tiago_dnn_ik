# tiago_dnn_ik

Different methods for learning IK with TIAGo.

## Step 1: Data generation
See instructions [here](tiago_data_generation/datagen_info.md).

## Step 2: Data preprocessing
Run `data_preprocessing.py` from tiago_data_generation scripts. This creates four files named `x_train`, `x_test`, `y_train` and `y_test` in the data folder. 

## Step 3: Model building and training
To build and train the DNN models, tensorflow is needed. The docker image can be downloaded with:
```bash
docker pull tensorflow/tensorflow:latest-gpu
```
Then, run the image from the `tiago_dnn_ik` folder and upload the neccessary files:
```bash
docker run -it --network=host --rm --volume ${PWD}:/home/tiago_dnn_ik --gpus all --name tensorflow_container tensorflow/tensorflow:latest-gpu /bin/bash
```
Move to the desired DNN model folder and run the training with python. Parameters can be modified by changing the corresponding yaml.
You may need to install some additional python modules:
```bash
pip install pyyaml
pip install pydot
pip install wandb
apt update && apt install graphviz
```