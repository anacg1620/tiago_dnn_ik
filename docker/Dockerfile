FROM tensorflow/tensorflow:2.14.0-gpu
# later tensorflow versions give forward compatibility error

# dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 graphviz

# python dependencies
RUN pip install -q -U \
pyyaml pydot wandb keras-tuner

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all