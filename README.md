# tiago_dnn_ik

Different methods for learning IK with TIAGo and how to deploy them.

## [tiago_dnn_ik](tiago_dnn_ik/README.md)

This directory contains the necessary code to generate and preprocess training, validation and test datasets, build the models and train them.

## [tiago_dnn_inference](tiago_dnn_inference/README.md)

This directory contains the necessary code to load a pre-trained model, receive a Cartesian target, perform the IK inference and send back the results to the controller.

## Deployment

To deploy, launch simulation and tiago_dnn_inference. Both are connected through a ROS topic. If you want to send a target, use:

```bash
rostopic pub -1 /infer/state_dnn geometry_msgs/Pose "position:
  x: 0.0
  y: 0.0
  z: 0.0
orientation:
  x: 0.0
  y: 0.0
  z: 0.0
  w: 0.0"
```
The inference package will compute the joint space positions and send them to the controller, which will move the arm.
