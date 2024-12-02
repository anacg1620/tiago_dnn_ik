# tiago_data_generation

TiAGO data generation for learning IK. Creates a CSV file with joint angles and end-effector positions.

## Launch simulation and controller

To launch tiago_data_generation AND simulation, run:
```bash
source ~/.virtualenvs/rocker/bin/activate
```

From within `tiago-2021-accessrobot-sim/`:
- With NVIDIA:

    ```bash
    rocker --home --network=host --user --volume ${PWD}:/tiago_public_ws/src/tiago-2021-accessrobot-sim --x11 --privileged ghcr.io/roboticslab-uc3m/tiago-2021-accessrobot-sim:latest --nvidia
    ```

- Else:

    ```bash
    rocker --home --network=host --user --volume ${PWD}:/tiago_public_ws/src/tiago-2021-accessrobot-sim --x11 --privileged ghcr.io/roboticslab-uc3m/tiago-2021-accessrobot-sim:latest --devices /dev/dri/card0
    ```

Make sure you ran the previous line from within `./tiago-2021-accessrobot`:
```bash
[ -f /tiago_public_ws/src/tiago-2021-accessrobot-sim/README.md ] && echo "OK!" || echo "FAIL! Make sure you ran the previous line from within 'tiago-2021-accessrobot-sim/'"
```

Move `tiago_data_generation` to `tiago_public_ws/src`, then:
```bash
cd /tiago_public_ws
sudo catkin build
source /tiago_public_ws/devel/setup.bash
roslaunch tiago_data_generation tiago_gazebo.launch
```

Simulation launches without visualization and writes a csv file with the following structure:
```csv
arm_1_joint, arm_2_joint, arm_3_joint, arm_4_joint, arm_5_joint, arm_6_joint, arm_7_joint, end_effector_x, end_effector_y, end_effector_z
```

The file can be found in `/home/username/.ros` and its name can be set from datagen_controller.yaml, as well as the number of points to be generated.
