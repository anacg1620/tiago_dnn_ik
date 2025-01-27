#!/usr/bin/env python

import argparse
import random
import csv
from tqdm import tqdm
from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.utils import plot_utils as p_utils


def show_info(robot):
    robot.show_robot_info()

    thetas = [0, 0, 0, 0, 0, 0, 0]
    robot.set_transform(thetas)

    _, ax = p_utils.init_3d_figure("FK", visible_axis=True)
    p_utils.plot_robot(ax=ax, 
                robot=robot,
                geom="visual",
                only_visible_geom=True,
                alpha=1)
    p_utils.show_figure()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", help="choose number of samples to generate", type=int)
    parser.add_argument("--file", help="choose resulting csv filename", type=str)
    parser.add_argument("--orient", help="choose orientation (quaternion, matrix or none)")
    args = parser.parse_args()

    file_path = 'urdf/tiago/tiago.urdf'
    robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
    joint_names = robot.get_all_active_joint_names()
    # show_info(robot)

    limits = {}
    for joint in joint_names:
        limits[joint] = {'lower': robot.joints[joint].limit[0], 'upper': robot.joints[joint].limit[1]}

    # generate positions
    i = 0
    with open(f'data/{args.file}.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if args.orient == 'quaternion':
            csvwriter.writerow(['arm_1','arm_2','arm_3','arm_4','arm_5','arm_6','arm_7','ee_x','ee_y','ee_z',
                               'ee_quat_x','ee_quat_y','ee_quat_z','ee_quat_w'])
        elif args.orient == 'matrix':
            csvwriter.writerow(['arm_1','arm_2','arm_3','arm_4','arm_5','arm_6','arm_7','ee_x','ee_y','ee_z',
                               'ee_rot_00','ee_rot_01','ee_rot_02','ee_rot_10','ee_rot_11','ee_rot_12','ee_rot_20','ee_rot_21','ee_rot_22'])
        else:
            csvwriter.writerow(['arm_1','arm_2','arm_3','arm_4','arm_5','arm_6','arm_7','ee_x','ee_y','ee_z'])

        pbar = tqdm(total=args.num)
        while i < args.num:
            pos = []
            for joint in joint_names:
                pos.append(random.uniform(limits[joint]['lower'], limits[joint]['upper']))

            fk = robot.forward_kin(pos)['arm_7_link']
            jointpos = ','.join([str(i) for i in pos])
            fkpos = ','.join([str(i) for i in fk.pos])

            if fk.pos[2] > 0.3:
                if args.orient == 'quaternion':
                    csvwriter.writerow([jointpos, fkpos, ','.join([str(i) for i in fk.rot])])
                elif args.orient == 'matrix':
                    csvwriter.writerow([jointpos, fkpos, ','.join([str(i) for i in fk.rotation_matrix])])
                else:
                    csvwriter.writerow([jointpos, fkpos])

                i += 1
                pbar.update(1)

        pbar.close()
