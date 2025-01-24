from pykin.robots.single_arm import SingleArm


def get_robot(name):
    robot_name = name
    file_path = '../assets/urdf/' + robot_name + '/' + robot_name + '.urdf'
    robot = SingleArm(file_path)
    return robot

robot = get_robot("tiago")
robot.show_robot_info()
