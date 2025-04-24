import numpy as np


ROBOT_NAME = "go2"
OBJECT_NAME = "Go2"
P_IMU_IN_VICON = np.array([ 0.0,  0.0625, -0.02])
R_IMU_IN_VICON = np.array(
    [[ 0.99420275, -0.07009067, -0.08153646],
    [ 0.06836363,  0.99737686, -0.02378686],
    [ 0.08298981,  0.01807484,  0.99638647],]
    )

STAND_UP_JOINT_POS = np.array([
    0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763,
    0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763
], dtype=float)

#    0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8,
STAND_DOWN_JOINT_POS = np.array([
    0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455,
    1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
],dtype=float)

P_IMU_IN_BASE = np.array([-0.02557, 0, 0.04232])
R_IMU_IN_BASE = np.eye(3)

CONTROL_FREQ = 100
Kp = 30
Kd = 2.25

# Multiply gain on the real robot
scale_gains = 1.5