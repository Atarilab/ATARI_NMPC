import time
import mujoco
import mujoco.viewer
import numpy as np
import pinocchio
from .utils import scnGeomManager

from robot_descriptions.loaders.mujoco import load_robot_description

class KinematicRobotVisualizer:
    def __init__(self, model,  vis_joint_frames=True):
        self.model = model
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.scn_geom_manager = None
        self.static_objects = None
        self.dynamic_objects = {}
        self.show_joints = vis_joint_frames

    def initialize_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
        self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0

        self.model.vis.scale.framelength = 0.4
        self.model.vis.scale.framewidth = 0.02
        self.model.vis.scale.com = 0.1 

        self.viewer.sync()
        self.viewer.user_scn.ngeom = 0
        self.scn_geom_manager = scnGeomManager(self.viewer)

        # for i in range(self.model.ngeom):
        #     self.model.geom_rgba[i][3] = 0.3

    def add_static_objects(self, objects):
        """
        Initializes static objects in the scene.

        Args:
            objects: List of tuples containing object types ('mesh', 'box') and parameters.
        """
        self.static_objects = objects

    def add_dynamic_objects(self, objects):
        for obj_type, params in objects:
            if obj_type == 'box':
                pos, size, rotation, rgba = params
                self.dynamic_objects[obj_type] = {'position': np.array(pos), 
                                                  'size': np.array(size), 
                                                  'rotation': np.array(rotation), 
                                                  'rgba': rgba}

    def update_scene(self):
        mujoco.mj_fwdPosition(self.model, self.data)
        self.scn_geom_manager.clear()
        self._add_world_frame()
        if self.show_joints:
            self._add_joint_frames()
        if self.static_objects is not None:
            self._initialize_static_objects()
        if self.dynamic_objects:
            self._update_dynamic_objects()
        self.scn_geom_manager.update()
        self.viewer.sync()

    def _add_world_frame(self):
        self.scn_geom_manager.add_frame([0.0, 0.0, 0.0], np.eye(3), 0.1)

    def _add_joint_frames(self):
        for i in range(self.model.njnt):
            pos = self.data.xanchor[i]
            xmat = self.data.xmat[i]
            joint_axis = self.data.xaxis[i]
            self.scn_geom_manager.add_frame(pos, xmat, 0.075, joint_axis)

    def _initialize_static_objects(self):
        for obj_type, params in self.static_objects:
            if obj_type == 'box':
                pos, size, rotation, rgba = params
                self.scn_geom_manager.add_box(pos, size, rotation, rgba)
            elif obj_type == 'mesh':
                mesh_path, pos, rotation, rgba = params
                self.scn_geom_manager.add_mesh(mesh_path, pos, rotation, rgba)

    def _update_dynamic_objects(self):
        # Update dynamic objects continuously
        for obj_type, attributes in self.dynamic_objects.items():
            if obj_type == 'box':
                pos = attributes['position']
                rotation = attributes['rotation']
                size = attributes['size']
                rgba = attributes['rgba']
                self.scn_geom_manager.add_box(pos, size, rotation, rgba)

    def visualize_configuration(self, base_pose, joint_positions):
        self.initialize_viewer()
        self.data.qpos[0:7] = base_pose
        self.data.qpos[7:] = joint_positions
        
        while self.viewer.is_running():
            self.update_scene()

    def visualize_trajectory(self, robot_trajectory=None, box_trajectory=None, 
                             dt_trajectory=None):
        self.initialize_viewer()
        trajectory_index = 0

        if robot_trajectory is None:
            trajectory_length = len(box_trajectory)
        else:
            trajectory_length = len(robot_trajectory)

        while self.viewer.is_running():
            if trajectory_index >= trajectory_length:
                trajectory_index = 0

            if robot_trajectory:
                base_pose, joint_positions = robot_trajectory[trajectory_index]

                self.data.qpos[0:7] = base_pose
                self.data.qpos[7:] = joint_positions

            if box_trajectory:
                box_pos, box_rot = box_trajectory[trajectory_index]
                if 'box' in self.dynamic_objects:
                    self.dynamic_objects['box']['position'] = np.array(box_pos)
                    self.dynamic_objects['box']['rotation'] = np.array(box_rot)
            self.update_scene()
            if dt_trajectory:
                time.sleep(dt_trajectory[trajectory_index])
            else:
                time.sleep(0.1)
            trajectory_index += 1

def vis_robot(com_trajectory, joint_trajectory, timestep_trajectory, box_trajectory=None):

    assert len(com_trajectory) == len(joint_trajectory)

    model = load_robot_description("go2_mj_description", variant="scene")

    visualizer = KinematicRobotVisualizer(model, vis_joint_frames=True)

    dynamic_objects = [
        ('box', ([0.75, 0, 0.1], [0.05, 0.05, 0.05], np.eye(3), [0.5, 0.5, 0.5, 1.0])),
    ]
    visualizer.add_dynamic_objects(dynamic_objects)

    # robot_base_pose = [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]
    # joint_positions = [1] * visualizer.model.nu
    # visualizer.visualize_configuration(robot_base_pose, joint_positions)

    robot_trajectory = []
    trajectory_steps = len(com_trajectory)
    for i in range(trajectory_steps):
        base_pos = com_trajectory[i][:3]
        base_mat = pinocchio.rpy.rpyToMatrix(com_trajectory[i][3],
                                                com_trajectory[i][4],
                                                com_trajectory[i][5])
        base_quat = pinocchio.Quaternion(base_mat)
        base_pose = [base_pos[0], base_pos[1], base_pos[2], 
                     base_quat.w, base_quat.x, base_quat.y, base_quat.z]
        joint_positions = joint_trajectory[i]
        robot_trajectory.append((base_pose, joint_positions))

    if box_trajectory:
        box_trajectory_ = []
        for i in range(trajectory_steps):
            box_pos = box_trajectory[i][:3]
            box_rot = pinocchio.rpy.rpyToMatrix(box_trajectory[i][3], 
                                                box_trajectory[i][4], 
                                                box_trajectory[i][5]) 
            box_trajectory_.append((box_pos, box_rot))
    else:
        box_trajectory_ = None

    visualizer.visualize_trajectory(robot_trajectory, box_trajectory_, timestep_trajectory) 

def vis_box(com_trajectory, timestep_trajectory: list=[]):

    world_model_xml = """
    <mujoco>
    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
        rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="2 2" texuniform="true"
        reflectance=".2"/>
    </asset>

    <worldbody>
        <light pos="0 0 1" mode="trackcom"/>
        <geom name="ground" type="plane" pos="0 0 0" size="20 20 .1" material="grid" solimp=".99 .99 .01" solref=".001 1"/>
    </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(world_model_xml)

    visualizer = KinematicRobotVisualizer(model, vis_joint_frames=True)

    dynamic_objects = [
        ('box', ([0.75, 0, 0.1], [0.05, 0.05, 0.05], np.eye(3), [0.5, 0.5, 0.5, 1.0])),
    ]
    visualizer.add_dynamic_objects(dynamic_objects)

    # robot_base_pose = [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]
    # joint_positions = [1] * visualizer.model.nu
    # visualizer.visualize_configuration(robot_base_pose, joint_positions)

    box_trajectory = []
    for i in range(len(com_trajectory)):
        box_pos = com_trajectory[i][:3]
        box_rot = pinocchio.rpy.rpyToMatrix(com_trajectory[i][3], 
                                            com_trajectory[i][4], 
                                            com_trajectory[i][5]) 
        box_trajectory.append((box_pos, box_rot))

    visualizer.visualize_trajectory(box_trajectory = box_trajectory, dt_trajectory= timestep_trajectory)
