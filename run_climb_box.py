import numpy as np
from typing import Any, List
import time
import sys
import pinocchio as pin

from sdk_controller.abstract import SDKController
from sdk_controller.robots import Go2

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.utils.crc import CRC

from mpc_climb_box import mpc_close_loop, HEIGHT, EDGE, OFFSET, SIM_DT, AcyclicMPC, sequence
from search.utils.save import load_phase_sequence_from_yaml
from scene.primitives import Surface, Box

class MPC_SDK_ClimbBox(SDKController):
    def __init__(self,
                 simulate : bool,
                 mpc : AcyclicMPC,
                 robot_config,
                 solution_dir : str,
                 box_size_xyz : List[float],
                 xml_path = "",
                 ):
        self.mpc = mpc
        self.mpc.scale_joint = np.repeat([1.3, 1.15, 1.], 4)
        
        super().__init__(simulate, robot_config, xml_path)
        self.box_size_xyz = np.array(box_size_xyz)
        self.solution_dir = solution_dir
        self.is_contact_plan_set = False
        
        # No base orientation limit
        self.safety.base_orientation_limit = None
        self.safety.joint_limits = {}
        self.surfaces = []
        
    def get_contact_patch(
        self,
        contact_seq: np.ndarray,
        contact_patches: List[List[int]],
        surfaces: List[Surface],
    ):
        """
        Determines the contact surface properties (center, normal, size) for each foot at each time node.
        """
        n_feet, n_nodes = contact_seq.shape

        # Initialize output arrays with zeros
        surface_centers = np.zeros((n_feet, n_nodes, 3))
        surface_rot = np.zeros((n_feet, n_nodes, 3, 3))
        surface_sizes = np.zeros((n_feet, n_nodes, 2))
        
        for i_foot in range(n_feet):
            cnt_phase = 0
            last_in_cnt = False
            for i_node in range(n_nodes):
                if contact_seq[i_foot, i_node] == 1:  # Foot is in contact
                    last_in_cnt = True
                else:
                    # If break cnt, go to next phase
                    if last_in_cnt:
                        cnt_phase += 1
                    last_in_cnt = False
                    
                # Select a surface from the valid patches
                i = min(cnt_phase, len(contact_patches[i_foot])-1)
                id_surf = contact_patches[i_foot][i]
                surface = surfaces[id_surf]

                # Assign surface properties
                surface_centers[i_foot, i_node] = surface.center
                surface_rot[i_foot, i_node] = surface.rot
                surface_sizes[i_foot, i_node] = [surface.size_x, surface.size_y]             

        return surface_centers, surface_rot, surface_sizes
    
    def get_sequence_patches_from_path(self, path, node_per_phase : int):
        seq = np.array([list(phase[1]) for phase in path]).T.repeat(node_per_phase, axis=-1)
        seq = np.concatenate([seq, seq[:, None, -1]], axis=-1)
        
        n_eeff = len(path[-1][1])
        patches_dict = {i_eeff : [] for i_eeff in range(n_eeff)}
        last_cnt = {i_eeff : 0 for i_eeff in range(n_eeff)}
        for phase in path:
            _, cnt, patch = phase
            i_patch = 0
            for i_eeff, c in enumerate(cnt):
                if c:
                    if last_cnt[i_eeff] == 0:
                        patches_dict[i_eeff].append(patch[i_patch])
                last_cnt[i_eeff] = c
                i_patch += c
        patches = list(patches_dict.values())
        return seq, patches
    
    def udpate_surfaces_pos(self, x_offset : float):        
        W_T_B = np.eye(4, 4)
        W_T_B[:3, 3] = self._q[:3]
        W_T_B[:3, :3] = pin.Quaternion(
                w=self._q[3],
                x=self._q[4],
                y=self._q[5],
                z=self._q[6]).toRotationMatrix()
        
        B_T_Box = np.eye(4, 4)
        B_T_Box[0, 3] = x_offset + self.box_size_xyz[0] / 2.
        
        # Compute the box center position based on the base position
        W_T_Box = W_T_B @ B_T_Box
        
        W_T_Box[2, 3] = self.box_size_xyz[-1] / 2.
        center = np.round(W_T_Box[:3, 3], 3)
        euler = np.round(pin.rpy.matrixToRpy(W_T_Box[:3, :3]), 3)
        euler[:2] = 0.
        print(self._q[:3], center, euler)
        box = Box(center, self.box_size_xyz / 2., euler)
        top_box = box.get_surfaces()[0]
        plane = Surface(np.array([0., 0., 0.]), np.array([0., 0., 1.]), rot=np.eye(3), size_x=1e1, size_y=1e1)

        self.surfaces = [plane, top_box]
        
    def init_contact_plan(self):
        # Load contact plan from solution
        # Set MPC contact plan
        node_sequence, nodes_per_phase = load_phase_sequence_from_yaml(self.solution_dir)

        nodes_per_phase = 7
        start_phase = 0 # if node_sequence[0] else 1
        cnt_sequence, patches = self.get_sequence_patches_from_path(sequence[start_phase:], nodes_per_phase)

        patch_center, patch_rot, patch_size = self.get_contact_patch(cnt_sequence, patches, self.surfaces)
        self.mpc.set_cnt_plan(
            cnt_sequence,
            patch_center,
            patch_rot,
            patch_size
        )
        self.is_contact_plan_set = True
        print("Contact plan set.")
        
    def update_motor_cmd(self, time):
        if len(self.surfaces) == 0:
            self.udpate_surfaces_pos(OFFSET + 0.08)
            self.init_contact_plan()
            
        torques_ff = self.mpc._compute_torques_ff(time, self._q, self._v)
        if self.mpc.first_solve or not self.is_contact_plan_set:
            if not self.is_contact_plan_set:
                print("Contact plan needs to be set.")
                
            # Stand up
            phase = 1.
            for i in range(self.nu):
                self.cmd.motor_cmd[i].q = phase * self.robot_config.STAND_UP_JOINT_POS[i] + (
                    1 - phase) * self.robot_config.STAND_DOWN_JOINT_POS[i]
                self.cmd.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0
                self.cmd.motor_cmd[i].dq = 0.0
                self.cmd.motor_cmd[i].kd = 3.5
                self.cmd.motor_cmd[i].tau = 0.0
                
        else:
            step = self.mpc.plan_step - 1
            self.mpc.tau_full.append(torques_ff)
            scale = 1. if self.simulate else self.robot_config.scale_gains
            for i, tau in enumerate(torques_ff, start=6):
                i_act = self.joint_dof2act_id[i]
                i_act = self.joint_dof2act_id[i]
                self.cmd.motor_cmd[i_act].q = self.mpc.q_plan[step, i]
                self.cmd.motor_cmd[i_act].kp = self.mpc.scale_joint[i-6] * self.robot_config.Kp * scale
                self.cmd.motor_cmd[i_act].dq = self.mpc.v_plan[step, i]
                self.cmd.motor_cmd[i_act].kd = self.mpc.scale_joint[i-6] * self.robot_config.Kd * scale
                max_tau = self.safety.torque_limits[i_act]
                self.cmd.motor_cmd[i_act].tau = np.clip(tau, -max_tau, max_tau)

    def reset_controller(self):
        print("reset controller")
        self.mpc.reset()
        self.surfaces = []
        self.init_contact_plan()

input("Press enter to start")
runing_time = 0.0

VICON_IP = "192.168.123.100:801"
SOLUTION_DIR = "data/climb_box_success_0/close_loop_iteration_73"

if __name__ == "__main__":
    from sdk_controller.robots import Go2
    from sdk_controller.joystick import JoystickPublisher
    from sdk_controller.vicon_publisher import ViconHighStatePublisher
    
    if len(sys.argv) <2:
        ChannelFactoryInitialize(1, "lo")
        simulate = True
        
    else:
        ChannelFactoryInitialize(0, sys.argv[1])
        
        joystick = JoystickPublisher(device_id=0, js_type="xbox")
        vicon = ViconHighStatePublisher(
            vicon_ip=VICON_IP,
            object_name=Go2.OBJECT_NAME,
            publish_freq=Go2.CONTROL_FREQ,
        )
        simulate = False
    
    sdk_controller = MPC_SDK_ClimbBox(
        simulate,
        mpc_close_loop,
        Go2,
        SOLUTION_DIR,
        [EDGE, 2 * EDGE, HEIGHT],
        )

    dt = 1 / Go2.CONTROL_FREQ
    try:
        while True:
            step_start = time.perf_counter()
            
            sdk_controller.send_motor_command(round(runing_time, 3))
            
            if runing_time > 10.:
                sdk_controller.stand_up_running = False
                sdk_controller.controller_running = False
                sdk_controller.damping_running = True
            elif runing_time > 3.5:
                sdk_controller.stand_up_running = False
                sdk_controller.controller_running = True
            elif runing_time > 0.1:
                sdk_controller.stand_up_running = True
                
            runing_time += dt
            time_until_next_step = dt - (time.perf_counter() - step_start)

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
    except KeyboardInterrupt:
        print(mpc_close_loop.print_timings())
        mpc_close_loop.plot_traj("q")
        mpc_close_loop.plot_traj("v")
        mpc_close_loop.show_plots()
        pass