import pyvicon_datastream as pv
import numpy as np
import time
import pinocchio as pin

from sdk_controller.topics import TOPIC_HIGHSTATE

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_

from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.utils.thread import RecurrentThread

class ViconHighStatePublisher:
    def __init__(self,
                 vicon_ip : str,
                 object_name : str,
                 publish_freq : int = 100,
                 ):
        self.vicon_ip = vicon_ip
        self.object_name = object_name
        self.publish_freq = publish_freq
        self.hight_state_dt = 1.0 / self.publish_freq
        
        self.is_connected = False
        self.vicon_client = pv.PyViconDatastream()
        self.connect()

        if self.is_connected == True:
            self.vicon_client.enable_segment_data()
            self.vicon_client.set_stream_mode(pv.StreamMode.ServerPush)
            self.vicon_client.set_axis_mapping(pv.Direction.Forward, pv.Direction.Left, pv.Direction.Up)
            self.vicon_frame_rate = self.vicon_client.get_frame_rate()
            if self.vicon_frame_rate > 0:
                self.vicon_dt = (1.0 / self.vicon_frame_rate)
            else:
                self.vicon_dt = self.hight_state_dt * 2.
        else:
            raise RuntimeError(f"Failed to connect to Vicon at {self.vicon_ip}")
        
        self.high_state = unitree_go_msg_dds__SportModeState_()
        self.high_state_puber = ChannelPublisher(TOPIC_HIGHSTATE, SportModeState_)
        self.high_state_puber.Init()
        self.HighStateThread = RecurrentThread(
            interval=self.hight_state_dt, target=self.PublishHighState, name="base_highstate"
        )
        self.ViconClientThread = RecurrentThread(
            interval=self.vicon_dt, target=self.UpdateViconClient, name="vicon_client"
        )
        self.frame_number = -1
        self.t, self.prev_t, self.prev_prev_t = 0., 0., 0.
        self.p, self.prev_p, self.prev_prev_p = np.zeros(3), np.zeros(3), np.zeros(3)
        self.q, self.prev_q, self.prev_prev_q = np.zeros(4), np.zeros(4), np.zeros(4) # w, x, y, z
        self.v = np.zeros(3)
        self.w = np.zeros(3)
        
        self.start()
        
    def start(self):
        self.HighStateThread.Start()
        self.ViconClientThread.Start()
        self.HighStateThread
        
        timout = 5
        t = 0
        sleep = 0.1
        
        while t < timout:
            if self.frame_number > 0:
                print("Vicon HighState Publisher started")
                return
            time.sleep(sleep)
            t += sleep
            
        raise TimeoutError(f"Vicon HighState Publisher failed to start. No messages received.")       
        
    def connect(self, ip=None):
        if ip is not None: # set
            print(f"Changing IP of Vicon Host to: {ip}")
            self.vicon_ip = ip
        
        ret = self.vicon_client.connect(self.vicon_ip)
        if ret != pv.Result.Success:
            print(f"Connection to {self.vicon_ip} failed")
            self.is_connected = False
        else:
            print(f"Connection to {self.vicon_ip} successful")
            self.is_connected = True
        return self.is_connected
    
    def _update_angular_velocity(self):
        # local angular velocity
        dt = self.t - self.prev_prev_t
        self.prev_w = self.w

        # first order finite difference for quaternions
        self.w = (2 / dt) * np.array([
            self.prev_prev_q[0]*self.q[1] - self.prev_prev_q[1]*self.q[0] - self.prev_prev_q[2]*self.q[3] + self.prev_prev_q[3]*self.q[2],
            self.prev_prev_q[0]*self.q[2] + self.prev_prev_q[1]*self.q[3] - self.prev_prev_q[2]*self.q[0] - self.prev_prev_q[3]*self.q[1],
            self.prev_prev_q[0]*self.q[3] - self.prev_prev_q[1]*self.q[2] + self.prev_prev_q[2]*self.q[1] - self.prev_prev_q[3]*self.q[0]
            ])

        self.w[np.abs(self.w) < 0.02] = 0.0
        # Apply second-order filtering to linear velocity
        alpha = 0.3  # Smoothing factor (adjust as needed)
        self.w = alpha * self.w + (1 - alpha) * self.prev_w
        
    def _update_linear_velocity(self):
        # global linear velocity
        avg_dt = (self.t - self.prev_prev_t) / 2.
        # Second order finite difference
        self.prev_v = self.v
        self.v = (3 * self.p - 4 * self.prev_p + self.prev_prev_p) / (2 * avg_dt)

        # Apply second-order filtering to linear velocity
        alpha = 0.3  # Smoothing factor (adjust as needed)
        self.v = alpha * self.v + (1 - alpha) * self.prev_v
        
    def _update_p_q_t(self, new_p : np.ndarray, new_q : np.ndarray, new_t : float):
        # time
        self.prev_prev_t = self.prev_t
        self.prev_t = self.t
        self.t = new_t
        # position
        self.prev_prev_p = self.prev_p
        self.prev_p = self.p
        self.p = new_p / 1000.
        # quaternion
        self.prev_prev_q = self.prev_q
        self.prev_q = self.q
        self.q = new_q
        
    def UpdateViconClient(self):
        if self.is_connected == True:
            
            frame = self.vicon_client.get_frame()
            if frame == pv.Result.Success:
                frame_number = self.vicon_client.get_frame_number()
                # if new frame
                if frame_number > self.frame_number:
                    self.frame_number = frame_number
                        
                    subject_count = self.vicon_client.get_subject_count()

                    for subj_idx in range(subject_count):
                        subject_name = self.vicon_client.get_subject_name(subj_idx)
                        if subject_name != self.object_name: #Skip objects we are not interessted in
                            continue
                        
                        segment_count = self.vicon_client.get_segment_count(self.object_name)
                        for seg_idx in range(segment_count):

                            segment_name = self.vicon_client.get_segment_name(self.object_name, seg_idx)
                            segment_global_translation = self.vicon_client.get_segment_global_translation(self.object_name, segment_name)
                            segment_global_quaternion     = self.vicon_client.get_segment_global_quaternion(self.object_name, segment_name)
                            latency = self.vicon_client.get_latency_total()

                            if segment_global_translation is not None and segment_global_quaternion is not None:
                                t = time.time() - latency
                                self._update_p_q_t(segment_global_translation, segment_global_quaternion, t)

                                if self.prev_t > 0.:
                                    self._update_angular_velocity()
                                if self.prev_prev_t > 0.:
                                    self._update_linear_velocity()
                                    
                                return True
        return False
                            
    def PublishHighState(self):
        
        self.high_state.position[0] = self.p[0]
        self.high_state.position[1] = self.p[1]
        self.high_state.position[2] = self.p[2]
        
        self.high_state.velocity[0] = self.v[0]
        self.high_state.velocity[1] = self.v[1]
        self.high_state.velocity[2] = self.v[2]
        
        self.high_state.imu_state.quaternion[0] = self.q[0]
        self.high_state.imu_state.quaternion[1] = self.q[1]
        self.high_state.imu_state.quaternion[2] = self.q[2]
        self.high_state.imu_state.quaternion[3] = self.q[3]
        
        self.high_state.imu_state.gyroscope[0] = self.w[0]
        self.high_state.imu_state.gyroscope[1] = self.w[1]
        self.high_state.imu_state.gyroscope[2] = self.w[2]

        self.high_state_puber.Write(self.high_state)
        
class ViconRecorder(ViconHighStatePublisher):
    def __init__(self, vicon_ip : str, object_name : str):
        self.data = {
            "p": [],
            "q": [],
            "t": [],
            "v": [],
            "w": [],
        }
        super().__init__(vicon_ip, object_name)

    
    def UpdateViconClient(self):
        # If new frame
        if super().UpdateViconClient():
            self.data["p"].append(self.p)
            self.data["q"].append(self.q)
            self.data["t"].append(self.t)
            self.data["v"].append(self.v)
            self.data["w"].append(self.w)
            return True
        return False
    
    def save_data(self, filename: str):
        np.savez(
            filename,
            p=np.array(self.data["p"]),
            q=np.array(self.data["q"]),
            t=np.array(self.data["t"]),
            v=np.array(self.data["v"]),
            w=np.array(self.data["w"]),
        )
    
    def plot_data(self):
        import matplotlib.pyplot as plt

        # Convert data to numpy arrays for easier handling
        p = np.array(self.data["p"])
        q = np.array(self.data["q"])
        t = np.array(self.data["t"])
        v = np.array(self.data["v"])
        w = np.array(self.data["w"])

        # Plot position (p)
        plt.figure(figsize=(10, 6))
        plt.plot(t, p[:, 0], label="p_x")
        plt.plot(t, p[:, 1], label="p_y")
        plt.plot(t, p[:, 2], label="p_z")
        plt.title("Position (p) vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.legend()
        plt.grid()
        plt.show()

        # Plot quaternion (q)
        plt.figure(figsize=(10, 6))
        plt.plot(t, q[:, 0], label="q_w")
        plt.plot(t, q[:, 1], label="q_x")
        plt.plot(t, q[:, 2], label="q_y")
        plt.plot(t, q[:, 3], label="q_z")
        plt.title("Quaternion (q) vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Quaternion")
        plt.legend()
        plt.grid()
        plt.show()

        # Plot linear velocity (v)
        plt.figure(figsize=(10, 6))
        plt.plot(t, v[:, 0], label="v_x")
        plt.plot(t, v[:, 1], label="v_y")
        plt.plot(t, v[:, 2], label="v_z")
        plt.title("Linear Velocity (v) vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.legend()
        plt.grid()
        plt.show()

        # Plot angular velocity (w)
        plt.figure(figsize=(10, 6))
        plt.plot(t, w[:, 0], label="w_x")
        plt.plot(t, w[:, 1], label="w_y")
        plt.plot(t, w[:, 2], label="w_z")
        plt.title("Angular Velocity (w) vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity (rad/s)")
        plt.legend()
        plt.grid()
        plt.show()
        
        
if __name__ == "__main__":
    VICON_IP = "192.168.123.100:801"
    OBJECT_NAME = "Go2"
    RECORD_TIME = 15
    
    ChannelFactoryInitialize(1, "lo")
    vicon = ViconRecorder(VICON_IP, OBJECT_NAME)
    time.sleep(RECORD_TIME)
    vicon.plot_data()
    
    FILE_NAME = "data_recording.npz"
    vicon.save_data(FILE_NAME)