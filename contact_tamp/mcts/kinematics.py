from typing import Any, List
from numpy.core.multiarray import array as array
import pinocchio
import time
import numpy as np
import mujoco
from utils.visuals import position_3d_callback
import os
from numpy.linalg import norm
from multiprocessing import Pool
from scipy.linalg import solve 

from mj_pin_wrapper.abstract.robot import QuadrupedWrapperAbstract
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from robots.configs import Go2Config
        
class QuadrupedKinematicFeasibility():
    # Greedy parameters for IK solver
    EPS    = 3.5e-2
    IT_MAX = 150
    DT     = 0.4
    DT_INCR   = 1. / IT_MAX
    DAMP   = 1e-14
    CLIP_LIN_VELOCITY = 0.045
    CLIP_ANG_VELOCITY = 0.35
    CLIP_J_VELOCITY = 0.4
    CLIP_ARRAY = np.array(
        [CLIP_LIN_VELOCITY] * 3 +
        [CLIP_ANG_VELOCITY] * 3 +
        [CLIP_J_VELOCITY] * 12
    )
    """
    Helper class to check if kinematic feasibility 
    for a quadruped robot.
    """
    def __init__(self,
                 robot: QuadrupedWrapperAbstract,
                 num_threads: int = -1,
                 **kwargs) -> None:
        self.robot = robot
        self.robot.reset()
        
        # Optionals
        optional_args = {
            "eps" : QuadrupedKinematicFeasibility.EPS,
            "it_max" : QuadrupedKinematicFeasibility.IT_MAX,
            "dt" : QuadrupedKinematicFeasibility.DT
        }
        optional_args.update(kwargs)
        for k, v in optional_args.items(): setattr(self, k, v)
        
        # Get initial position
        self.q0, _ = self.robot.get_pin_state()
        
        # Maximum number of cores if not specified
        if num_threads < 1:
            num_threads = len(os.sched_getaffinity(0)) // 2
        self.num_threads = num_threads
        
        # Max reachability of the feet
        self.max_reach = self._compute_reachability()

        # Load collision geometries
        self.geom_model = pinocchio.buildGeomFromUrdf(
            self.robot.pin_model,
            self.robot.path_urdf,
            pinocchio.GeometryType.COLLISION)
        
        # Add collisition pairs
        self.geom_model.addAllCollisionPairs()
        self.geom_data = pinocchio.GeometryData(self.geom_model)
        
        self.configuration = []
            
        feet_pin_frame_name = robot.pin_feet_frame_name       
        self.frame_ids_feet = [self.robot.pin_model.getFrameId(frame_name) for frame_name in feet_pin_frame_name]
    
    def _compute_reachability(self) -> float:
        """
        Compute the maximum distamce that the robot can 
        possibely reach.

        Returns:
            float: Maximum distance.
        """
        # Go in neutral position
        q_old, _ = self.robot.get_pin_state()
        q_neutral = np.zeros_like(q_old)
        self.robot.reset(q_neutral)
        
        # We assume legs are straight in neutral position
        feet_position_world = self.robot.get_pin_feet_position_world()
        max_reach = np.linalg.norm(feet_position_world, axis=-1).max()
        
        # reset robot
        self.robot.reset(q_old)
        
        return max_reach
        
    def is_feasible(self,
                    desired_feet_pos: np.ndarray,
                    allow_crossed_legs: bool = False,
                    check_collision: bool = True,
                    check_inverse_kinematics: bool = True,
                    ) -> np.ndarray:
        """
        Check if the feet can reach the desired positions.
        The robot position is assumed to be at the center of the 4 desired
        feet positions.

        Args:
            - desired_feet_pos (np.ndarray): Feet positions to reach. Shape [N, 4, 3].
            - allow_crossed_legs (bool, optional): Allow positions where the legs are crossing. Defaults to False.
            - check_collision (bool): If solution found, check self-collision.
            - check_inverse_kinematics (bool): Check if kinematically feasible with inverse kinematics.
        Returns:
            np.ndarray: Quadruped can reach the 4 contact locations simultaneously. Shape [Nr].
        """
        # Initialize feasible array to True for all positions
        feasible = np.full(len(desired_feet_pos), True, dtype=bool)

        # Check reachability and update feasibility
        # reachable = self._check_reachability(desired_feet_pos, scale_reach)
        # feasible[~reachable] = False

        # Check legs crossing condition if necessary
        if not allow_crossed_legs:
            feasible[feasible] &= self._check_cross_legs(desired_feet_pos[feasible])

        # Perform inverse kinematics on feasible positions
        if check_inverse_kinematics:
            ik_feasible, configurations = self._inverse_kinematics(desired_feet_pos[feasible])
            feasible[feasible] &= ik_feasible

            # Check for collisions if necessary
            if check_collision:
                feasible[feasible] &= self._self_check_collision(configurations)
  
        return feasible
        
    def reachable_locations(self,
                            current_location: np.array,
                            foot_pos: np.ndarray,
                            scale_reach: float = 0.6,
                            ) -> np.ndarray:
        """
        Returns contact locations reachable by the robot.

        Args:
            - current_location (np.ndarray): Current base location in world. Shape [3]
            - feet_pos (np.ndarray): Feet positions as array. Shape [N, 3]
            - scale_reach (foat): Scale the reach of the robot to prune distant locations.

        Returns:
            np.ndarray: Index of the reachable foot positions. Shape [Nr]
        """
        # Compute distance to center of the feet
        feet_pos_centered = foot_pos - current_location[np.newaxis, :]        # [N, 3]
        distance_center_to_feet = np.linalg.norm(feet_pos_centered, axis=-1)  # [N, 1]
        
        # True if a foot if reachable
        reachable = distance_center_to_feet < self.max_reach * scale_reach

        # Reachable contact locations index
        reachable_id = np.nonzero(reachable)[0]
        
        return reachable_id

    def _check_cross_legs(self,
                          feet_pos: np.ndarray,
                          ) -> np.ndarray:
        """
        Checks if the given feet positions result in a configuration where
        legs are crossed.

        Args:
            feet_pos (np.ndarray): feet positions in world frame. Shape [N, 4, 3]

        Returns:
            np.ndarray: Feet positions where legs are not crossing. Shape [Nc, 4, 3]
        """
        #
        # FL -- FR
        #  |    |
        #  |    |
        #  |    |
        # RL -- RR
        #
        # 1) Lines FL-RL and FR-RR should never cross
        # 2) Lines FL-FR and RL-RR should never cross
        #
        # Algo from https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
        
        if feet_pos.shape[0] == 0: return np.empty((0,), dtype=bool)
        
        A, B, C, D = np.split(feet_pos, 4, axis=1) # FL, FR, RL, RR

        D_A = D - A
        D_B = D - B
        C_A = C - A
        C_B = C - B
        B_A = B - A
        D_C = D - C
        B_C = - C_B
        
        # Counter Clockwise
        ccw_ACD = (D_A[:, :, 1] * C_A[:, :, 0] > C_A[:, :, 1] * D_A[:, :, 0])
        ccw_BCD = (D_B[:, :, 1] * C_B[:, :, 0] > C_B[:, :, 1] * D_B[:, :, 0])
        ccw_ABC = (C_A[:, :, 1] * B_A[:, :, 0] > B_A[:, :, 1] * C_A[:, :, 0])
        ccw_ABD = (D_A[:, :, 1] * B_A[:, :, 0] > B_A[:, :, 1] * D_A[:, :, 0])
        ccw_CBD = (D_C[:, :, 1] * B_C[:, :, 0] > B_C[:, :, 1] * D_C[:, :, 0])
        ccw_ACB = (B_A[:, :, 1] * C_A[:, :, 0] > C_A[:, :, 1] * B_A[:, :, 0])
        
        # Checks
        check1 = (ccw_ACD != ccw_BCD) & (ccw_ABC != ccw_ABD)
        check2 = (ccw_ABD != ccw_CBD) & (ccw_ACB != ccw_ACD)

        # Check if A, B, C or D are not at the same locations
        non_zero = lambda a : np.sum(a, axis=-1) != 0.
        id_different_locations = (
            non_zero(D_A) &
            non_zero(D_B) &
            non_zero(D_C) &
            non_zero(C_A) &
            non_zero(C_B) &
            non_zero(B_A)
        )
        
        id_not_crossing = np.bitwise_not(check1 | check2).squeeze() & id_different_locations.squeeze()

        return id_not_crossing

    @staticmethod
    def _inverse_kinematics_single(
        pin_model,
        pin_data,
        q0: np.ndarray,
        desired_feet_pos: np.ndarray,
        frame_ids_feet: List[int],
        dt: float = 0.3,
        eps: float = 1.0e-2,
        it_max: int = 150
        ) -> np.ndarray:
        """
        Perform closed-loop inverse kinematics (CLIK)
        to check if desired_feet_pos are reachable for only one set of positions.

        Args:
            - desired_feet_pos (np.ndarray): Desired feet positions. Shape [4, 3]
        Returns:
            bool: Reachable feet positions using CLIK.
        """
        # From https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_d-inverse-kinematics.html
        
        success = False
        i = 0
        dt_ = dt
        q = q0.copy()
        q[:3] = np.mean(desired_feet_pos, axis=0)
        q[2] += q0[2] + 0.1

        # Desired feet position SE3
        assert desired_feet_pos.shape == (4, 3)
        oMdes_feet = [pinocchio.SE3(np.eye(3), pos) for pos in desired_feet_pos]

        while True:
            if i >= it_max:
                success = False
                break

            pinocchio.forwardKinematics(pin_model, pin_data, q)
            pinocchio.updateFramePlacements(pin_model, pin_data)

            err = np.zeros(6 * len(frame_ids_feet))
            J_full = np.zeros((6 * len(frame_ids_feet), pin_model.nv))

            for k, (frame_id, oMdes) in enumerate(zip(frame_ids_feet, oMdes_feet)):
                oMcurrent = pin_data.oMf[frame_id]
                dMi = oMdes.actInv(oMcurrent)
                err[6*k:6*k+3] = pinocchio.log(dMi).vector[:3] # Just position error
                J_full[6*k:6*(k+1), :] = pinocchio.computeFrameJacobian(pin_model, pin_data, q, frame_id)
            
            if norm(err) < eps:
                success = True
                break

            v = - J_full.T.dot(solve(J_full.dot(J_full.T) + \
                  QuadrupedKinematicFeasibility.DAMP * np.eye(J_full.shape[0]), err))
            v = np.clip(v, -QuadrupedKinematicFeasibility.CLIP_ARRAY, QuadrupedKinematicFeasibility.CLIP_ARRAY)
            v[3:6] += np.random.randn(3) / 50. # Hack to make the orientation more stable
            q = pinocchio.integrate(pin_model, q, v * dt_)

            dt_ += QuadrupedKinematicFeasibility.DT_INCR
            i += 1
            
        return (success, q)
    
    def _inverse_kinematics(self, desired_feet_pos: np.ndarray) -> np.ndarray:
        """
        Perform closed-loop inverse kinematics (CLIK)
        to check if desired_feet_pos are reachable.

        Args:
            - desired_feet_pos (np.ndarray): Desired feet positions. Shape [N, 4, 3]
        Returns:
            np.ndarray: Reachable feet positions using CLIK.
        """
        if desired_feet_pos.shape[0] == 0: return np.empty((0,), dtype=bool), np.empty((0,), dtype=np.float32)
        
        with Pool(processes=self.num_threads) as pool:
            # Bool array
            ik_feasible_q = pool.starmap(
                # Function
                self._inverse_kinematics_single,
                # Args
                [
                    (self.robot.pin_model,
                    self.robot.pin_data,
                    self.q0,
                    feet_pos,
                    self.frame_ids_feet,
                    self.dt,
                    self.eps,
                    self.it_max,)
                    for feet_pos in desired_feet_pos
                ]
            )
        
            # Split into two separate lists using list comprehensions
            bool_list = [item[0] for item in ik_feasible_q]
            array_list = [item[1] for item in ik_feasible_q if item[0]]

            # Convert the lists to NumPy arrays
            ik_feasible = np.array(bool_list)
            q_array = np.array(array_list)

        return ik_feasible, q_array   

    def _self_check_collision(self, q_array: np.ndarray) -> np.ndarray:
        """
        Checks if the robot is in self collision for the given 
        configuration.
        Multi threaded version.
        
        Args:
            - q_array (np.ndarray): Given robot configurations. Shape [N, nq]
            
        Returns:
            np.ndarray: True if self-collision. Shape [N]
        """
        if q_array.shape[0] == 0: return np.empty((0,), dtype=bool)

        collision_free_list = []
        for q in q_array:

            collision_free = True
            
            pinocchio.computeCollisions(
                self.robot.pin_model,
                self.robot.pin_data,
                self.geom_model,
                self.geom_data,
                q,
                False)

            for cr in self.geom_data.collisionResults: 
                if cr.isCollision():
                    collision_free = False
                    break
                
            collision_free_list.append(collision_free)

            if collision_free:
                self.configuration.append(q)
        # Bool array
        return np.array(collision_free_list, dtype=bool)


if __name__ == "__main__":
    
    cfg = Go2Config

    ### Paths
    URDF_path,\
    xml_string,\
    package_dir = RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir, models_path="./robots")

    ### Load robot
    robot = QuadrupedWrapperAbstract(
        URDF_path,
        xml_string,
        package_dir,
        rotor_inertia=cfg.rotor_inertia,
        gear_ratio=cfg.gear_ratio,
        foot_size=cfg.foot_size,
        )
    robot.reset()

    model  = robot.pin_model
    data  = robot.pin_data
    
    ### Create desired feet positions
    feet_pin_frame_name = robot.pin_feet_frame_name       
    frame_ids_feet = [model.getFrameId(frame_name) for frame_name in feet_pin_frame_name]
    feet_position_world = np.expand_dims(robot.get_pin_feet_position_world(), 0)
    
    N = 1000
    scale = 0.03
    displacement = np.random.randn(N, 4, 3)  * scale
    desired_feet_position = feet_position_world + displacement

    ### Check if the position is Kinematically feasible
    kinematics = QuadrupedKinematicFeasibility(robot, 10)
    
    t = time.time()
    feasible = kinematics.is_feasible(desired_feet_position, check_collision=True)
    run_time = (time.time() - t)
    run_time_it_ms = run_time * 1000 / N
    
    print("Total run time", run_time, "s")
    print(run_time_it_ms, "ms per desired pos")
    print("Kinematically feasible:", np.sum(feasible) / N, "%")
    
    # Render the first feasible solution
    q = kinematics.configuration[0]
    id = np.where(feasible)[0][0]
    mj_q = q.copy()
    mj_q[3:7] = q[6], q[3], q[4], q[5]
    robot.reset(mj_q)

    with mujoco.viewer.launch_passive(robot.mj_model, robot.mj_data) as viewer:
        
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        
        viewer.sync()
        sim_start_time = time.time()
        while (viewer.is_running()):

            viewer.sync()
            position_3d_callback(viewer, desired_feet_position[id])
            time.sleep(1e-3)
