import numpy as np
import mujoco

def friction_cone_approximation(mu, p_cone, R_cone, n=4):

    height = 0.2
    radius = mu * height
    
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    T_cone_0 = np.eye(4)
    T_cone_0[:3, :3] = R_cone
    T_cone_0[:3, 3] = p_cone

    points = np.array([
        [radius * np.cos(angle), radius * np.sin(angle), height] for angle in angles
    ])
    points = T_cone_0 @ np.hstack([points, np.ones((n, 1))]).T
    points = points.T[:, :3]
    crown_origins = np.roll(points, 1, axis=0)

    wall_lengths = []
    wall_rotations = []
    for i in range(len(points)):
        length, rotation_matrix = point_to_vector_representation(points[i], p_cone)
        wall_lengths.append(length)
        wall_rotations.append(rotation_matrix)

    crown_lengths = []
    crown_rotations = []
    for i in range(len(points)):
        crown_length, crown_rotation = point_to_vector_representation(points[i], crown_origins[i])
        crown_lengths.append(crown_length)
        crown_rotations.append(crown_rotation)
    
    return wall_lengths, wall_rotations, crown_lengths, crown_rotations, crown_origins

def point_to_vector_representation(point, origin=[0, 0, 0]):
    """
    Mujoco represents vectors in the scn_geom api in terms of:
    - length
    - rotation matrix
    - origin
    This function converts a vector given its origin and tip in the world
    to the representation used by the mujoco scn_geom api.
    Args:
        point: The tip of the vector in the world frame.
        origin: The origin of the vector in the world frame.
    Returns:
        length: The length of the vector.
        rotation_matrix: The rotation matrix that aligns the vector with the world frame.
    """
    origin = np.array(origin)
    vector = np.array(point) - origin
    
    length = np.linalg.norm(vector)
    
    if length != 0:
        unit_vector = vector / length
    else:
        unit_vector = vector
    
    rotation_matrix = axis_to_rotation_matrix(unit_vector)
            
    return length, rotation_matrix

class scnGeomManager:
    def __init__(self, viewer):
        self.viewer = viewer

        self.viewer.user_scn.ngeom = 0
        self.total_geoms = 0


    def add_friction_cone(self, p_cone, R_cone, f_coeff, rgba=[1, 0, 0, 1.0]):
        """
        Adds a friction cone visualization at the given position and orientation.
        This is a bit hacky but cones are not exposed to the mujoco python API.
        There is some work to add cone geoms, and it is quite easy but you have to 
        modify the mujoco source code and recompile it. Anyways this is a quick hack
        to visualize cones via a wireframe. To see the issue on github related a proper 
        implementation of cone visualization see: 
        https://github.com/google-deepmind/mujoco/issues/667
        Args:
            p_cone: The position of the cone in the world frame.
            R_cone: The orientation of the cone in the world frame.
            f_coeff: The friction coefficient of the cone.
            rgba: The color of the cone.
        """

        wall_lengths, wall_rotations, crown_lengths, crown_rotations, origins = \
            friction_cone_approximation(f_coeff, p_cone, R_cone, n = 40)
        
        # Add the walls of the cone
        for i, (length, rotation) in enumerate(zip(wall_lengths, wall_rotations)):
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[i+self.total_geoms],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=[0.0, 0.0, length],
                pos=p_cone,
                mat=rotation.flatten(),
                rgba=rgba
            )

        # Add the crown of the cone
        for i, (length, rotation, origin) in enumerate(zip(crown_lengths, crown_rotations, origins)):
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[i+len(wall_lengths)+self.total_geoms],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=[0.0, 0.0, length],
                pos=origin,
                mat=rotation.flatten(),
                rgba=rgba
            )

        self.total_geoms += len(wall_lengths) + len(crown_lengths)

    def add_force_vector(self, position, force, length_scale=0.1, rgba=[0.2, 0.8, 0.2, 0.8]):
        """
        Adds a force vector visualization at the given position and direction.
        Args:
            position: The position of the force vector in the world frame.
            force: The force vector in the world frame.
            length_scale: The scale of the force vector.
            rgba: The color of the force vector.
        """
        
        length = np.linalg.norm(force) * length_scale
        direction = force / (np.linalg.norm(force) + 1e-6)
        rotation_matrix = axis_to_rotation_matrix(direction)
        rotation_flattened = rotation_matrix.flatten()

        mujoco.mjv_initGeom(
            self.viewer.user_scn.geoms[self.total_geoms],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, length],
            pos=position,
            mat=rotation_flattened,
            rgba=rgba
        )
        self.total_geoms += 1

    def add_box(self, position, size, rotation=np.eye(3), rgba=[0.5, 0.5, 0.5, 1.0]):
        """
        Adds a box at the given position, with the specified size and orientation.
        Args:
            position: The position of the box in the world frame (3D vector).
            size: The size of the box (3D vector, representing half-lengths in x, y, z).
            rotation: A 3x3 rotation matrix for the box orientation (default is identity).
            rgba: The color of the box (default is grey).
        """
        mujoco.mjv_initGeom(
            self.viewer.user_scn.geoms[self.total_geoms],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,  # Box size is defined by half-lengths in each dimension
            pos=position,  # The position of the box in the world frame
            mat=rotation.flatten(),  # Flatten the rotation matrix for Mujoco
            rgba=rgba  # Box color
        )
        self.total_geoms += 1

    def add_plane(self, position=[0, 0, 0], normal=[0, 0, 1], size=[20, 20, 0.01], rgba=[0.6796875, 0.79296875, 0.9765625, 1.0]):
        """
        Adds a plane at the given position with a normal and size. 
        By default, it creates a horizontal plane at the origin.

        Args:
            position: The position of the center of the plane (default is [0, 0, 0]).
            normal: The normal vector of the plane (default is pointing upwards [0, 0, 1]).
            size: The size of the plane (half-lengths in x and y directions, default is [1, 1, 0.01]).
            rgba: The color of the plane (default is grey).
        """
        # Create rotation matrix from normal vector
        rotation_matrix = axis_to_rotation_matrix(normal)
        rotation_flattened = rotation_matrix.flatten()

        mujoco.mjv_initGeom(
            self.viewer.user_scn.geoms[self.total_geoms],
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=size,  # Plane size (half-extent in x, y, and a small z component)
            pos=position,  # Plane center
            mat=rotation_flattened,  # Rotation matrix to align the plane
            rgba=rgba  # Color
        )
        self.total_geoms += 1

    def add_line(self, position=[0, 0, 0], rot=[1, 0, 0, 0, 1, 0, 0, 0, 1], 
                 size=[0.0, 0.0, 1.0], rgba=[0.6796875, 0.79296875, 0.9765625, 1.0]):
        """
        Adds a plane at the given position with a normal and size. 
        By default, it creates a horizontal plane at the origin.

        Args:
            position: The position of one end of the line.
            rot: A flattened rotation matrix (9,1).
            size: Length of the line ([1, 1, 0.01]).
            rgba: The color of the plane (default is grey).
        """

        mujoco.mjv_initGeom(
            self.viewer.user_scn.geoms[self.total_geoms],
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=size,
            pos=position,
            mat=rot,
            rgba=rgba
        )
        self.total_geoms += 1

    def add_frame(self, pos, xmat, size=0.1, axis=None):
        """
        Adds a frame at the given position and orientation.
        Args:
            pos: The position of the frame in the world frame.
            xmat: The orientation of the frame in the world frame (3,3 matrix).
            size: The size of the frame.
            axis: The axis in world frame. This option is useful to  
                  visualise joints.
        """
        xmat = np.array(xmat).reshape(3, 3)

        if axis is not None:
            jaxis_mat = axis_to_rotation_matrix(axis)
            xmat = jaxis_mat

        x_rot = axis_to_rotation_matrix(xmat[:, 0])
        y_rot = axis_to_rotation_matrix(xmat[:, 1])
        z_rot = axis_to_rotation_matrix(xmat[:, 2])

        self.add_line(pos, x_rot.flatten(), [0.0, 0.0, size], [1.0, 0.0, 0.0, 1.0])  # Red for X-axis
        self.add_line(pos, y_rot.flatten(), [0.0, 0.0, size], [0.0, 1.0, 0.0, 1.0])  # Green for Y-axis
        self.add_line(pos, z_rot.flatten(), [0.0, 0.0, size], [0.0, 0.0, 1.0, 1.0])  # Blue for Z-axis

    def clear(self):
        """
        Clears all objects managed by this class.
        """
        self.viewer.user_scn.ngeom = 0
        self.total_geoms = 0

    def update(self):
        """
        Updates the viewer to reflect changes.
        """
        self.viewer.user_scn.ngeom = self.total_geoms

def axis_to_rotation_matrix(axis, ref_axis=np.array([0, 0, 1])):
    """
    Converts a given axis to a rotation matrix that aligns the reference axis with the given axis.

    Args:
    axis (array-like): The target axis to align with.
    ref_axis (array-like): The reference axis to be aligned. Default is [0, 0, 1].

    Returns:
    numpy.ndarray: A 3x3 rotation matrix.
    """
    axis = np.array(axis)
    ref_axis = np.array(ref_axis)
    axis = axis / np.linalg.norm(axis)
    ref_axis = ref_axis / np.linalg.norm(ref_axis)

    # Check if vectors are already aligned
    if np.allclose(axis, ref_axis):
        return np.eye(3)

    v = np.cross(ref_axis, axis)
    c = np.dot(ref_axis, axis)
    s = np.linalg.norm(v)

    if s == 0:  # Vectors are either the same or opposite
        # If the vectors are opposite, we need to find a perpendicular axis for the 180-degree rotation
        perp_axis = np.array([1, 0, 0]) if not np.allclose(ref_axis, [1, 0, 0]) else np.array([0, 1, 0])
        perp_axis = perp_axis - perp_axis.dot(ref_axis) * ref_axis  # Make it perpendicular
        perp_axis /= np.linalg.norm(perp_axis)
        k = np.array([
            [0, -perp_axis[2], perp_axis[1]],
            [perp_axis[2], 0, -perp_axis[0]],
            [-perp_axis[1], perp_axis[0], 0]
        ])
        return np.eye(3) + k @ k * 2  # 180-degree rotation

    k = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    # Rodrigues' rotation formula
    rotation_matrix = np.eye(3) + k + ((1 - c) / (s ** 2)) * np.dot(k, k)
    return rotation_matrix