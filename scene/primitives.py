import numpy as np
import mujoco

def euler_to_rot(euler_xyz) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, euler_xyz, "xyz")
    mat_flat = np.zeros(9)
    mujoco.mju_quat2Mat(mat_flat, q)
    return mat_flat.reshape(3, 3, order="A")
    
class Surface:
    """Represents a planar surface with a center, normal, and size."""
    def __init__(self, center: np.ndarray, normal: np.ndarray, rot : np.ndarray, size_x: float, size_y: float):
        self.center = np.array(center)
        # Normalized normal
        self.normal = np.round(np.array(normal) / np.linalg.norm(normal), 4)
        # Rotation matrix
        self.rot = np.round(rot, 4)
        self.size_x = size_x
        self.size_y = size_y

    def __repr__(self):
        return f"Surface(center={self.center}, normal={self.normal}, R={self.rot}, size_x={self.size_x}, size_y={self.size_y})"

class Box:
    """Extracts surfaces from a rotated box given its position, size, and orientation."""
    def __init__(self, pos: list, size: list, euler: list):
        """
        :param pos: Box center position [x, y, z]
        :param size: Half-size of the box [dx, dy, dz]
        :param euler: Euler angles [roll, pitch, yaw] in radians
        """
        self.pos = np.array(pos)
        self.size = np.array(size)
        self.rot = euler_to_rot(euler)
        
    def get_surfaces(self):
        """Returns a list of six rotated surfaces (center, normal, size_x, size_y) of the box."""
        dx, dy, dz = self.size  # Half-dimensions of the box

        # Define local surface centers and normals before rotation
        local_surfaces = [
            ([0., 0., dz], np.eye(3), dx, dy),   # Top: Normal is [0, 0, 1], identity rotation matrix
            ([0., 0., -dz], np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), dx, dy),  # Bottom: Normal is [0, 0, -1]
            ([dx, 0., 0.], np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), dy, dz),   # Front: Normal is [1, 0, 0]
            ([-dx, 0., 0.], np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]), dy, dz),  # Back: Normal is [-1, 0, 0]
            ([0., dy, 0.], np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), dx, dz),   # Right: Normal is [0, 1, 0]
            ([0., -dy, 0.], np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), dx, dz),  # Left: Normal is [0, -1, 0]
        ]
        # Apply rotation to centers and normals
        # Compute size in world frame
        rotated_surfaces = [
            Surface(
                center=self.pos + self.rot @ center, 
                normal=self.rot @ R[:, -1],
                rot=self.rot @ R,
                size_x=np.dot(np.array([size_x, 0., 0.]), self.rot[0, :]), 
                size_y=np.dot(np.array([0., size_y, 0.]), self.rot[1, :])
            ) for center, R, size_x, size_y in local_surfaces
        ]

        return rotated_surfaces

if __name__ == "__main__":
    # Example usage
    edge = 0.2
    height = 0.3
    euler_angles = [0.0, np.pi / 6, np.pi / 4]  # Roll, pitch, yaw in radians
    
    box = Box(
        pos=[0., 0., height], 
        size=[edge, edge, height],
        euler=euler_angles,
    )

    surfaces = box.get_surfaces()

    for surface in surfaces:
        print(surface)
