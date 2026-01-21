import mujoco
import mujoco.viewer
import numpy as np
import time
from utils import point_to_vector_representation

def friction_cone_approximation(mu, n=4):

    height = 0.2
    radius = mu * height
    
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    points = np.array([
        [radius * np.cos(angle), radius * np.sin(angle), height] for angle in angles
    ])

    lengths = []
    rotations = []
    for i in range(len(points)):
        length, rotation_matrix = point_to_vector_representation(points[i])
        lengths.append(length)
        rotations.append(rotation_matrix)
    
    return lengths, rotations

cube_model_xml = """
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

# Load the model
model = mujoco.MjModel.from_xml_string(cube_model_xml)

# Initialize the MuJoCo model and data
data = mujoco.MjData(model)

# Simulation parameters
sim_step = 0
sim_dt = 0.1
model.opt.timestep = sim_dt

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    # Disable some rendering features for simplicity
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

    viewer.sync()
    print([attr for attr in dir(mujoco.renderer._enums.mjtCone) if not attr.startswith("__")])
    print([attr for attr in dir(mujoco._render.mjr_render) if not attr.startswith("__")])

    while viewer.is_running():

        # Reset the number of geometries
        viewer.user_scn.ngeom = 0

        # Define positions, sizes, and colors for the box and arrow
        positions = [
            np.array([0.0, 0.0, 0.0]),  # Position for the box
            np.array([-0.3, 0.3, 0.0]),  # Position for the arrow
        ]

        sizes = [
            [0.0, 0.0, 1.0],  # Box size
            [0.02, 0.02, 0.5],       # Arrow size
        ]

        colors = [
            [1, 0, 0, 1],  # Red for the box
            [0, 0, 1, 1],  # Blue for the arrow
        ]

        geom_types = [
            mujoco.mjtGeom.mjGEOM_LINE,   # Box
            mujoco.mjtGeom.mjGEOM_ARROW, # Arrow
        ]

        # Add the box and arrow to the scene
        for i, (pos, size, color, geom_type) in enumerate(zip(positions, sizes, colors, geom_types)):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=geom_type,
                size=size,
                pos=pos,
                mat=np.eye(3).flatten(),  # Identity matrix for orientation
                rgba=color
            )

        # Update the number of geometries in the scene
        viewer.user_scn.ngeom = len(positions)

        # Sync the viewer to display the new geometries
        viewer.sync()

        time.sleep(sim_dt)  # Slow down the loop for better visualization

