import numpy as np
from typing import List
from mj_pin.simulator import Simulator
from scene.primitives import Box, Surface
from scene.utils import vis_surfaces_normal, save_to_yaml, load_from_yaml, scene_file_exists

SCENE_NAME = "climb_box"

def setup_scene(
    sim : Simulator,
    height : float = 0.1,
    edge : float = 0.4,
    offset : float = 0.38,
    save_dir : str = "",
    vis_normal : bool = False) -> List[Surface]:

    if save_dir:
        if not scene_file_exists(save_dir):
            save_to_yaml(
                save_dir,
                data={
                    "name" : SCENE_NAME,
                    "height" : height,
                    "edge" : edge,
                    "offset" : offset
                }
            )
        else:
            sim.edit.reset()
            data = load_from_yaml(save_dir)
            height = data["height"]
            edge = data["edge"]
            offset = data["offset"]
    
    ################## Box

    pos =   [offset + edge/2., 0., height/2.]
    size =  [edge/2., edge/1.5, height/2.]
    euler = [0., 0., 0.]
    box = Box(pos, size, euler)
    plane = Surface(np.array([0., 0., 0.]), np.array([0., 0., 1.]), rot=np.eye(3), size_x=1e1, size_y=1e1)

    all_surfaces = [plane] + box.get_surfaces()
    # floor, top, side
    AVAILABLE_SURFACES = [0, 1, 4]
    surfaces = [all_surfaces[i] for i in AVAILABLE_SURFACES]
    
    if vis_normal:
        vis_surfaces_normal(sim, all_surfaces)

    ################## Simulator
    sim.edit.add_box(pos, size, euler, name="goal")

    return surfaces