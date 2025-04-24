from typing import List
from mj_pin.simulator import Simulator
from scene.primitives import Surface
import yaml
import os

SCALE_NORMAL = 0.08
RADIUS = 0.008
N_SHPERE = 8
SCENE_FILE = "scene.yaml"

# Viz surfaces
def vis_surfaces_normal(sim : Simulator, surfaces : List[Surface]):
    for s in  surfaces:
        # normal and center
        for i in range(N_SHPERE):
            normal_v = s.center + s.normal * SCALE_NORMAL * (i/N_SHPERE)
            # center
            if i == 0:
                sim.edit.add_sphere(normal_v, RADIUS*2, color="black")
            # normal
            sim.edit.add_sphere(normal_v, RADIUS, color="red")
            
def save_to_yaml(dir_path : str, data : dict):
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, SCENE_FILE)
    
    with open(file_path, "w") as file:
        yaml.safe_dump(data, file, default_flow_style=False)
        
def load_from_yaml(dir_path :str) -> dict:
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, SCENE_FILE)
    
    with open(file_path, "r") as file:
        d = yaml.safe_load(file)
    return d

def scene_file_exists(dir_path:str) -> bool:
    return os.path.exists(
        os.path.join(dir_path, SCENE_FILE)
    )