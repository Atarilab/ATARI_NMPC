import yaml
import os

FILE_NAME = "sequence.yaml"
PHASE_KEY = "phases"
NODES_KEY = "nodes_per_phase"

def save_phase_sequence_to_yaml(dir_path : str, phase_sequence : list, nodes_per_phase : int = 0):
    os.makedirs(dir_path, exist_ok=True)
    # Save phase sequence to a YAML file
    if not os.path.isdir(dir_path):
        raise ValueError(f"{dir_path} should be a directory.")
    
    file_path = os.path.join(dir_path, FILE_NAME)
    data = {
        PHASE_KEY : phase_sequence,
        NODES_KEY : nodes_per_phase
    }
    with open(file_path, "w") as file:
        yaml.safe_dump(data, file, default_flow_style=False)
    # print("Sequence saved to ", file_path)
        
def load_phase_sequence_from_yaml(dir_path : str) -> tuple[list, int]:
    # Save phase sequence to a YAML file
    if not os.path.isdir(dir_path):
        raise ValueError(f"{dir_path} should be a directory.")
    
    file_path = os.path.join(dir_path, FILE_NAME)
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
        
    phase_sequence = [
        (phase[0], tuple(phase[1]), tuple(phase[2]))
        for phase 
        in data[PHASE_KEY]
    ]
    nodes_per_phase = data[NODES_KEY]
    return phase_sequence, nodes_per_phase