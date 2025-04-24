import subprocess
import time
import os
import psutil
import sys

def kill_other_python3():
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if (
                proc.info['pid'] != current_pid and
                'python3' in proc.info['name'].lower() and
                'python3' in ' '.join(proc.info['cmdline'])
            ):
                proc.terminate()  # or proc.kill() for force
                print(f"Killed: PID {proc.pid} CMD: {' '.join(proc.info['cmdline'])}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

def update_config_file( scene : str = "", use_joystick : int = 1, simulate_dt = 1/500, viewer_dt = 1/12):
    config_path = "/home/atari/unitree_mujoco/simulate_python/config.py"

    try:
        with open(config_path, "r") as file:
            lines = file.readlines()

        with open(config_path, "w") as file:
            for line in lines:
                if line.startswith("USE_JOYSTICK"):
                    file.write(f"USE_JOYSTICK = {use_joystick}\n")
                elif line.startswith("ROBOT_SCENE") and scene:
                    file.write(f"ROBOT_SCENE = '{scene}'\n")
                elif line.startswith("SIMULATE_DT"):
                    file.write(f"SIMULATE_DT = {float(simulate_dt)}\n")
                elif line.startswith("VIEWER_DT"):
                    file.write(f"VIEWER_DT = {float(viewer_dt)}\n")
                else:
                    file.write(line)
        
        print(f"Updated config file: USE_JOYSTICK={use_joystick}, SCENE='{scene}'")
    except Exception as e:
        print(f"Error updating config file: {e}")
    
def run_unitree_mujoco_simulation(scene : str, use_joystick : int = 1):
    path = "/home/atari/unitree_mujoco/simulate_python/"
    script = "unitree_mujoco.py"
    

    try:
        kill_other_python3()
        time.sleep(0.1)
        update_config_file(scene, use_joystick)
        subprocess.run(["python3", script], cwd=path, check=True)
        process = subprocess.Popen(["python3", script], cwd=path)
        process.wait()

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the simulation script: {e}")
            
if __name__ == "__main__":

    scene_path = "" if len(sys.argv) < 1 else sys.argv[1]
    use_joystick = True if len(sys.argv) < 2 else int(sys.argv[2])

    run_unitree_mujoco_simulation(scene_path, use_joystick)