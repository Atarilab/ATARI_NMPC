# ATARI_NMPC
Atarilab NMPC

## Project Description
ATARI_NMPC is a Nonlinear Model Predictive Control (NMPC) framework designed for quadruped robots. It leverages the Acados solver for efficient optimization and Pinocchio for robot dynamics. The framework is modular, allowing easy adaptation to different quadruped robots by configuring URDF paths, gait parameters, and control settings.

## Project Dependencies

### Docker Image

To ease the installation process, a `Dockerfile` is provided [here](.devcontainer/Dockerfile).
It was configured to be used with the [Dev Container extension](https://code.visualstudio.com/docs/devcontainers/containers) of VSCode.
Once the repo opened in VSCode simply build the container (Ctrl+Shift+P, Rebuild Container Without Cache). Nothing more should be needed to setup the environment.

### Local Instal

- Conda environment is provided in `environment.yml`
    ```bash
    conda env create -n atari_nmpc -f environment.yml python=3.10
    ```
- [Acados](https://docs.acados.org/) with python interface.
    ```bash
    git clone https://github.com/acados/acados.git
    cd acados
    git submodule update --recursive --init
    ```
    ```bash
    mkdir -p build
    cd build
    cmake -DACADOS_WITH_QPOASES=ON ..
    make install -j4
    ```
    ```bash
    pip install -e ../../acados/interfaces/acados_template
    ```
    ```
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
    export ACADOS_SOURCE_DIR="<acados_root>"
    ```
- [mj_pin_utils](https://github.com/Atarilab/mj_pin_utils): Utils for MuJoCo simulator.
    ```bash
    git clone https://github.com/Atarilab/mj_pin_utils.git
    # In the Conda environment
    pip install -e ./mj_pin_utils
    ```

## Running the MPC

### Without controller

Simply run `main.py`. It will run the mpc real-time in close-loop on a forward trotting task.
One can push the robot in the MuJoCo viewer.
The first time you run it will take some time (minutes) as the solver has to be compiled. Once the MPC has been compiled once, you can set `recompile=False` in the [config_opt.py](mpc_controller/config/quadruped/mpc_opt.py) file.
```
python3 main.py
```

### Deploy on the real robot

The framework to deploy on the real robot can be tested in simulation first, simply by running a simulation node instead of the real robot. We used the [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) repo to do so.

**A Xbox joystick is required for the following steps!**
See the [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) repo to use a different controller.

#### Test in simulation first

1. First run the simulation node.
```
python3 run_unitree_mujoco.py
```
2. Then run the controller node.
```
python3 run_sdk_controller.py
```

On the Xbox controller:
- Y: stand up
- A: stand down
- B: damping mode
- X: run the controller
- joystick left: direction
- joystick right: orientation

#### On the real robot

Set the vicon IP in [run_sdk_controller.py](run_sdk_controller.py).

To run on the real robot, simply specify the name of the network card connected to the robot (here *enp3s0* as an example).
```
python3 run_sdk_controller.py enp3s0
```


## Configuration Files
### GaitConfig
Defines the gait parameters for the robot:
- `gait_name`: Name of the gait.
- `nominal_period`: Nominal period of the gait cycle.
- `stance_ratio`: Ratio of the gait cycle where the foot is in contact.
- `phase_offset`: Phase offset between legs.
- `nom_height`: Nominal height of the robot.
- `step_height`: Step height during the swing phase.

### MPCOptConfig
Defines the optimization parameters for the MPC:
- `time_horizon`: Time horizon for the optimization.
- `n_nodes`: Number of optimization nodes.
- `opt_dt_scale`: Time bounds between nodes.
- `replanning_freq`: Frequency of replanning.
- `real_time_it`: Enable real-time iterations.
- `enable_time_opt`: Enable time optimization.
- `enable_impact_dyn`: Enable impact dynamics.
- `cnt_patch_restriction`: Constrain end-effector locations within a patch.
- `opt_peak`: Use peak constraints.
- `max_iter`: Maximum SQP iterations.
- `warm_start_sol`: Warm start states and inputs with the last solution.
- `warm_start_nlp`: Warm start solver IP outer loop.
- `warm_start_qp`: Warm start solver IP inner loop.
- `hpipm_mode`: HPIPM mode.
- `recompile`: Recompile solver.
- `use_cython`: Use Cython in the solver.
- `max_qp_iter`: Maximum QP iterations for one SQP step.
- `nlp_tol`: Outer loop SQP tolerance.
- `qp_tol`: Inner loop interior point method tolerance.
- `Kp`: Gain on joint position for torque PD.
- `Kd`: Gain on joint velocities for torque PD.
- `use_delay`: Take into account replanning time.

### MPCCostConfig
Defines the cost parameters for the MPC:
- `W_e_base`: Terminal cost weights for base position, orientation, and velocity.
- `W_base`: Running cost weights for base position, orientation, and velocity.
- `W_joint`: Running cost weights for joint positions and velocities.
- `W_e_joint`: Terminal cost weights for joint positions and velocities.
- `W_acc`: Running cost weights for joint accelerations.
- `W_swing`: Running cost weights for end-effector motion.
- `W_cnt_f_reg`: Force regularization weights for each foot.
- `W_foot_pos_constr_stab`: Weight constraint for contact horizontal velocity.
- `W_foot_displacement`: Foot displacement penalization weights.
- `cnt_radius`: Default contact radius for contact restriction.
- `time_opt`: Time optimization cost.
- `reg_eps`: Regularization running cost.
- `reg_eps_e`: Regularization terminal cost.

## Main Script
The main script (`main.py`) initializes the simulation environment, loads the robot model, and sets up the MPC controller. It defines a `ReferenceVisualCallback` class for visualizing the reference trajectory and contact locations. The script runs the simulation for a specified duration and visualizes the results.

## MPC and Solver
The `LocomotionMPC` class in the `mpc_controller` module sets up the MPC controller using the provided configurations. It initializes the solver, sets up the reference trajectory, and updates the solver with the current state and contact plan. The `QuadrupedAcadosSolver` class in the `utils/solver.py` file extends the `AcadosSolverHelper` class to define the specific dynamics and cost functions for the quadruped robot.

## Implementing MPC on a New Robot
To implement the MPC on a new robot, follow these steps:
1. **Model**: Make sure that your MuJoCo and Pinocchio models can be loaded. The Pinocchio model should be without root joint as it will be added in the solver.
2. **Feet Frame Names**: Update the `feet_frame_names` parameter with the end-effector frame names of the new robot.
3. **Configurations**: Define a new `GaitConfig`, `MPCOptConfig`, `MPCCostConfig` with appropriate parameters for the new robot.
4. **Simulation Initialization**: Update the main script to load the new robot model and initialize the `LocomotionMPC` with the new configurations.

Use `print_info` argument of the MPC for debugging purposes.


## TODO

### Solver
- [ ] euler to tengent space orientation representation
- [ ] other contact model (e.g. humanoid feet)
- [ ] linearize cone constraint, see if performance improved
  
### Code optimzation
- [x] use acados `set_flat` to init the solver
- [ ] eventually C++ implementation

### MPC
- [ ] keyboard velocity control
- [ ] gait switching
- [ ] tune better `horizon`, `n_nodes`, and `qp_iter` in `MPCConfigOpt`. Seems that 1s horizon is too much.
- [ ] try with dt time optimization with `enable_time_opt = True` (this needs to be fixed)
- [ ] add a plot comparing planed state and robot state.
- [ ] add new robots 
