# ATARI_NMPC
Atarilab NMPC

## Project Description
ATARI_NMPC is a Nonlinear Model Predictive Control (NMPC) framework designed for quadruped robots. It leverages the Acados solver for efficient optimization and Pinocchio for robot dynamics. The framework is modular, allowing easy adaptation to different quadruped robots by configuring URDF paths, gait parameters, and control settings.
This branch was created and modified to adapt NMPC for controlling bipedal robots in a similar manner. Incremental modifications have been made to some source files in the main and contact_tamp directories, primarily by inheriting and creating subclasses to redefine functions while preserving as much of the original quadruped-specific definitions and code implementations as possible. The specific modifications are clearly outlined in the latter part of the README.

## Project Dependencies
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


## Change for Biped
**Closed loop is not properly working yet, work in progress**

### mpc.py
Biped Configuration:
Switched from quadruped to biped by importing and using get_biped_config and BipedAcadosSolver, and updated the default gait name to "trot_biped".
Contact Planner Adjustments:
Modified parameters for the "raibert" contact planner, notably increasing the foot size from 0.0085 to 0.02 (with a TODO note for review).
State Reference Update:
Expanded base_ref_vel_tracking from a 6-element to a 12-element vector to match the biped model requirements.
### config_abstract.py
remove the assertion of W_acc is 12
### mpc_cost.py
Add cost weight for biped control
### mpc_gait.py
Add biped trot gait
### mpc_opt.py
Add MPCCyclicBiped
### utils.py
Add get_biped_config
### solver.py
Dynamics Model:
Replace the quadruped dynamics with a biped version to accommodate different state and contact models.
Solver & Config:
Switch from quadruped-specific acados solver and configuration to biped-specific ones.
Cost & Constraints:
Adjust cost weights and contact restrictions to reflect biped locomotion characteristics.
Reference Setup:
Update the computation of base and joint references, and adjust initial feet positions for the biped stance.
### main.py
Biped Model Update:
Changed feet frame names and file paths (URDF/XML) to reference the biped robot.
Mujoco Integration:
Created an MjModel from the biped XML and initialized the joint state accordingly.
Controller Initialization:
Instantiated the MPC controller with the biped-specific joint reference and configuration.
CLI & Mode Adjustments:
Updated default simulation mode and arguments for biped testing.
### contact_tamp
    Src folder:
    Add the model of G1 with removed hand
    point_contact.py:
    Add defination of plane contact as following changes
        Linear Constraint:
        Calculates the signed distance between the contact point and the plane along the plane normal, then extracts the tangential (projected) velocity error—similar to point contact.
        Angular Constraint:
        Computes a rotational error by taking the logarithm of the contact frame’s rotation matrix (pin.log3(R_foot)) and scales it by a gain. It then also considers the angular velocity error of the contact frame.
        Concatenation:
        Linear and angular components are vertically concatenated to form a comprehensive kinematic constraint that ensures both proper foot placement and orientation relative to the contact plane.
