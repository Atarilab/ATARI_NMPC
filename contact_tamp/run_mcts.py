import numpy as np
from mcts.mcts_stepping_stones import MCTSSteppingStonesKin

if __name__ == "__main__":

    start = [23, 9, 21, 7]
    goal = [27, 13, 25, 11]
   
   # get current random state of numpy
    state = np.random.get_state()
        
    ### Load robot
    robot = None
    
    ### Controller
    optimzier = None

    mcts = MCTSSteppingStonesKin(
        simulator,
        simulation_steps=2,
        alpha_exploration=0.0,
        C=0.01,
        W=5.,
        max_solution_search=3,
        print_info=True,
        n_threads_kin=1,
        n_threads_sim=1,
        use_inverse_kinematics=False,
    )

    mcts.search(start, goal, num_iterations=10000)
    
    for fn_name, timings in mcts.get_timings().items():
        print(fn_name, timings)
    
    print(mcts.statistics)
    print(mcts.solutions)

    # Run Visualization