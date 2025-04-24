import os
from mj_pin.abstract import DataRecorder, VisualCallback
from mpc_controller.mpc import LocomotionMPC

### DataRecorder
class SteppingStonesDataRecorder_Visual(DataRecorder, VisualCallback):
    KEYS = ["time", "q", "v", "feet_pos_w", "target_cnt_w"]
    def __init__(self, mpc : LocomotionMPC, record_dir = "", record_step = 1):
        DataRecorder.__init__(self, record_dir, record_step)
        VisualCallback.__init__(self)
        self.mpc = mpc
        self.i_gait_cycle = 0
        self.sphere_radius = 0.01
        self.n_cnt_plan = len(self.mpc.contact_planner._contact_locations)
        
    def _new_gait_cycle(self) -> bool:
        """
        Returns true when sim step corresponds to a new gait
        cycle of the NMPC.
        """
        return self.mpc.current_opt_node >= self.i_gait_cycle * self.mpc.contact_planner.nodes_per_cycle
        
    def record(self, mj_data, **kwargs):
        # Record every new gait cycle
        if self._new_gait_cycle():           
            # Record state position
            self.data["q"].append(mj_data.qpos.copy())
            # Record state velocity
            self.data["v"].append(mj_data.qvel.copy())
            # Record feet position in world
            feet_pos_w = self.mpc.solver.dyn.get_feet_position_w()
            self.data["feet_pos_w"].append(feet_pos_w)
            # Record all future cnt locations in the plan, keep the last one
            i = min(self.i_gait_cycle, self.n_cnt_plan - 1)
            cnt_plan_full = self.mpc.contact_planner._contact_locations[i:]
            self.data["target_cnt_w"].append(cnt_plan_full)
            
            # Next gait cycle
            self.i_gait_cycle += 1
             
    def add_visuals(self, mj_data):
        i = min(self.i_gait_cycle, self.n_cnt_plan - 1)
        cnt_plan_full = self.mpc.contact_planner._contact_locations[i:]
        for pos_feet in cnt_plan_full:
            for i, pos in enumerate(pos_feet):
                self.add_sphere(pos, self.sphere_radius, self.colors.id(i))

    def reset(self) -> None:
        """
        Reset the recorder data.
        """
        self.data = dict.fromkeys(SteppingStonesDataRecorder_Visual.KEYS, list())

    def save(self) -> None:
        """
        Save the recorded data to a file in the specified directory.
        """
        if not self.record_dir:
            self.record_dir = os.getcwd()
        # Uncomment to save data
        # os.makedirs(self.record_dir, exist_ok=True)

        timestamp = self.get_date_time_str()
        file_path = os.path.join(self.record_dir, f"simulation_data_{timestamp}.npz")

        try:
            print(self.data["time"][:10])
            # Uncomment to save data
            # np.savez(file_path, **self.data)
            print(f"Data successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving data: {e}")