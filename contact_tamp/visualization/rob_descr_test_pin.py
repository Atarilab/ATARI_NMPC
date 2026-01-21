import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

def print_actuated_joints_with_limits(model):
    # Iterate over all joints in the model

    # List to store actuated joint IDs
    actuated_joint_ids = []

    # Loop over all joints in the model
    for joint_id, joint in enumerate(model.joints):
        # Check if the joint is actuated (i.e., if it's part of the generalized velocities 'v' vector)
        if joint_id > 0 and model.idx_vs[joint_id] != -1:  # Ignore the root joint (ID 0)
            if joint.shortname() not in ["JointModelFreeFlyer", "JointModelRevoluteUnbounded"]:
                actuated_joint_ids.append(joint_id)

    # Print the actuated joint IDs
    print("Actuated joint IDs:", actuated_joint_ids)

    for joint_id in actuated_joint_ids:
        joint_name = model.names[joint_id]
        # Find the corresponding link (frame) name associated with the joint
        link_name = None
        for frame in model.frames:
            if frame.type == pin.FrameType.BODY and frame.parentJoint == joint_id:
                link_name = frame.name
                break
        
        # Get position limits (qmin and qmax) and effort limit for the joint
        position_limit_min = model.lowerPositionLimit[robot.model.idx_qs.tolist()[joint_id]]
        position_limit_max = model.upperPositionLimit[robot.model.idx_qs.tolist()[joint_id]]
        effort_limit = model.effortLimit[robot.model.idx_vs.tolist()[joint_id]]

        # Print the joint information
        print(f"Actuated Joint {joint_id}: {joint_name} (Link: {link_name})")
        print(f"Position Limits: [{position_limit_min}, {position_limit_max}]  Effort Limit: {effort_limit}")

if __name__ == "__main__":
    
    robot = load_robot_description("go2_description", pin.JointModelFreeFlyer())
    model = robot.model

    print_actuated_joints_with_limits(model)