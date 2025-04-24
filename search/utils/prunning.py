import numpy as np

def get_id_positions_within_radius(pos: np.ndarray, all_pos: np.ndarray, max_distance: float) -> list[np.ndarray]:
    """
    Return a list of stones indices with a distance to
    each position in <pos> below <max_distance>. 

    Args:
    pos (np.ndarray): positions to check reachability from. Shape [N, 3]
    all_pos (np.ndarray): all possible positions. Shape [M, 3]
    max_distance (float): maximum distance to consider reachable.

    Returns:
    np.ndarray: array of reachable indices for each position in pos. Shape [N, K]
    """
    # Compute distance from each pos to all_pos
    distance_matrix = np.linalg.norm(pos[:, np.newaxis, :] - all_pos[np.newaxis, :, :], axis=-1)  # [N, M]
    # Determine reachable positions
    reachable_matrix = distance_matrix < max_distance
    # Get indices of reachable positions
    reachable_indices = [np.nonzero(reachable_matrix[i])[0] for i in range(pos.shape[0])]
    return reachable_indices

def check_distance_to_center(node_pos : np.ndarray, max_distance : float) -> np.ndarray:
    """
    Checks if the given node positions have a distance to their centers below
    max_distance.
    
    Args:
        node_pos (np.ndarray): node positions in world frame. Shape [N, n_feet, 3]
    """      
    center_pos = np.mean(node_pos, axis=1)
    distance_to_center = np.linalg.norm(node_pos - center_pos[:, np.newaxis, :], axis=-1)
    within_distance = np.all(distance_to_center < max_distance, axis=1)
    return within_distance

def check_crossing(all_pos : np.ndarray) -> np.ndarray:
    """
    Checks if the given node positions result in a configuration where
    legs are crossed.

    Args:
        all_pos (np.ndarray): node positions in world frame. Shape [N, n_feet, 3]

    Returns:
        np.ndarray: node positions where legs are not crossing. Shape [Nc, n_feet, 3]
    """
    #
    # FL -- FR
    #  |    |
    #  |    |
    #  |    |
    # RL -- RR
    #
    # 1) Lines FL-RL and FR-RR should never cross
    # 2) Lines FL-FR and RL-RR should never cross
    #
    # Algo from https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    
    if all_pos.shape[0] == 0: return np.empty((0,), dtype=bool)
    
    A, B, C, D = np.split(all_pos, 4, axis=1) # FL, FR, RL, RR

    D_A = D - A
    D_B = D - B
    C_A = C - A
    C_B = C - B
    B_A = B - A
    D_C = D - C
    B_C = - C_B
    
    # Counter Clockwise
    ccw_ACD = (D_A[:, :, 1] * C_A[:, :, 0] > C_A[:, :, 1] * D_A[:, :, 0])
    ccw_BCD = (D_B[:, :, 1] * C_B[:, :, 0] > C_B[:, :, 1] * D_B[:, :, 0])
    ccw_ABC = (C_A[:, :, 1] * B_A[:, :, 0] > B_A[:, :, 1] * C_A[:, :, 0])
    ccw_ABD = (D_A[:, :, 1] * B_A[:, :, 0] > B_A[:, :, 1] * D_A[:, :, 0])
    ccw_CBD = (D_C[:, :, 1] * B_C[:, :, 0] > B_C[:, :, 1] * D_C[:, :, 0])
    ccw_ACB = (B_A[:, :, 1] * C_A[:, :, 0] > C_A[:, :, 1] * B_A[:, :, 0])
    
    # Checks
    check1 = (ccw_ACD != ccw_BCD) & (ccw_ABC != ccw_ABD)
    check2 = (ccw_ABD != ccw_CBD) & (ccw_ACB != ccw_ACD)

    # Check if A, B, C or D are not at the same Nodes
    non_zero = lambda a : np.any(a != 0., axis=-1)
    id_different_Nodes = (
        non_zero(D_A) &
        non_zero(D_B) &
        non_zero(D_C) &
        non_zero(C_A) &
        non_zero(C_B) &
        non_zero(B_A)
    )
    
    id_not_crossing = np.bitwise_not(check1 | check2).squeeze() & id_different_Nodes.squeeze()

    return id_not_crossing
    