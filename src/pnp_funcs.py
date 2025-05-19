import cv2
import numpy as np
from calibration_funcs import load_checkerboard_config, board_points, detect_board, draw_checkerboard
from i308_utils import show_images
from typing import List

def get_poses(left_imgs: List[np.ndarray], calibration_results: dict,
            checkerboard_path: str, print_info: bool = False, show_boards: bool = False) -> List[np.ndarray]:
    """
    Get the poses of the left images.
    Function provided by professors of course I308.
    ----------
    Parameters:
        - left_imgs (List[np.ndarray]): List of left images.
        - calibration_results (dict): Calibration results.
            - left_K (np.ndarray): Left camera intrinsic matrix.
            - left_dist (np.ndarray): Left camera distortion coefficients.
            - right_K (np.ndarray): Right camera intrinsic matrix.
            - right_dist (np.ndarray): Right camera distortion coefficients.
            - R (np.ndarray): Rotation matrix between the two cameras.
            - T (np.ndarray): Translation vector between the two cameras.
            - E (np.ndarray): Essential matrix.
            - F (np.ndarray): Fundamental matrix.
            - image_size (Tuple[int, int]): The size of the images.
        - checkerboard_path (str): Path to the checkerboard configuration file.
        - print_info (bool): Whether to print the information.
        - show_boards (bool): Whether to show the boards.
    ----------
    Returns:
        - poses (List[np.ndarray]): List of poses.
    """
    poses = []
    for i, left_img in enumerate(left_imgs):
        left_img = left_img.copy() 
        left_img_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        checkerboard, square_size_mm = load_checkerboard_config(checkerboard_path)
        left_found, left_corners = detect_board(checkerboard, left_img_gray)
        if print_info:
            print(f"left corners found for image {i+1}: {left_found}")
        if show_boards:
            draw_settings = {
            "corner_radius": 10,
            "corner_thickness": 5,
            "line_thickness": 4
            }
            left_image = draw_checkerboard(left_img, checkerboard, left_corners, True, **draw_settings)
            right_image = draw_checkerboard(left_img, checkerboard, left_corners,  True, **draw_settings)
            show_images([left_image, right_image], ["left", "right"])

        object_3dpoints = board_points(checkerboard)

        object_3dpoints_mm = object_3dpoints * square_size_mm

        ret, rvec, tvec = cv2.solvePnP(object_3dpoints_mm,
                                    left_corners,
                                    calibration_results["left_K"],
                                    calibration_results["left_dist"],
                                    flags=cv2.SOLVEPNP_IPPE)
        c_R_o = cv2.Rodrigues(rvec)
        c_T_o = np.column_stack((c_R_o[0], tvec))
        c_T_o = np.vstack((c_T_o, [0, 0, 0, 1]))
        o_T_c = np.linalg.inv(c_T_o)
        poses.append(o_T_c)
    return poses