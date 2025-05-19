import cv2
import numpy as np
from utils import get_images, print_calib_info, write_pickle
from i308_utils import *
from typing import List, Tuple

def load_checkerboard_config(config_path: str) -> Tuple[Tuple[int, int], float]:
    """
    Load the checkerboard configuration from a file.
    The file should be formatted as:
    checkerboard = (rows, cols)
    square_size_mm = size
    Function provided by professors of course I308.
    -----------
    Parameters:
        - config_path (str): Path to the configuration file.
    ----------
    Returns:
        - Tuple[Tuple[int, int], float]: The checkerboard size (rows, cols) and the square size in mm.
    """
    context = {}
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                exec(line, {}, context)
            except Exception as e:
                raise ValueError(f"Error parsing line '{line}': {e}")
    if "checkerboard" not in context or "square_size_mm" not in context:
        raise ValueError("checkerboard and square_size_mm must be defined in the config file")
    return context["checkerboard"], context["square_size_mm"]

def board_points(checkerboard: Tuple[int, int]) -> np.ndarray:
    """
    Generate the 3D points of the checkerboard corners in the world coordinate system.
    The points are generated in the Z=0 plane.
    Function provided by professors of course I308.
    ----------
    Parameters:
        - checkerboard (Tuple[int, int]): The size of the checkerboard (rows, cols).
    ----------
    Returns:
        - np.ndarray: The 3D points of the checkerboard corners in the world coordinate system.
    """
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

    return objp

def get_points_from_images(left_images: List[np.ndarray], right_images: List[np.ndarray],
                        checkerboard: Tuple[int, int], checkerboard_world_points_mm: np.ndarray,
                        print_info=False, show_boards=False
                        ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
    """
    Get the points from the images and the world points.
    Parameters:
        - left_images (List[np.ndarray]): List of left images.
        - right_images (List[np.ndarray]): List of right images.
        - checkerboard (Tuple[int, int]): The size of the checkerboard (rows, cols).
        - checkerboard_world_points_mm (np.ndarray): The 3D points of the checkerboard corners in the world coordinate system.
        - print_info (bool): Whether to print information about the processing.
        - show_boards (bool): Whether to show the detected checkerboards.
    ----------
    Returns:
        - world_points (List[np.ndarray]): The 3D points of the checkerboard corners in the world coordinate system.
        - left_images_points (List[np.ndarray]): The 2D points of the checkerboard corners in the left images.
        - right_images_points (List[np.ndarray]): The 2D points of the checkerboard corners in the right images.
        - image_size (Tuple[int, int]): The size of the images.
    """
    image_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    world_points = []
    left_images_points = []
    right_images_points = []

    left_images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in left_images]
    right_images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in right_images]
    
    for idx, (left_image, right_image) in enumerate(zip(left_images_gray, right_images_gray)):
        if print_info:
            print(f"processing image pair {idx + 1} of {len(left_images)}")

        # get the images sizes
        left_size = (left_image.shape[1], left_image.shape[0])
        right_size = (right_image.shape[1], right_image.shape[0])

        # checks that images sizes match
        if left_size != right_size:
            raise Exception(f"left and right images sizes differ: left {left_size} / right {right_size}")
            
        if image_size is None:
            # remembers the images size
            image_size = left_size
        else:
            if image_size != left_size:
                raise Exception(f"there are images with different sizes: {image_size} vs {left_size}")

        # finds the checkerboard in each image
        left_found, left_corners = cv2.findChessboardCorners(left_image, checkerboard)
        right_found, right_corners = cv2.findChessboardCorners(right_image, checkerboard)

        if not left_found or not right_found:
            print("warning, checkerboard was not found")
            continue

        # checkerboard was found in both images.
        # let's improve the found corners
        left_corners = cv2.cornerSubPix(left_image, left_corners, (7, 7), (-1,-1), criteria)
        right_corners = cv2.cornerSubPix(right_image, right_corners, (7, 7), (-1,-1), criteria)

        left_images_points.append(left_corners)
        right_images_points.append(right_corners)

        if show_boards:
            draw_settings = {
            "corner_radius": 10,
            "corner_thickness": 5,
            "line_thickness": 4
            }

            left_image = draw_checkerboard(left_images[idx], checkerboard, left_corners, True, **draw_settings)
            right_image = draw_checkerboard(right_images[idx], checkerboard, right_corners,  True, **draw_settings)
            show_images([left_image, right_image], ["left", "right"], show=True)

        world_points.append(checkerboard_world_points_mm)
    return world_points, left_images_points, right_images_points, image_size

def calib_complete(calib_info: dict, print_info: bool = False, show_boards: bool = False) -> dict:
    """
    Perform the complete calibration process using the provided images and checkerboard configuration.
    The function will load the images, detect the checkerboard corners, and perform stereo calibration.
    ----------
    Parameters:
        - calib_info (dict): A dictionary containing the calibration information.
            - images_path (str): Path to the images folder.
            - images_prefix (str): Prefix for the images.
            - checkerboard_path (str): Path to the checkerboard configuration file.
            - calib_pickle_path (str): Path to save the calibration results.
        - print_info (bool): Whether to print information about the processing.
        - show_boards (bool): Whether to show the detected checkerboards.
    ----------
    Returns:
        - dict: A dictionary containing the calibration results.
            - left_K (np.ndarray): Left camera intrinsic matrix.
            - left_dist (np.ndarray): Left camera distortion coefficients.
            - right_K (np.ndarray): Right camera intrinsic matrix.
            - right_dist (np.ndarray): Right camera distortion coefficients.
            - R (np.ndarray): Rotation matrix between the two cameras.
            - T (np.ndarray): Translation vector between the two cameras.
            - E (np.ndarray): Essential matrix.
            - F (np.ndarray): Fundamental matrix.
            - image_size (Tuple[int, int]): The size of the images.
    """
    images_path = calib_info["images_path"]
    images_prefix = calib_info["images_prefix"]
    checkerboard_path = calib_info["checkerboard_path"]
    left_images, right_images = get_images(images_path, images_prefix, print_info=print_info)
    checkerboard, square_size_mm = load_checkerboard_config(checkerboard_path)
    if print_info:
        print("checkerboard =", checkerboard)
        print("square size (mm) =", square_size_mm)
    checkerboard_world_points_mm = board_points(checkerboard) * square_size_mm
    world_points, left_images_points, right_images_points, image_size = get_points_from_images(left_images, right_images, checkerboard, checkerboard_world_points_mm, print_info, show_boards)
    err, left_K, left_dist, right_K, right_dist, R, T, E, F = cv2.stereoCalibrate(
        world_points, 
        left_images_points, 
        right_images_points, 
        None, 
        None, 
        None, 
        None, 
        image_size, 
        flags=0
    )
    calibration_results = {
    'left_K': left_K,
    'left_dist': left_dist,
    'right_K': right_K,
    'right_dist': right_dist,
    'R': R,
    'T': T,
    'E': E,
    'F': F,
    'image_size': image_size,
    }
    if print_info:
        print_calib_info(calibration_results)
    return calibration_results

def write_complete_calib_pickle(calib_info: dict, print_info: bool = False)-> str:
    """
    Calls the `calib_complete` function to perform the calibration and then saves the results to a file.
    ----------
    Parameters:
         - calib_info (dict): A dictionary containing the calibration information.
            - images_path (str): Path to the images folder.
            - images_prefix (str): Prefix for the images.
            - checkerboard_path (str): Path to the checkerboard configuration file.
            - calib_pickle_path (str): Path to save the calibration results.
        - print_info (bool): Whether to print information about the processing.
    ----------
    Returns:
        - str: The path to the saved calibration results file.
    """
    calib_pickle_path = calib_info["calib_pickle_path"]
    if calib_pickle_path is None:
        raise ValueError("calib_pickle_path must be defined in the config file")
    
    calibration_results = calib_complete(calib_info, print_info=print_info)
    file_path = write_pickle(calib_pickle_path, calibration_results)
    if print_info:
        print(f"calibration results saved in {file_path}")
    return file_path

def detect_board(CHECKERBOARD: Tuple[int, int], gray: np.ndarray,
                criteria: Tuple[int, int, float] = None,
                subpix_win: Tuple[int, int] = (7, 7)) -> Tuple[bool, np.ndarray]:
    """
    Detects the checkerboard corners in the given image.
    Function provided by professors of course I308.
    ----------
    Parameters:
        - CHECKERBOARD (Tuple[int, int]): The size of the checkerboard (rows, cols).
        - gray (np.ndarray): The grayscale image where the corners will be detected.
        - criteria (Tuple[int, int, float]): The termination criteria for corner sub-pixel refinement.
        - subpix_win (Tuple[int, int]): The window size for corner sub-pixel refinement.
    ----------
    Returns:
        - ret (bool): True if the corners were found, False otherwise.
        - corners (np.ndarray): The detected corners in the image.
    """
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        # let's refine its coordinates
        if criteria is None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # refining pixel coordinates for given 2d points.
        corners = cv2.cornerSubPix(gray, corners, subpix_win, (-1, -1), criteria)

    return ret, corners

def draw_checkerboard(
        image: np.ndarray,
        board_size: Tuple[int, int],
        corners: np.ndarray,
        found: bool,
        line_thickness: int =2,
        corner_radius: int = 5,
        corner_thickness: int = 2,
        line_color: Tuple[int, int, int] = (0, 255, 0),
        circles_color: Tuple[int, int, int] = (0, 255, 255)
):
    """
    Draws detected checkerboard.
    Function provided by professors of course I308.
    ----------
    Parameters:
        - image (np.ndarray): The image where the corners will be drawn.
        - board_size (Tuple[int, int]): The size of the chessboard (rows, columns).
        - corners (np.ndarray): The detected corners from cv2.findChessboardCorners.
        - line_thickness (int): Thickness of the lines connecting the corners.
        - corner_radius (int): Radius of the circles at each corner.
        - corner_thickness (int): Thickness of the circles at each corner.
        - line_color (Tuple[int, int, int]): Color of the lines connecting the corners.
        - circles_color (Tuple[int, int, int]): Color of the circles at each corner.
    ----------
    Returns:
        - np.ndarray: The image with the drawn checkerboard.
    """

    if not found:
        return image

    # Ensure corners are in integer format for drawing
    corners = corners.astype(int)

    # Draw lines connecting corners
    for i in range(board_size[1]):
        for j in range(board_size[0] - 1):
            idx1 = i * board_size[0] + j
            idx2 = i * board_size[0] + (j + 1)
            cv2.line(image, tuple(corners[idx1][0]), tuple(corners[idx2][0]), line_color, line_thickness)

    for i in range(board_size[1] - 1):
        for j in range(board_size[0]):
            idx1 = i * board_size[0] + j
            idx2 = (i + 1) * board_size[0] + j
            cv2.line(image, tuple(corners[idx1][0]), tuple(corners[idx2][0]), line_color, line_thickness)

    # Draw circles at each corner
    for corner in corners:
        cv2.circle(image, tuple(corner[0]), corner_radius, circles_color, corner_thickness)

    return image