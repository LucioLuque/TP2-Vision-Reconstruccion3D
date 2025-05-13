import os
import cv2
import numpy as np
from utils import get_images, print_calib_info, write_pickle
from i308_utils import *

def load_checkerboard_config(config_path):
    """
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

def board_points(checkerboard):
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

    return objp

def get_points_from_images(left_images, right_images, checkerboard, checkerboard_world_points_mm, print_info=False, show_boards=False):
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

        # acumulo los corners detectados
        left_images_points.append(left_corners)
        right_images_points.append(right_corners)

        if show_boards:
            # draw the checkerboard
            draw_settings = {
            "corner_radius": 10,
            "corner_thickness": 5,
            "line_thickness": 4
            }

            left_image = draw_checkerboard(left_images[idx], checkerboard, left_corners, True, **draw_settings)
            right_image = draw_checkerboard(right_images[idx], checkerboard, right_corners,  True, **draw_settings)
            show_images([left_image, right_image], ["left", "right"], show=True)

        # acumulo los puntos del mundo
        world_points.append(checkerboard_world_points_mm)
    return world_points, left_images_points, right_images_points, image_size

def calib_complete(calib_info, print_info=False, show_boards=False):
    """
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

def write_complete_calib_pickle(calib_info, print_info=False):
    calib_pickle_path = calib_info["calib_pickle_path"]
    if calib_pickle_path is None:
        raise ValueError("calib_pickle_path must be defined in the config file")
    
    calibration_results = calib_complete(calib_info, print_info=print_info)
    file_path = write_pickle(calib_pickle_path, calibration_results)
    if print_info:
        print(f"calibration results saved in {file_path}")
    return file_path

def detect_board(CHECKERBOARD, gray, criteria=None, subpix_win=(7, 7)):

    # shape = gray.shape[::-1]
    # gray = cv2.blur(gray, (5, 5))

    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        # we've found the corners
        # let's refine its coordinates
        # objpoints.append(objp)
        if criteria is None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # refining pixel coordinates for given 2d points.
        corners = cv2.cornerSubPix(gray, corners, subpix_win, (-1, -1), criteria)
        # pass
        # imgpoints.append(corners2)

    return ret, corners

def draw_checkerboard(
        image,
        board_size,
        corners,
        found,
        line_thickness=2,
        corner_radius=5,
        corner_thickness=2,
        line_color=(255, 0, 255),
        circles_color=(0, 255, 255)
):
    """
        Draws detected checkerboard.

        Parameters:
        - image: The image where the corners will be drawn.
        - corners: The detected corners from cv2.findChessboardCorners.
        - board_size: The size of the chessboard (rows, columns).
        - line_thickness: Thickness of the lines connecting the corners.
        - corner_radius: Radius of the circles at each corner.
        - corner_thickness: Thickness of the circles at each corner.
        - color: Color of the lines and circles (B, G, R).
    """

    if not found:
        return image

    # Ensure corners are in integer format for drawing
    corners = corners.astype(int)

    # Draw lines connecting corners
    # line_color = color  # (0, 255, 0)
    # circles_color = (255, 0, 0)
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