import os
import cv2
import numpy as np
from utils import get_images_path, print_calib_info, write_pickle

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


def get_points_from_images(left_file_names, right_file_names, checkerboard, checkerboard_world_points_mm, print_info=False):
    image_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    world_points = []
    left_images = []
    right_images = []
    left_images_points = []
    right_images_points = []

    for left_file_name, right_file_name in zip(left_file_names, right_file_names):
        if print_info:
            print("processing", os.path.normpath(left_file_name),  os.path.normpath(right_file_name))

        # read left and right images
        left_image = cv2.imread(left_file_name, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(right_file_name, cv2.IMREAD_GRAYSCALE)

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
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        left_corners = cv2.cornerSubPix(left_image, left_corners, (7, 7), (-1,-1), criteria)
        right_corners = cv2.cornerSubPix(right_image, right_corners, (7, 7), (-1,-1), criteria)

        # acumulo las imagenes
        left_images.append(left_image)
        right_images.append(right_image)

        # acumulo los corners detectados
        left_images_points.append(left_corners)
        right_images_points.append(right_corners)

        # acumulo los puntos del mundo
        world_points.append(checkerboard_world_points_mm)
    return world_points, left_images_points, right_images_points, image_size

def calib_complete(calib_info, print_info=False):
    images_path = calib_info["images_path"]
    images_prefix = calib_info["images_prefix"]
    checkerboard_path = calib_info["checkerboard_path"]
    left_file_names, right_file_names = get_images_path(images_path, images_prefix, print_info=print_info)
    checkerboard, square_size_mm = load_checkerboard_config(checkerboard_path)
    if print_info:
        print("checkerboard =", checkerboard)
        print("square size (mm) =", square_size_mm)
    checkerboard_world_points_mm = board_points(checkerboard) * square_size_mm
    world_points, left_images_points, right_images_points, image_size = get_points_from_images(left_file_names, right_file_names, checkerboard, checkerboard_world_points_mm, print_info=print_info)
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