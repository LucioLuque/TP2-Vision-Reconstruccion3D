
from disparity.method_cre_stereo import CREStereo
from disparity.methods import Calibration, InputPair, Config
import numpy as np
import os
from pathlib import Path
import cv2
from typing import List

def get_calibration_disparity(w: int, h: int, calibration_results: dict) -> Calibration:
    """
    Get the calibration for the disparity method.
    ----------
    Parameters:
        - w (int): Width of the images.
        - h (int): Height of the images.
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
    ----------
    Returns:
        - calibration (Calibration): Calibration object.
    """
    left_K = calibration_results["left_K"]
    fx = left_K[0][0]
    fy = left_K[1][1]
    cx0 = left_K[0][2]
    cy0 = left_K[1][2]
    T = calibration_results["T"]
    baseline = np.linalg.norm(T)

    calibration = Calibration(**{
        "width": w,
        "height": h,
        "baseline_meters": baseline / 1000,
        "fx": fx,
        "fy": fy,
        "cx0": cx0,
        "cx1": cx0,
        "cy": cy0,
        "depth_range": [0.05, 20],
        "left_image_rect_normalized": [0, 0, 1, 1]
    })
    return calibration

def get_disparity_image(left_image: np.ndarray, right_image: np.ndarray, calibration: Calibration) -> np.ndarray:
    """
    Get the disparity image from the left and right images using the CREStereo method.
    ----------
    Parameters:
        - left_image (np.ndarray): Left image.
        - right_image (np.ndarray): Right image.
        - calibration (Calibration): Calibration object.
    ----------
    Returns:
        - disparity (np.ndarray): Disparity image.
    """
    models_path = "models"
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    models_path = Path.home() / ".cache" / "stereodemo" / "models"
    models_path = Path(models_path)

    left_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
    pair = InputPair(left_rgb, right_rgb, calibration)
    config = Config(models_path=models_path)

    method = CREStereo(config)
    method.parameters["Shape"].set_value("1280x720")
    method.parameters["Mode"].set_value("combined")
    method.parameters["Iterations"].set_value("10")
    disparity = method.compute_disparity(pair)
    return disparity.disparity_pixels


def get_disparity_images(left_images: List[np.ndarray], right_images: List[np.ndarray],
                        calibration_results: dict) -> List[np.ndarray]:
    """
    Get the disparity images from the left and right images using the CREStereo method.
    Calls get_disparity_image for each pair of images.
    ----------
    Parameters:
        - left_images (List[np.ndarray]): List of left images.
        - right_images (List[np.ndarray]): List of right images.
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
    ----------
    Returns:
        - disparities (List[np.ndarray]): List of disparity images.
    """
    w, h = left_images[0].shape[1], left_images[0].shape[0]
    calibration = get_calibration_disparity(w, h, calibration_results)
    disparities = []
    for left_image, right_image in zip(left_images, right_images):
        disparity = get_disparity_image(left_image, right_image, calibration)
        disparities.append(disparity)
    return disparities

def compute_depth(disparity_map: np.ndarray, f: float, B: float, default: float = 1000.0):
    """
    Compute the depth map from the disparity map.
    Function provided by professors of course I308.
    ----------
    Parameters:
        - disparity_map (np.ndarray): Disparity map.
        - f (float): Focal length in pixels.
        - B (float): Baseline in meters.
        - default (float): Default value for invalid disparity points.
    ----------
    Returns:
        - depth_map (np.ndarray): Depth map.
    """
    disparity_map = disparity_map.copy()
    mask_invalid = (disparity_map <= 0)
    
    #Z = f * B / disparity
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)
    depth_map[~mask_invalid] = (f * B) / disparity_map[~mask_invalid]
    depth_map[mask_invalid] = default
    
    return depth_map