import cv2
import pickle
from calibration_funcs import calib_complete
from utils import write_pickle, print_rect_info, compare_rectfied_effect
from typing import List, Tuple
import numpy as np

def stereo_rectify(calibration_results: dict):
    """
    Compute the stereo rectification transformation using the calibration results.
    ----------
    Parameters:
        - calibration_results (dict): A dictionary containing the calibration results.
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
        - R1 (np.ndarray): Rectification transformation for the left camera.
        - R2 (np.ndarray): Rectification transformation for the right camera.
        - P1 (np.ndarray): Projection matrix for the left camera after rectification.
        - P2 (np.ndarray): Projection matrix for the right camera after rectification.
        - Q (np.ndarray): Disparity-to-depth mapping matrix.
        - validRoi1 (Tuple[int, int, int, int]): Valid region of interest for the left camera.
        - validRoi2 (Tuple[int, int, int, int]): Valid region of interest for the right camera.
    """
    left_K = calibration_results["left_K"]
    left_dist = calibration_results["left_dist"]
    right_K = calibration_results["right_K"]
    right_dist = calibration_results["right_dist"]
    R = calibration_results["R"]
    T = calibration_results["T"]
    image_size = calibration_results["image_size"]

    # Compute the rectification transformation
    R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(
        left_K,
        left_dist,
        right_K,
        right_dist,
        image_size,
        R,
        T,
        alpha=0
    )

    return R1, R2, P1, P2, Q, validRoi1, validRoi2

def undistort_map(calibration_results: dict, R1: np.ndarray, P1: np.ndarray, R2: np.ndarray,
                    P2: np.ndarray):   

    """
    Undistort and rectify the images using the calibration results.
    ----------
    Parameters:
        - calibration_results (dict): A dictionary containing the calibration results.
            - left_K (np.ndarray): Left camera intrinsic matrix.
            - left_dist (np.ndarray): Left camera distortion coefficients.
            - right_K (np.ndarray): Right camera intrinsic matrix.
            - right_dist (np.ndarray): Right camera distortion coefficients.
            - image_size (Tuple[int, int]): The size of the images.
        - R1 (np.ndarray): Rectification transformation for the left camera.
        - P1 (np.ndarray): Projection matrix for the left camera after rectification.
        - R2 (np.ndarray): Rectification transformation for the right camera.
        - P2 (np.ndarray): Projection matrix for the right camera after rectification.
    ----------
    Returns:
        - left_map_x (np.ndarray): Map for the x-coordinates of the left camera.
        - left_map_y (np.ndarray): Map for the y-coordinates of the left camera.
        - right_map_x (np.ndarray): Map for the x-coordinates of the right camera.
        - right_map_y (np.ndarray): Map for the y-coordinates of the right camera.
    """
    left_K = calibration_results["left_K"]
    left_dist = calibration_results["left_dist"]
    right_K = calibration_results["right_K"]
    right_dist = calibration_results["right_dist"]
    image_size = calibration_results["image_size"]
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(left_K, left_dist, R1, P1, image_size, cv2.CV_32FC1)
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(right_K, right_dist, R2, P2, image_size, cv2.CV_32FC1)

    return left_map_x, left_map_y, right_map_x, right_map_y


def rectify_and_undistort_map(calibration_results: dict, print_info: bool = False) -> dict:
    """
    Rectify and undistort the images using the calibration results.
    ----------
    Parameters:
        - calibration_results (dict): A dictionary containing the calibration results.
            - left_K (np.ndarray): Left camera intrinsic matrix.
            - left_dist (np.ndarray): Left camera distortion coefficients.
            - right_K (np.ndarray): Right camera intrinsic matrix.
            - right_dist (np.ndarray): Right camera distortion coefficients.
            - image_size (Tuple[int, int]): The size of the images.
        - print_info (bool): Whether to print the rectification information.
    ----------
    Returns:
        - rectification_results (dict): A dictionary containing the rectification results.
            - R1 (np.ndarray): Rectification transformation for the left camera.
            - R2 (np.ndarray): Rectification transformation for the right camera.
            - P1 (np.ndarray): Projection matrix for the left camera after rectification.
            - P2 (np.ndarray): Projection matrix for the right camera after rectification.
            - Q (np.ndarray): Disparity-to-depth mapping matrix.
            - validRoi1 (Tuple[int, int, int, int]): Valid region of interest for the left camera.
            - validRoi2 (Tuple[int, int, int, int]): Valid region of interest for the right camera.
            - left_map_x (np.ndarray): Map for the x-coordinates of the left camera.
            - left_map_y (np.ndarray): Map for the y-coordinates of the left camera.
            - right_map_x (np.ndarray): Map for the x-coordinates of the right camera.
            - right_map_y (np.ndarray): Map for the y-coordinates of the right camera.
    """
    R1, R2, P1, P2, Q, validRoi1, validRoi2 = stereo_rectify(calibration_results)
    left_map_x, left_map_y, right_map_x, right_map_y = undistort_map(calibration_results, R1, P1, R2, P2)

    rectification_results = {
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "validRoi1": validRoi1,
        "validRoi2": validRoi2,
        "left_map_x": left_map_x,
        "left_map_y": left_map_y,
        "right_map_x": right_map_x,
        "right_map_y": right_map_y
    }

    if print_info:
        print_rect_info(rectification_results)

    return rectification_results

def complete_rectification(calib_info, print_info=False, show_boards=False) -> dict:
    """
    The function to complete the rectification process.
    Calls the calibration function (calib_complete) and the rectification function (rectify_and_undistort_map).
    ----------
    Parameters:
        - calib_info (dict): A dictionary containing the calibration information.
            - calib_done (bool): Whether the calibration is done or not.
            - calib_pickle_path (str): The path to the calibration pickle file.
            - left_images (List[str]): List of paths to the left images.
            - right_images (List[str]): List of paths to the right images.
        - print_info (bool): Whether to print the rectification information.
        - show_boards (bool): Whether to show the calibration boards or not.
    ----------
    Returns:
        - rectification_results (dict): A dictionary containing the rectification results.
            - R1 (np.ndarray): Rectification transformation for the left camera.
            - R2 (np.ndarray): Rectification transformation for the right camera.
            - P1 (np.ndarray): Projection matrix for the left camera after rectification.
            - P2 (np.ndarray): Projection matrix for the right camera after rectification.
            - Q (np.ndarray): Disparity-to-depth mapping matrix.
            - validRoi1 (Tuple[int, int, int, int]): Valid region of interest for the left camera.
            - validRoi2 (Tuple[int, int, int, int]): Valid region of interest for the right camera.
            - left_map_x (np.ndarray): Map for the x-coordinates of the left camera.
            - left_map_y (np.ndarray): Map for the y-coordinates of the left camera.
            - right_map_x (np.ndarray): Map for the x-coordinates of the right camera.
            - right_map_y (np.ndarray): Map for the y-coordinates of the right camera.
    """
    if calib_info["calib_done"] == True:
        pickle_path = calib_info["calib_pickle_path"]
        with open(pickle_path, "rb") as f:
            calibration_results = pickle.loads(f.read())
        if print_info:
            print(f"calibration already done, loading from {pickle_path}")
    else:
        calibration_results = calib_complete(calib_info, print_info=print_info, show_boards=show_boards)
    rectification_results = rectify_and_undistort_map(calibration_results, print_info=print_info)

    return rectification_results

def write_complete_rect_pickle(calib_info: dict, rect_pickle_path: str,
                               print_info: bool = False, show_boards: bool = False):
    """
    Computes the rectification results and saves them to a pickle file.
    Calls the complete_rectification function and the write_pickle function.
    ----------
    Parameters:
        - calib_info (dict): A dictionary containing the calibration information.
            - calib_done (bool): Whether the calibration is done or not.
            - calib_pickle_path (str): The path to the calibration pickle file.
            - left_images (List[str]): List of paths to the left images.
            - right_images (List[str]): List of paths to the right images.
        - rect_pickle_path (str): The path to save the rectification results pickle file.
        - print_info (bool): Whether to print the rectification information.
        - show_boards (bool): Whether to show the calibration boards or not.
    ----------
    Returns:
        - file_path (str): The path to the saved rectification results pickle file.
    """
    rectification_results = complete_rectification(calib_info, print_info=print_info, show_boards=show_boards)
    file_path = write_pickle(rect_pickle_path, rectification_results)
    if print_info:
        print(f"Rectification results saved to {file_path}")
    return file_path

def rectify_images(left_images: List[np.ndarray], right_images: List[np.ndarray],
                     rectification_results: dict, show: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Rectify the left and right images using the rectification results.
    ----------
    Parameters:
        - left_images (List[np.ndarray]): List of left images to be rectified.
        - right_images (List[np.ndarray]): List of right images to be rectified.
        - rectification_results (dict): A dictionary containing the rectification results.
            - left_map_x (np.ndarray): Map for the x-coordinates of the left camera.
            - left_map_y (np.ndarray): Map for the y-coordinates of the left camera.
            - right_map_x (np.ndarray): Map for the x-coordinates of the right camera.
            - right_map_y (np.ndarray): Map for the y-coordinates of the right camera.
        - show (bool): Whether to show the rectified images or not.
    ----------
    Returns:
        - left_rectified_images (List[np.ndarray]): List of rectified left images.
        - right_rectified_images (List[np.ndarray]): List of rectified right images.
    """
    left_map_x, left_map_y = rectification_results["left_map_x"], rectification_results["left_map_y"]
    right_map_x, right_map_y = rectification_results["right_map_x"], rectification_results["right_map_y"]
    left_rectified_images = []
    right_rectified_images = []

    for left_image, right_image in zip(left_images, right_images):
        left_rectified_image = cv2.remap(left_image, left_map_x, left_map_y, cv2.INTER_LINEAR)
        right_rectified_image = cv2.remap(right_image, right_map_x, right_map_y, cv2.INTER_LINEAR)
        left_rectified_images.append(left_rectified_image)
        right_rectified_images.append(right_rectified_image)
        if show:
            compare_rectfied_effect(left_image, right_image, left_rectified_image, right_rectified_image)
    
    return left_rectified_images, right_rectified_images