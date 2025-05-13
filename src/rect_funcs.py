import cv2
import os
import pickle

from calib_funcs import calib_complete
from utils import write_pickle, print_rect_info, compare_rectfied_effect



def stereo_rectify(calibration_results):
    """

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

def undistort_map(calibration_results, R1, P1, R2, P2):

    """
    Undistort and rectify the images using the calibration results.
    """
    left_K = calibration_results["left_K"]
    left_dist = calibration_results["left_dist"]
    right_K = calibration_results["right_K"]
    right_dist = calibration_results["right_dist"]
    image_size = calibration_results["image_size"]
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(left_K, left_dist, R1, P1, image_size, cv2.CV_32FC1)
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(right_K, right_dist, R2, P2, image_size, cv2.CV_32FC1)

    return left_map_x, left_map_y, right_map_x, right_map_y


def rectify_and_undistort_map(calibration_results, print_info=False):
    """
    Rectify and undistort the images using the calibration results.
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

def complete_rectification(calib_info, print_info=False, show_boards=False):
    """
    Complete the stereo rectification process.
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

#deberia dejar la opcion si es con pickle de calibracion o hacer la calibracion!
def write_complete_rect_pickle(calib_info, rect_pickle_path, print_info=False, show_boards=False):
    """
    Write the complete rectification results to a pickle file.
    """
    rectification_results = complete_rectification(calib_info, print_info=print_info, show_boards=show_boards)
    file_path = write_pickle(rect_pickle_path, rectification_results)
    if print_info:
        print(f"Rectification results saved to {file_path}")
    return file_path

def rectify_images(left_images, right_images, rectification_results, show=False):
    """
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