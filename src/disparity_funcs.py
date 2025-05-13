
from disparity.method_cre_stereo import CREStereo
# from disparity.method_opencv_bm import StereoBM, StereoSGBM
from disparity.methods import Calibration, InputPair, Config
import numpy as np
import os
from pathlib import Path
import cv2

def get_calibration_disparity(w, h, calibration_results):
    """
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

def get_disparity_image(left_image, right_image, calibration):
    """
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


def get_disparity_images(left_images, right_images, calibration_results):
    """
    """
    w, h = left_images[0].shape[1], left_images[0].shape[0]
    calibration = get_calibration_disparity(w, h, calibration_results)
    disparities = []
    for i in range(len(left_images)):
        left_image = left_images[i]
        right_image = right_images[i]
        disparity = get_disparity_image(left_image, right_image, calibration)
        disparities.append(disparity)
    return disparities
    

