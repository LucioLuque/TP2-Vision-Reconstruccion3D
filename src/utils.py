import os
import re
import glob
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import copy
from i308_utils import show_images
from typing import List, Tuple

def numeric_sort(file_name:str)-> int:
    """
    Extract the numeric part from the file name.
    -----------
    Parameters:
        - file_name (str): The file name to extract the number from.
    -----------
    Returns:
        - int: The extracted number.
    """
    base = os.path.basename(file_name)
    match = re.search(r"_(\d+)\.\w+$", base)
    return int(match.group(1))

def get_images_path(directory: str, prefix: str = "calib",
                    ext: str = "jpg", print_info: bool=False) -> Tuple[List[str], List[str]]:
    """
    Get the paths of the left and right images from the specified directory, 
    with the specified prefix and extension. Based on a function provided by I308 instructors
    -----------
    Parameters:
        - directory (str): The directory to search for images.
        - prefix (str): The prefix of the image files.
        - ext (str): The extension of the image files.
        - print_info (bool): Whether to print the found images information.
    -----------
    Returns:
        - Tuple[List[str], List[str]]: The paths of the left and right images.
    """
    if len(prefix) > 0 and prefix[-1] != "_":
        prefix += "_"
    left_pattern = os.path.join(directory, f"{prefix}left_*.{ext}")
    right_pattern = os.path.join(directory, f"{prefix}right_*.{ext}")
    left_file_names = sorted(glob.glob(left_pattern), key=numeric_sort)
    right_file_names = sorted(glob.glob(right_pattern), key=numeric_sort)
    num_left = len(left_file_names)
    num_right = len(right_file_names)
    if  num_left != num_right:
        raise Exception(f"the number of files (left {num_left} / right{num_right}) doesn't match")
    if print_info:
        print(f"Found {num_left} left images and {num_right} right images")
        print(f"First left image: {os.path.normpath(left_file_names[0])}")
        print(f"First right image: {os.path.normpath(right_file_names[0])}")
    return left_file_names, right_file_names

def get_images_from_paths(left_imgs_names: List[str], right_imgs_names: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Get the left and right images from the specified paths.
    -----------
    Parameters:
        - left_imgs_names (List[str]): The paths of the left images.
        - right_imgs_names (List[str]): The paths of the right images.
    -----------
    Returns:
        - Tuple[List[np.ndarray], List[np.ndarray]]: The left and right images list.
    """
    left_imgs = [cv2.imread(img) for img in left_imgs_names]
    right_imgs = [cv2.imread(img) for img in right_imgs_names]
    return left_imgs, right_imgs

def get_images(directory: str, prefix: str = "calib", ext: str = "jpg",
               print_info: bool=False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Get the left and right images from the specified directory, with
    the specified prefix and extension.
    -----------
    Parameters:
        - directory (str): The directory to search for images.
        - prefix (str): The prefix of the image files.
        - ext (str): The extension of the image files.
        - print_info (bool): Whether to print the found images information.
    -----------
    Returns:
        - Tuple[List[np.ndarray], List[np.ndarray]]: The left and right images list.
    """
    left_file_names, right_file_names = get_images_path(directory, prefix, ext, print_info)
    left_imgs, right_imgs = get_images_from_paths(left_file_names, right_file_names)
    return left_imgs, right_imgs

def write_pickle(path: str, data: object) -> str:
    """
    Write the data to a pickle file.
    -----------
    Parameters:
        - path (str): The path to the file.
        - data (object): The data to write.
    -----------
    Returns:
        - str: The path to the file.
    """
    path = os.path.normpath(path)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path

def np_print(np_array: np.ndarray) -> str:
    """
    Convert a numpy array to a string representation.
    Function provided by professors of course I308.
    -----------
    Parameters:
        - np_array (np.ndarray): The numpy array to convert.
    -----------
    Returns:
        - str: The string representation of the numpy array.
    """
    h, w = np_array.shape
    if h == 1 or w == 1:
        num_fmt = "{:.6f}"
    else:
        num_fmt = "{:.3f}"

    str_array = "[\n" + ",\n".join([
        "\t[" + ",\t".join([num_fmt.format(v).rjust(10, ' ') for v in row]) + "]"
        for row in np_array
    ]) + "\n]"
    ret = "np.array(" + str_array + ")"
    return ret

def print_calib_info(calibration_results: dict) -> None:
    """
    Print the calibration results.
    Based on a function provided by I308 instructors.
    -----------
    Parameters:
        - calibration_results (dict): The calibration results.
    -----------
    Returns:
        - None
    """
    to_print = [

    "# Left camera Intrinsics:",
    ("left_K", calibration_results["left_K"]),
    ("left_dist", calibration_results["left_dist"]),
    "# Right camera Intrinsics:",
    ("right_K", calibration_results["right_K"]),
    ("right_dist", calibration_results["right_dist"]),

    "# Rotation:",
    ("R", calibration_results["R"]),

    "# Translation:",
    ("T", calibration_results["T"]),
    
    "# Essential Matrix:",
    ("E", calibration_results["E"]),
    
    "# Fundamental Matrix:",
    ("F", calibration_results["F"]),
            
    ]
    print("# STEREO CALIBRATION")
    for line in to_print:
        if isinstance(line, str):   
            print(line)
        else:
            var_name, np_array = line
            print(f"{var_name} = {np_print(np_array)}\n")

def print_rect_info(rectification_results: dict) -> None:
    """
    Print the rectification results.
    Based on a function provided by I308 instructors.
    -----------
    Parameters:
        - rectification_results (dict): The rectification results.
    -----------
    Returns:
        - None
    """
    to_print = [
        "# Left camera Rectification:",
        ("R1", rectification_results["R1"]),
        ("P1", rectification_results["P1"]),
        ("validRoi1", rectification_results["validRoi1"]),

        "# Right camera Rectification:",
        ("R2", rectification_results["R2"]),
        ("P2", rectification_results["P2"]),
        ("validRoi2", rectification_results["validRoi2"]),

        "# Disparity-to-depth mapping matrix Q:",
        ("Q", rectification_results["Q"]),

        "# Rectification maps (left):",
        ("left_map_x.shape", rectification_results["left_map_x"].shape),
        ("left_map_y.shape", rectification_results["left_map_y"].shape),

        "# Rectification maps (right):",
        ("right_map_x.shape", rectification_results["right_map_x"].shape),
        ("right_map_y.shape", rectification_results["right_map_y"].shape),
    ]

    print("# STEREO RECTIFICATION")
    for line in to_print:
        if isinstance(line, str):
            print(line)
        else:
            var_name, value = line
            if isinstance(value, tuple):
                print(f"{var_name} = {value}\n")
            else:
                print(f"{var_name} = {np_print(value)}\n")

def compare_rectfied_effect(left_image: np.ndarray, right_image: np.ndarray,
                            left_rectified_image: np.ndarray, right_rectified_image: np.ndarray) -> None:
    """
    Compare the rectified images with the original images.
    Based on a function provided by I308 instructors.
    -----------
    Parameters:
        - left_image (np.ndarray): The left image.
        - right_image (np.ndarray): The right image.
        - left_rectified_image (np.ndarray): The left rectified image.
        - right_rectified_image (np.ndarray): The right rectified image.
    -----------
    Returns:
        - None
    """
    fig, axes = show_images([
        left_image, right_image
    ], [
        "left", "right"
    ], show=False)

    for y, c in zip([255, 470, 830], ['r', 'b', 'c']):
        axes[0].axhline(y=y, color=c, linestyle='-', linewidth=0.5)
        axes[1].axhline(y=y, color=c, linestyle='-', linewidth=0.5)
    plt.show()


    fig, axes = show_images([
        left_rectified_image, right_rectified_image
    ], [
        "left rectified", "right rectified"
    ], show=False)

    for y, c in zip([255, 470, 830], ['r', 'b', 'c']):
        axes[0].axhline(y=y, color=c, linestyle='-', linewidth=0.5)
        axes[1].axhline(y=y, color=c, linestyle='-', linewidth=0.5)
    plt.show()

def save_images(path: str, left_images: List[np.ndarray], right_images: List[np.ndarray],
                name: str = "") -> None:
    """
    Save the images to the specified path.
    -----------
    Parameters:
        - path (str): The path to save the images.
        - left_images (List[np.ndarray]): The left images.
        - right_images (List[np.ndarray]): The right images.
        - name (str): The name after the prefix left/right.
    -----------
    Returns:
        - None
    """
    if len(name) > 0:
            name = f"_{name}"
    for i, (left_image, right_image) in enumerate(zip(left_images, right_images)):
        left_path = os.path.join(path, f"left{name}_{i}.jpg")
        right_path = os.path.join(path, f"right{name}_{i}.jpg")
        cv2.imwrite(left_path, left_image)
        cv2.imwrite(right_path, right_image)

def draw_registration_result(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
                            transformation: np.ndarray, show: bool=False,
                            save_path: str=None) -> None:
    """
    Draw the registration result.
    Based on a function in https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
    -----------
    Parameters:
        - source (o3d.geometry.PointCloud): The source point cloud.
        - target (o3d.geometry.PointCloud): The target point cloud.
        - transformation (np.ndarray): The transformation matrix.
        - show (bool): Whether to show the result.
        - save_path (str): The path to save the result if given.
    -----------
    Returns:
        - None
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    if show:
        print("yellow source \nblue target")
        o3d.visualization.draw_geometries([source_temp, target_temp])
    if save_path:
        o3d.io.write_point_cloud(save_path, source_temp+target_temp)