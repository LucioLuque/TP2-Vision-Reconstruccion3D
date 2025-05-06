import os
import re
import glob
import pickle


def numeric_sort(file_path):
    base = os.path.basename(file_path)
    match = re.search(r"_(\d+)\.jpg$", base)
    return int(match.group(1))

def get_images_path(directory, prefix="calib", print_info=False):
    """
    
    """
    if len(prefix) > 0 and prefix[-1] != "_":
        prefix += "_"
    left_pattern = os.path.join(directory, f"{prefix}left_*.jpg")
    right_pattern = os.path.join(directory, f"{prefix}right_*.jpg")

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

def write_pickle(path, data):
    """
    """
    path = os.path.normpath(path)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path

def np_print(np_array):
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

def print_calib_info(calibration_results):
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

def print_rect_info(rectification_results):
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
