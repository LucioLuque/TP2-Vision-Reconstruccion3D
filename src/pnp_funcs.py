import cv2
import numpy as np

from calib_funcs import load_checkerboard_config, board_points, detect_board

def get_matrixs(left_imgs, calibration_results, checkerboard_path):
    o_T_c_list = []
    for i in range(len(left_imgs)):
        left_img = cv2.cvtColor(left_imgs[i], cv2.COLOR_BGR2GRAY)
        checkerboard, square_size_mm = load_checkerboard_config(checkerboard_path)

        left_found, left_corners = detect_board(checkerboard, left_img)

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
        o_T_c_list.append(o_T_c)
    return o_T_c_list