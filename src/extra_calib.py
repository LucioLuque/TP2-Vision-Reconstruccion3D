import numpy as np
import cv2
import glob


def detect_boards(directory, CHECKERBOARD, show=False, wait=0, criteria=None):

    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = board_points(CHECKERBOARD)

    # Extracting path of individual image stored in a given directory
    images = glob.glob(directory)
    shape = None
    for fname in images:
        print("processing", fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        shape = gray.shape[::-1]

        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret:
            objpoints.append(objp)

            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            img = draw_checkerboard(img, CHECKERBOARD, corners2, ret)

        if show:
            cv2.imshow('img', img)
            k = cv2.waitKey(wait)
        else:
            k = 0
        if k == ord('q'):
            break

    return shape, objpoints, imgpoints



def do_calib(img_shape, obj_points, world_points):
    print("num_points", len(obj_points))
    print("calibrating...")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        world_points,
        img_shape,
        None, None
    )

    np.set_printoptions(suppress=True)
    # print("Camera matrix : \n")
    # print(mtx.round(3))
    # print("dist : \n")
    # print(dist)

    print("# Intrinsic parameters")
    print("K = ", np_print( mtx ))

    print("")

    print("dist_coeffs = ", np_print(dist))

    return mtx, dist


def calib_zhang(object_points, world_points):
    import zhang

    n = len(object_points)
    first = object_points[0]
    m = first.shape[1]

    object_points = np.array(object_points).reshape(n, m, 3)
    world_points = np.array(world_points).reshape(n, m, 2)

    homographies = [zhang.compute_homography(wp, ip) for wp, ip in
                    # zip(world_points, object_points)
                    zip(object_points, world_points)
                    ]

    mint = zhang.intrinsic_from_homographies(homographies)

    extrinsics = [zhang.extrinsics_from_homography(H, mint) for H in homographies]
    for i, (R, t) in enumerate(extrinsics):
        print(f"Extrinsics for image {i + 1}:\nR:\n{R}\nt:\n{t}\n")

    return mint




if __name__ == "__main__":

    # Defining the dimensions of checkerboard
    directory = './cam3_stereo_images/calib_left_*.jpg'
    CHECKERBOARD = (10, 7)

    img_shape, obj_points, world_points = detect_boards(
        directory,
        CHECKERBOARD, show=True, wait=1
    )

    # calib using CV
    mint, dist = do_calib(img_shape, obj_points, world_points)

    # this code compares "hand made" zhang calibration method
    # # calib using zhang
    # mint2 = calib_zhang(obj_points, world_points)
    #
    # print(mint.round())
    # print(mint2.round())
