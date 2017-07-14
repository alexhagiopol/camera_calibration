import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
#%matplotlib qt


def compute_point_locations(image_directory, dims, debug=False):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((dims[0]*dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:dims[1], 0:dims[0]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(os.path.join(image_directory, '*.jpg'))

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            if debug:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (8, 6), corners, ret)
                # write_name = 'corners_found'+str(idx)+'.jpg'
                # cv2.imwrite(write_name, img)
                cv2.imshow('img', img)
                cv2.waitKey(500)
    if debug:
        cv2.destroyAllWindows()
    return (objpoints, imgpoints)
