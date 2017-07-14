import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.pyplot as plt


def compute_point_locations(calibration_image_directory, dims, visualize=False):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((dims[0]*dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:dims[1], 0:dims[0]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(os.path.join(calibration_image_directory, '*.jpg'))
    shape=None
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if idx == 0:
            shape=img.shape
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            if visualize:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (8, 6), corners, ret)
                cv2.imshow(fname, img)
                cv2.waitKey(500)
    if visualize:
        cv2.destroyAllWindows()
    return (objpoints, imgpoints, shape)


def compute_calibration_matrix(objpoints, imgpoints, shape, test_image_directory):
    # Test undistortion on an image
    img_size = (shape[1], shape[0])
    # Do camera calibration given object points and image points
    ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return (camera_matrix, distortion_coefficients)


def undistort_image(image, camera_matrix, distortion_coefficients, save_image_path=None, save_pickle_path=None, visualize=False):
    undistorted_image = cv2.undistort(image, camera_matrix, distortion_coefficients, None, camera_matrix)
    if save_image_path:
        cv2.imwrite(save_image_path,undistorted_image)

    if visualize:
        # Visualize undistortion
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(undistorted_image)
        ax2.set_title('Undistorted Image', fontsize=30)
        plt.show()
        plt.close()

    if save_pickle_path:
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle['mtx'] = camera_matrix
        dist_pickle['dist'] = distortion_coefficients
        pickle.dump( dist_pickle, open( save_pickle_path, 'wb' ) )

if __name__ == '__main__':
    # Example code for usage:
    calibration_image_directory = 'example_calibration_images'
    test_image_directory = 'example_test_images'
    (objpoints, imgpoints, shape) = compute_point_locations(calibration_image_directory, (6, 8), visualize=False)
    (camera_matrix, distortion_coefficients) = compute_calibration_matrix(objpoints, imgpoints, shape, test_image_directory)
    test_images = glob.glob(os.path.join(test_image_directory, '*.jpg'))
    for index, test_image_filename in enumerate(test_images):
        test_image = cv2.imread(test_image_filename)
        undistorted_image = undistort_image(test_image, camera_matrix, distortion_coefficients, save_image_path=test_image_filename[:-4]+'undistorted.jpg', visualize=True)