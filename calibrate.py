import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.pyplot as plt
import argparse


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
        ret, corners = cv2.findChessboardCorners(gray, (dims[1], dims[0]), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            if visualize:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (dims[1], dims[0]), corners, ret)
                cv2.imshow(fname, img)
                cv2.waitKey(500)
    if visualize:
        cv2.destroyAllWindows()
    return (objpoints, imgpoints, shape)


def compute_calibration_matrix(objpoints, imgpoints, shape, save_pickle_path=None):
    # Test undistortion on an image
    img_size = (shape[1], shape[0])
    # Do camera calibration given object points and image points
    ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    if save_pickle_path:
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        pickle_dict = {}
        pickle_dict['camera_matrix'] = camera_matrix
        pickle_dict['distortion_coefficients'] = distortion_coefficients
        pickle.dump(pickle_dict, open(save_pickle_path, 'wb'))

    return (camera_matrix, distortion_coefficients)


def undistort_image(image, camera_matrix, distortion_coefficients, save_image_path=None, visualize=False):
    undistorted_image = cv2.undistort(image, camera_matrix, distortion_coefficients, None, camera_matrix)
    if save_image_path:
        cv2.imwrite(save_image_path,undistorted_image)

    if visualize:
        # Visualize undistortion
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=14)
        ax2.imshow(undistorted_image)
        ax2.set_title('Undistorted Image', fontsize=14)
        plt.show()
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate camera given chessboard images')
    parser.add_argument('--chessboard-height', '-ch', dest='height', type=int, required=True, help='Required int: Height of chessboard used in calibration.' )
    parser.add_argument('--chessboard-width', '-cw', dest='width', type=int, required=True, help='Required int: Width of chessboard used in calibration.' )
    parser.add_argument('--calibration-image-dir', '-cd', dest='calibration_dir', type=str, required=True, help='Required str: Directory containing chessboard images for calibration.' )
    parser.add_argument('--test-image-dir', '-td', dest='test_dir', type=str, required=False, default=None, help='Optional str: Directory containing test images for undistortion visualization.' )
    parser.add_argument('--save-pickle-path', '-p', dest='pickle_path', type=str, required=False, default=None, help='Optional str: Path of pickle file in which to save camera matrix and distortion coeffs.' )
    args = parser.parse_args()

    (objpoints, imgpoints, shape) = compute_point_locations(args.calibration_dir, (args.height, args.width), visualize=False)
    (camera_matrix, distortion_coefficients) = compute_calibration_matrix(objpoints, imgpoints, shape, save_pickle_path=args.pickle_path)

    if args.test_dir:
        test_images = glob.glob(os.path.join(args.test_dir, '*.jpg'))
        for index, test_image_filename in enumerate(test_images):
            test_image = cv2.imread(test_image_filename)
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            undistorted_image = undistort_image(test_image, camera_matrix, distortion_coefficients, visualize=True)
