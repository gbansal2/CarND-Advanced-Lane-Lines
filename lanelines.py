import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os

#1.  Calibrate the camera

#Read the calibration images filenames
images = glob.glob('camera_cal/calibration*.jpg')

#Arrays to store object points and image points
objpoints = []
imgpoints = []

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) 

#Iterate through images to get the calibration matrix
for fname in images:
    #read each image
    img = mpimg.imread(fname)

    #convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #fnd the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

test_images = glob.glob('test_images/test*.jpg')
outpath = 'output_images'

from pipeline import apply_undist,getSobel
from pipeline import abs_sobel_thresh,mag_thresh,dir_threshold
from pipeline import color_transform

for fname in test_images:
    #read each image
    img = mpimg.imread(fname)

    #undistort the img
    undist = apply_undist(img, mtx, dist)

    #write the undist image
    #save_fname = os.path.join(outpath, 'undist_'+os.path.basename(fname))
    #cv2.imwrite(save_fname, undist)

    #apply sobel
    sobelx, sobely = getSobel(undist, sobel_kernel=15)

    # Apply each of the thresholding functions
    gradx, grady = abs_sobel_thresh(sobelx, sobely, thresh=(30, 100))
    mag_binary = mag_thresh(sobelx, sobely, thresh=(30, 100))
    dir_binary = dir_threshold(sobelx, sobely, thresh=(0.7, 1.3))

    combined_grad = np.zeros_like(dir_binary)
    combined_grad[((gradx == 1) & (grady == 1)) & ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    #write the gradient binary image
    #save_fname = os.path.join(outpath, 'gradient_'+os.path.basename(fname))
    #plt.figure()
    #plt.imshow(combined_grad,cmap='gray')
    #plt.savefig(save_fname)

    #Apply color transform
    schannel_thres = color_transform(undist, thresh=(170,255))
    combined_grad_color = np.zeros_like(schannel_thres)
    combined_grad_color[(combined_grad == 1) | (schannel_thres == 1)] = 1

    #write the gradient and color thresholding binary image
    save_fname = os.path.join(outpath, 'gradient_color_thres_'+os.path.basename(fname))
    plt.figure()
    plt.imshow(combined_grad_color,cmap='gray')
    plt.savefig(save_fname)




    
