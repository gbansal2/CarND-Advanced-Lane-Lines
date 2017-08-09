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

objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2) 

#Iterate through images to get the calibration matrix
for fname in images:
    #read each image
    img = mpimg.imread(fname)

    #convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #fnd the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

outpath = 'output_images'

for fname in images:
    #read each image
    img = mpimg.imread(fname)

    #undistort the img
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    #write the undistorted image 
    save_fname = os.path.join(outpath, 'undist_'+os.path.basename(fname))
    cv2.imwrite(save_fname, undist)
    
