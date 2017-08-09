import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def getSobel(image, sobel_kernel=3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    return sobelx, sobely

def abs_sobel_thresh(sobelx, sobely, thresh=(0, 255)):
    abs_sobel = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    abs_binary_x = np.copy(sxbinary)

    abs_sobel = np.absolute(sobely)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    abs_binary_y = np.copy(sxbinary)
    return abs_binary_x, abs_binary_y

def mag_thresh(sobelx, sobely, thresh=(0, 255)):
    mag_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    mag_binary = np.copy(sxbinary)
    return mag_binary

def dir_threshold(sobelx, sobely, thresh=(0, np.pi/2)):
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir_sobel = np.arctan2(abs_sobely,abs_sobelx)
    sxbinary = np.zeros_like(dir_sobel)
    sxbinary[(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] = 1
    dir_binary = np.copy(sxbinary) # Remove this line
    return dir_binary

def apply_undist(img, mtx, dist):
    final = cv2.undistort(img, mtx, dist, None, mtx)
    return final

def color_transform(img, thresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return s_binary


