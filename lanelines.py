import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os

from pipeline import apply_undist,getSobel
from pipeline import abs_sobel_thresh,mag_thresh,dir_threshold
from pipeline import color_transform,region_of_interest,b_thres
from finding_lines import find_lines

frame_count = 0

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        #self.best_fit = None  
        self.best_left_fit = None  
        self.best_right_fit = None  
        #polynomial coefficients for the most recent fit
        #self.current_fit = [np.array([False])]  
        self.current_left_fit = [np.array([False])]  
        self.current_right_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature_left = None 
        self.radius_of_curvature_right = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        self.frame_count = 0

tracker = Line()


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

outpath = 'output_images'

for fname in images:
    #read each image
    img = mpimg.imread(fname)

    #undistort the img
    undist = apply_undist(img, mtx, dist)

    #write the undist image
    save_fname = os.path.join(outpath, 'undist_cal_'+os.path.basename(fname))
    cv2.imwrite(save_fname, undist)

test_images = glob.glob('test_images/test*.jpg')


#Get perspective transformation matrix
str_img =  mpimg.imread('test_images/straight_lines1.jpg')
undist_str_img = apply_undist(str_img, mtx, dist)
#src = np.float32([[200,720],[620,430],[650,430],[1120,720]])
src = np.float32([[200,720],[590,450],[680,450],[1120,720]])
dst = np.float32([[420,720],[420,0],[870,0],[870,720]])
#dst = np.float32([[300,720],[300,0],[1000,0],[1000,720]])
cv2.polylines(undist_str_img, np.int32([src]), True, (255,0,0), thickness=5)
cv2.polylines(undist_str_img, np.int32([dst]), True, (0,0,255), thickness=5)
#cv2.imwrite("undist_perslines_str_img.png",undist_str_img)
#cv2.imshow("undist_str_img",undist_str_img)
#cv2.waitKey(0)
M = cv2.getPerspectiveTransform(src,dst)
Minv = cv2.getPerspectiveTransform(dst,src)

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
    gradx, grady = abs_sobel_thresh(sobelx, sobely, thresh=(20, 130))
    mag_binary = mag_thresh(sobelx, sobely, thresh=(20, 130))
    dir_binary = dir_threshold(sobelx, sobely, thresh=(0.7, 1.3))

    combined_grad = np.zeros_like(dir_binary)
    combined_grad[((gradx == 1) & (grady == 1)) & ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    #write the gradient binary image
    #save_fname = os.path.join(outpath, 'gradient_'+os.path.basename(fname))
    #plt.figure()
    plt.imshow(undist,cmap='gray')
    plt.show()
    #plt.savefig(save_fname)

    #Apply color transform
    bchannel_thres = b_thres(undist, thresh=(220,255))
    schannel_thres = color_transform(undist, thresh=(150,255))
    combined_grad_color1 = np.zeros_like(schannel_thres)
    combined_grad_color1[(combined_grad == 1) | ((schannel_thres == 1) | (bchannel_thres == 1))] = 1
    #color_binary = np.dstack((np.zeros_like(schannel_thres),combined_grad, schannel_thres, bchannel_thres))
    color_binary = np.dstack((np.zeros_like(schannel_thres),combined_grad, schannel_thres))

    #write the gradient and color thresholding binary image
    #save_fname = os.path.join(outpath, 'gradient_color_thres_img'+os.path.basename(fname))
    #plt.figure()
    #plt.subplot(121)
    plt.imshow(color_binary)
    #plt.subplot(122)
    #plt.imshow(undist)
    plt.show()
    #plt.savefig(save_fname)

    #Apply masking
    vertices = np.array([[(100,720),(600,450),(750,450),(1200,720)]],dtype=np.int32)
    combined_grad_color = region_of_interest(combined_grad_color1, vertices)

    plt.imshow(combined_grad_color)
    plt.show()

    #Apply perspective transform
    img_size = (combined_grad_color.shape[1],combined_grad_color.shape[0])
    pers_binary = cv2.warpPerspective(combined_grad_color, M, img_size)
    save_fname = os.path.join(outpath, 'pers_binary_'+os.path.basename(fname))
    #plt.figure()
    plt.imshow(pers_binary,cmap='gray')
    plt.show()
    #plt.savefig(save_fname)

    #Apply fit
    #print(pers_binary.shape)
    [left_fitx, right_fitx, ploty,
            left_curverad, right_curverad,offset_m] = find_lines(pers_binary,fname,tracker,frame_count)

    frame_count = frame_count+1

    warp_zero = np.zeros_like(pers_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    cv2.putText(result,'radius of curavture: (L): %6.2f m (R): %6.2f m' %(left_curverad,right_curverad),(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(result,'offset from center: %6.2f m ' %(offset_m),(50,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    #plt.figure()
    plt.imshow(result)
    #plt.text(600,150,'left curvature   = %6.2f m\nright curvature = %6.2f m \n' %(left_curverad,right_curverad), color='white')
    plt.show()
    save_fname = os.path.join('output_images', 'rewarp_lines_'+os.path.basename(fname))
    #plt.savefig(save_fname)
    #plt.close(fig2)

def process_image(image):
    #undistort the img
    undist = apply_undist(image, mtx, dist)

    #apply sobel
    sobelx, sobely = getSobel(undist, sobel_kernel=15)

    # Apply each of the thresholding functions
    gradx, grady = abs_sobel_thresh(sobelx, sobely, thresh=(20, 100))
    mag_binary = mag_thresh(sobelx, sobely, thresh=(20, 100))
    dir_binary = dir_threshold(sobelx, sobely, thresh=(0.7, 1.3))

    combined_grad = np.zeros_like(dir_binary)
    combined_grad[((gradx == 1) & (grady == 1)) & ((mag_binary == 1) & (dir_binary == 1))] = 1

    #Apply color transform
    #schannel_thres = color_transform(undist, thresh=(130,255))
    #combined_grad_color1 = np.zeros_like(schannel_thres)
    #combined_grad_color1[(combined_grad == 1) | (schannel_thres == 1)] = 1
    bchannel_thres = b_thres(undist, thresh=(200,255))
    schannel_thres = color_transform(undist, thresh=(130,255))
    combined_grad_color1 = np.zeros_like(schannel_thres)
    combined_grad_color1[(combined_grad == 1) | ((schannel_thres == 1) | (bchannel_thres == 1))] = 1
    color_binary = np.dstack((np.zeros_like(schannel_thres),combined_grad, schannel_thres))

    #Apply masking
    vertices = np.array([[(100,720),(600,450),(750,450),(1200,720)]],dtype=np.int32)
    combined_grad_color = region_of_interest(combined_grad_color1, vertices)

    #Apply perspective transform
    img_size = (combined_grad_color.shape[1],combined_grad_color.shape[0])
    pers_binary = cv2.warpPerspective(combined_grad_color, M, img_size)

    #Apply fit
    [left_fitx, right_fitx, ploty,
            left_curverad, right_curverad,offset_m] = find_lines(pers_binary,fname,tracker,tracker.frame_count)
    #print("frame number = ", frame_count)

    tracker.frame_count = tracker.frame_count+1

    warp_zero = np.zeros_like(pers_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    cv2.putText(result,'radius of curavture: (L): %6.2f m (R): %6.2f m' %(left_curverad,right_curverad),(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(result,'offset from center: %6.2f m ' %(offset_m),(50,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)


    return result


    
