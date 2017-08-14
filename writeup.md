## Advanced Lane Finding Project

[//]: # ( ### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other # method and submit a pdf if you prefer.)

[//]: # (---)

[//]: # (**Advanced Lane Finding Project**)

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1a]: ./camera_cal/calibration1.jpg "distorted image"
[image1b]: ./output_images/undist_cal_calibration1.jpg "undistorted image"
[image2a]: ./test_images/test1.jpg "Road Transformed"
[image2b]: ./output_images/undist_test1.jpg "Undistorted Road Transformed"
[image3]: ./output_images/gradient_color_thres_imgtest1.jpg "Binary Example"
[image4]: ./output_images/undist_perslines_str_img.png "Warp Example"
[image5]: ./output_images/pers_binary_test1.jpg "Perspective binary test image"
[image6]: ./output_images/lines_test1.jpg "lane lines"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. This document is the Writeup / README for this project and describes in detail the steps followed to arrive at the solution.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines #41 through #68 of the file called `lanelines.py`. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function. An example of the original distorted image and an undistorted image is given below.

Distorted:
![image1a]

Undistorted:
![image1b]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2a]

Using the camera calibration matrix obtained by using the chess board images as described in the above section, we can undistort this image. The undistorted image looks like:

![alt text][image2b]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #117 through #137 in `lanelines.py`).  A bunch of utility functions for obtaining gradients and color thresholds are written in file called `pipeline.py`.
For gradient thresholding, I first applied sobel filters in x and y directions to obtain gradients of the image. I thresholded the gradients, as well as magnitude and direction of the gradient. 

For color thresholding, I transformed the image into HLS space, and applied thresholding to the S-channel. 

Here's an example of my output for this step, for the same test image shown in Step 1 above. The original image is also shown. 
In the binary image, the blue color corresponds to color threshold, and green color corresponds to gradient threshold.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a call to function cv2.getPerspectiveTransform which takes points from the source and destination planes and outputs a transformation matrix. For this project, we need both the forward and the inverse transformation matrices. This code appears on lines #92 to #103 in `lanelines.py` file.

I chose to hardcode the source and destination points in the following manner:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 420, 720      | 
| 590, 450      | 420, 0        |
| 680, 450      | 870, 0        |
| 1120, 720     | 870, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. This is shown in the image below. The blue lines are for source and red lines are for destination.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Using the perspective transformation matrix, I transformed the test images into a `top-view` image. An example of test image looks like this:

![alt text][image5]

Then using the algorithm provided in the lectures, I identified the pixels corresponding to lane lines. This method consists of two steps:

1. Generate a histogram of bottom half pixels of the image.

2. Use a sliding window method to identify the pixels corresponding to peaks in the histogram.

Finally, a second-order polynomial fit is generated to approximate the pixels in the lane lines. All this code is present in the file `finding_lines.py`.

Here's an image of the detected lane lines in the test image:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

After identifying the lane lines and fitting a polynomial to the lane lines, I next calculate the radius of curvature of the lines. For this, the polynomial fit must be made in the physical coordinantes and not in the pixel coordinates. For this, the pixels are scaled to physical dimensions, and a new fit is generated. Then using the formula provided in video lectures, I calculate the radius of curvature. 

This part of the code is also in the file `finding_lines.py`. 

The radius of curvature values are embedded in the output test images. 



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
