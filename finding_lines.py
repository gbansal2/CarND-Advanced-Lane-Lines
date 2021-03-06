import numpy as np
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
import os

def find_lines(binary_warped,fname,tracker,frame_count):

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
# Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    if tracker.detected == False:
        print("Full search")
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
        nwindows = 9
    # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
    # Set the width of the windows +/- margin
        margin = 100
    # Set minimum number of pixels found to recenter window
        minpix = 50
    # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

    # Step through the windows one by one
        for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2) 

    # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]


        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        #print(lefty.shape[0])
        #print(leftx.shape[0])
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        ratio_second_deg = abs(right_fit[0]/left_fit[0])
        #print("ratio_second_deg = ", ratio_second_deg)

        y_pix = binary_warped.shape[0]
        left_lane = left_fit[0]*y_pix**2 + left_fit[1]*y_pix + left_fit[2]
        right_lane = right_fit[0]*y_pix**2 + right_fit[1]*y_pix + right_fit[2]
        lane_width = np.absolute((right_lane - left_lane)*xm_per_pix)

        print("left_curverad = ", left_curverad,
                "right_curverad = ", right_curverad,
                "ratio_second_deg = ", ratio_second_deg,
                "lane width = ", lane_width)

#For first frame
        if (frame_count == 0):
            tracker.detected = True
            tracker.current_left_fit = left_fit
            tracker.current_right_fit = right_fit
            tracker.best_left_fit = left_fit
            tracker.best_right_fit = right_fit
            tracker.radius_of_curvature_left = left_curverad
            tracker.radius_of_curvature_right = right_curverad
        else:
            if ((left_curverad > 15000.0) or (left_curverad < 200.0)
                    or (right_curverad > 15000.0) or (right_curverad < 200.0)  
                    or (ratio_second_deg > 8.0) or
                    (ratio_second_deg < 0.2) or
                    (lane_width < 2.0) or 
                    (lane_width > 4.0)):
                tracker.detected = False
                print("Using previous good fit")
                tracker.current_left_fit = left_fit
                tracker.current_right_fit = right_fit
                left_fit = tracker.best_left_fit
                right_fit = tracker.best_right_fit
                left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
                left_curverad = tracker.radius_of_curvature_left
                right_curverad = tracker.radius_of_curvature_right
            else:
                tracker.detected = True
                print("Using new good fit")
                tracker.current_left_fit = left_fit
                tracker.current_right_fit = right_fit
                tracker.best_left_fit = left_fit
                tracker.best_right_fit = right_fit
                tracker.radius_of_curvature_left = left_curverad
                tracker.radius_of_curvature_right = right_curverad



        #compute center of camera (pixels)
        car_pos_pix = binary_warped.shape[1]/2.0
        y_pix = binary_warped.shape[0]
        left_lane = left_fit[0]*y_pix**2 + left_fit[1]*y_pix + left_fit[2]
        right_lane = right_fit[0]*y_pix**2 + right_fit[1]*y_pix + right_fit[2]
        center_lane = (left_lane + right_lane)/2.0
        offset_pix = (car_pos_pix - center_lane)
        offset_m = offset_pix*xm_per_pix
        #print("left_lane = ", left_lane,
        #        "right_lane = ", right_lane,
        #        "car_ps = ", car_pos_pix,
        #        "offset_pix = ", offset_pix,
        #        "offset_m = ", offset_m)




        # Now our radius of curvature is in meters
        print(left_curverad, 'm', right_curverad, 'm')
        # Example values: 632.1 m    626.2 m

        plt.figure()
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        ##plt.text(700,200,'left curvature  = %6.2f m \nright_curvature = %6.2f m \n' %(left_curverad,right_curverad), color='white')
        #plt.show()
        save_fname = os.path.join('output_images', 'fits_'+os.path.basename(fname))
        plt.savefig(save_fname)
        #plt.close(fig1)

        return [left_fitx, right_fitx, ploty, left_curverad, right_curverad,offset_m]
    else:
        print("Margin search")
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100

        left_fit = tracker.current_left_fit 
        right_fit = tracker.current_right_fit

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

	# Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
	# Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
	# Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        #print(lefty.shape[0])
        #print(leftx.shape[0])
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        ratio_second_deg = abs(right_fit[0]/left_fit[0])
        #print("ratio_second_deg = ", ratio_second_deg)

        y_pix = binary_warped.shape[0]
        left_lane = left_fit[0]*y_pix**2 + left_fit[1]*y_pix + left_fit[2]
        right_lane = right_fit[0]*y_pix**2 + right_fit[1]*y_pix + right_fit[2]
        lane_width = np.absolute(right_lane - left_lane)*xm_per_pix
        #US highway lane width is 3.7m

        print("left_curverad = ", left_curverad,
                "right_curverad = ", right_curverad,
                "ratio_second_deg = ", ratio_second_deg,
                "lane_width = ", lane_width)

        if ((left_curverad > 15000.0) or (left_curverad < 200.0)
                or (right_curverad > 15000.0) or (right_curverad < 200.0)  
                or (ratio_second_deg > 8.0) or
                (ratio_second_deg < 0.2) or
                (lane_width < 2.0) or 
                (lane_width > 4.0)):
            tracker.detected = False
            print("Using prev fit")

            tracker.current_left_fit = left_fit
            tracker.current_right_fit = right_fit

            left_fit = tracker.best_left_fit
            right_fit = tracker.best_right_fit

            left_curverad = tracker.radius_of_curvature_left
            right_curverad = tracker.radius_of_curvature_right

            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        else:
            tracker.detected = True

            tracker.current_left_fit = left_fit
            tracker.current_right_fit = right_fit
            tracker.best_left_fit = left_fit
            tracker.best_right_fit = right_fit
            tracker.radius_of_curvature_left = left_curverad
            tracker.radius_of_curvature_right = right_curverad
            print("Using new good fit")

        # Now our radius of curvature is in meters
        print(left_curverad, 'm', right_curverad, 'm')

        car_pos_pix = binary_warped.shape[1]/2.0
        y_pix = binary_warped.shape[0]
        left_lane = left_fit[0]*y_pix**2 + left_fit[1]*y_pix + left_fit[2]
        right_lane = right_fit[0]*y_pix**2 + right_fit[1]*y_pix + right_fit[2]
        center_lane = (left_lane + right_lane)/2.0
        offset_pix = (car_pos_pix - center_lane)
        offset_m = offset_pix*xm_per_pix

        fig1 = plt.figure()
        plt.imshow(out_img)
        #plt.show()
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        ##plt.text(700,200,'left curvature  = %6.2f m \nright_curvature = %6.2f m \n' %(left_curverad,right_curverad), color='white')
        #plt.show()
        save_fname = os.path.join('output_images', 'fits_'+os.path.basename(fname))
        plt.savefig(save_fname)
        plt.close(fig1)

        return [left_fitx, right_fitx, ploty, left_curverad, right_curverad,offset_m]

