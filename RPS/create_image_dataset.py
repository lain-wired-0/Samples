# use opencv to capture images 
# organize captured images into file paths
# image_dataset -> hand_gesture_1 -> img
#                                 -> img
#               -> hand_gesture_2 -> img


import cv2
from imutils.video import VideoStream
import imutils
import argparse
import time
import os
import numpy as np
from numpy import percentile

import skin_color


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='name of dataset you want to save in')
ap.add_argument('-i', '--image', required=True, help='name of image you\'re recording')
ap.add_argument('-n', '--number-of-images', required=True, help='number of images you want saved in the dataset')
ap.add_argument('-p', '--dataset-path', required=True, help='out path for dataset folder to reside in')
args = vars(ap.parse_args())

# make folder for image dataset
dataset_path = args['dataset_path'] + os.sep + args['dataset']
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
img_dataset_path = dataset_path + os.sep + args['image']
if not os.path.exists(img_dataset_path):
    os.makedirs(img_dataset_path)

# start videostream
vs = cv2.VideoCapture(0)
time.sleep(1.0)

frame_count = 0
num_images_taken = 1
start_download_frame = 500
area_lis = []
hsv_range = None

# auto edge detection
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

# process frame to create mask that detects skin color
def preprocess(roi_img, hsv_range):
    blur = cv2.GaussianBlur(roi_img, (3,3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

    lower_color = np.array([108, 23, 82], dtype = "uint8")
    upper_color = np.array([179, 255, 255], dtype = "uint8")
    # lower_color = np.array(hsv_range[0], dtype = "uint8")
    # upper_color = np.array(hsv_range[1], dtype = "uint8")

    mask = cv2.inRange(hsv, lower_color, upper_color)
    blur = cv2.medianBlur(mask, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    hsv_d = cv2.dilate(blur, kernel)

    return hsv_d

# test preprocess to test a new hsv_range
def preprocess_2(roi_img, hsv_range):
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_RGB2HSV)
    lower_color = np.array([0,30,60], dtype = "uint8")
    upper_color = np.array([20,150,255], dtype = "uint8")
    mask = cv2.inRange(hsv, lower_color, upper_color)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return mask

# extract only parts of frame with skin, rest of frame is thresholded, black
def extractSkin(image):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Defining HSV Threadholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)
    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)
    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


while True:
    # get frame
    _, frame = vs.read()
    frame = imutils.resize(frame, width=100)

    # draw rectangle for hand to be in
    # if hsv_range == None:
    #     cv2.putText(frame, 'Put hand in box', (30, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
    #     # [y:y+h, x:x+w]
    #     roi=frame[6:25, 6:25]
    #     cv2.rectangle(frame,(5,5),(25,25),(0,255,0),1)

    # take picture of hand to extract hand color
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('c'):
    #     roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    #     rgb_colors = skin_color.get_color(roi, 1)
    #     print(rgb_colors)
    #     hsv_colors = skin_color.rgb_to_hsv(rgb_colors)
    #     print(hsv_colors)
    #     hsv_range = skin_color.get_hsv_range(hsv_colors, 40)
    #     print(hsv_range)

    # threshold frame and feed it into preprocess to get better skin detection
    frame_copy = extractSkin(frame)
    mask = preprocess(frame_copy, hsv_range)

    # grab contours
    contours= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(contours)

    # if there is a contour prediction
    max_contour = None
    if len(cnts) != 0:
        # get contour with biggest area
        max_contour = max(cnts, key=cv2.contourArea)
        # get list of contour areas for 5number summary
        area_lis.append(cv2.contourArea(max_contour))
        # draw contour with filled color
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)#cv2.FILLED) 

        height, width, _ = frame.shape
        min_x, min_y = width, height
        max_x = max_y = 0
        # computes the bounding box for the contour, and draws it on the frame
        (x,y,w,h) = cv2.boundingRect(max_contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        if w > 80 and h > 80:
            roi=frame[y:y+h, x:x+h]
        elif max_x - min_x > 0 and max_y - min_y > 0:
            roi=frame[min_y:max_y, min_x:max_x]
        
    # fill outside of contour with black
    # frame[np.where((frame!=[0,255,0]).all(axis=2))] = [0,0,0]

    # save image in correct dataset path
    # frame_count += 1
    # if frame_count >= start_download_frame:
    #     quartiles = percentile(area_lis, [25, 50, 75])
    #     print(min(area_lis), quartiles[0], quartiles[1], quartiles[2], max(area_lis))

    #     if frame_count % 2 == 0 and (quartiles[0]<cv2.contourArea(max_contour)<quartiles[2]):
    #         cv2.resize(roi, (55,55), interpolation = cv2.INTER_AREA)
    #         cv2.imwrite(img_dataset_path + os.sep + 'img_' + str(num_images_taken) + '.png', roi)
    #         num_images_taken += 1

    # draw rectangles
    if w > 80 and h > 80:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 1)
    elif max_x - min_x > 0 and max_y - min_y > 0:
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 1)

    # show frame
    cv2.imshow('Frame', frame)

    # exit conditions
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or num_images_taken == int(args['number_of_images']) + 1:
        break


cv2.destroyAllWindows()
vs.release()