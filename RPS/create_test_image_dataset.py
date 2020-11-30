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
ap.add_argument('-i', '--image', required=True, help='name of image you\'re recording')
ap.add_argument('-n', '--number-of-images', required=True, help='number of images you want saved in the dataset')
ap.add_argument('-p', '--dataset-path', required=True, help='out path for dataset folder to reside in')
args = vars(ap.parse_args())

# make folder for image dataset
dataset_path = args['dataset_path'] + os.sep + 'Test_Dataset'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
img_dataset_path = dataset_path + os.sep + args['image']
if not os.path.exists(img_dataset_path):
    os.makedirs(img_dataset_path)

# start videostream
vs = cv2.VideoCapture(0)
time.sleep(1.0)
# vs.set(cv2.CAP_PROP_FRAME_WIDTH, 100)
# vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)

# keep track of frame number for naming purposes
frame_count = 0
num_images_taken = 1


hsv_range = None


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

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

while True:
    _, frame = vs.read()
    frame = imutils.resize(frame, width=100)

    # if hsv_range == None:
    #     cv2.putText(frame, 'Put hand in box', (30, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
    #     # [y:y+h, x:x+w]
    #     roi=frame[6:25, 6:25]
    #     cv2.rectangle(frame,(5,5),(25,25),(0,255,0),1)


    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('c'):
    #     roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    #     rgb_colors = skin_color.get_color(roi, 1)
    #     print(rgb_colors)
    #     hsv_colors = skin_color.rgb_to_hsv(rgb_colors)
    #     print(hsv_colors)
    #     hsv_range = skin_color.get_hsv_range(hsv_colors, 40)
    #     print(hsv_range)

    
    mask = preprocess(frame, hsv_range)

    contours= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(contours)

    max_contour = None
    if len(cnts) != 0:
        # get contour with biggest area
        max_contour = max(cnts, key=cv2.contourArea)
        # draw contour with filled color
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), cv2.FILLED) 

        # fill outside of contour with black
        # frame[np.where((frame!=[0,255,0]).all(axis=2))] = [0,0,0]


    cv2.imshow('Frame', frame)

    # save image in correct dataset path
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
            cv2.imwrite(img_dataset_path + os.sep + 'img_' + str(num_images_taken) + '.png', frame)
            num_images_taken += 1

    # exit conditions
    if key == ord('q') or num_images_taken == int(args['number_of_images']) + 1:
        break


cv2.destroyAllWindows()
vs.release()