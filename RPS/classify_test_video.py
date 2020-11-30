import cv2
from imutils.video import VideoStream
import imutils
import argparse
import time
import os
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import pickle
import os
import random
from imutils import paths
from imutils.video import VideoStream
import time

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', help='path to saved model')
ap.add_argument('-l', '--labels', help='path to saved binarized labels')
args = vars(ap.parse_args())


vs = cv2.VideoCapture(0)
time.sleep(1.0)

model = load_model(args['model'])
lb = pickle.loads(open(args['labels'], 'rb').read())

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

def predict_img(model, lb, img):
    # classifying input image
    probabilities = model.predict(img)[0]
    # print(probabilities)
    index = np.argmax(probabilities)       # argmax: Returns the indices of the maximum values along an axis.
    # print(lb.classes_, index)
    label = lb.classes_[index]
    return probabilities[index], label

while True:
    _, frame = vs.read()
    frame = imutils.resize(frame, width=100)

    frame_copy = extractSkin(frame)
    hsv_range = None
    mask = preprocess(frame_copy, hsv_range)

    # fill outside of contour with black
    frame[np.where((frame!=[0,255,0]).all(axis=2))] = [0,0,0]

    contours= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(contours)

    max_contour = None
    if len(cnts) != 0:
        # get contour with biggest area
        max_contour = max(cnts, key=cv2.contourArea)
        # draw contour with filled color
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), cv2.FILLED) 

        height, width, _ = frame.shape
        min_x, min_y = width, height
        max_x = max_y = 0
        # computes the bounding box for the contour, and draws it on the frame
        (x,y,w,h) = cv2.boundingRect(max_contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        if w > 80 and h > 80:
            roi=frame[y:y+h, x:x+h]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 1)
        

        if max_x - min_x > 0 and max_y - min_y > 0:
            roi=frame[min_y:max_y, min_x:max_x]
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 1)

        roi = roi.astype('float') / 255.0
        roi = cv2.resize(roi, (55, 55))
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        # print(predict_img(model, lb, roi))
        prob, label = predict_img(model, lb, roi)
        text = str(prob) + ' - ' + label
    cv2.putText(frame, str(prob), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)

    cv2.imshow('Prediction', frame)

    # exit conditions
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cv2.destroyAllWindows()
vs.release()