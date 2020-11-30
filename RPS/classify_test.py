from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import random
from imutils import paths
from imutils.video import VideoStream
import time

def predict_img(model, lb, img):
    # classifying input image
    probabilities = model.predict(img)[0]
    print(probabilities)
    index = np.argmax(probabilities)       # argmax: Returns the indices of the maximum values along an axis.
    print(lb.classes_, index)
    label = lb.classes_[index]
    return probabilities[index], label


ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', help='path to saved model')
ap.add_argument('-l', '--labels', help='path to saved binarized labels')
ap.add_argument('-i', '--image', help='path to image')
args = vars(ap.parse_args())

# load network and labels
print('[INFO] loading network...')
model = load_model(args['model'])
lb = pickle.loads(open(args['labels'], 'rb').read())

img = args['image']
img = cv2.imread(img)
img = img.astype('float') / 255.0
img = cv2.resize(img, (100, 100))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
print(predict_img(model, lb, img))