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


ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help='path to trained model')
ap.add_argument('-l', '--labelbin', required=True, help='path to label binarizer')
args = vars(ap.parse_args())

# image_paths = list(paths.list_images('/Users/mandywoo/Documents/cnn_projects/fruits360-kaggle/fruits/fruits-360/Test'))
image_paths = list(paths.list_images('fruits360-kaggle/dataset/Backgrounds'))
random.seed(42)
random.shuffle(image_paths)

correct_count = 0

for image_path in image_paths:
    # read image
    # image = cv2.imread(args['image'])
    image = cv2.imread(image_path)
    output = image.copy()

    # preprocess image for classification
    image = cv2.resize(image, (100, 100))
    image = image.astype('float') / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # load trained convolutional neural network and label binarizer
    print('[INFO] loading network...')
    model = load_model(args['model'])
    lb = pickle.loads(open(args['labelbin'], 'rb').read())

    # classify input image
    print('[INFO] classifying image...')
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]


    filename = image_path.split(os.path.sep)[-2]
    # correct = 'correct' if filename.rfind(label) != -1 else 'incorrect'
    if filename.rfind(label) != -1:
        correct = 'correct'
        correct_count += 1
    else:
        correct = 'incorrect'


    # build and draw label on image
    label = '{}: {:.2f}% ({})'.format(label, proba[idx] * 100, correct)
    output = imutils.resize(output, width=400)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print('[INFO] {}'.format(label))
    cv2.imshow('Output', output)
    cv2.waitKey(0)

print('Correct Percentage: ' + str(correct_count/len(image_paths)))
cv2.waitKey(0)






