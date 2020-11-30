# so figures can be saved in background
import matplotlib
matplotlib.use('Agg')

from keras.preprocessing.image import ImageDataGenerator   # for data augmentation
from keras.optimizers import Adam   # optimizer used to train network
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array
# allows us to input set of class labels, transform labels into one-hot encoded vectors, 
# then allow us to take an integer class label prediction from Keras CNN and transform it back into a human-readable label
from sklearn.preprocessing import LabelBinarizer    
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from cnn import RPS_CNN
from batch_size import FindBatchSize

import time

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to dataset')
ap.add_argument('-m', '--model', required=True, help='path to save model')
ap.add_argument('-l', '--labels', required=True, help='path to save label binarizer')
ap.add_argument('-g', '--graph', required=True, help='path to save plot')
args = vars(ap.parse_args())

EPOCHS = 100    # num of epochs to train for (how many times the network sees each training example and learns patterns from it)
INIT_LR = 1e-3  # initial learning rate (1e-3 is default for Adam optimizer)
BS = FindBatchSize(RPS_CNN)      # batch size (we will pass batches of images into the network for training)
IMAGE_DIMS = (55, 55, 3)    # image dimensions 
PATIENCE = 10


print('[INFO] loading images...')
image_paths = sorted(list(paths.list_images(args['dataset'])))
random.seed(42)
random.shuffle(image_paths)

# store image and corresponding label
data = []
labels = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    cv2.imshow("img", image)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    label = image_path.split(os.sep)[-2]
    labels.append(label)


# scale raw pixel intensities to range [0, 1] for normalisation
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)
print('[INFO] data matrix: {:.2f}MB'.format(data.nbytes / (1024 * 1000.0)))

# binarize labels
print('[INFO] binarizing labels')
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# split training data into training - 80% validation - 20%
print('[INFO] splitting data...')
train_x, val_x, train_y, val_y = train_test_split(data, labels, test_size=0.2, random_state=42)

# create model
print('[INFO] creating model...')
model = RPS_CNN.build_cnn(IMAGE_DIMS[1], IMAGE_DIMS[0], IMAGE_DIMS[2], len(lb.classes_))
# optimizer = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
optimizer = SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# augment data to increase variety in dataset and prevent overfitting
print('[INFO] augmenting data...')
aug_data=ImageDataGenerator(featurewise_center=False, #set input mean to 0
                           samplewise_center=False,  #set each sample mean to 0
                           featurewise_std_normalization=False, #divide input datas to std
                           samplewise_std_normalization=False,  #divide each datas to own std
                           zca_whitening=False,  #dimension reduction
                           rotation_range=20,    #rotate 5 degree
                           zoom_range=0.5,        #zoom in-out 5%
                           width_shift_range=0.5, #shift 5%
                           height_shift_range=0.5,
                           shear_range=0.15,
                           horizontal_flip=False,  #randomly flip images
                           vertical_flip=True,
                           )
aug_data.fit(train_x)

# make patient early stopping
print('[INFO] initiating tools for early stopping and saving')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
mc = ModelCheckpoint(args['model'], monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# train network
print('[INFO] training network')
history = model.fit_generator(aug_data.flow(train_x, train_y, batch_size=BS), 
                                            validation_data=(val_x, val_y),
                                            steps_per_epoch=len(train_x) // BS,
                                            epochs=EPOCHS,
                                            verbose=1,
                                            callbacks=[es, mc])

# saving network
print('[INFO] saving network...')
model.save(args['model'])

# saving label binarizer
print('[INFO] saving label binarizer')
f = open(args['labels'], 'wb')
f.write(pickle.dumps(lb))
f.close()

# plot training loss and accuracy
print('[INFO] plot graph')
plt.style.use('ggplot')
plt.figure()
if es.stopped_epoch != 0:
    stopped_epoch = es.stopped_epoch + 1
else:
    stopped_epoch = EPOCHS
plt.plot(np.arange(0, stopped_epoch), history.history['loss'], label='train_loss')
plt.plot(np.arange(0, stopped_epoch), history.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, stopped_epoch), history.history['acc'], label='train_acc')
plt.plot(np.arange(0, stopped_epoch), history.history['val_acc'], label='val_acc')
plt.title('Train Loss Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='upper left')
plt.savefig(args['graph'])