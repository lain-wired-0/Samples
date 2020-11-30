# so figures can be saved in background
import matplotlib
matplotlib.use('Agg')

from keras.preprocessing.image import ImageDataGenerator   # for data augmentation
from keras.optimizers import Adam   # optimizer used to train network
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

from cnn import Fruit_CNN

EPOCHS = 25 #100    # num of epochs to train for (how many times the network sees each training example and learns patterns from it)
INIT_LR = 1e-3  # initial learning rate (1e-3 is default for Adam optimizer)
BS = 32   #64      # batch size (we will pass batches of images into the network for training)
IMAGE_DIMS = (100, 100, 3)    # image dimensions (96x96 pixels, 3 channels)

print('[INFO] loading images....')
train_image_paths = sorted(list(paths.list_images('/Users/mandywoo/Documents/cnn_projects/fruits360-kaggle/fruits/fruits-360/Training')))
random.seed(42)
random.shuffle(train_image_paths)

test_image_paths = sorted(list(paths.list_images('/Users/mandywoo/Documents/cnn_projects/fruits360-kaggle/fruits/fruits-360/Test')))
random.seed(42)
random.shuffle(test_image_paths)

# store training data and corresponding labels
train_data = []
train_labels = []
for train_image_path in train_image_paths:
    image = cv2.imread(train_image_path)
    image = img_to_array(image)
    train_data.append(image)

    label = train_image_path.split(os.path.sep)[-2]
    train_labels.append(label)

# store testing data and corresponding labels
# testing data is for testing the finished model
test_data = []
test_labels = []
for test_image_path in test_image_paths:
    image = cv2.imread(test_image_path)
    image = img_to_array(image)
    test_data.append(image)

    label = test_image_path.split(os.path.sep)[-2]
    test_labels.append(label)

# scale raw pixel intensities to range [0, 1] for normalisation
# training data
train_data = np.array(train_data, dtype='float') / 255.0
train_labels = np.array(train_labels)
print('[INFO] train data matrix: {:.2f}MB'.format(train_data.nbytes / (1024 * 1000.0)))
# test data
test_data = np.array(test_data, dtype='float') / 255.0
test_labels = np.array(test_labels)
print('[INFO] test data matrix: {:.2f}MB'.format(test_data.nbytes / (1024 * 1000.0)))

# binarize labels
print('[INFO] binarizing labels')
lb_train = LabelBinarizer()
lb_test = LabelBinarizer()
train_labels = lb_train.fit_transform(train_labels)
test_labels = lb_test.fit_transform(test_labels)

# split training data into training-80% and validation-20%
print('[INFO] split data')
train_x, val_x, train_y, val_y = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# create model
print('[INFO] creating model')
model = Fruit_CNN.build_cnn(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=len(lb_train.classes_))
optimizer = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# augment data by flipping, rotating, zooming, shifting to prevent overfitting
print('[INFO] augmenting data')
aug_data=ImageDataGenerator(featurewise_center=False, #set input mean to 0
                           samplewise_center=False,  #set each sample mean to 0
                           featurewise_std_normalization=False, #divide input datas to std
                           samplewise_std_normalization=False,  #divide each datas to own std
                           zca_whitening=False,  #dimension reduction
                           rotation_range=0.5,    #rotate 5 degree
                           zoom_range=0.5,        #zoom in-out 5%
                           width_shift_range=0.5, #shift 5%
                           height_shift_range=0.5,
                           horizontal_flip=False,  #randomly flip images
                           vertical_flip=False,
                           )
aug_data.fit(train_x)

# train network
print('[INFO] training network')
history = model.fit_generator(aug_data.flow(train_x, train_y, batch_size=BS), 
                            validation_data=(val_x, val_y),
                            steps_per_epoch=len(train_x) // BS,
                            epochs=EPOCHS,
                            verbose=1)

# saving model
print('[INFO] serializing network...')
model.save('/Users/mandywoo/Documents/cnn_projects/fruits360-kaggle/fruits_cnn.model')

# save label binarizer to disk
print('[INFO] serializing training label binarizer...')
f = open('/Users/mandywoo/Documents/cnn_projects/fruits360-kaggle/lb_train.pickle', 'wb')
f.write(pickle.dumps(lb_train))
f.close()

# plot training loss and accuracy
plt.style.use('ggplot')
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), history.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), history.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, N), history.history['acc'], label='train_acc')
plt.plot(np.arange(0, N), history.history['val_acc'], label='val_acc')
plt.title('Train Loss Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='upper left')
plt.savefig('/Users/mandywoo/Documents/cnn_projects/fruits360-kaggle/plot_1.png')