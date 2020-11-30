from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class Fruit_CNN:
    @staticmethod
    def build_cnn(width, height, depth, classes):
        model = Sequential()
        input_shape = (height, width, depth)

        channel_dimension = -1

        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            channel_dimension= 1

        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
        # model.add(BatchNormalization(axis=channel_dimension)) # so that model converges faster
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, padding='same',kernel_size=(3, 3), activation='relu'))
        # model.add(BatchNormalization(axis=channel_dimension))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # model.add(Conv2D(128, padding='same',kernel_size=(3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        # model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))

        return model