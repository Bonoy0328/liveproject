import keras
from keras.models import Sequential
from keras.layers import Flatten
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import regularizers, optimizers
import numpy as np
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
# download and split the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
  
# break training set into training and validation sets
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

mean = np.mean(x_train)
std = np.std(x_train)
x_train = (x_train-mean)/(std+1e-7)
x_valid = (x_valid-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

# one-hot encode the labels in train, valid, and test datasets
# we use ??to_categorical?? function in keras
num_classes = 10
y_train = np_utils.to_categorical(y_train,num_classes)
y_valid = np_utils.to_categorical(y_valid,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)

# data augmentation
datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
        )
# compute the data augmentation on the training set
datagen.fit(x_train)

base_hidden_units = 32
weight_decay = 1e-4

model = Sequential()

model.add(Conv2D(base_hidden_units,kernel_size = 3,padding = "same",kernel_regularizer=regularizers.l2(weight_decay),input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(base_hidden_units,kernel_size=3,padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(base_hidden_units *2 ,kernel_size=3,padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(base_hidden_units * 2,kernel_size=3,padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(10,activation='softmax'))

model.summary()

batch_size = 128
epochs = 125

checkpointer = ModelCheckpoint(filepath='model.100epochs.hdf5',save_best_only=True)

optimizer = keras.optimizers.Adam(lr=0.0001,decay=1e-6)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

if os.path.exists("model.100epochs.hdf5"):
    model.load_weights("model.100epochs.hdf5")

history = model.fit(datagen.flow(x_train,y_train,batch_size=batch_size),callbacks=[checkpointer],
                              steps_per_epoch=x_train.shape[0]/batch_size,validation_data=(x_valid,y_valid))


