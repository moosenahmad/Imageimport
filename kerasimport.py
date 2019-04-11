import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


train_path = 'images/train'
valid_path = 'images/valid'
test_path = 'images/test'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(448,448), classes=['stop', 'yield'], batch_size = 3)
valid_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(448,448), classes=['stop', 'yield'], batch_size = 3)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(448,448), classes=['stop', 'yield'], batch_size = 8)

imgs, labels = next(train_batches)

model = Sequential([
	Conv2D(64, (3, 3), activation='relu', input_shape=(448,448,3)),
	Flatten(),
	Dense(2, activation='softmax'),
])

model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=4, validation_data = valid_batches, validation_steps = 4, epochs=5,verbose=2)

test_imgs, test_labels = next(test_batches)

test_labels = test_labels[:,0]

predictions = model.predict_generator(test_batches, steps=1, verbose=0)

print(predictions)
