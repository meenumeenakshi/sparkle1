# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os 
from keras.models import Sequential # Initialise our neural network model as a sequential network
from keras.layers import Conv2D # Convolution operation
from keras.layers import MaxPooling2D # Maxpooling function
from keras.layers import Flatten # Converting 2D arrays into a 1D linear vector.
from keras.layers import Dense # Perform the full connection of the neural network
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
from PIL import Image
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from skimage import io, transform
def cnn_classifier():
    cnn = Sequential()
    cnn.add(Conv2D(32,(3,3),input_shape=(50,50,3), padding='same', activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn.add(Conv2D(64, (3,3), padding='same', activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn.add(Flatten())
    cnn.add(Dense(500, activation = 'relu'))
    cnn.add(Dense(2, activation = 'sigmoid'))
    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    print("cnn.summary()")
    return cnn
def reshaped_image(image):
    return transform.resize(image,(50, 50, 3))
def load_images_from_folder():
    Images = os.listdir("C:\wasteseg")
    train_images = []
    train_labels = []
    for image in Images:
            l = [0,0] 
            if image.find('Bio') != -1:
                path = os.path.join("C:\wasteseg\train", image)
                img = cv2.imread(path)
                train_images.append(reshaped_image(img))
                l = [1,0] 
                train_labels.append(l)
            if image.find('non_bio') != -1:
                path = os.path.join("C:\wasteseg\train", image)
                img = cv2.imread(path)
                       # print img_src
                train_images.append(reshaped_image(img))
                l = [0,1] 
                train_labels.append(l)
    return np.array(train_images), np.array(train_labels)
def train_test_split(train_images, train_labels, fraction):
    index = int(len(train_images)*fraction)
    return train_images[:index], train_labels[:index],train_images[index:],train_labels[index:]

train_images, train_labels = load_images_from_folder()
test_data, test_labels = load_images_from_folder()
fraction = 0.8
train_data = train_test_split(train_images,train_labels,fraction) 
train_labels = train_test_split(train_images, train_labels, fraction)
print ("Train data size: ", len(train_images.shape))
print ("Test data size: ", len(test_data.shape))

cnn = cnn_classifier()
print("train data shape:",train_images.shape)
print("test data shape:",test_data.shape)
idx = np.random.permutation(train_images.shape[0])
print(idx,"67")
cnn.fit(train_images[idx], train_labels[idx], batch_size = 64, epochs = 10)
predicted_test_labels = np.argmax(cnn.predict(test_data), axis=1)
test_labels = np.argmax(test_labels, axis=1)

print ("Actual test labels:", test_labels)
print ("Predicted test labels:", predicted_test_labels)
print ("Accuracy score:", accuracy_score(test_labels, predicted_test_labels))