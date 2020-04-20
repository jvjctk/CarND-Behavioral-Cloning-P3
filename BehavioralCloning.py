#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
import cv2
import numpy as np

from PIL import Image
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout

from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K


# In[3]:


lines=[]

car_images=[]
steering_angles=[]
with open('data/driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
       
        steering_center = float(row[3])
        
        # create adjusted steering measurements for the side camera images
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        filename_center = row[0].split('/')[-1]
        filename_left = row[1].split('/')[-1]
        filename_right = row[2].split('/')[-1]
        filepath_center = 'data/IMG/' + filename_center
        filepath_left = 'data/IMG/' + filename_left 
        filepath_right = 'data/IMG/' + filename_right

        img_center = np.asarray(Image.open(filepath_center))
        img_left = np.asarray(Image.open(filepath_left))
        img_right = np.asarray(Image.open(filepath_right))
        car_images.extend([img_center, img_left, img_right])
        steering_angles.extend([steering_center, steering_left, steering_right])

y_train = np.array(steering_angles) # training labels
X_train=np.array(car_images)   # training image pixels


# In[4]:


import tensorflow as tf
def preprocess(image):  # preprocess image
    return tf.image.resize_images(image, (200, 66))


# In[5]:


model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(preprocess))
model.add(Lambda(lambda x: (x/ 255.0 - 0.5)))
model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2),activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(5, 5),strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2),activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3) ,activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3),activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1))
print(model.summary())


# In[6]:


model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=2)
model.save('model.h5')

