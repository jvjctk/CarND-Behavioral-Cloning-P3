#!/usr/bin/env python
# coding: utf-8

# In[47]:


import csv
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout

from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K


# In[28]:


lines = []
with open('data/driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)


# In[38]:


# only for visualization
col_names = ['center', 'left', 'right',
                'steering', 'throttle', 'brake', 'speed']
data_reader = pd.DataFrame(lines, columns=col_names)
data_reader.iloc[:5]


# In[46]:


# steering angle visualization
data_reader["steering"] = pd.to_numeric(data_reader["steering"])
data_reader['steering'].plot.hist(bins=25)


# In[97]:


# visualizing random images
from random import seed
from random import randint

fig,axs = plt.subplots(5, 3, figsize=(20, 15))
axs = axs.ravel()
for number in range(5):
    value = randint(0, 2000)
    img = plt.imread(data_reader["left"][value][14:])
    axs[number*3].imshow(img)
    axs[number*3].set_title('left camera')
    axs[number*3].axis('off')
    
    img = plt.imread(data_reader["center"][value][14:])
    axs[number*3+1].imshow(img)
    axs[number*3+1].set_title('center camera')
    axs[number*3+1].axis('off')
    
    img = plt.imread(data_reader["right"][value][14:])
    axs[number*3+2].imshow(img)  
    axs[number*3+2].set_title('right camera')
    axs[number*3+2].axis('off')


# In[106]:


# visualizing flipped image

fig,axs = plt.subplots(3, 2, figsize=(20, 15))
axs = axs.ravel()
count = 0

value = randint(0, 2000)

camera = ['left', 'center', 'right']

for idx ,side in enumerate(camera):
    img = plt.imread(data_reader[side][value][14:])
    axs[idx*2].imshow(img)
    axs[idx*2].set_title("{} camera".format(side))
    axs[idx*2].axis('off')

    img_flipped = np.fliplr(img)
    axs[idx*2+1].imshow(img_flipped)
    axs[idx*2+1].set_title("{} camera flipped".format(side))
    axs[idx*2+1].axis('off')
    


# In[93]:


car_images=[]
steering_angles=[]

for line in lines:
    
    steering_center = float(line[3])
        
    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # read in images from center, left and right cameras
    filename_center = line[0].split('/')[-1]
    filename_left = line[1].split('/')[-1]
    filename_right = line[2].split('/')[-1]
    filepath_center = 'data/IMG/' + filename_center
    filepath_left = 'data/IMG/' + filename_left 
    filepath_right = 'data/IMG/' + filename_right

    img_center = np.asarray(Image.open(filepath_center))
    img_left = np.asarray(Image.open(filepath_left))
    img_right = np.asarray(Image.open(filepath_right))
        
    car_images.extend([img_center, img_left, img_right])
    steering_angles.extend([steering_center, steering_left, steering_right])
    
    img_center_flipped = np.fliplr(img_center)
    img_left_flipped = np.fliplr(img_left)
    img_right_flipped = np.fliplr(img_right)
    
    steering_center_flipped = -steering_center
    steering_left_flipped = -steering_left
    steering_right_flipped = -steering_right

    car_images.extend([img_center_flipped, img_left_flipped, img_right_flipped])
    steering_angles.extend([steering_center_flipped, steering_left_flipped, steering_right_flipped])
    
y_train = np.array(steering_angles) # training labels
X_train=np.array(car_images)   # training image pixels


# In[94]:


#printing lenght of image
print (len(steering_angles))


# In[114]:


# defining model

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x/ 255.0 - 0.5)))
model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2),activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(5, 5),strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2),activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3) ,activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3),activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1))
print(model.summary())


# In[6]:


model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=3)
model.save('model.h5')

