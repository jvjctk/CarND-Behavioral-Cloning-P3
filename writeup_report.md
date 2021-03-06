# **Behavioral Cloning** 

## Writeup 



**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 32 and 128 (model.py lines 165 -169) 

The model includes RELU layers to introduce nonlinearity (code line 165 - 169), and the data is normalized in the model using a Keras lambda layer (code line 164). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 171, 173, 175). 

The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line ##).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to drive the vehicle autonomously in the simulator.

My first step was to use a convolution neural network model similar to the 'end_to_end_dl' by Nvidia. The is given in their website. http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf. 
I thought this model might be appropriate because it is used in the same application.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I added dropout layers in between fully connected layers. It reduced connections reduced the overfitting behaviour

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.  To improve the driving behavior in these cases, I added/subtracted corrections to the steering angle as .2

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 

#### Layer (type)                 Output Shape              Param  
_________________________________________________________________
cropping2d_3 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
lambda_4 (Lambda)            (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 20, 77, 36)        21636     
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_35 (Conv2D)           (None, 4, 33, 64)         36928     
_________________________________________________________________
flatten_7 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 8448)              0         
_________________________________________________________________
dense_25 (Dense)             (None, 100)               844900    
_________________________________________________________________
dropout_5 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_26 (Dense)             (None, 50)                5050      
_________________________________________________________________
dropout_6 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_11 (Dense)             (None, 10)                510       
_________________________________________________________________
#### dense_12 (Dense)             (None, 1)                 11        
#### Total params: 981,819
#### Trainable params: 981,819
##### Non-trainable params: 0
_________________________________________________________________

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text](./images/cameraimages.JPG)


To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text](./images/flippedimage.JPG)


After the collection process, I had 37218 number of data points. Each image has a initial dimension of 160x320x3. I preprocessed this data by 50 pixels from above and 20 pixels from below. Then each image has the dimension of 90x320x3


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by trial and error method. I used an adam optimizer so that manually training the learning rate wasn't necessary.
