[image1]: ./images/modelloss.jpg "Accuracy per Epoch and most Error Prone Classes"


# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains files for the Behavioral Cloning Project from udacity nano degree.

In this project, I use what I've learned about deep neural networks and convolutional neural networks to clone driving behavior. I trained, validatde and tested a model using Keras. The model outputs a steering angle to control and keep a simulated vehicle over a predefined path.

# Main Goals of the Project
Main goals of the project were as follows:
* Develop and implement a deep neural netowrk to be able to clone driving behavior of a human
* Collect driving data for training the developed model
* Train and test the developed model
* Using the simulator, verify that developed model is able to autonomously drive the car

# Important files
Important files here include
- **model.py** : It contains the script to create and train the model using the collected in subdirectories of **./data**
- **drive.py** : Provided by [Udacity](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/drive.py) for model testing
- **model.h5** : Contains a trained convolution neural network.
- **writeup_report.md** : Results summary
- **images/VideoOut.mp4** a video of simulator driving the car autonomously around the track

# Train data accumulation
To begin with some data was provided by Udacity for training. My initial estimate was that this would be enough and I would not need to collect more since collecting 
quality data on the simulator over remote desktop and without joystick is not easily possible. Furthermore, the provided data was augmented by following steps:

* image from center camera was flipped with steering angle reversed
* image from left camera was used with a correction factor of 0.2
* image from right camera was used with a correction factor of -0.2

This way every acquired data point resulted in four training images. 

However, I found out that training using this data and nVidia model was not enough to successfully drive the car on the track autonommously. 

So, I recorded two more rounds on track 1 and one round on track two. After this I saw major improvement in the results. Same techniques for data augmentation were 
also applied to the recorded data. 



# Model Summary 


```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
_________________________________________________________________
```


The model I am using is a variant of nVidia model. I started with full nVidia model as shown above. After training it for 3 
epochs the car can autonomously complete a lap of track1 as can be seen here:  [video](images/videoOut.mp4). There were no 
dropout layers for regularization so the epochs were kept low (3) to avoid overfitting. 

I started with this model and iteratively reduced the complexity reducing the complexity as long as the validation loss 
remained below 0.07. With this model the car was able to drive around the track 1 autonomously. 

After this I explored further and attempted to reduce the size of this model. 
With the strategy of trial and error I was iteratively able to reduce this model to the following version which is still capable of driving autonomously. 
It seems reducing it any further by removing a layer or reducing number of filters in convoulutional layers renders it incapable of controlling the for a 
whole lap of track 1. 


```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 32, 160, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 78, 24)        1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 5, 37, 36)         21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 3, 35, 48)         15600     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 1, 33, 64)         27712     
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 20)                2020      
_________________________________________________________________
dropout_2 (Dropout)          (None, 20)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                210       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 280,313
Trainable params: 280,313
Non-trainable params: 0
_________________________________________________________________
```

Notice the reduction in number of trainable params from approximately 1 million to around 300,000. This results in a noticeable speed up in network training.
In addition, to further speed-up the network training, the size of the input image is reduced to half along each axis in the first lambda layer. 
This results in 4 times less pixel per image. In total, the training time for second model is reduced by alomst a factor of two, though I did not actually measure it.

For parameter tune-up: As I used Adam optimizer, I did not need to optimize the learning rate manually.

The video output for the second version of model is uploaded here: [video](images/videoOut2.mp4)

In the following daigram I shows the training beahvior of the model.


![alt text][image1]


