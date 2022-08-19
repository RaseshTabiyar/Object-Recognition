# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:18:31 2021

@author: Jagriti
"""

"""The dataset we will be using is the CIFAR-10 dataset, which consists
 of 60,000 32x32 color images in 10 classes, with 6000 images per class.
 There are 50000 training images and 10000 test images.
 
 https://arxiv.org/pdf/1412.6806.pdf     """
 

from keras.datasets import cifar10
data= cifar10.load_data()  
(X_train, y_train), (X_test, y_test) = data

"""-----------------------------------------------------------------"""
# DETERMINE DATASET CHARACTERISTICS
print('Training Images: {}'.format(X_train.shape))
print('Testing Images: {}'.format(X_test.shape))

import matplotlib.pyplot as plt

# create a grid of 3x3 images
for i in range(0,9):
    plt.subplot(330 + 1 + i)
    img = X_train[i]
    plt.imshow(img)
    
plt.show() 

"""-----------------------------------------------------------------"""
#PREPROCESSING THE DATASET
# fix random seed for reproducibility
import numpy as np
seed = 6
np.random.seed(seed)  

# normalize the inputs
X_train = X_train / X_train.max()
X_test = X_test / X_test.max()


"""-----------------------------------------------------------------"""
"""The class labels are a single integer value (0-9).
 What we really want is a one-hot vector of length ten.
 For example, the class label of 6 should be denoted as
 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]. 
 We can accomplish this using the np_utils.to_categorical() function """
 
 
from keras.utils import np_utils 
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)
num_classes = Y_test.shape[1]




"""-----------------------------------------------------------------"""
#BUILDING CNN

""" The three base networks used for classification on CIFAR-10 and CIFAR-100.

Input 32 × 32 RGB image
MODEL C
 
3 × 3 conv. 96 ReLU
3 × 3 conv. 96 ReLU
3 × 3 max-pooling stride 2

3 × 3 conv. 192 ReLU 
3 × 3 conv. 192 ReLU
3 × 3 max-pooling stride 2

3 × 3 conv. 192 ReLU
1 × 1 conv. 192 ReLU
1 × 1 conv. 10 ReLU
global averaging over 6 × 6 spatial dimensions
10 or 100-way softmax

""" 



from keras.models import Sequential
from keras.layers import Dropout, Activation, Conv2D, GlobalAveragePooling2D


def allcnn(weights=None):
    # dDEFINING SEQUENTIAL MODEL
    model = Sequential()

    # add model layers - Convolution2D, Activation, Dropout
    model.add(Conv2D(96, (3, 3), padding = 'same', input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding = 'same', strides = (2,2)))
    model.add(Dropout(0.5))  #TO PREVENT OVERFITTING

    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding = 'same', strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1), padding = 'valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1), padding = 'valid'))

    # add GlobalAveragePooling2D layer with Softmax activation
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    
    # load the weights
    if weights:
        model.load_weights(weights)
    
    
    return model



"""-----------------------------------------------------------------"""
#DEFINING HYPERPARAMETERS
learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9

model = allcnn()


# DEFINING OPTIMIZER & COMPILING THE MODEL
from keras.optimizers import SGD #stochaistic gradient descent
sgd = SGD(lr=learning_rate, decay=weight_decay, momentum=momentum, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print (model.summary())


"""batch size simply means you're going to run 32 images.
And then after those 32 images you can update the loss or update the
parameters instead of updating them individually after every single image."""


model.fit(X_train, Y_train, validation_data=(X_test, Y_test), 
          epochs=350, batch_size=32, verbose = 1)



"""-----------------------------------------------------------------"""
#PREDICTIONS

classes = range(0,10)

names = ['airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck']

# zip the names and classes to make a dictionary of class_labels
class_labels = dict(zip(classes, names))

# generate batch of 9 images to predict
batch = X_test[100:109]
labels = np.argmax(Y_test[100:109],axis=-1)

# make predictions
predictions = model.predict(batch, verbose = 1)

# convert class probabilities to class labels
class_result = np.argmax(predictions,axis=-1)

# create a grid of 3x3 images
fig, axs = plt.subplots(3, 3, figsize = (15, 6))
fig.subplots_adjust(hspace = 1)
axs = axs.flatten()

for i, img in enumerate(batch):

    # determine label for each prediction, set title
    for key, value in class_labels.items():
        if class_result[i] == key:
            title = 'Prediction: {}\nActual: {}'.format(class_labels[key], class_labels[labels[i]])
            axs[i].set_title(title)
            axs[i].axes.get_xaxis().set_visible(False)
            axs[i].axes.get_yaxis().set_visible(False)
            
    # plot the image
    axs[i].imshow(img)
    

plt.show()














































