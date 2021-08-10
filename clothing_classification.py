#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 20:32:27 2020

@author: dandorci
"""
#%% imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#%% load clothing grayscale image data (train and test)

####################################
# download the following files here: https://www.kaggle.com/zalando-research/fashionmnist
####################################

train_data = pd.read_csv("fashion-mnist_train.csv")

test_data = pd.read_csv("fashion-mnist_test.csv")


# map label values to designated clothing item based off, provided info
# 0 T-shirt/Top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle Boot

clothing_labels = ["T-shirt/Top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]

#%% features and labels


#----------------------- TRAINING DATA -----------------------#

# get the amount of training examples in the data set
num_train_examples = np.shape(train_data)[0]

# first column coresponds to label so drop it, the remaining columns are every pixel in a given image
train_features = np.array(train_data.iloc[:,1:])

# reshape to caputure the shape of each image, each image is 28x28 pixels
pixels_per_direction = 28

train_features_matrix = np.reshape(train_features, [num_train_examples, pixels_per_direction, pixels_per_direction])

tf_matrix = tf.reshape(train_features, [num_train_examples, pixels_per_direction, pixels_per_direction], name=None)

# selecting the first column (via the column name) to create a training data labels array
train_labels = np.array([train_data["label"]])

# reshape the labels into a column (a row was returned in the line above)
train_labels = np.reshape(train_labels, [num_train_examples,1])


#------------------------- TEST DATA -------------------------#

# get the amount of test samples in the data set
num_test_samples = np.shape(test_data)[0]

# first column coresponds to label so drop it, the remaining columns are every pixel in a given image
test_features = np.array(test_data.iloc[:,1:])

test_features_matrix= np.reshape(test_features, [num_test_samples, pixels_per_direction, pixels_per_direction])

tf_matrix = tf.reshape(test_features, [num_test_samples, pixels_per_direction, pixels_per_direction], name=None)

# selecting the first column (via the column name) to create a test data labels array
test_labels = np.array([test_data["label"]])

# reshape the labels into a column (a row was returned in the line above)
test_labels = np.reshape(test_labels, [num_test_samples,1])



#%% plot first example
fig1 = plt.figure()
pic = np.array(train_features_matrix[0,:,:])
label = train_labels[0]
plt.title(clothing_labels[label[0]])
plt.imshow(pic, cmap='Greys', interpolation='none')
plt.show()


#%% normalize train data 

norm_train = train_features/255
features_matrix_norm = np.reshape(norm_train, [num_train_examples, pixels_per_direction, pixels_per_direction, 1])


#%% CNN Model

# max pooling: reduce dimensionality by taking max value of a given patch size, perserves spatial invariance

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', activation=tf.nn.relu, input_shape=(pixels_per_direction, pixels_per_direction, 1), data_format="channels_last"),
        tf.keras.layers.MaxPooling2D((2,2), strides=2),
        tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2,2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

#%% Compile model


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


      
#%% train model
epochs = 5
history = model.fit(features_matrix_norm, train_labels, batch_size=32 ,epochs=epochs, validation_split=0.1)

#%% visualize the traing/validation metrics

# set x-axis tick marks to epochs
x = np.linspace(1,epochs, epochs)

# loss plot
plt.title("Loss vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(x, history.history['loss'])
plt.plot(x, history.history['val_loss'])
plt.xticks(x)
axes = plt.gca()
axes.set_ylim([0,max(max(history.history['loss']), max(history.history['val_loss']))])
plt.legend(['Training Set', 'Validation Set']) 
plt.show()

# accuracy plot
plt.title("Accuracy vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(x, history.history['accuracy'])
plt.plot(x, history.history['val_accuracy'])
plt.xticks(x)
plt.legend(['Training Set', 'Validation Set'], loc='upper left') 
plt.show()


#%% normalize test data 

norm_test = np.array(test_features/255)
features_matrix_norm_test = np.reshape(norm_test, [num_test_samples, pixels_per_direction, pixels_per_direction, 1])


#%% evaluate on test data

test_loss, test_accuracy = model.evaluate(features_matrix_norm_test, test_labels)
print('accuracy on test data:', test_accuracy)



#%% make predictions on random test data

test_used = []
for i in range(0, 10):
    while True:
        
        test_number = random.randint(0, num_test_samples)
        if test_number in test_used:
            continue
        
        test_used.append(test_number)
        break
            
    rand_test = np.array(norm_test[test_number,:])
    rand_test = np.reshape(rand_test, [1, pixels_per_direction, pixels_per_direction, 1])
    
    prediction = model.predict(rand_test)
    
    print("----------------------------------------------------")
    print("test {} label: ".format(test_number), clothing_labels[test_labels[test_number][0]])
    print("test {} prediction: ".format(test_number), clothing_labels[np.argmax(prediction)])









