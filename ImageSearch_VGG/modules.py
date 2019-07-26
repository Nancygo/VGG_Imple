#!/usr/bin/env python
# coding: utf-8

# In[10]:


from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import activations, Flatten, Dense, Dropout
from keras import backend as K

class smallVGGnet:
    def __init__(self, width, height, depth, classes):
        self.width = width
        self.height = height
        self.depth = depth
        self.classes = classes
        
    def build(self):
        width = self.width
        height = self.height
        depth = self.depth
        classes = self.classes
        
        #Initializing model with input shape
        model = Sequential()
        inputshape = (height, width, depth)
        chan_dim = -1
        
        #Add layer to Model
        model.add(Conv2D(32, (3,3), padding = 'same', input_shape = inputshape, activation = 'relu'))
        model.add(BatchNormalization(axis = chan_dim))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(rate = 0.2))
        
        #Add 2nd layer to Deep CNN
        model.add(Conv2D(64, (3,3), activation='relu', padding = 'same'))
        model.add(BatchNormalization(axis = chan_dim))
        model.add(Conv2D(64,(3,3), padding = 'same', activation='relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(rate = 0.2))
        
        #Add 3rd Layer to Deep CNN
        model.add(Conv2D(128, (3,3), padding = 'same', activation='relu'))
        model.add(BatchNormalization(axis = chan_dim))
        model.add(Conv2D(128,(3,3), padding = 'same', activation = 'relu'))
        model.add(BatchNormalization(axis =chan_dim))
        model.add(Conv2D(128, (3,3), padding= 'same', activation='relu'))
        model.add(BatchNormalization(axis =chan_dim))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(rate = 0.2))
        
        #Add Fully Connected NN 
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(rate = 0.2))
        
        #Softmax cassifier for output layer with classes as no of output classes
        model.add(Dense(classes, activation = 'softmax'))
        
        #Return model
        return model

