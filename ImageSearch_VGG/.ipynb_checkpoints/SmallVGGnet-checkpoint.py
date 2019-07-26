{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import Sequential\n",
    "from keras.layers.convolutional import Conv2D, MaxPooling2D\n",
    "from keras.layers.normalization import BatchNormalization\n",
    "from keras.layers.core import activations, Flatten, Dense, Dropout\n",
    "from keras import backend as K\n",
    "\n",
    "class smallVGGnet:\n",
    "    @staticmethod\n",
    "    def build(width, height, depth, classes):\n",
    "        #Initializing model with input shape\n",
    "        model = Sequential()\n",
    "        inputshape = (height, width, depth)\n",
    "        chan_dim = -1\n",
    "        \n",
    "        #Flexibility to use another backend\n",
    "        if K.image_data_format == 'channels_first': #If theano backend\n",
    "            inputshape = (depth, height, width)\n",
    "            chan_dim = 1\n",
    "        \n",
    "        #Add layer to Model\n",
    "        model.add(Conv2D(32, (3,3), padding = 'same', input_shape = inputshape, activations = 'relu'))\n",
    "        model.add(BatchNormalization(axis = chan_dim))\n",
    "        model.add(MaxPooling2D(pool_size = (2,2)))\n",
    "        model.add(Dropout(rate = 0.2))\n",
    "        \n",
    "        #Add 2nd layer to VGGNet\n",
    "        model.add(Conv2D(64, (3,3), activationsivations='relu', padding = 'same'))\n",
    "        model.add(BatchNormalization(axis = chan_dim))\n",
    "        model.add(Conv2D(64,(3,3), padding = 'same', activations='relu'))\n",
    "        model.add(BatchNormalization(axis=chan_dim))\n",
    "        model.add(MaxPooling2D(pool_size = (2,2)))\n",
    "        model.add(Dropout(rate = 0.2))\n",
    "        \n",
    "        #Add 3rd Layer to VGGNet\n",
    "        model.add(Conv2D(128, (3,3), padding = 'same', activations='relu'))\n",
    "        model.add(BatchNormalization(axis = chan_dim))\n",
    "        model.add(Conv2D(128,(3,3), padding = 'same', activations = 'relu'))\n",
    "        model.add(BatchNormalization(axis==chan_dim))\n",
    "        model.add(Conv2D(128, (3,3), padding= 'same', activations='relu'))\n",
    "        model.add(BatchNormalization(axis==chan_dim))\n",
    "        model.add(MaxPooling2D(pool_size = (2,2)))\n",
    "        model.add(Dropout(rate = 0.2))\n",
    "        \n",
    "        #Add Fully Connected NN \n",
    "        model.add(Flatten())\n",
    "        model.add(Dense(512, activations='relu'))\n",
    "        model.add(BatchNormalization())\n",
    "        model.add(Dropout(rate = 0.2))\n",
    "        \n",
    "        #Softmax cassifier for output layer with classes as no of output classes\n",
    "        model.add(Dense(classes, activationsvations = 'softmax'))\n",
    "        \n",
    "        #Return model\n",
    "        return model"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
