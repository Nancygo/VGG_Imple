#!/usr/bin/env python
# coding: utf-8

# In[154]:


from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, MaxPool2D
from keras.optimizers import SGD
from imutils import paths
from sklearn.metrics import classification_report
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import pickle
import cv2
import os
import random

#initialize data and labels
x_data =[]
label_ids={}
y_label = []
current_id = 0
#Getching Image data drirectory path
image_dir_path = os.path.join(os.getcwd(),'Images') 
#image_dir = sorted(list(paths.list_images(image_dir_path))) 
random.seed(42)
random.shuffle(image_dir)

#Looping over input images
for root, dirs, files in os.walk(image_dir_path):
    for file in files:
        if file.endswith(".pgm"):
            img_path = os.path.join(root, file)
            label = os.path.basename(root)
            #Adding label to label dictionary
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            #Image resizing to 60% keeping aspect ration intact, 3685 pixel size of image
            scale_percentage = 60
            dim = ((int(image.shape[1]*scale_percentage/100)), (int(image.shape[0]*scale_percentage/100)))
            image = cv2.resize(image, dim).flatten()
            x_data.append(image)
            y_label.append(id_)

#Scaling image pixel from (0,255) to (0,1)
x_data = np.array(x_data,dtype = "float")/255.0
y_label = np.array(y_label)
#print(x_data.shape) #size of input layer 

#Partition 25% data randomly for testing
(train_x, test_x, train_y, test_y) = train_test_split(x_data, y_label, test_size = 0.25, random_state = 42)

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

#train_y_1 = to_categorical(train_y, num_classes=len(set(train_y)))
print(train_y.shape)
print(train_x.shape)
print(test_x.shape)
print(lb.classes_)

#Define 3685*1024*512*no_of_classes Keras Model
model = Sequential()
model.add(Dense(1024, input_shape = (3685,), activation = "sigmoid"))
model.add(Dense(512, activation= "sigmoid"))
model.add(Dense(len((lb.classes_)),activation= "softmax"))

#Initial LEarning rate and training epoch
INIT_LR = 0.02
var_epoch = 75

#train the network
optimize = SGD(lr = INIT_LR)
model.compile(loss ="categorical_crossentropy", optimizer=optimize, metrics=["accuracy"])
model_fit = model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), epochs=var_epoch, batch_size=32)

predictions = model.predict(test_x,batch_size =32)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1)))


N = np.arange(0,var_epoch)
plt.style.use('ggplot')
plt.figure()
plt.plot(N, model_fit.history["val_loss"], label = "validation_loss")
plt.plot(N,model_fit.history["loss"], label = "loss")
plt.plot(N,model_fit.history["acc"], label = "accuracy")
plt.plot(N, model_fit.history["val_acc"], label = "val_Acc")
plt.title("Simple NN")
plt.xlabel("no. of epochs")
plt.ylabel("loss/accuracy")
plt.legend()
model.save(os.path.join(os.getcwd(),"model\simpleNN_model"))
print(os.getcwd())
f = open(os.path.join(os.getcwd(),"model\simpleNN_labelbinarizer"), mode = "wb")
f.write(pickle.dump(lb))
f.close()

