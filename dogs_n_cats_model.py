# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,Activation,Dropout,MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

! wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip

!unzip kagglecatsanddogs_3367a.zip

DATADIR = "PetImages"
categories = ["Dog", "Cat"]

'''for category in categories:
    
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):
        
        img_array = cv2.imread(os.path.join(path,img), cv2.IRMEAD_GRAYSCALE)
   '''

img_size = 50

training_data = []

def create_training_data():
    
    for category in categories:
        path = os.path.join(DATADIR,category)
        class_num = categories.index(category)
        
        for img in os.listdir(path):
            
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size,img_size))
                training_data.append([new_array,class_num])
            
            except Exception as e:
                
                pass
            
create_training_data()

print(len(training_data))

import random

random.shuffle(training_data)

X = []
Y = []

for features,lables in training_data:
    X.append(features)
    Y.append(lables)
    
X = np.array(X).reshape(-1 , img_size, img_size, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y,pickle_out)
pickle_out.close()

data = pickle.load(open("X.pickle", "rb"))
data[1]



"""  TensorBoard config for Google Colab  """

!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip

LOG_DIR = './log'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)

get_ipython().system_raw('./ngrok http 6006 &')

! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

"""  MODEL """

name = ''
X = pickle.load(open("X.pickle","rb"))
Y = pickle.load(open("Y.pickle","rb"))

X = X/255.0

n = 0
dense_layers = [1,2,3]
layer_sizes = [32,64,128]
conv_layers = [1,2,3]

for dense_layer in dense_layers:
    for conv_layer in conv_layers:
        for layer_size in layer_sizes:
            n += 1
            print("EPOCH:",n)
            
            name = "{}conv-{}nodes-{}dense-{}".format(conv_layer,layer_size,dense_layer,int(time.time()))
            tensorboard = TensorBoard(log_dir='./log/{}'.format(name))
            
            model = Sequential()

            model.add(Conv2D(layer_size , (3,3) , input_shape = X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size = (2,2)))
            
            for l in range(conv_layer - 1):

                model.add(Conv2D(layer_size , (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size = (2,2)))

            model.add(Flatten())
            
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation("sigmoid"))              #Try with Softmax and Tanh

            model.compile(loss = "binary_crossentropy",
                          optimizer = "adam",
                          metrics = ['accuracy'])         #Try with categorical_crossentropy

            model.fit(X,Y, batch_size = 32, validation_split = 0.1, epochs = 3, callbacks = [tensorboard])

