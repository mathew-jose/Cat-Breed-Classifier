#!/usr/bin/env python
# coding: utf-8

# In[33]:


X=[]

#import os
#os.sys.path
import math
import glob
import cv2
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import tf_utils
from tensorflow import keras
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_uniform
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:















X_train=np.array(X_train)
Y_train=np.array(Y_train)
X_test=np.array(X_test)
Y_test=np.array(Y_test)
Y_train[4]


# In[36]:


index = 4
plt.imshow(X_test[index])
print ("y = " + str(Y_test[index]))
Y_test.shape


# In[37]:


#X_Train_flatten = X_train.reshape(X_train.shape[0], -1).T
#Y_Train_flatten = Y_Train.reshape(Y_train.shape[0], -1).T
#X_Train = X_train/255.
#Y_train = X_test_flatten/255.



import tensorflow as tf

vocab = ['British ShortHair Cat','Persian Cat','Maine Coon Cat','Siamese Cat','Bombay Cat','Chartreux Cat']

input = tf.placeholder(dtype=tf.string, shape=[356,1])
matches = tf.stack([tf.equal(input, s) for s in vocab], axis=-1)
onehot = tf.cast(matches, tf.int8)

with tf.Session() as sess:
    out = sess.run(onehot, feed_dict={input: Y_train})
    
Y_train=out

input1 = tf.placeholder(dtype=tf.string, shape=[26,1])
matches1 = tf.stack([tf.equal(input1, s) for s in vocab], axis=-1)
onehot1 = tf.cast(matches1, tf.int8)

with tf.Session() as sess:
    out1 = sess.run(onehot1, feed_dict={input1: Y_test})
    
Y_test=out1
#Y_test = convert_to_one_hot(Y_test_orig, 6)


# In[38]:


Y_train=Y_train.astype('float32')
Y_test=Y_test.astype('float32')

X_train=X_train/255
X_test=X_test/255


# In[39]:


Y_train = Y_train.reshape(Y_train.shape[0], -1)  # The "-1" makes reshape flatten the remaining dimensions
Y_test = Y_test.reshape(Y_test.shape[0], -1)  # The "-1" makes reshape flatten the remaining dimensions


# In[8]:


def cat_model(input_shape):
    
    X_input = Input(input_shape)
    #X = ZeroPadding2D((3, 3))(X_input)

    # X=X.astype('float32')
    X = Conv2D(filters = 96, kernel_size = (11, 11), strides = (4,4), padding = 'valid')(X_input)
    X = Activation('relu')(X)

    X = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='max_pool')(X)
    X = Conv2D(filters = 256, kernel_size = (5, 5), strides = (1,1), padding = 'same')(X)


    X = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='max_pool1')(X)
    X = Conv2D(filters = 384, kernel_size = (3, 3), strides = (1,1), padding = 'same')(X)

    X = Conv2D(filters = 384, kernel_size = (3, 3), strides = (1,1), padding = 'same')(X)
    X = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1,1), padding = 'same')(X)
    X = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='max_poo')(X)


    X = Flatten()(X)
    X = Dense(9216,input_shape=(227*227*3,),activation='relu', name='fc')(X)
    X = Dense(4096,activation='relu', name='fc4')(X)
    X = Dense(4096,activation='relu', name='fc1')(X)
    X = Dense(1000,activation='relu', name='fc2')(X)
    X = Dense(6,activation='softmax', name='fc3')(X)
    global graph

    graph = tf.get_default_graph()
    model = Model(inputs = X_input, outputs = X, name='cat_model')


    return model


# In[9]:


Cat_Model = cat_model((227,227,3))


# In[10]:


from keras import optimizers

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
Cat_Model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics = ["accuracy"])


# In[11]:


X_train.shape


# In[12]:


Y_test.shape


# In[13]:


global graph

graph = tf.get_default_graph()
Cat_Model.fit(x = X_train, y =Y_train, epochs = 9, batch_size= 5)


# In[40]:


preds = Cat_Model.evaluate(x=X_test,y=Y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))




