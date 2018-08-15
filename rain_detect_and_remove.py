
# coding: utf-8

# In[1]:


import sklearn.model_selection
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Input
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from PIL import Image
from keras.layers.merge import add
import os
from matplotlib.image import imread
from keras import optimizers
import cv2 as cv
import keras.backend as K
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt


# In[2]:


#model=Sequential()
def build(input):
    input=Input(shape=input)
    alpha_1=Conv2D(16,(3,3),strides=1,padding='same',input_shape=(64,64,3))(input)
    beta_1=BatchNormalization()(alpha_1)
    add_layer=Activation('relu')(beta_1)
    for i in range(12):
        a1=Conv2D(16,(3,3),strides=1,padding='same')(add_layer)
        b1=BatchNormalization()(a1)
        c1=Activation('relu')(b1)
    
        d1=Conv2D(16,(3,3),strides=1,padding='same')(c1)
        e1=BatchNormalization()(d1)
        f1=Activation('relu')(e1)
        add_layer=add([f1,add_layer])
    
    _output=Conv2D(3,(3,3),strides=1,padding='same')(add_layer)
    _output=BatchNormalization()(_output)
    _output=Activation('relu')(_output)
    return Model(inputs=input,outputs=_output)


# In[3]:


path="/home/goyal/Downloads/working/CVPR17_training_code/data_generation/image"
path_rainy="/home/goyal/Downloads/working/CVPR17_training_code/data_generation/image/input"
path_clean="/home/goyal/Downloads/working/CVPR17_training_code/data_generation/image/label"
files_r=os.listdir(path_rainy)
files_c=os.listdir(path_clean)
x_train=[]
y_train=[]
for fi in files_r:
    baarish=imread(path_rainy+'/'+fi)
    saaf=imread(path_clean+'/'+fi)
    baarish=baarish/255.0
    saaf=saaf/255.0
    x_train.append(baarish)
    y_train.append(saaf)
x_train=np.array(x_train)
y_train=np.array(y_train)


# In[13]:



'''n_epochs=3
learning_rate=0.1
batch_size=20
sizes=int((len(files_r))/batch_size)
sizes=int((4*sizes)/5)
print(sizes)

for j in range(1,n_epochs+1):
    Training_loss=0
    if(j>1):
        learning_rate=0.01*learning_rate
    else:
        learning_rate=0.1*learning_rate
        
    for im in range(sizes):
        x_batch=[]
        y_batch=[]
        for i in range(im*batch_size,(im+1)*batch_size):
            baarish=imread(path_rainy+'/'+files_r[i])
            saaf=imread(path_clean+'/'+files_c[i])
            x_batch.append(baarish)
            y_batch.append(saaf)
        loss=model.train(x_batch,y_batch,loss='mean_squared_error')
         #print(sizes)
    for j in range(1,n_epochs+1):
        Training_loss=0
        if(j>1):
            learning_rate=0.01*learning_rate
        else:
            learning_rate=0.1*learning_rate
        
        for im in range(sizes):
            x_batch=[]
            y_batch=[]
            for i in range(im*batch_size,(im+1)*batch_size):
                baarish=imread(path_rainy+'/'+files_r[i])
                saaf=imread(path_clean+'/'+files_c[i])
                x_batch.append(baarish)
                y_batch.append(saaf)    
            x_batch=np.array(x_batch)
            y_batch=np.array(y_batch)
            print(np.array(x_batch).shape)
            loss=model.train_on_batch(x_batch,y_batch)
            print(loss)
    '''


# In[4]:


x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x_train,y_train,test_size=0.1,random_state=42)


# In[9]:


if __name__=='__main__':
    #loss=np.mean(np.square(labels-output))
    val=ssim(x_test[0].astype('float32'),y_test[0].astype('float32'),win_size=5,multichannel=True)
    print(val)
    input_shape=[64,64,3]
    model=build(input_shape)
    model.summary()
    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    batch_s=40
    size=int(len(x_train)/batch_s)
    for epochs in range(1):
        print(size)
        training_loss=0
        count=0
        for i in range(20):
            x_batch=x_train[i*batch_s:(i+1)*batch_s,:,:,:]
            y_batch=y_train[i*batch_s:(i+1)*batch_s,:,:,:]
            training_loss=model.train_on_batch(x_batch,y_batch)
            print(training_loss," ",count)
            count+=1
        y_pred=model.predict(x_test,batch_size=64,verbose=1)
        vals=0
        count=0
        for i in range(len(x_test)):
            count+=1
            vals+=ssim(y_test[i,:,:,:].astype('float32'),y_pred[i,:,:,:].astype('float32'),win_size=5,multichannel=True)
            print("ssim index",vals/count)
            
        print("epoch %d is completed %d",epochs,training_loss)    
    #model.fit(x_train,y_train,batch_size=20,epochs=3,verbose=1,validation_split=0)


# In[55]:


vals=0
count=0
for i in range(len(x_test)):
    count+=1
    vals+=ssim(y_test[i,:,:,:].astype('float32'),y_pred[i,:,:,:].astype('float32'),win_size=11,multichannel=True)
    print("ssim index",vals/count)


# In[27]:


val=ssim(x_test[8].astype('float32'),y_test[8].astype('float32'),win_size=11,multichannel=True)
print(val)


# In[4]:


plt.subplot(1,2,1)     
plt.imshow(y_train[9])      
plt.title('input')

plt.subplot(1,2,2)    
plt.imshow(x_train[9])
plt.title('output')

plt.show()

