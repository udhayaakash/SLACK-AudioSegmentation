#!/usr/bin/env python
# coding: utf-8

# ## Import Statements

# In[51]:


import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from PIL import Image
import pathlib
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Keras
import keras

import warnings
warnings.filterwarnings('ignore')

from python_speech_features import mfcc as mfccc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from pandas import DataFrame


# ## Data Preprocessing

# In[3]:


header = ''
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()


# In[ ]:


file = open('data1.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)


# ### Preparation on dataset

# In[10]:


genres = 'NS S'.split()
for g in genres:
    for filename in os.listdir(f'./{g}'):
        songname = f'./{g}/{filename}'
        (rate,sig) = wav.read(songname)
        mfcc = mfccc(sig,rate,numcep=20)    
        for e in mfcc:
            to_append = f''
            for z in e:
                to_append += f' {z}'
            to_append += f' {g}'
            file = open('data2.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
            file.close()


# ## Data Visualization

# In[52]:


d = pd.read_csv('./data/data1.csv')
data = DataFrame.drop_duplicates(d)
data.tail()


# In[ ]:





# In[56]:


genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)


# In[57]:


scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[59]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(2, activation='softmax'))


# In[60]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[61]:


history = model.fit(X_train,y_train,epochs=5,batch_size=64)


# In[66]:


test_loss, test_acc = model.evaluate(X_test,y_test)

print('test_acc: ',test_acc)


# In[63]:


# save model and architecture to single file
model.save("model_128_128.h5")
print("Saved model to disk")


# In[14]:


from pydub import AudioSegment
from keras.models import load_model
from python_speech_features import mfcc as mfccc
import scipy.io.wavfile as wav

def predict(f):
    
    model = load_model('model_final.h5')
    
    (rate,sig) = wav.read(f)
    mfcc = mfccc(sig,rate,numcep=20)
    pred = model.predict_classes(mfcc)

    newAudio = AudioSegment.from_wav(f)
    
    a=[]
    time = 0
    starttime = 0
    pointer = pred[0]
    prev = pred[0]
    for i in range(len(pred)-1):
        if(pointer!=prev):
            if(pointer==1):
                if(pred[i+1]==0):
                    pointer = pred[i+1]
                    time = time + 10
                    continue
                else:
                    starttime = time

            elif (pointer == 0):
                if(pred[i+1]==1):
                    pointer = pred[i+1]
                    time = time + 10
                    continue
                else:
                    a.append([starttime,time-10])
                print('Segment at {}ms'.format(time-10))
            prev=pointer

        pointer = pred[i+1]
        time = time + 10

    if(prev==1 and pred[-1]==1):
        a.append([starttime,time-10])
        print('Segment at {}ms'.format(time-10))
        
    for i,j in enumerate(a):
    
        t1 = j[0]
        t2 = j[1]

        audio = newAudio[t1:t2]
        audio.export('./'+str(i)+f[2:], format="wav")


# In[ ]:


predict('./katl_dep_0927_20170105_sr_001.wav')


# In[ ]:




