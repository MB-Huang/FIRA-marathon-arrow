import cv2
import sys
import os
# from PIL import Image
# from PIL import ImageFont
# from PIL import ImageDraw
import os
import csv
import h5py
import numpy as np
import keras
np.random.seed(1337)  # for reproducibility
#from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout,Input,Dense,Reshape,Permute, Activation, Convolution2D, MaxPooling2D,GRU,Flatten,LSTM,TimeDistributed ,concatenate
from keras.optimizers import Adam,SGD,RMSprop,Adagrad
# from keras import optimizers
from keras.models import load_model
from keras.models import Model
#from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras import backend as K
import numpy as np

# import pyttsx

# from gtts import gTTS
import os
# import pygame
# from pygame import mixer # Load the required library
data_augmentation=True
num_classes = 4
index=0
width=220
height=220
num_channels=1#3
num_pic = 1703
X_Pre=np.zeros((1,height,width,num_channels))
X_Train=np.zeros((num_pic,height,width,num_channels))
Y_Train=np.zeros((num_pic,num_classes))

#numberof camera:0 1 ,2,lsusb
cap = cv2.VideoCapture(1)

cap.set(3,640)
cap.set(4,480)
# time.sleep(2) #need this step
cap.set(15, -8.0)

#160.0 x 120.0
#176.0 x 144.0
#320.0 x 240.0
#352.0 x 288.0
#640.0 x 480.0
#1024.0 x 768.0
#1280.0 x 1024.0
index=0
# size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
#         int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
# fourcc = cv2.cv.FOURCC(*'CVID')
# out = cv2.VideoWriter(filePath, fourcc, fps, size)
model = load_model('res_model_0811_20.h5')#testmodel
# model = load_model('proposedmodelII.h5')#
# model = load_model('proposedmodel.h5')#

result=np.zeros(4)


while (cap.isOpened()):
    ret, frame = cap.read()
  
    if ret == True:
        x=100 
        y=20 
        w=440 
        h=440
        
        res = frame[y:y+h, x:x+w]
        res=cv2.resize(res,(width,height))
        
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)  
        X_Pre[0,:,:,0] = np.array(gray)
        X_Pre/=255
        y_proba=model.predict(X_Pre,64,0)
        y_max_pro=y_proba.max(axis=-1)
        y_result=y_proba.argmax(axis=-1)
        # cv2.putText(frame,printstring,(10,30), font, 1,(0,0,0),1,cv2.LINE_AA)
        cv2.namedWindow('camera')        # Create a named window
        # cv2.moveWindow('camera', 0,300) 
        #     
        if y_result ==0:
            printstring='left'
        elif y_result ==1:
            printstring='straight'
        elif y_result ==2:
            printstring='right'
        else:
            printstring='none'
        # printstring+= str(y_max_pro)
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(frame,printstring,(100,20), font, 0.8,(255,150,0),1,cv2.LINE_AA)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
        cv2.imshow('camera',frame)
        
        # print cap.get(cv2.CAP_PROP_FPS)
        # out.write(gray)
        # print y_result
        print y_proba
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
# out.release()
cv2.destroyAllWindows()
