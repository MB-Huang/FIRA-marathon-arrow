# from PIL import Image
# from PIL import ImageFont
# from PIL import ImageDraw

import os
import numpy as np
import csv
import cv2
import sys
np.random.seed(1337)  # for reproducibility
num_pic =3598
num_classes = 4
#load_pic_num=np.zeros(num_classes)
index=0
width=220
height=220
num_channels=1
isTrain = True
DATA_DIR = "DATA"

X_Train=np.zeros((num_pic,height,width,num_channels))
Y_Train=np.zeros((num_pic,num_classes))
# total=np.zeros((6))
for foldername1 in os.listdir(DATA_DIR):
    for filename  in os.listdir(os.path.join(DATA_DIR, foldername1)):
    
        # print(os.path.join(DATA_DIR, foldername1,filename))   

        img = cv2.imread(os.path.join(DATA_DIR, foldername1,filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x=100 
        y=20 
        w=440 
        h=440
        
        roi_gray = gray[y:y+h, x:x+w]

        res=cv2.resize(roi_gray,(width,height),interpolation=cv2.INTER_LINEAR)
        # print res.shape()
        # cv2.imshow('img',res)
        # cv2.waitKey(10)
        #if i==1:
        #    im.show()
        #im2 = im.resize((width, length), Image.BILINEAR)        
        # X_Train[index, :, :,0,i] = np.array(im2)                    
        X_Train[index, :, :,0] = np.array(res)
        # print X_Train[index,:,:,:]
        print foldername1
        Y_Train[index,int(foldername1[0])]=1
        print   Y_Train[index,:]
        index+=1
print index
                    #print(os.path.join(DATA_DIR,foldername1,foldername2,filename))

# print total
X_Train/=255
print X_Train
print Y_Train
if isTrain:
    f = file("Arrow_x_train_220.npy", "wb")
else:
    f = file("Arrow_x_test_320240.npy", "wb")
np.save(f, X_Train)
f.close()
if isTrain:
    f = file("Arrow_y_train_220.npy", "wb")
else:
    f = file("Arrow_y_test_320240.npy", "wb")
np.save(f, Y_Train)
f.close()