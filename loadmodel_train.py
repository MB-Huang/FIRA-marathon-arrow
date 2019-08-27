
from keras.models import Model,Sequential  
from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate,Activation,ZeroPadding2D  
from keras.layers import add,Flatten,Convolution2D,GlobalAveragePooling2D
import numpy as np  
seed = 7  
np.random.seed(seed)  
from keras import layers
from keras import models
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras import backend as K
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True
session = tf.Session(config=config)

# # session
KTF.set_session(session)
#
# image dimensions
#

data_augmentation=True
num_classes = 4
index=0
width=220
height=220
num_channels=1
num_pic = 4500

X_Train=np.zeros((num_pic,height,width,num_channels))
Y_Train=np.zeros((num_pic,num_classes))

f = file("Arrow_x_train_220.npy", "rb")
X_Train=np.load(f)
f.close()

f = file("Arrow_y_train_220.npy", "rb")
Y_Train=np.load(f)
f.close()
# print X_Train
# print Y_Train 


X_Train, Y_Train = shuffle(X_Train, Y_Train, random_state=2)
X_Train, Y_Train = shuffle(X_Train, Y_Train, random_state=4)
X_Train, Y_Train = shuffle(X_Train, Y_Train, random_state=6)
X_Train, Y_Train = shuffle(X_Train, Y_Train, random_state=8)

# model = load_model('res_model_0729_30.h5')#testmodel
model = load_model('res_model_0801_30.h5')#testmodel


adam = Adam(lr=1e-4)
model.compile(optimizer=adam,
loss='categorical_crossentropy',
metrics=['accuracy'])
if not data_augmentation:
    # We add metrics to get more results you want to see
    
    # plot_model(model, to_file='res18model_plot.png', show_shapes=True, show_layer_names=True)
    for i in range(1):
        model.fit(X_Train, Y_Train
        # ,validation_data=(X_Test,Y_Test)
        , epochs=20,verbose=2, batch_size=64, shuffle=True,)
        print('\n\n\n\n\n')
        save_path='res_model_0717_'+str(70)+'.h5'
        print(save_path)
        model.save(save_path)
        # model.save('my_model_0313_.h5')
        # loss,accuracy=model.evaluate(X_Test,Y_Test,batch_size=128,)
        # print('\ntest loss: ', loss)
        # print('\ntest accuracy: ', accuracy)
else:


    datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                zoom_range=0,
                rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=False,  # randomly flip images
                vertical_flip=False)  # randomly flip images
    datagen.fit(X_Train)

    model.fit_generator(datagen.flow(X_Train, Y_Train,
                batch_size=32),
                epochs=20,verbose=2,
                # validation_data=(X_Train[1001:1258,:,:,:],Y_Train[1001:1258,:]),
                shuffle=True,
                workers=4)

            
    save_path='res_model_0802_'+str(50)+'.h5'
    print(save_path)
    model.save(save_path)
       


