
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
from keras.preprocessing.image import ImageDataGenerator
# from sklearn.utils import shuffle
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
num_pic = 3598

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


# X_Train, Y_Train = shuffle(X_Train, Y_Train, random_state=0)
# X_Train, Y_Train = shuffle(X_Train, Y_Train, random_state=5)
# X_Train, Y_Train = shuffle(X_Train, Y_Train, random_state=3)
# X_Train, Y_Train = shuffle(X_Train, Y_Train, random_state=66)


def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):  
    if name is not None:  
        bn_name = name + '_bn'  
        conv_name = name + '_conv'  
    else:  
        bn_name = None  
        conv_name = None  
    x = BatchNormalization()(x)  
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)  
    # x = BatchNormalization()(x)
    return x  
  
def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):  
    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')  
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')  
    if with_conv_shortcut:  
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)  
        x = add([x,shortcut])  
        return x  
    else:  
        x = add([x,inpt])  
        return x  
  
inpt = Input(shape=(height,width,num_channels))  
x = ZeroPadding2D((0,0))(inpt)  
x = Conv2d_BN(x,nb_filter=32,kernel_size=(7,7),strides=(1,1),padding='same')  
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)  
# #(56,56,64)
x = Conv_Block(x,nb_filter=32,kernel_size=(5,5),strides=(1,1),with_conv_shortcut=False)    
# x = Conv_Block(x,nb_filter=64,kernel_size=(5,5),strides=(2,2),with_conv_shortcut=True)  
# x = Conv_Block(x,nb_filter=64,kernel_size=(5,5),strides=(2,2),with_conv_shortcut=True)
x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)  
# x = Conv2d_BN(x,nb_filter=64,kernel_size=(5,5),strides=(1,1),padding='same')  
x = Conv_Block(x,nb_filter=64,kernel_size=(3,3),strides=(1,1),with_conv_shortcut=True)  
x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)  

# x = Conv_Block(x,nb_filter=64,kernel_size=(3,3),strides=(1,1),with_conv_shortcut=False)  
# x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)  
x = Conv_Block(x,nb_filter=64,kernel_size=(3,3),strides=(1,1),with_conv_shortcut=False)  
x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)  
x = Conv_Block(x,nb_filter=128,kernel_size=(3,3),strides=(1,1),with_conv_shortcut=True)  
x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)  

# x = Conv_Block(x,nb_filter=64,kernel_size=(3,3),strides=(1,1))  
# x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)    
# x = Conv2d_BN(x,nb_filter=64,kernel_size=(5,5),strides=(1,1),padding='same')  

# x = Conv_Block(x,nb_filter=32,kernel_size=(7,7))
# x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)    
# x = Conv2d_BN(x,nb_filter=128,kernel_size=(3,3),strides=(1,1),padding='same')  

# x = Conv_Block(x,nb_filter=64,kernel_size=(3,3),with_conv_shortcut=False)
# x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)    
#(28,28,128)  
# x = Conv_Block(x,nb_filter=256,kernel_size=(3,3),strides=(1,1),with_conv_shortcut=True)  
# x = Dropout(0.2)(x)  

# x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))  
# x = Dropout(0.2)(x)  

# x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
# x = Dropout(0.2)(x)  

# x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))  
# x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
# x = Conv2d_BN(x,nb_filter=256,kernel_size=(3,3),strides=(1,1),padding='same')  

# x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)    
# x = Conv2d_BN(x,nb_filter=512,kernel_size=(3,3),strides=(1,1),padding='same')  
# x = Conv2d_BN(x,nb_filter=1024,kernel_size=(3,3),strides=(1,1),padding='same')  

# (14,14,256)  
# x = Conv_Block(x,nb_filter=256,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True) 
# X = Conv2D(256,3,1)(x)
# x = Conv2d_BN(x,nb_filter=256,kernel_size=(3,3),strides=(1,1),padding='same')  
# x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
# x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
# x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
# x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
# x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
# (7,7,512)  
# x = Conv_Block(x,nb_filter=512,kernel_size=(3,3),strides=(1,1),with_conv_shortcut=True)  
# x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))  
# x = Conv_Block(x,nb_filter=512,kernel_size=(3,3)) 

# x = Conv2D(512,3,padding='same',strides=1,activation='relu')(x)  
# x = Conv2D(1024,3,padding='same',strides=1,activation='relu')(x)  
 

# model = Sequential()
# model.add(Convolution2D(
#     batch_input_shape=(None,width, length,num_channels),
#     filters=32,
#     kernel_size=7,#5
#     strides=2,
#     padding='same',     # Padding method
#     data_format='channels_last',
# ))
# model.add(Activation('relu'))
# # model.add(MaxPooling2D(2,2, 'same'))
# model.add(BatchNormalization())
# # model.add(MaxPooling2D(2,2, 'same'))
# model.add(Convolution2D(32, 5, strides=2, padding='same', data_format='channels_last'))#5
# model.add(Activation('relu'))
# # model.add(MaxPooling2D(2,2, 'same'))
# model.add(BatchNormalization())
# # model.add(MaxPooling2D(2,2, 'same'))
# model.add(Convolution2D(64, 5, strides=2, padding='same', data_format='channels_last'))#5
# model.add(Activation('relu'))
# # model.add(MaxPooling2D(2,2, 'same'))
# # model.add(BatchNormalization())
# # model.add(MaxPooling2D(2,2, 'same'))
# model.add(Convolution2D(64, 3, strides=2, padding='same', data_format='channels_last'))#5
# model.add(Activation('relu'))
# # model.add(MaxPooling2D(2,2, 'same'))
# # model.add(BatchNormalization())
# # model.add(MaxPooling2D(2,2, 'same'))
# model.add(Convolution2D(64, 3, strides=2, padding='same', data_format='channels_last'))#5
# model.add(Activation('relu'))
# # model.add(BatchNormalization())

# model.add(Convolution2D(128, 3, strides=1, padding='same', data_format='channels_last'))#5
# model.add(Activation('relu'))
# model.add(Convolution2D(128, 3, strides=1, padding='same', data_format='channels_last'))#5
# model.add(Activation('sigmoid'))
# # model.add(Convolution2D(128, 3, strides=1, padding='same', data_format='channels_last'))#5
# # model.add(Activation('sigmoid'))

# model.add(Flatten())
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

# x = AveragePooling2D(pool_size=(8,8))(x)  
x = Flatten()(x)
x = Dropout(0.4)(x)  
x = Dense(4,activation='softmax')(x)  
  
model = Model(inputs=inpt,outputs=x)  
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])  
model.summary()  



# model.save('nontrain.h5')
# print(model.summary())

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
        save_path='res_model_0717_'+str(20)+'.h5'
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

            
    save_path='res_model_0811_'+str(20)+'.h5'
    print(save_path)
    model.save(save_path)
       


