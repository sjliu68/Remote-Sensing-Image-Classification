from __future__ import print_function
import keras
from keras.models import Model
from keras.layers import concatenate, Dense, Dropout, Flatten, Add, SpatialDropout2D, Conv3D
from keras.layers import Conv2D, MaxPooling2D, Input, Activation,AveragePooling2D,BatchNormalization
from keras.layers import MaxPooling3D, AveragePooling3D
from keras import backend as K
from keras import regularizers
from keras import initializers
from keras.initializers import he_normal, RandomNormal
from keras.layers import multiply, GlobalAveragePooling2D, GlobalAveragePooling3D
from keras.layers.core import Reshape, Dropout

def DCCNN(band, imx, ncla1):
    input1 = Input(shape=(imx,imx,band))

    # define network
    conv01 = Conv2D(128,kernel_size=(1,1),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv02 = Conv2D(128,kernel_size=(3,3),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv03 = Conv2D(128,kernel_size=(5,5),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn1 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    bn2 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv0 = Conv2D(128,kernel_size=(1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv11 = Conv2D(128,kernel_size=(1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv12 = Conv2D(128,kernel_size=(1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv21 = Conv2D(128,kernel_size=(1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv22 = Conv2D(128,kernel_size=(1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv31 = Conv2D(128,kernel_size=(1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv32 = Conv2D(128,kernel_size=(1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv33 = Conv2D(128,kernel_size=(1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
#    
    fc1 = Dense(ncla1,activation='softmax',name='output1',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    # x1
    x1 = conv01(input1)
    x2 = conv02(input1)
    x3 = conv03(input1)
    x1 = MaxPooling2D(pool_size=(5,5))(x1)
    x2 = MaxPooling2D(pool_size=(3,3))(x2)
    x1 = concatenate([x1,x2,x3],axis=-1)
    
    x1 = Activation('relu')(x1)
    x1 = bn1(x1)
    x1 = conv0(x1)
    
    x11 = Activation('relu')(x1)
    x11 = bn2(x11)
    x11 = conv11(x11)
    x11 = Activation('relu')(x11)
    x11 = conv12(x11)
    x1 = Add()([x1,x11])
    
    x11 = Activation('relu')(x1)
    x11 = conv21(x11)
    x11 = Activation('relu')(x11)
    x11 = conv22(x11)
    x1 = Add()([x1,x11])
    
    x1 = Activation('relu')(x1)
    x1 = conv31(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(0.5)(x1)
    x1 = conv32(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(0.5)(x1)
    x1 = conv33(x1)
    
    x1 = Flatten()(x1)
    pre1 = fc1(x1)

    model1 = Model(inputs=input1, outputs=pre1)
    return model1
    

def DBMA(band, ncla1):
    input1 = Input(shape=(7,7,band,1))
    
    ## spectral branch
    conv11 = Conv3D(24,kernel_size=(1,1,7),strides=(1,1,2),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    
    bn12 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv12 = Conv3D(24,kernel_size=(1,1,7),strides=(1,1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    
    bn13 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv13 = Conv3D(24,kernel_size=(1,1,7),strides=(1,1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    
    bn14 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv14 = Conv3D(24,kernel_size=(1,1,7),strides=(1,1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn15 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv15 = Conv3D(60,kernel_size=(1,1,4),strides=(1,1,1),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    fc11 = Dense(30,activation=None,kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    fc12 = Dense(60,activation=None,kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    
    ## spatial branch
    conv21 = Conv3D(24,kernel_size=(1,1,band),strides=(1,1,1),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    
    bn22 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv22 = Conv3D(12,kernel_size=(3,3,1),strides=(1,1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    
    bn23 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv23 = Conv3D(12,kernel_size=(3,3,1),strides=(1,1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    
    bn24 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv24 = Conv3D(12,kernel_size=(3,3,1),strides=(1,1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn25 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv25 = Conv3D(24,kernel_size=(3,3,1),strides=(1,1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv26 = Conv3D(1,activation=None,kernel_size=(3,3,2),strides=(1,1,2),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    
    fc = Dense(ncla1,activation='softmax',name='output1',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    
    # spectral 
    x1 = conv11(input1)
    
    x11 = bn12(x1)
    x11 = Activation('relu')(x11)
    x11 = conv12(x11)
    
    x12 = concatenate([x1,x11],axis=-1)
    x12 = bn13(x12)
    x12 = Activation('relu')(x12)
    x12 = conv13(x12)
    
    x13 = concatenate([x1,x11,x12],axis=-1)
    x13 = bn14(x13)
    x13 = Activation('relu')(x13)
    x13 = conv14(x13)
    
    x14 = concatenate([x1,x11,x12,x13],axis=-1)
    x14 = bn15(x14)
    x14 = Activation('relu')(x14)
    x14 = conv15(x14)
    
    x1_max = MaxPooling3D(pool_size=(7,7,1))(x14)
    x1_avg = AveragePooling3D(pool_size=(7,7,1))(x14)
    
    x1_max = fc11(x1_max)
    x1_max = fc12(x1_max)
    
    x1_avg = fc11(x1_avg)
    x1_avg = fc12(x1_avg)
    
    x1 = Add()([x1_max,x1_avg])
    x1 = Activation('sigmoid')(x1)
    x1 = multiply([x1,x14])
    x1 = GlobalAveragePooling3D()(x1)
    
    # spatial
    x2 = conv21(input1)
    x21 = bn22(x2)
    x21 = Activation('relu')(x21)
    x21 = conv22(x21)
    
    x22 = concatenate([x2,x21],axis=-1)
    x22 = bn23(x22)
    x22 = Activation('relu')(x22)
    x22 = conv23(x22)
    
    x23 = concatenate([x2,x21,x22],axis=-1)
    x23 = bn24(x23)
    x23 = Activation('relu')(x23)
    x23 = conv24(x23)
    
    x24 = concatenate([x2,x21,x22,x23],axis=-1)
    x24 = Reshape(target_shape=(7,7,60,1))(x24)
    
    x2_max = MaxPooling3D(pool_size=(1,1,60))(x24)
    x2_avg = AveragePooling3D(pool_size=(1,1,60))(x24)
    
    x2_max = Reshape(target_shape=(7,7,1))(x2_max)
    x2_avg = Reshape(target_shape=(7,7,1))(x2_avg)
    
    x25 = concatenate([x2_max,x2_avg],axis=-1)
    x25 = Reshape(target_shape=(7,7,2,1))(x25)
    x25 = conv26(x25)
    x25 = Activation('sigmoid')(x25)
    
    x2 = multiply([x24,x25])
    x2 = Reshape(target_shape=(7,7,1,60))(x2)
    x2 = GlobalAveragePooling3D()(x2)
    
    x = concatenate([x1,x2],axis=-1)
    pre = fc(x)
    
    model = Model(inputs=input1, outputs=pre)
    return model

def resnet99_avg_se(band, imx, ncla1, l=1):
    input1 = Input(shape=(imx,imx,band))

    # define network
    conv0x = Conv2D(32,kernel_size=(3,3),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv0 = Conv2D(32,kernel_size=(3,3),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn11 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv11 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv12 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    fc11 = Dense(4,activation=None,kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    fc12 = Dense(64,activation=None,kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    
    
    bn21 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv21 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv22 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    fc21 = Dense(4,activation=None,kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    fc22 = Dense(64,activation=None,kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    
    
    fc1 = Dense(ncla1,activation='softmax',name='output1',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    # x1
    x1 = conv0(input1)
    x1x = conv0x(input1)
#    x1 = MaxPooling2D(pool_size=(2,2))(x1)
#    x1x = MaxPooling2D(pool_size=(2,2))(x1x)
    x1 = concatenate([x1,x1x],axis=-1)
    x11 = bn11(x1)
    x11 = Activation('relu')(x11)
    x11 = conv11(x11)
    x11 = Activation('relu')(x11)
    x11 = conv12(x11)
    x12 = GlobalAveragePooling2D()(x11)
    x12 = fc11(x12)
    x12 = fc12(x12)
    x12 = Activation('sigmoid')(x12)
    x11 = multiply([x11,x12])
    x1 = Add()([x1,x11])
    
    if l==2:
        x11 = bn21(x1)
        x11 = Activation('relu')(x11)
        x11 = conv21(x11)
        x11 = Activation('relu')(x11)
        x11 = conv22(x11)
        x12 = GlobalAveragePooling2D()(x11)
        x12 = fc11(x12)
        x12 = fc12(x12)
        x12 = Activation('sigmoid')(x12)
        x11 = multiply([x11,x12])
        x1 = Add()([x1,x11])
    
    x1 = GlobalAveragePooling2D()(x1)
    
#    x1 = Flatten()(x1)
    pre1 = fc1(x1)

    model1 = Model(inputs=input1, outputs=pre1)
    return model1

def resnet99_avg(band, imx, ncla1, l=1):
    input1 = Input(shape=(imx,imx,band))

    # define network
    conv0x = Conv2D(32,kernel_size=(3,3),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv0 = Conv2D(32,kernel_size=(3,3),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn11 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv11 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv12 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn21 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv21 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv22 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    
    fc1 = Dense(ncla1,activation='softmax',name='output1',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    # x1
    x1 = conv0(input1)
    x1x = conv0x(input1)
#    x1 = MaxPooling2D(pool_size=(2,2))(x1)
#    x1x = MaxPooling2D(pool_size=(2,2))(x1x)
    x1 = concatenate([x1,x1x],axis=-1)
    x11 = bn11(x1)
    x11 = Activation('relu')(x11)
    x11 = conv11(x11)
    x11 = Activation('relu')(x11)
    x11 = conv12(x11)
    x1 = Add()([x1,x11])
    
    if l==2:
        x11 = bn21(x1)
        x11 = Activation('relu')(x11)
        x11 = conv21(x11)
        x11 = Activation('relu')(x11)
        x11 = conv22(x11)
        x1 = Add()([x1,x11])
    
    x1 = GlobalAveragePooling2D()(x1)
    
#    x1 = Flatten()(x1)
    pre1 = fc1(x1)

    model1 = Model(inputs=input1, outputs=pre1)
    return model1


def resnet99(band, ncla1):
    input1 = Input(shape=(9,9,band))

    # define network
    conv0x = Conv2D(32,kernel_size=(3,3),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv0 = Conv2D(32,kernel_size=(3,3),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn11 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv11 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv12 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn21 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv21 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv22 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    
    fc1 = Dense(ncla1,activation='softmax',name='output1',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    # x1
    x1 = conv0(input1)
    x1x = conv0x(input1)
#    x1 = MaxPooling2D(pool_size=(2,2))(x1)
#    x1x = MaxPooling2D(pool_size=(2,2))(x1x)
    x1 = concatenate([x1,x1x],axis=-1)
    x11 = bn11(x1)
    x11 = Activation('relu')(x11)
    x11 = conv11(x11)
    x11 = Activation('relu')(x11)
    x11 = conv12(x11)
    x1 = Add()([x1,x11])
    
#    x11 = bn21(x1)
#    x11 = Activation('relu')(x11)
#    x11 = conv21(x11)
#    x11 = Activation('relu')(x11)
#    x11 = conv22(x11)
#    x1 = Add()([x1,x11])
    
    x1 = Flatten()(x1)
    pre1 = fc1(x1)

    model1 = Model(inputs=input1, outputs=pre1)
    return model1

def wcrn3D(band, ncla1):
    input1 = Input(shape=(5,5,band))

    # define network
    conv0x = Conv2D(64,kernel_size=(1,1,7),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv0 = Conv2D(64,kernel_size=(3,3,1),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn11 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv11 = Conv2D(128,kernel_size=(1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv12 = Conv2D(128,kernel_size=(1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    fc1 = Dense(ncla1,activation='softmax',name='output1',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    # x1
    x1 = conv0(input1)
    x1x = conv0x(input1)
    x1 = MaxPooling2D(pool_size=(3,3))(x1)
    x1x = MaxPooling2D(pool_size=(5,5))(x1x)
    x1 = concatenate([x1,x1x],axis=-1)
    x11 = bn11(x1)
    x11 = Activation('relu')(x11)
    x11 = conv11(x11)
    x11 = Activation('relu')(x11)
    x11 = conv12(x11)
    x1 = Add()([x1,x11])
    
    x1 = Flatten()(x1)
    pre1 = fc1(x1)

    model1 = Model(inputs=input1, outputs=pre1)
    return model1

def wcrn(band, ncla1):
    input1 = Input(shape=(5,5,band))

    # define network
    conv0x = Conv2D(64,kernel_size=(1,1),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv0 = Conv2D(64,kernel_size=(3,3),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn11 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv11 = Conv2D(128,kernel_size=(1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv12 = Conv2D(128,kernel_size=(1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
#    
    fc1 = Dense(ncla1,activation='softmax',name='output1',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    # x1
    x1 = conv0(input1)
    x1x = conv0x(input1)
    x1 = MaxPooling2D(pool_size=(3,3))(x1)
    x1x = MaxPooling2D(pool_size=(5,5))(x1x)
    x1 = concatenate([x1,x1x],axis=-1)
    x11 = bn11(x1)
    x11 = Activation('relu')(x11)
    x11 = conv11(x11)
    x11 = Activation('relu')(x11)
    x11 = conv12(x11)
    x1 = Add()([x1,x11])
    
    x1 = Flatten()(x1)
    pre1 = fc1(x1)

    model1 = Model(inputs=input1, outputs=pre1)
    return model1