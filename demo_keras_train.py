# -*- coding: utf-8 -*-
"""

Last updated on Aug 12 2019
@author: Shengjie Liu
@Email: liushengjie0756@gmail.com

This script reads *.npy samples for training.
"""

import numpy as np
from scipy import stats
import rscls
import matplotlib.pyplot as plt
import time
import networks as nw
from keras.utils import to_categorical
import keras
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras import backend as K
import argparse
#from sklearn.covariance import GraphicalLassoCV
from keras.callbacks import EarlyStopping
from scipy.io import loadmat
import gdal

## data location
im1_file = 'F:/t0809/data/yubei0603.tif'
gt1_file = 'yubei2.tif'

## number of training samples per class
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--nos', type=int, default = 5)
args = parser.parse_args()
num_per_cls1 = args.nos

## network configuration
patch = 5  # If patch==5, WCRN. If patch==7, BDMA. If patch==9, ResNet-avg.
vbs = 1  # if vbs==0, training in silent mode
bsz1 = 20  # batch size
ensemble = 1  # if ensemble>1, snapshot ensemble activated 
# if loss not decrease for 5 epoches, stop training 
early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=2)


# for Monte Carlo runs  
seedx = [0,1,2,3,4,5,6,7,8,9]
seedi = 0  # default seed = 0

saved = {}  # save confusion matrix for raw classification
saved_p = {}  # save confusion matrix after object-based refinement

if True:
    for seedi in range(0,1):
        time1 = int(time.time())
        K.clear_session()  # clear session before next loop 

        x1_train,y1_train = np.load('x_train.npy'),np.load('y_train.npy')
        cls1 = y1_train.max()+1
        y1_train = to_categorical(y1_train)  # to one-hot labels
        

        ''' training part '''
        im1z = x1_train.shape[-1]
        if patch == 7:
            model1 = nw.DBMA(im1z,cls1) # 3D CNN, samples are 5-dimensional
            x1_train = x1_train.reshape(x1_train.shape[0],patch,patch,im1z,-1)
        elif patch == 5:
            model1 = nw.wcrn(im1z,cls1) # WCRN
            # model1 = nw.DCCNN(im1z,patch,cls1) # DCCNN
        elif patch == 9:
            model1 = nw.resnet99_avg(im1z,patch,cls1,l=1)
        else:
#            print('using resnet_avg')
            model1 = nw.resnet99_avg(im1z,patch,cls1,l=1)
        time2 = int(time.time())
        
        # first train the model with lr=1.0
        print('start training')
        model1.compile(loss=categorical_crossentropy,optimizer=Adadelta(lr=1.0),metrics=['accuracy'])
        model1.fit(x1_train,y1_train,batch_size=bsz1,epochs=170,verbose=vbs,shuffle=True,callbacks=[early_stopping])
            
        # then train the model with lr=0.1
        model1.compile(loss=categorical_crossentropy,optimizer=Adadelta(lr=0.1),metrics=['accuracy'])   
        model1.fit(x1_train,y1_train,batch_size=bsz1,epochs=30,verbose=vbs,shuffle=True,callbacks=[early_stopping])
        time3 = int(time.time()) # training time
        print('training time:',time3-time2)
        model1.save('model'+str(time3)[-5:]+'.h5') # uncomment to save model
        