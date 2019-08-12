# -*- coding: utf-8 -*-
"""

Last updated on Aug 12 2019
@author: Shengjie Liu
@Email: liushengjie0756@gmail.com

This implementation reads *.tif files
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
vbs = 0  # if vbs==0, training in silent mode
bsz1 = 20  # batch size
ensemble = 1  # if ensemble>1, snapshot ensemble activated 
# if loss not decrease for 5 epoches, stop training
early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=2)


# for Monte Carlo runs  
seedx = [0,1,2,3,4,5,6,7,8,9]
seedi = 0  # default seed = 0

saved = {}  # save confusion matrix for raw classification
saved_p = {}  # save confusion matrix after object-based refinement

# 6509,8210

# bgx,bgy,imx,imy
subsets = [(0,0,6509,4105),
           (0,4105,6509,4105)]


def setGeo(geotransform,bgx,bgy):
    reset0 = geotransform[0] + bgx*geotransform[1]
    reset3 = geotransform[3] + bgy*geotransform[5]
    reset = (reset0,geotransform[1],geotransform[2],
             reset3,geotransform[4],geotransform[5])
    return reset


if True:
    for subset in subsets:
        time1 = int(time.time())
        K.clear_session()  # clear session before next loop 
        print(subset)
            
        gt = gdal.Open(gt1_file,gdal.GA_ReadOnly)
        im = gdal.Open(im1_file,gdal.GA_ReadOnly)
        projection = gt.GetProjection()
        geotransform = gt.GetGeoTransform()
        newgeo = setGeo(geotransform,subset[0],subset[1])
        gt = gt.ReadAsArray(subset[0],subset[1],subset[2],subset[3])
        im = im.ReadAsArray(subset[0],subset[1],subset[2],subset[3])
        im = im.transpose(1,2,0)
        cls1 = gt.max()

        im1x,im1y,im1z = im.shape
        im = np.float32(im)
        
        # kind of normalization, the max DN of the original image is 10000
        # the goal is to make the DNs range from (-2,2)
        im = im/5000.0
        
        # initilize controller 
        c1 = rscls.rscls(im,gt,cls=cls1)
        c1.padding(patch)  
        
        # random seed for Monte Carlo runs
        np.random.seed(seedx[seedi])
        x2_train,y2_train = c1.test_sample()
        print(x2_train.shape)
        try:
            x1_train = np.concatenate([x1_train,x2_train],axis=0)
            y1_train = np.concatenate([y1_train,y2_train],axis=0)
            del x2_train,y2_train
        except:
            print('If this line printed twice, failed to concatenate')
            x1_train = x2_train
            y1_train = y2_train
            
        np.save('x_train.npy',x1_train)
        np.save('y_train.npy',y1_train)
        