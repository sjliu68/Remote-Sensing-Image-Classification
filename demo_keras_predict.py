# -*- coding: utf-8 -*-
"""
Last updated on Aug 12 2019
@author: Shengjie Liu
@Email: liushengjie0756@gmail.com

This script loads a model and then predict on the whole image.
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
from keras.models import load_model

## data location
im1_file = 'F:/t0809/data/yubei0603.tif'
gt1_file = 'yubei2.tif'

# load model created by demo_keras_predict.py
model_file = 'model99079.h5'

## network configuration
patch = 5  # If patch==5, WCRN. If patch==7, BDMA. If patch==9, ResNet-avg.
vbs = 1  # if vbs==0, training in silent mode
bsz1 = 20  # batch size
ensemble = 1  # if ensemble>1, snapshot ensemble activated 

# for Monte Carlo runs  
seedx = [0,1,2,3,4,5,6,7,8,9]
seedi = 0  # default seed = 0

saved = {}  # save confusion matrix for raw classification

bgx,bgy,imx,imy = 1000,1000,5000,5000

def setGeo(geotransform,bgx,bgy):
    reset0 = geotransform[0] + bgx*geotransform[1]
    reset3 = geotransform[3] + bgy*geotransform[5]
    reset = (reset0,geotransform[1],geotransform[2],
             reset3,geotransform[4],geotransform[5])
    return reset

if True:
    for seedi in range(0,1):
        time1 = int(time.time())
        K.clear_session()  # clear session before next loop 
        print('seed'+str(seedi)+','+str(seedx[seedi]))
            
        gt = gdal.Open(gt1_file,gdal.GA_ReadOnly) # dataset
        im = gdal.Open(im1_file,gdal.GA_ReadOnly)
        projection = gt.GetProjection()
        geotransform = gt.GetGeoTransform()
        newgeo = setGeo(geotransform,bgx,bgy)
        gt = gt.ReadAsArray(bgx,bgy,imx,imy)
        im = im.ReadAsArray(bgx,bgy,imx,imy)
        im = im.transpose(1,2,0)
        cls1 = gt.max()

        im1x,im1y,im1z = im.shape
        im = np.float32(im)
        
        # kind of normalization, the max DN of the original image is 8000
        im = im/5000.0
        
        # initilize controller 
        c1 = rscls.rscls(im,gt,cls=cls1)
        c1.padding(patch)  
        
        model1 = load_model(model_file)
        time3 = int(time.time())
        
        # predict part
        pre_all_1 = []
        for i in range(ensemble):
            pre_rows_1 = []
            # uncomment below if snapshot ensemble activated
            # model1.fit(x1_train,y1_train,batch_size=bsz1,epochs=2,verbose=vbs,shuffle=True)
            for j in range(im1x):
                if j%100==0:
                    print(j)
                #print(j) uncomment to monitor predicing stages
                sam_row = c1.all_sample_row(j)
                if patch == 7:
                    sam_row = sam_row.reshape(sam_row.shape[0],patch,patch,im1z,1)
                pre_row1 = np.argmax(model1.predict(sam_row),axis=1)
                pre_row1 = pre_row1.reshape(1,im1y)
                pre_rows_1.append(pre_row1)
            pre_all_1.append(np.array(pre_rows_1))
            
        time4 = int(time.time())
        print('predict time:',time4-time3) # predict time

        # classification map and confusion matrix for raw classification
        pre_all_1 = np.array(pre_all_1).reshape(ensemble,im1x,im1y)
        pre1 = np.int8(stats.mode(pre_all_1,axis=0)[0]).reshape(im1x,im1y)
        result11 = rscls.gtcfm(pre1+1,c1.gt+1,cls1)
        saved[str(seedi)+'a'] = result11
        rscls.save_cmap(pre1, 'jet', 'pre.png')
        
        # save as geocode-tif
        name = 'predict_'+str(time4)[-5:]
        outdata = gdal.GetDriverByName('GTiff').Create(name+'.tif', im1y, im1x, 1, gdal.GDT_UInt16)
        outdata.SetGeoTransform(newgeo)
        outdata.SetProjection(projection)
        outdata.GetRasterBand(1).WriteArray(pre1+1)
        outdata.FlushCache() ##saves to disk!!
        outdata = None