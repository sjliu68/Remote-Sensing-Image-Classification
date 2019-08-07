# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 01:36:12 2019

@author: SJ
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
from sklearn.decomposition import PCA,IncrementalPCA
#from sklearn.covariance import GraphicalLassoCV
from keras.callbacks import EarlyStopping
from scipy.io import loadmat
from skimage.segmentation import felzenszwalb

## data location
im1_file = 'PaviaU.mat'
gt1_file = 'PaviaU_gt.mat'

## number of training samples per class
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--nos', type=int, default = 10)
args = parser.parse_args()
num_per_cls1 = args.nos

## network configuration
patch = 7  # if patch==5, WCRN. if patch==7, BDMA. if patch==9, ResNet-avg.
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

if True:
    for seedi in range(0,1):
        time1 = int(time.time())
        K.clear_session()  # clear session before next loop 
        print('seed'+str(seedi)+','+str(seedx[seedi]))
            
        gt = loadmat(gt1_file)['paviaU_gt']
        im = loadmat(im1_file)['paviaU']
        cls1 = gt.max()

        im1x,im1y,im1z = im.shape
        im = np.float32(im)
        
        # segmentation on top-3 PCs
        estimator = IncrementalPCA(n_components=3)
        estimator = PCA(n_components=3)
        im = im.reshape(im1x*im1y,-1)
        im2 = estimator.fit_transform(im)
        im2 = im2.reshape(im1x,im1y,3)
        seg = felzenszwalb(im2, scale=0.5, sigma=0.8, min_size=5, multichannel=True)
        im = im.reshape(im1x,im1y,im1z)
        
        # kind of normalization, the max DN of the original image is 8000
        im = im/5000.0
        
        # initilize controller 
        c1 = rscls.rscls(im,gt,cls=cls1)
        c1.padding(patch)  
        c1.locate_obj(seg)  # locate superpixels
        
        # random seed for Monte Carlo runs
        np.random.seed(seedx[seedi])
        x1_train,y1_train = c1.train_sample(num_per_cls1)  # load train samples
        x1_train,y1_train = rscls.make_sample(x1_train,y1_train)  # augmentation
        y1_train = to_categorical(y1_train)  # to one-hot labels
        

        ''' training part '''
        im1z = im.shape[2]
        if patch == 7:
            model1 = nw.DBMA(im1z,cls1) # 3D CNN, samples are 5-dimensional
            x1_train = x1_train.reshape(x1_train.shape[0],patch,patch,im1z,-1)
        elif patch == 5:
            model1 = nw.wcrn(im1z,cls1) # WCRN
        elif patch == 9:
            model1 = nw.resnet99_avg(im1z,patch,cls1,l=1)
        else:
#            print('using resnet_avg')
            model1 = nw.resnet99_avg(im1z,patch,cls1,l=1)
        time2 = int(time.time())
        
        # first train the model with lr=1.0
        model1.compile(loss=categorical_crossentropy,optimizer=Adadelta(lr=1.0),metrics=['accuracy'])
        model1.fit(x1_train,y1_train,batch_size=bsz1,epochs=170,verbose=vbs,shuffle=True)
            
        # then train the model with lr=0.1
        model1.compile(loss=categorical_crossentropy,optimizer=Adadelta(lr=0.1),metrics=['accuracy'])   
        model1.fit(x1_train,y1_train,batch_size=bsz1,epochs=30,verbose=vbs,shuffle=True,callbacks=[early_stopping])
        time3 = int(time.time()) # training time
        print('training time:',time3-time2)
        #model1.save('model'+str(time3)[-5:]+'.h5') # uncomment to save model
        
        # predict part
        pre_all_1 = []
        for i in range(ensemble):
            pre_rows_1 = []
            # uncomment below if snapshot ensemble activated
            # model1.fit(x1_train,y1_train,batch_size=bsz1,epochs=2,verbose=vbs,shuffle=True)
            for j in range(im1x):
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
        
        # after object-based refinement
        pcmap = rscls.obpc(c1.seg,pre1,c1.obj)
        result12 = rscls.gtcfm(pcmap+1,c1.gt+1,cls1)
        saved_p[str(seedi)+'b'] = result12
        rscls.save_cmap(pre1, 'jet', 'pre.png')    
        rscls.save_cmap(pcmap, 'jet', 'pcmap.png')  
        
