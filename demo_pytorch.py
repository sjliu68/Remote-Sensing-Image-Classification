# -*- coding: utf-8 -*-
"""
Created on Tue Aug 6 03:32:28 2019

@author: Shengjie Liu
@Email: liuishengjie0756@gmail.com
"""

import numpy as np
import rscls
from scipy import stats
import time
import torch 
import torch.nn as nn
import torch.utils.data as Data 
from skimage.segmentation import felzenszwalb
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Training configuration
imfile = 'paviaU_im.npy'  # width*height*channel
gtfile = 'paviaU_gt.npy'  # width*height, classes begin from 1 with background=0

ensemble = 1  # times of snapshot ensemble
num_per_cls = 10  # number of samples per class
bsz = 20  # batch size
patch = 5  # sample size: 5*5*channel
vbs = 1  # show training process

# Monte Carlo runs 
seedx = [0,1,2,3,4,5,6,7,8,9]
seedi = 0  # default seed is 0 

# network definition 
# wide contextual residual network (WCRN)
class WCRN(nn.Module):
    def __init__(self, num_classes=9):
        super(WCRN, self).__init__()

        self.conv1a = nn.Conv2d(103, 64, kernel_size=3, stride=1, padding=0)
        self.conv1b = nn.Conv2d(103, 64, kernel_size=1, stride=1, padding=0)
        self.maxp1 = nn.MaxPool2d(kernel_size = 3)
        self.maxp2 = nn.MaxPool2d(kernel_size = 5)
        
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2a = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)

        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        out = self.conv1a(x)
        out1 = self.conv1b(x)
        out = self.maxp1(out)
        out1 = self.maxp2(out1)
        
        out = torch.cat((out,out1),1)
        
        out1 = self.bn1(out)
        out1 = nn.ReLU()(out1)
        out1 = self.conv2a(out1)
        out1 = nn.ReLU()(out1)
        out1 = self.conv2b(out1)
        
        out = torch.add(out,out1)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        
        return out


#%% initilize controller and prepare training and testing samples
for seedi in range(1): # for Monte Carlo runs
    print('random seed:',seedi)
    _ls = []
    if True:
        gt = np.load(gtfile)
        cls1 = gt.max()
        im = np.load(imfile)
        imx,imy,imz = im.shape
        c = rscls.rscls(im,gt,cls=cls1)
        c.padding(patch)
        c.normalize(style='-11')
    
        np.random.seed(seedx[seedi])
        x_train,y_train = c.train_sample(num_per_cls)
        x_train,y_train = rscls.make_sample(x_train,y_train)
    
        x_test,y_test = c.test_sample()
        
    # segmentation
    seg = felzenszwalb(im[:,:,[30,50,90]],scale=0.5,sigma=0.8,
                       min_size=5,multichannel=True) 
    c.locate_obj(seg)  # locate samples in superpixels
    
    # pytorch input: (None,channel,width,height)
    x_train = np.transpose(x_train, (0,3,1,2))  
    x_test = np.transpose(x_test, (0,3,1,2))
        
    # convert np.array to torch.tensor
    x_train,y_train = torch.from_numpy(x_train),torch.from_numpy(y_train)
    x_test,y_test = torch.from_numpy(x_test),torch.from_numpy(y_test)
    
    # keep it in case of errors
    y_test = y_test.long()
    y_train = y_train.long()
    
    # define dataset for training and testing
    train_set = Data.TensorDataset(x_train,y_train) 
    test_set = Data.TensorDataset(x_test,y_test)
    
    train_loader = Data.DataLoader(
            dataset = train_set,
            batch_size = bsz,
            shuffle = True,
            num_workers = 0,
            )
    
    test_loader = Data.DataLoader(
            dataset = test_set,
            batch_size = bsz,
            shuffle = False,
            num_workers = 0,
            )
    
#%%  begin training
    time1 = int(time.time())
    model = WCRN(cls1)
    model.to(device)  # using gpu or cpu
    criterion = nn.CrossEntropyLoss()
    
    # train the model using lr=1.0
    train_model = True
    if train_model:
        lr = 1.0
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
        model.train()
        total_step = len(train_loader)
        num_epochs = 25
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        
#%% train the model using lr=0.8
        model.train()
        lr = 0.8
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
        total_step = len(train_loader)
        num_epochs = 15
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device) # sample to gpu/cpu
                labels = labels.to(device) # label to gpu/cpu
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (i+1) % 100 == 0: # print training
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                    
#%% Train the model using lr=0.1
        lr = 0.1
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
        total_step = len(train_loader)
        num_epochs = 10
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        time2 = int(time.time())
        print('training time:',time2-time1,'s')

#%% Test the model
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
        
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Test Accuracy of the model: {} %'.format(100 * correct / total))
            
#%% predict image
        time3 = int(time.time())
        pre_all_1 = []
        model.eval()
        with torch.no_grad():
            for i in range(ensemble):
                pre_rows_1 = []
                
                # uncommment if ensemble>1
                # model1.fit(x1_train,y1_train,batch_size=bsz1,epochs=2,verbose=vbs,shuffle=True)
                for j in range(imx):
                    # print(j)  # monitor predicting stages
                    sam_row = c.all_sample_row(j)
                    sam_row = np.transpose(sam_row, (0,3,1,2))
                    pre_row1 = model(torch.from_numpy(sam_row).to(device))
                    pre_row1 = np.argmax(np.array(pre_row1.cpu()),axis=1)
                    pre_row1 = pre_row1.reshape(1,imy)
                    pre_rows_1.append(pre_row1)
                pre_all_1.append(np.array(pre_rows_1))
            
        time4 = int(time.time())
        print('predicted time:',time4-time3,'s')

        # raw classification
        pre_all_1 = np.array(pre_all_1).reshape(ensemble,imx,imy)
        pre1 = np.int8(stats.mode(pre_all_1,axis=0)[0]).reshape(imx,imy)
        result11 = rscls.gtcfm(pre1+1,c.gt+1,cls1)
        
        # after post processin using superpixel-based refinement 
        pcmap = rscls.obpc(c.seg,pre1,c.obj)
        result12 = rscls.gtcfm(pcmap+1,c.gt+1,cls1)
        rscls.save_cmap(pre1,'jet','pre.png')    
        rscls.save_cmap(pcmap,'jet','pcmap.png')  

