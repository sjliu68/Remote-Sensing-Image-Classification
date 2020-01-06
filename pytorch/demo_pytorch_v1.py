from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR,MultiStepLR
import torch.utils.data as Data 
import torchnet
import rscls
import numpy as np
from scipy import stats
import time

#%% arguments
#Net = torchnet.wcrn
Net = torchnet.resnet99_avg
imfile = 'paU_im.npy'
gtfile = 'paU_gt.npy'
patch = 9
vbs = 1

seedx = [0,1,2,3,4,5,6,7,8,9]
seedi = 0
criterion = nn.CrossEntropyLoss()

#%%
def train(args, model, device, train_loader, optimizer, epoch, vbs=0):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if vbs==0:
            continue
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#%% begin training
#if True:
oa = []
for seedi in range(10): # for Monte Carlo runs
    print('random seed:',seedi)
    parser = argparse.ArgumentParser(description='PyTorch PaviaU')
    parser.add_argument('--nps', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
#%%
    time1 = time.time()
    gt = np.load(gtfile)
    cls1 = gt.max()
    im = np.load(imfile)
    im = np.float32(im)
    im = im/5000.0
    imx,imy,imz = im.shape
    c = rscls.rscls(im,gt,cls=cls1)
    c.padding(patch)
#    c.normalize(style='01')
    
    np.random.seed(seedx[seedi])
    x_train,y_train = c.train_sample(args.nps)
    x_train,y_train = rscls.make_sample(x_train,y_train)
    x_test,y_test = c.test_sample()
    x_train = np.transpose(x_train, (0,3,1,2))  
    x_test = np.transpose(x_test, (0,3,1,2))
    
    x_train,y_train = torch.from_numpy(x_train),torch.from_numpy(y_train)
    x_test,y_test = torch.from_numpy(x_test),torch.from_numpy(y_test)

    y_test = y_test.long()
    y_train = y_train.long()
    
    train_set = Data.TensorDataset(x_train,y_train) 
    test_set = Data.TensorDataset(x_test,y_test)
    
    train_loader = Data.DataLoader(
            dataset = train_set,
            batch_size = args.batch_size,
            shuffle = True,
            **kwargs
            )
    
    test_loader = Data.DataLoader(
            dataset = test_set,
            batch_size = args.test_batch_size,
            shuffle = False,
            **kwargs
            )
    
    time2 = int(time.time())
    print('load time:',time2-time1,'s')
    
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = MultiStepLR(optimizer, milestones=[170,200], gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, vbs=vbs)
        #test(args, model, device, test_loader)
        scheduler.step()
        
    time3 = int(time.time())
    print('train time:',time3-time2,'s')
    
    # single test
#    test(args,model,device,test_loader)
    time4 = int(time.time())
#    print('test time:',time4-time3,'s')
    
    # predict
    pre_all_1 = []
    model.eval()
    with torch.no_grad():
        ensemble = 1
        for i in range(ensemble):
            pre_rows_1 = []
            for j in range(imx):
                # print(j)  # monitor predicting stages
                sam_row = c.all_sample_row(j)
                sam_row = np.transpose(sam_row, (0,3,1,2))
                pre_row1 = model(torch.from_numpy(sam_row).to(device))
                pre_row1 = np.argmax(np.array(pre_row1.cpu()),axis=1)
                pre_row1 = pre_row1.reshape(1,imy)
                pre_rows_1.append(pre_row1)
            pre_all_1.append(np.array(pre_rows_1))
            
    time5 = int(time.time())
    print('predicted time:',time5-time4,'s')
    
    pre_all_1 = np.array(pre_all_1).reshape(ensemble,imx,imy)
    pre1 = np.int8(stats.mode(pre_all_1,axis=0)[0]).reshape(imx,imy)
    result11 = rscls.gtcfm(pre1+1,c.gt+1,cls1)
    oa.append(result11[-1,0])
    rscls.save_cmap(pre1,'jet','pre.png')

    
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
        
#%%
oa2 = np.array(oa)
print(oa2.mean(),oa2.std())
