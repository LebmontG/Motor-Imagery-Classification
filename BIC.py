'''
Classification of Motor Imagery EEG signal
2022.5.21 Z.Gan
'''

import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from scipy.fftpack import fft,ifft
import scipy.fftpack as fftpack
import gc
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn
from Transformer import *
from EEGnet import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from tqdm import tqdm
import time
import os,sys
#import mne

# data loading
train_path='./train/'
test_path='./test/'
train_x,train_y,test_x,test_y=[],[],[],[]
for i in range(4):
    tmp=np.load(train_path+'S'+str(i+1)+'.npz')
    label=tmp.files #X,y
    train_x.append(tmp[label[0]])
    train_y.append(tmp[label[1]])
    tmp=np.load(test_path+'S'+str(i+5)+'.npz')
    test_x.append(tmp[label[0]])
train_x,train_y=np.vstack(train_x),np.hstack(train_y)
test_x=np.vstack(test_x)
train_x.shape,test_x.shape,train_y.shape #((800, 13, 750), (800,))
#shuffle
ind=np.random.permutation(np.arange(len(train_x)))
train_x,train_y=train_x[ind],train_y[ind]
del tmp
gc.collect()

# data mining
ex=[]
for x in train_x:
    for ele in x:
        for v in ele:
            if v>100:
                ex.append(ele)
                break
        break
print(len(ex))
plt.plot(ex[0])

# data mining/PCA
plt.plot(train_x[0][0])
pca=PCA(n_components=2)
tx=np.transpose(np.round(train_x).astype('int64'),(0,2,1))
lx=np.transpose(np.round(test_x).astype('int64'),(0,2,1))
Lx,Tx=[],[]
for i in range(len(tx)):
    pca.fit(tx[i])
    Tx.append(pca.transform(tx[i]).flatten())
    pca.fit(lx[i])
    Lx.append(pca.transform(lx[i]).flatten())
#split
div=len(train_x)*9//10
tx,vx=np.split(Tx,[div])
ty,vy=np.split(train_y,[div])
tx,vx,lx=torch.LongTensor(tx),torch.LongTensor(vx),torch.LongTensor(Lx)
del Tx,Lx
gc.collect()
# M distance
def mah(x1,y1,x2,y2):
    x1=np.array([np.ravel(ele) for ele in x1])
    x2=np.array([np.ravel(ele) for ele in x2])
    x1_c0,x1_c1,x2_c0,x2_c1=x1[y1==0],x1[y1==1],x2[y2==0],x2[y2==1]
    def cal_dis(m1,m2):
        m1,m2=sum(m1)/len(m1),sum(m2)/len(m2)
        al=np.vstack((m1,m2))
        D = np.cov(al.T)+0.01  # covariance
        invD = np.linalg.inv(D)  # covariance inverse
        tp=m1-m2
        return np.sqrt(np.dot(np.dot(tp,invD),tp.T))
    return (cal_dis(x1_c0,x2_c0)+cal_dis(x1_c1,x2_c1))/2

# data mining/DFT
def DFT(x):
    res=[]
    for i in range(len(x)):
        tmp=[]
        for c in x[i]:
            tmp.append(fft(c)[:len(c)//2][:120])
            #tmp.append(abs(fft(c)[:len(c)//2][120:]))
        #print(len(tmp),tmp[0].shape);input()
        res.append(tmp)
    return np.array(res)
tmp=DFT(train_x)
plt.plot(tmp[0][0])
t=ifft(tmp[0][0])
plt.plot(t)
tmp=EA(tmp)
plt.plot(tmp[0][0])
a=[]
for i in range(len(tmp)):
    a.append([ele[120:].mean() for ele in tmp[i]])
loc=[]
for i in range(len(a)):
    for j in range(13):
        if a[i][j]>10:
            loc.append(i)
print(len(loc)) #319/10400
s_t=train_x[0][0]
plt.plot(s_t)
s_f=abs(fft(s_t)[:len(s_t)//2])
plt.plot(s_f[:150])

# Transformer
#train_x=np.round(train_x).astype('int32')
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#tx,ty=torch.LongTensor(train_x).to(device).flatten(1),torch.LongTensor(train_y).to(device)
vy,ty=torch.LongTensor(vy).unsqueeze(1),torch.LongTensor(ty).unsqueeze(1)
tx,vx=tx-tx.min(),vx-vx.min()
dic_len_enc=max(tx.max().item(),vx.max().item())
dic_len_dec=2 # 2 categories
bs=12
sten_len=tx.shape[-1] # source sentences length
tar_len=1 # target sentences length
m=transformer(dic_len_enc+1,dic_len_dec,0,0,
              d_word_vec=32, d_model=32, d_inner=128,
              d_k=32, d_v=32,n_position=len(tx[0]))
data=TensorDataset(tx,ty)
loader=DataLoader(
    dataset=data,
    batch_size=bs,
    shuffle=False,
    num_workers=2)
loader_test=DataLoader(
    dataset=TensorDataset(vx,vy),
    batch_size=bs,
    shuffle=False,
    num_workers=2)
L=[]
for i in range(2):
    L.append(train_epoch(m,loader))
    time.sleep(20)
plt.plot(L)
ac=test(m,loader_test)

# EEGnet
DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE=torch.device('cpu')
loss_func = nn.CrossEntropyLoss().to(DEVICE)
c=4
m=eeg_base(2)
m=EEGNet(inp_dim=[13,750],chns=[c//2,c,c*2],
                 groups=[c//4,c//4],classes_num=2,
                 pooling=[6,6],dp=0.1,k_size=64).to(DEVICE)
optimizer =optim.Adam(m.parameters())
tx,ty=torch.Tensor(train_x).unsqueeze(1),torch.LongTensor(train_y)
loader=DataLoader(
    dataset=TensorDataset(tx,ty),
    batch_size=1,
    shuffle=False,
    num_workers=2)
for epoch in range(1):
    for step, (b_x, b_y) in enumerate(loader):
        #print(b_x.shape,b_y.shape);input()
        b_x, b_y = b_x.to(DEVICE), b_y.to(DEVICE)
        output =m(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # test_out, last_layer =m(vx.to(DEVICE))
    # pred_y = torch.max(test_out.cpu(), 1)[1].data.numpy()
    # accuracy = float((pred_y ==vy.data.numpy()).astype(int).sum()) / float(vy.size(0))
    # print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.3f' % accuracy)

# xgboost
params = {
    "max_depth":7,
    "booster":'gbtree',
    "n_estimators":300,
    "learning_rate":0.3,
#    "objective":'reg:linear',
    "seed":1000
}
div=len(train_x)*9//10
x=np.array([np.ravel(ele) for ele in train_x])
y=train_y[:div]
xt=x[div:]
x=x[:div]
yt=train_y[div:]
xgboost=xgb.XGBClassifier(**params)
xgboost.fit(x,y)
r=xgboost.predict(xt)
ac=sum(r==yt)/len(yt)
print(ac)
# plot.pyplot.scatter(np.arange(len(train_x[0][0])),train_x[0][0])
# plot.pyplot.plot(train_x[0][1])
# plot.pyplot.plot(train_x[0][0])
# l1=[]
# l2=[]
# for i in range(750):
#     if abs(train_x[0][0][i]-train_x[1][0][i])>3:
#         l1.append([i,train_x[0][0][i]])
#         l2.append([i,train_x[1][0][i]])

def eeg_plot(sig,domain):
    for i in range(len(sig)):
        ax=plt.subplot(13,1,13-i)
        ax.yaxis.set_major_locator(MultipleLocator(600))
        plt.plot(sig[i],color='blue', linewidth=1.5)
        plt.ylabel('c'+str(i+1))
        if i!=0:plt.xticks([])
        plt.yticks([])
        plt.xlim(0,len(sig[0]))
        if i==0:plt.xlabel(domain)
    plt.gcf().savefig('1.eps',format='eps')
    return
