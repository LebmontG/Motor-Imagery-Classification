import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.linalg import fractional_matrix_power

def EA(X):
    N = X.shape[0]
    R=sum([np.dot(X[i],X[i].T) for i in range(N)])
    R_ = fractional_matrix_power(R/N, -1/2)
    return np.array([np.dot(R_,X[i]) for i in range(N)])

def L2Loss(model):
    l2_loss = torch.tensor(0.0,requires_grad = True)
    for name,parma in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.5*0.1* torch.sum(torch.pow(parma,2)))
    return l2_loss

class EEGNet(nn.Module):
    def __init__(self,inp_dim=[13,750],chns=[12,24,24],
                 groups=[6,12],classes_num=2,
                 pooling=[6,6],dp=0.1,k_size=64
                 ):
        super(EEGNet, self).__init__()
        self.drop_out =dp
        
        self.block_1 = nn.Sequential(
            # nn.ZeroPad2d((left, right, up, bottom))
            # fill '0' in excepted directions
            #nn.ZeroPad2d((0,0, 0, 0)),
            nn.Conv2d(
                in_channels=1,
                out_channels=chns[0],
                kernel_size=(1,k_size),
                bias=False
            ),
            nn.BatchNorm2d(chns[0])
        )
        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=chns[0],
                out_channels=chns[1],
                kernel_size=(inp_dim[0], 1),
                groups=groups[0],
                bias=False
            ),
            nn.BatchNorm2d(chns[1]),
            nn.ELU(),
            nn.AvgPool2d((1,pooling[0])),
            nn.Dropout(self.drop_out)
        )
        
        self.block_3 = nn.Sequential(
            #nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
               in_channels=chns[1],
               out_channels=chns[1],
               kernel_size=(1,1),
               groups=groups[1],
               bias=False
            ), # (chns[2], 1, T//4)
            nn.Conv2d(
                in_channels=chns[1],
                out_channels=chns[2],
                kernel_size=(1, 1),
                bias=False
            ), # (16, 1, T//4)
            nn.BatchNorm2d(chns[2]),
            nn.ELU(),
            nn.AvgPool2d((1,pooling[1])), # (chns[2], 1, T//32)
            nn.Dropout(self.drop_out)
        )
        dim=chns[2]*((inp_dim[1]-k_size+1)//pooling[0]//pooling[1])
        self.out1 = nn.Linear(dim,dim//2)
        self.out2 = nn.Linear(dim//2,classes_num)
    
    def forward(self, x):
        x = self.block_1(x)
        #print(x.shape) # [1,chn[0],inp_dim[0],inp_dim[1]-k_size+1]
        x = self.block_2(x)
        #print(x.shape) # [1,chn[1], 1,~//pooling[0]]
        x = self.block_3(x)
        #print(x.shape);input() # [1,chn[1], 1,~//pooling[1]]
        
        x = x.view(x.size(0), -1)
        x = self.out2(self.out1(x))
        return F.softmax(x, dim=1), x   # return x for visualization