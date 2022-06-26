import torch
import torch.nn as nn
import torch.nn.functional as F


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
        dim=chns[1]*((inp_dim[1]-k_size+1)//pooling[0]//pooling[1])
        self.out = nn.Linear(dim,classes_num)
    
    def forward(self, x):
        x = self.block_1(x)
        #print(x.shape) # [1,chn[0],inp_dim[0],inp_dim[1]-k_size+1]
        x = self.block_2(x)
        #print(x.shape) # [1,chn[1], 1,~//pooling[0]]
        x = self.block_3(x)
        #print(x.shape);input() # [1,chn[1], 1,~//pooling[1]]
        
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return F.softmax(x, dim=1), x   # return x for visualization