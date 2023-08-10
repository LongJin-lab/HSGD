#create network
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils import data

class block(nn.Module):
    def __init__(self,planes):
        super(block, self).__init__()
        self.fc1 = nn.Linear(planes,planes)
        self.fc2 = nn.Linear(planes,planes)
        self.fc3 = nn.Linear(planes,planes)
        self.fc4 = nn.Linear(planes,planes)
        self.relu = nn.ReLU()
    def forward(self,a,b,c,d,x0,x1=None,x2=None,x3=None):

        x1 = self.fc1(x0)
        x2 = self.fc2(x1)
        x3 = self.fc3(x2)
        x4 = self.fc4(x3)
        x5= a*x0 + b*x1 + c*x3+ d*x4
        x5 = self.relu(x5)
        return x2,x3,x4,x5

class Net(nn.Module):
    def __init__(self,block,planes):
        super(Net,self).__init__()
        self.a = nn.Parameter(torch.Tensor(1).uniform_(0.4, 1),requires_grad=True)
        self.b = nn.Parameter(torch.Tensor(1).uniform_(-1, 1),requires_grad=True)
        self.c = nn.Parameter(torch.Tensor(1).uniform_(0,0.5),requires_grad=True)
        self.block1 = block(planes)
        self.block2 = block(planes)
        self.block3 = block(planes)
        self.block4 = block(planes)
        self.block5 = block(planes)
        self.block6 = block(planes)
 
        self.fc = nn.Linear(planes,1)
        # self.softmax = nn.Softmax(dim = 1)
    def forward(self,x):
        x_1,x_2,x_3,out = self.block1(1+self.c,-2*self.c,self.c,self.b,x)
        x_1,x_2,x_3,out = self.block2(1+self.c,-2*self.c,self.c,self.b,x_1,x_2,x_3,out)
        x_1,x_2,x_3,out = self.block3(1+self.c,-2*self.c,self.c,self.b,x_1,x_2,x_3,out)
        x_1,x_2,x_3,out = self.block4(1+self.c,-2*self.c,self.c,self.b,x_1,x_2,x_3,out)
        x_1,x_2,x_3,out = self.block5(1+self.c,-2*self.c,self.c,self.b,x_1,x_2,x_3,out)
        _,_,_,out = self.block6(1+self.c,-2*self.c,self.c,self.b,x_1,x_2,x_3,out)
        out = self.fc(out)
        # return self.softmax(out)
        return out