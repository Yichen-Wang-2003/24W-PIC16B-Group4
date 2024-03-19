# Description: This file contains the code for the 1D convolutional Neural network for our project.
# We first import all necessary packages and libraries, specifically torch related ones
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor 
import sqlite3
import math

# below is the code chunk for the 1D convolutional Neural network for our project. 

class Convolution1D(nn.Module):
    def __init__(self):
        '''
        Convolutional Neural Network with 1D convolutions
        params:
        - input_channels: number of input channels
        - output_channels: number of output channels
        - kernel_size: size of the kernel
        - stride: stride of the kernel
        - padding: padding of the kernel
        '''
        super(Convolution1D, self).__init__()
        
        # Initial Convolution
        # 22 input channels (features), 64 output channels, 3x3 kernel
        self.conv1 = nn.Conv1d(in_channels=22, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Batch Normalization with 64 features
        self.bninit = nn.BatchNorm1d(64)

        # Residual chunk 1
        self.res1_conv1 = nn.Conv1d(64, 64, 3, padding=1)
        self.res1_bn1 = nn.BatchNorm1d(64)
        self.res1_conv2 = nn.Conv1d(64, 64, 3, padding=1)
        self.res1_bn2 = nn.BatchNorm1d(64)
        
        # Residual chunk 2
        self.res2_conv1 = nn.Conv1d(64, 128, 3, padding=1, stride=2)  # Reduce dimensionality
        self.res2_bn1 = nn.BatchNorm1d(128)
        self.res2_conv2 = nn.Conv1d(128, 128, 3, padding=1)
        self.res2_bn2 = nn.BatchNorm1d(128)
        self.res2_shortcut = nn.Conv1d(64, 128, 1, stride=2)  # Shortcut to match dimensions


        # Dropout layer
        self.dropout = nn.Dropout(0.3)
        # Final global pooling and fully connected layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)
        
    def forward(self, x):
        # Initial conv layer
        x = F.relu(self.bninit(self.conv1(x)))
        
        # Residual Chunk 1
        res1 = self.res1_conv1(x)
        res1 = F.relu(self.res1_bn1(res1))
        res1 = self.res1_conv2(res1)
        x = F.relu(x + res1)
        
        # Residual Chunk 2
        res2 = self.res2_conv1(x)
        res2 = F.relu(self.res2_bn1(res2))
        res2 = self.res2_conv2(res2)
        shortcut = self.res2_shortcut(x)
        x = F.relu(res2 + shortcut)

        # Final layers
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        
        return x
