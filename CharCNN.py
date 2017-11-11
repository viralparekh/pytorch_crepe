import os
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms



class CharCNN(nn.Module):
    #Model params
    vocab_size=69
    maxlen = 256
    nb_filter = 256                     #Filters for conv layers
    dense_outputs = 1024                #Number of units in the dense layer
    filter_kernels = [7, 7, 3, 3, 3, 3] #Conv layer kernel size
    cat_output = 4                      #Number of units in the final output layer. Number of classes.

    def __init__(self):
        super(CharCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(self.vocab_size,self.nb_filter,kernel_size=self.filter_kernels[0]),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(self.nb_filter,self.nb_filter,kernel_size=self.filter_kernels[1]),     
            nn.ReLU(),     
            nn.MaxPool1d(3),
            nn.Conv1d(self.nb_filter,self.nb_filter,kernel_size=self.filter_kernels[2]),    
            nn.ReLU(), 
            nn.Conv1d(self.nb_filter,self.nb_filter,kernel_size=self.filter_kernels[3]),    
            nn.ReLU(), 
            nn.Conv1d(self.nb_filter,self.nb_filter,kernel_size=self.filter_kernels[4]),    
            nn.ReLU(), 
            nn.Conv1d(self.nb_filter,self.nb_filter,kernel_size=self.filter_kernels[5]),    
            nn.ReLU(), 
            nn.MaxPool1d(3),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*5,self.dense_outputs),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.dense_outputs, self.dense_outputs),
            nn.ReLU(inplace=True),
            nn.Linear(self.dense_outputs, self.cat_output),
            #nn.Softmax(), #commented because CrossEntropyLoss performs softmax
        )             



    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*5)
        x = self.classifier(x)
        return x
    