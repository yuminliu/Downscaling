#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:46:46 2019

@author: yumin
"""
"""
implement SRCNN
"""
import torch
from torch import nn

#class SRCNN_BLOCK(nn.Module):
#    def __init__(self, input_channels=1,output_channels=1):
#        super(SRCNN_BLOCK, self).__init__()
#
#        layer1 = nn.Sequential(nn.Conv2d(in_channels=input_channels,out_channels=64,kernel_size=(9,9),stride=1,padding=4),
#                               nn.ReLU(inplace=True))
#        layer2 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(1,1),stride=1,padding=0),
#                               nn.ReLU(inplace=True))
#        layer3 = nn.Conv2d(in_channels=32,out_channels=output_channels,kernel_size=(5,5),stride=1,padding=2)
#        
#        self.block = nn.Sequential(layer1,layer2,layer3)
#        
#    def get_block(self):
#        
#        return self.block

def srcnn_block(input_channels=1,output_channels=1):
    
    layer1 = nn.Sequential(nn.Conv2d(in_channels=input_channels,out_channels=64,kernel_size=(9,9),stride=1,padding=4),
                               nn.ReLU(inplace=True))
    layer2 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(1,1),stride=1,padding=0),
                               nn.ReLU(inplace=True))
    layer3 = nn.Conv2d(in_channels=32,out_channels=output_channels,kernel_size=(5,5),stride=1,padding=2)
    
    block = nn.Sequential(layer1,layer2,layer3)
    
    return block
    
    
    


class SRCNN(nn.Module):
    def __init__(self, input_channels=2,output_channels=1):
        super(SRCNN, self).__init__()

        layer1 = nn.Sequential(nn.Conv2d(in_channels=input_channels,out_channels=64,kernel_size=(9,9),stride=1,padding=4),
                               nn.ReLU(inplace=True))
        layer2 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(1,1),stride=1,padding=0),
                               nn.ReLU(inplace=True))
        layer3 = nn.Conv2d(in_channels=32,out_channels=output_channels,kernel_size=(5,5),stride=1,padding=2)
        
        self.layers = nn.Sequential(layer1,layer2,layer3)

    def forward(self, x):

        return self.layers(x)
    
    
class StackedSRCNN(nn.Module):
    def __init__(self, input_channels=2,output_channels=1,num_block=3,num_feature=2,base_scale_factor=2):
        super(StackedSRCNN, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_block = num_block # how many SRCNNs stacked together
        self.num_feature = num_feature
        self.base_scale_factor = base_scale_factor
        
        #srcnn_layers = []
        self.upsample1 = nn.Upsample(scale_factor=self.base_scale_factor,mode='bilinear',align_corners=False)
        self.SRCNN1 = SRCNN(input_channels=self.input_channels,output_channels=self.num_feature-1)
        #srcnn_layers.append(self.upsample1,self.SRCNN1)
        if self.num_block>=2:
            self.upsample2 = nn.Upsample(scale_factor=self.base_scale_factor,mode='bilinear',align_corners=False)
            self.SRCNN2 = SRCNN(input_channels=self.num_feature,output_channels=self.num_feature-1)
            #srcnn_layers.append(nn.Sequential(self.upsample2,self.SRCNN2))
        if self.num_block==3:
            self.upsample3 = nn.Upsample(scale_factor=self.base_scale_factor,mode='bilinear',align_corners=False)
            self.SRCNN3 = SRCNN(input_channels=self.num_feature,output_channels=self.output_channels)
            #srcnn_layers.append(nn.Sequential(self.upsample3,self.SRCNN3))

    def forward(self, x, x2, x3=None, x4=None):
        #x1, x2, x3, x4 = x[0], x[1], x[2], x[3] # precipitation and elevation, respectively
        ## x1: [num_batch, C,H,W] = [num_batch, 1,H,W]
        
        #print('block0: x.size()={}'.format(x.size()))
        x = self.upsample1(x)
        x = self.SRCNN1(torch.cat((x,x2),dim=1)) ##[num_batch, 2,H,W]
        
        #print('block1: x.size()={}'.format(x.size()))
               
        if self.num_block>=2 and x3 is not None:
            x = self.upsample2(x)
            x = self.SRCNN2(torch.cat((x,x3),dim=1))
        
        #print('block2: x.size()={}'.format(x.size()))
        
        if self.num_block==3 and x4 is not None:
            x = self.upsample3(x)
            x = self.SRCNN3(torch.cat((x,x4),dim=1))
        
        #print('block3: x.size()={}'.format(x.size()))
        
        return x       
    
#model = StackedSRCNN()
#print('model={}\n'.format(model))    
#model_state_dict = model.state_dict()
    
    
    
#%% StackedSRCNN Version 0    
#class StackedSRCNN(nn.Module):
#    def __init__(self, input_channels=2,output_channels=1,num_block=3):
#        super(StackedSRCNN, self).__init__()
#        self.num_block = num_block # how many SRCNNs stacked together
#        
#        self.block1 = srcnn_block(input_channels=input_channels,output_channels=1)
#        self.block2 = None
#        self.block3 = None
#        if self.num_block>=2:
#            self.block2 = srcnn_block(input_channels=2,output_channels=1)
#        if self.num_block==3:
#            self.block3 = srcnn_block(input_channels=2,output_channels=output_channels)
#
#    def forward(self, x, x2, x3=None, x4=None):
#        #x1, x2, x3, x4 = x[0], x[1], x[2], x[3] # precipitation and elevation, respectively
#        ## x1: [num_batch, C,H,W] = [num_batch, 1,H,W]
#        
#        #print('block0: x.size()={}'.format(x.size()))
#        
#        x = self.block1(torch.cat((x,x2),dim=1)) ##[num_batch, 2,H,W]
#        
#        #print('block1: x.size()={}'.format(x.size()))
#        
#        
#        if self.num_block>=2:
#            x = nn.functional.interpolate(x,size=x3.size()[2:],mode='bilinear',align_corners=False)
#            x = self.block2(torch.cat((x,x3),dim=1))
#        
#        #print('block2: x.size()={}'.format(x.size()))
#        
#        if self.num_block==3:
#            x = nn.functional.interpolate(x,size=x4.size()[2:],mode='bilinear',align_corners=False)
#            x = self.block3(torch.cat((x,x4),dim=1))
#        
#        #print('block3: x.size()={}'.format(x.size()))
#        
#        return x   