#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:34:32 2019

@author: yumin
"""
import math
import torch
from torch import nn
#from fastai.layers import PixelShuffle_ICNR as PixelShuffle



torch.autograd.set_detect_anomaly(True)
class YNet30_test(nn.Module):
    def __init__(self, num_layers=15, num_features=64,input_channels=1,output_channels=1,scale=4,use_climatology=False):
        super(YNet30_test, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.scale = scale
        self.use_climatology = use_climatology

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, self.num_features, kernel_size=3, stride=1, padding=0),
                                         nn.ReLU(inplace=True)))
        for i in range(self.num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=0),
                                             nn.ReLU(inplace=True)))

        for i in range(self.num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, padding=1,output_padding=1),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(self.num_features,self.num_features,kernel_size=3,padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, output_padding=1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(self.num_features,self.input_channels,kernel_size=3,stride=1,padding=1)))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

        self.subpixel_conv_layer = nn.Sequential(nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True),
                                                 nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False),
                                                 nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True))
        #self.upsample = nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False)
        if self.use_climatology:
            self.fusion_layer = nn.Sequential(nn.Conv2d(2*self.input_channels+2,self.num_features,kernel_size=3,stride=1,padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.num_features,self.output_channels,kernel_size=1,stride=1,padding=0))#,
                                              #nn.ReLU(inplace=True))

    def forward(self, x, x2=None, x3=None):
        residual = x
        #residual_up = nn.functional.interpolate(residual,scale_factor=self.scale,mode='bilinear',align_corners=False)
        #residual_up = self.upsample(x)
        
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)
        #print('after conv: x.size()={}\n'.format(x.size()))
        
        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)
        #print('torch.sum(x)={}'.format(torch.sum(x)))
        #print('after convtrans: x.size()={},residual.size()={}'.format(x.size(),residual.size()))
        #x += residual
        x = x+residual
        x = self.relu(x)
        #print('before subpixel conv: x.size()={}\n'.format(x.size()))
        x = self.subpixel_conv_layer(x)
        #print('after subpixel conv: x.size()={}\n'.format(x.size()))
        #x = x+residual_up
        #x = x+x3
        x = torch.cat([x,x3],dim=1)
        #print('x2.size()={}\n'.format(x2.size()))
               
        if self.use_climatology and (x2 is not None):
            x = self.fusion_layer(torch.cat([x,x2],dim=1)) # [Nbatch,Nchannel,Nlat,Nlon]
            #print('using fusion')
        
        return x



















#%% Version 3
torch.autograd.set_detect_anomaly(True)
class YNet30(nn.Module):
    def __init__(self, num_layers=15, num_features=64,input_channels=1,output_channels=1,scale=4,use_climatology=False):
        super(YNet30, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.scale = scale
        self.use_climatology = use_climatology

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, self.num_features, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(self.num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(self.num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(self.num_features,self.num_features,kernel_size=3,padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, output_padding=0),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(self.num_features,self.input_channels,kernel_size=3,stride=1,padding=1)))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

        self.subpixel_conv_layer = nn.Sequential(nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True),
                                                 nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False),
                                                 nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True))
        #self.upsample = nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False)
        if self.use_climatology:
            self.fusion_layer = nn.Sequential(nn.Conv2d(2*self.input_channels+2,self.num_features,kernel_size=3,stride=1,padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.num_features,self.output_channels,kernel_size=1,stride=1,padding=0))#,
                                              #nn.ReLU(inplace=True))

    def forward(self, x, x2=None, x3=None):
        residual = x
        #residual_up = nn.functional.interpolate(residual,scale_factor=self.scale,mode='bilinear',align_corners=False)
        #residual_up = self.upsample(x)
        
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)
        #print('after conv: x.size()={}\n'.format(x.size()))
        
        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)
        #print('torch.sum(x)={}'.format(torch.sum(x)))
        #print('after convtrans: x.size()={},residual.size()={}'.format(x.size(),residual.size()))
        #x += residual
        x = x+residual
        x = self.relu(x)
        #print('before subpixel conv: x.size()={}\n'.format(x.size()))
        x = self.subpixel_conv_layer(x)
        #print('after subpixel conv: x.size()={}\n'.format(x.size()))
        #x = x+residual_up
        #x = x+x3
        x = torch.cat([x,x3],dim=1)
        #print('x2.size()={}\n'.format(x2.size()))
               
        if self.use_climatology and (x2 is not None):
            x = self.fusion_layer(torch.cat([x,x2],dim=1)) # [Nbatch,Nchannel,Nlat,Nlon]
            #print('using fusion')
        
        return x

#%% Version 2,
# =============================================================================
# class YNet30(nn.Module):
#     def __init__(self, num_layers=15, num_features=64,input_channels=1,output_channels=1,scale=4,use_climatology=False):
#         super(YNet30, self).__init__()
#         self.num_layers = num_layers
#         self.num_features = num_features
#         self.input_channels = input_channels
#         self.output_channels = output_channels
#         self.scale = scale
#         self.use_climatology = use_climatology
# 
#         conv_layers = []
#         deconv_layers = []
# 
#         conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, self.num_features, kernel_size=3, stride=1, padding=1),
#                                          nn.ReLU(inplace=True)))
#         for i in range(self.num_layers - 1):
#             conv_layers.append(nn.Sequential(nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1),
#                                              nn.ReLU(inplace=True)))
# 
#         for i in range(self.num_layers - 1):
#             deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, padding=1),
#                                                nn.ReLU(inplace=True),
#                                                nn.Conv2d(self.num_features,self.num_features,kernel_size=3,padding=1),
#                                                nn.ReLU(inplace=True)))
#         deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, output_padding=0),
#                                            nn.ReLU(inplace=True),
#                                            nn.Conv2d(self.num_features,self.input_channels,kernel_size=3,stride=1,padding=1)))
# 
#         self.conv_layers = nn.Sequential(*conv_layers)
#         self.deconv_layers = nn.Sequential(*deconv_layers)
#         self.relu = nn.ReLU(inplace=True)
# 
#         self.subpixel_conv_layer = nn.Sequential(nn.Conv2d(self.input_channels,self.output_channels*(self.scale**2),kernel_size=3,stride=1,padding=1),
#                                                  nn.ReLU(inplace=True),
#                                                  #nn.PixelShuffle(self.scale),
#                                                  #PixelShuffle(ni=self.output_channels*(self.scale**2),nf=self.output_channels,scale=self.scale,blur=True),
#                                                  #nn.ReLU(inplace=True),
#                                                  nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False),
#                                                  nn.Conv2d(self.output_channels*(self.scale**2),self.output_channels,kernel_size=3,stride=1,padding=1),
#                                                  nn.ReLU(inplace=True))
#         if self.use_climatology:
#             self.fusion_layer = nn.Sequential(nn.Conv2d(self.output_channels+2,self.output_channels,kernel_size=3,stride=1,padding=1),
#                                               nn.ReLU(inplace=True))
# 
#     def forward(self, x, x2=None):
#         residual = x
# 
#         conv_feats = []
#         for i in range(self.num_layers):
#             x = self.conv_layers[i](x)
#             if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
#                 conv_feats.append(x)
#         #print('after conv: x.size()={}\n'.format(x.size()))
#         conv_feats_idx = 0
#         for i in range(self.num_layers):
#             x = self.deconv_layers[i](x)
#             if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
#                 conv_feat = conv_feats[-(conv_feats_idx + 1)]
#                 conv_feats_idx += 1
#                 x = x + conv_feat
#                 x = self.relu(x)
# 
#         #print('after convtrans: x.size()={},residual.size()={}'.format(x.size(),residual.size()))
#         x += residual
#         x = self.relu(x)
#         #print('before subpixel conv: x.size()={}\n'.format(x.size()))
#         x = self.subpixel_conv_layer(x)
#         #print('after subpixel conv: x.size()={}\n'.format(x.size()))
#         
#         #print('x2.size()={}\n'.format(x2.size()))
#                
#         if self.use_climatology and (x2 is not None):
#             x = self.fusion_layer(torch.cat([x,x2],dim=1)) # [Nbatch,Nchannel,Nlat,Nlon]
#             #print('using fusion')
#         
#         return x
# 
# =============================================================================


#import torch
#class MeanLayer(torch.nn.Module):
#    def __init__(self, dim,keepdim=False,out=None):
#        """
#        In the constructor we instantiate two nn.Linear modules and assign them as
#        member variables.
#        """
#        super(MeanLayer, self).__init__()
#        self.dim = dim
#        self.keepdim = keepdim
#        self.out = out
#
#    def forward(self, x):
#        """
#        In the forward function we accept a Tensor of input data and we must return
#        a Tensor of output data. We can use Modules defined in the constructor as
#        well as arbitrary operators on Tensors.
#        """
#        x = torch.mean(x,dim=self.dim,keepdim=self.keepdim,out=self.out)
#        
#        return x



#%% Version 1, no checkerboard effect
#class YNet30(nn.Module):
#    def __init__(self, num_layers=15, num_features=64,input_channels=1,output_channels=1,scale=4):
#        super(YNet30, self).__init__()
#        self.num_layers = num_layers
#        self.num_features = num_features
#        self.input_channels = input_channels
#        self.output_channels = output_channels
#        self.scale = scale
#
#        conv_layers = []
#        deconv_layers = []
#
#        conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, self.num_features, kernel_size=3, stride=1, padding=1),
#                                         nn.ReLU(inplace=True)))
#        for i in range(self.num_layers - 1):
#            conv_layers.append(nn.Sequential(nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1),
#                                             nn.ReLU(inplace=True)))
#
#        for i in range(self.num_layers - 1):
#            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, padding=1),
#                                               nn.ReLU(inplace=True),
#                                               nn.Conv2d(self.num_features,self.num_features,kernel_size=3,padding=1),
#                                               nn.ReLU(inplace=True)))
#        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, output_padding=0),
#                                           nn.ReLU(inplace=True),
#                                           nn.Conv2d(self.num_features,self.input_channels,kernel_size=3,stride=1,padding=1)))
#
#        self.conv_layers = nn.Sequential(*conv_layers)
#        self.deconv_layers = nn.Sequential(*deconv_layers)
#        self.relu = nn.ReLU(inplace=True)
#
#        self.subpixel_conv_layer = nn.Sequential(nn.Conv2d(self.input_channels,self.output_channels*(self.scale**2),kernel_size=3,stride=1,padding=1),
#                                                 nn.ReLU(inplace=True),
#                                                 #nn.PixelShuffle(self.scale),
#                                                 #PixelShuffle(ni=self.output_channels*(self.scale**2),nf=self.output_channels,scale=self.scale,blur=True),
#                                                 #nn.ReLU(inplace=True),
#                                                 nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False),
#                                                 nn.Conv2d(self.output_channels*(self.scale**2),self.output_channels,kernel_size=3,stride=1,padding=1),
#                                                 nn.ReLU(inplace=True))
#
#    def forward(self, x):
#        residual = x
#
#        conv_feats = []
#        for i in range(self.num_layers):
#            x = self.conv_layers[i](x)
#            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
#                conv_feats.append(x)
#        #print('after conv: x.size()={}\n'.format(x.size()))
#        conv_feats_idx = 0
#        for i in range(self.num_layers):
#            x = self.deconv_layers[i](x)
#            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
#                conv_feat = conv_feats[-(conv_feats_idx + 1)]
#                conv_feats_idx += 1
#                x = x + conv_feat
#                x = self.relu(x)
#
#        #print('after convtrans: x.size()={},residual.size()={}'.format(x.size(),residual.size()))
#        x += residual
#        x = self.relu(x)
#        #print('before subpixel conv: x.size()={}\n'.format(x.size()))
#        x = self.subpixel_conv_layer(x)
#        #print('after subpixel conv: x.size()={}\n'.format(x.size()))
#        return x
        

#%% Version 0, convTransose2d results in chessboard effect    
#class YNet30(nn.Module):
#    def __init__(self, num_layers=15, num_features=64,input_channels=1,output_channels=1,scale=4):
#        super(YNet30, self).__init__()
#        self.num_layers = num_layers
#        self.num_features = num_features
#        self.input_channels = input_channels
#        self.output_channels = output_channels
#        self.scale = scale
#
#        conv_layers = []
#        deconv_layers = []
#
#        conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, self.num_features, kernel_size=3, stride=1, padding=1),
#                                         nn.ReLU(inplace=True)))
#        for i in range(self.num_layers - 1):
#            conv_layers.append(nn.Sequential(nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1),
#                                             nn.ReLU(inplace=True)))
#
#        for i in range(self.num_layers - 1):
#            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, padding=1),
#                                               nn.ReLU(inplace=True),
#                                               nn.Conv2d(self.num_features,self.num_features,kernel_size=3,padding=1),
#                                               nn.ReLU(inplace=True)))
#        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, output_padding=0),
#                                           nn.ReLU(inplace=True),
#                                           nn.Conv2d(self.num_features,self.input_channels,kernel_size=3,stride=1,padding=1)))
#
#        self.conv_layers = nn.Sequential(*conv_layers)
#        self.deconv_layers = nn.Sequential(*deconv_layers)
#        self.relu = nn.ReLU(inplace=True)
#
#        self.subpixel_conv_layer = nn.Sequential(nn.Conv2d(self.input_channels,self.output_channels*(self.scale**2),kernel_size=3,stride=1,padding=1),
#                                                 #nn.PixelShuffle(self.scale),
#                                                 nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False),
#                                                 nn.Conv2d(self.output_channels*(self.scale**2),self.output_channels,kernel_size=3,stride=1,padding=1),
#                                                 nn.ReLU(inplace=True))
#
#    def forward(self, x):
#        residual = x
#
#        conv_feats = []
#        for i in range(self.num_layers):
#            x = self.conv_layers[i](x)
#            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
#                conv_feats.append(x)
#        #print('after conv: x.size()={}\n'.format(x.size()))
#        conv_feats_idx = 0
#        for i in range(self.num_layers):
#            x = self.deconv_layers[i](x)
#            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
#                conv_feat = conv_feats[-(conv_feats_idx + 1)]
#                conv_feats_idx += 1
#                x = x + conv_feat
#                x = self.relu(x)
#
#        #print('after convtrans: x.size()={},residual.size()={}'.format(x.size(),residual.size()))
#        x += residual
#        x = self.relu(x)
#        #print('before subpixel conv: x.size()={}\n'.format(x.size()))
#        x = self.subpixel_conv_layer(x)
#        #print('after subpixel conv: x.size()={}\n'.format(x.size()))
#        return x





#%% ESPCN 
# from https://github.com/leftthomas/ESPCN
import torch
#import torch.nn.functional as F
class ESPCNNet(nn.Module):
    def __init__(self, upscale_factor,input_channels=1,output_channels=1):
        super(ESPCNNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv1 = nn.Conv2d(self.input_channels, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, self.output_channels * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        #x = F.tanh(self.conv1(x))
        #x = F.tanh(self.conv2(x))
        #x = F.sigmoid(self.pixel_shuffle(self.conv3(x)))
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x
    
   
#%% original REDNet30
class REDNet30(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, num_layers=15, num_features=64):
        super(REDNet30, self).__init__()
        self.num_layers = num_layers
        self.input_channels = input_channels
        self.output_channels = output_channels

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, num_features, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features, self.output_channels, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        x += residual
        x = self.relu(x)

        return x    
    