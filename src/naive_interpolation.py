#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:53:19 2019

@author: yumin
"""
import os
import numpy as np
from skimage.transform import resize

datapath = '../data/Climate/PRISM_GCMdata/0.125by0.125/test/'
#filename = 'prism_gcm_log1p_prmean_monthly_0.125to1.0_195001-200512_USA_month637.npz'
filenames = [f for f in os.listdir(datapath) if f.endswith('.npz')]
filenames = sorted(filenames)
#filenames = [filenames[0]]
#%% read in data
losses = 0
for filename in filenames:
    data = np.load(datapath+filename)
    target = np.squeeze(data['prism']) #[1,Nlat,Nlon] --> [Nlat,Nlon]
    gcms = data['gcms'] # [Ngcm,Nlat,Nlon]
    #gcms = np.mean(gcms,axis=0) #[Nlat,Nlon]
    gcms = np.transpose(gcms,axes=(1,2,0)) # [Nlat,Nlon,Ngcm]
    ## bi-linear interpolation up sampling
    interpolated = resize(gcms,target.shape,order=1,preserve_range=True) #[Nlat,Nlon,Ngcm]
    #print('interpolated.shape={}'.format(interpolated.shape))
    
    interpolated = np.expm1(interpolated) #[Nlat,Nlon,Ngcm], unit: mm/day   
    interpolated = np.mean(interpolated,axis=2) # [Nlat,Nlon]    
    losses += np.mean((interpolated-target)**2)
    
losses = losses/len(filenames) # average over all months
print('interpolated MSE: {}'.format(losses))

diff_target_interpolated = np.abs(target-interpolated)

def plot():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(np.mean(np.expm1(gcms),axis=2))
    plt.title('input GCM')
    plt.xticks([], [])
    plt.xlabel('logitude')
    plt.yticks([], [])
    plt.ylabel('latitude')
    plt.show()
    
    fig = plt.figure()
    plt.imshow(interpolated)#.cpu())
    plt.title('interpolated result')
    plt.xticks([], [])
    plt.xlabel('logitude')
    plt.yticks([], [])
    plt.ylabel('latitude')
    plt.show()
    
    fig = plt.figure()
    plt.imshow(target)
    plt.title('groundtruth (GT)')
    plt.xticks([], [])
    plt.xlabel('logitude')
    plt.yticks([], [])
    plt.ylabel('latitude')
    plt.show()
    
    fig = plt.figure()
    plt.imshow(diff_target_interpolated)
    plt.title('abs(GT-interpolated)')
    plt.xticks([], [])
    plt.xlabel('logitude')
    plt.yticks([], [])
    plt.ylabel('latitude')
    plt.show()
    plt.colorbar(fraction=0.02)
    
plot()