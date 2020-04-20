#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:17:19 2019

@author: yumin
"""

import os
imgpaths = []

prefix = '/home/yumin/Desktop/DS/'
#prefix = '/export/thishost/yuminliu/DS/'
root_path = prefix + 'ImageNetData/ILSVRC/Data/CLS-LOC/test/'
for i, (root,dirs,files) in enumerate(os.walk(root_path)):
    #if i>=6: break
    for file in files:
        if file.endswith('.JPEG'):
            imgpaths.append([root,file])

#%%
from skimage.io import imread,imsave
from tqdm import tqdm
#import matplotlib.pyplot as plt
patch_size = (256,256)
savepath = prefix + 'ImageNetData_Processed/test/'
n = 0
folder = 1
for i, (path,filename) in enumerate(tqdm(imgpaths)):
    img = imread(path+'/'+filename)
    if len(img.shape)<3: 
        continue
    H,W,C = img.shape
    if H<patch_size[0] or W<patch_size[1]:
        continue
    #crop_h = np.random.randint(0,H-patch_size[0])
    #crop_w = np.random.randint(0,W-patch_size[1])
    crop_h = (H-patch_size[0])//2
    crop_w = (W-patch_size[1])//2
    img = img[crop_h:crop_h+patch_size[0],crop_w:crop_w+patch_size[1],:]   
    if not os.path.exists(savepath+str(folder)):
        os.makedirs(savepath+str(folder))
    #savename = filename.replace('.JPEG','.PNG')
    #imsave(savepath+str(folder)+'/'+filename.replace('.JPEG','.PNG'),img)
    imsave(savepath+str(folder)+'/'+filename,img)
    n += 1
    if n>=5000:
        n = 0
        folder += 1
    
    #if i>300:
    #    break
    
