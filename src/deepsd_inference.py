#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:58:24 2019

@author: yumin
"""

#%%
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
#import collections
from skimage.transform import resize
from srcnn import StackedSRCNN
from utils import AverageMeter

variable = 'ppt' #'tmin' #'tmax'
resolution = 0.125
folder = '2020-01-08_16.06.29.813876'
epoch = 299
num_block = 3 # num of SRCNN stacked to form StackedSRCNN
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
weightspath = '../results/Climate/PRISM_GCM/DeepSD/{}/train_together/{}by{}/{}/StackedSRCNN_epoch_{}.pth'.format(variable,resolution,resolution,folder,epoch)
#srcnnpath1 = '../results/DeepSD/0.5by0.5/2019-11-21_18.56.45.018462/SRCNN_epoch_{}.pth'.format(99)
#srcnnpath2 = '../results/DeepSD/0.25by0.25/2019-11-21_21.12.19.390243/SRCNN_epoch_{}.pth'.format(99)
#srcnnpath3 = '../results/DeepSD/0.125by0.125/2019-11-21_21.25.50.778898/SRCNN_epoch_{}.pth'.format(99)
#checkpoint_paths = [srcnnpath1,srcnnpath2,srcnnpath3]
#checkpoint_paths = checkpoint_paths[:num_block]
datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/test/'.format(variable,resolution,resolution)
#filename = 'prism_gcm_log1p_prmean_monthly_0.25to1.0_195001-200512_USA_month637.npz'
#filenames = [filename]
filenames = [f for f in os.listdir(datapath) if f.endswith('.npz')]
filenames = sorted(filenames)
savepath = '/'.join(weightspath.split('/')[:-1])+'/' #'../results/DeepSD/Inference/'

#%%
model = StackedSRCNN(input_channels=2,output_channels=1,num_block=num_block)
checkpoint = torch.load(weightspath, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['model_state_dict'])
#model_state_dict = collections.OrderedDict()
#for i,cp_path in enumerate(checkpoint_paths):
#    checkpoint = torch.load(cp_path,map_location=lambda storage,loc : storage)
#    for key in checkpoint['model_state_dict']:
#        model_state_dict['SRCNN{}.'.format(i+1)+key] = checkpoint['model_state_dict'][key]
#model.load_state_dict(model_state_dict)

model = model.to(device)
model.eval()

#%% read in data
total_losses = AverageMeter()
criterion = torch.nn.MSELoss()
losses = 0
preds = []
for filename in filenames:
    data = np.load(datapath+filename)
    gcms = np.mean(data['gcms'],axis=0) #[Nlat,Nlon]
    #gcms = resize(gcms,(2*gcms.shape[0],2*gcms.shape[1]),order=1,preserve_range=True)#[Nlat,Nlon]
    elevation = data['elevation'] #[1,Nlat,Nlon]
    
    gcms = gcms/5.0 #[0.0,1.0]
    elevation = elevation/10.0 #[0.0,1.0]
    
    elevation1 = resize(elevation[0,:,:],(2*gcms.shape[0],2*gcms.shape[1]),order=1,preserve_range=True)#[Nlat,Nlon]
    elevation2 = resize(elevation[0,:,:],(4*gcms.shape[0],4*gcms.shape[1]),order=1,preserve_range=True)#[Nlat,Nlon]
    
    input1 = gcms[np.newaxis,np.newaxis,...] # [Nlat,Nlon] --> [1,1,Nlat,Nlon]
    input2 = elevation1[np.newaxis,np.newaxis,...] # [Nlat,Nlon] --> [1,1,Nlat,Nlon]
    input3 = elevation2[np.newaxis,np.newaxis,...] # [Nlat,Nlon] --> [1,1,Nlat,Nlon]
    input4 = elevation[np.newaxis,...] # [1,Nlat,Nlon] --> [1,1,Nlat,Nlon]
    inputs = [input1,input2,input3,input4]
    
    target = torch.from_numpy(data['prism']).float() #[1,Nlat,Nlon]
    
    
    
    inputs = [torch.from_numpy(e).float() for e in inputs]
    inputs = [e.to(device) for e in inputs]
    target = target.to(device)
    #print('inputs.size()=\n{}\n'.format([e.size() for e in inputs]))
    #print('target.size()={}'.format(target.size()))
    
    with torch.no_grad():
        pred = model(*inputs) # [1,1,Nlat,Nlon]
    #print('output pred.size()={}'.format(pred.size())) # [1,1,Nlat,Nlon]
    
    pred = pred*5.0
    
    pred = pred.squeeze_().expm1_() # [Nlat,Nlon] unit: mm/day
    target = target.squeeze_().expm1_() # [Nlat,Nlon] unit: mm/day
    
    #pred = pred.squeeze_()*50.0 # [Nlat,Nlon] unit: Celsius
    #target = target.squeeze_()*50.0 # [Nlat,Nlon] unit: Celsius
    
    losses += torch.mean((pred-target)**2)
    loss = criterion(pred, target)
    total_losses.update(loss.item(),1)
    preds.append(pred)

preds = torch.stack(preds,dim=0) #[Ntest,Nlat,Nlon]
losses = losses/len(filenames) # average over all months
print('downscaling MSE: {}'.format(total_losses.avg))
print('losses MSE: {}'.format(losses))
    
diff_target_pred = torch.abs(target-pred)

if savepath and not os.path.exists(savepath):
    os.makedirs(savepath)

preds = preds.cpu().numpy()
if savepath:
    np.save(savepath+'pred_results_MSE{}.npy'.format(losses),preds)



    
def plot():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    #plt.imshow(np.expm1(gcms))
    plt.imshow(gcms*50.0)
    plt.title('input GCM')
    plt.xticks([], [])
    plt.xlabel('logitude')
    plt.yticks([], [])
    plt.ylabel('latitude')
    if savepath:
        plt.savefig(savepath+'input_gcm.png',dpi=1200,bbox_inches='tight')
    plt.show()
    
    fig = plt.figure()
    plt.imshow(pred.cpu())
    plt.title('predicted result')
    plt.xticks([], [])
    plt.xlabel('logitude')
    plt.yticks([], [])
    plt.ylabel('latitude')
    if savepath:
        plt.savefig(savepath+'pred_result.png',dpi=1200,bbox_inches='tight')
    plt.show()
    
    fig = plt.figure()
    plt.imshow(target.cpu())
    plt.title('groundtruth (GT)')
    plt.xticks([], [])
    plt.xlabel('logitude')
    plt.yticks([], [])
    plt.ylabel('latitude')
    if savepath:
        plt.savefig(savepath+'groundtruth.png',dpi=1200,bbox_inches='tight')
    plt.show()
    
    fig = plt.figure()
    plt.imshow(diff_target_pred.cpu())
    plt.title('abs(GT-pred)')
    plt.xticks([], [])
    plt.xlabel('logitude')
    plt.yticks([], [])
    plt.ylabel('latitude')
    if savepath:
        plt.savefig(savepath+'abs_diff.png',dpi=1200,bbox_inches='tight')
    plt.show()
    plt.colorbar(fraction=0.02)
#%%
def plot2():
    #### plot figures
    import matplotlib.pyplot as plt 
    fig,axs = plt.subplots(3,2)
    axs[0,0].imshow(target[0,:,:]/target[0,:,:].max())
    axs[0,0].set_title('groundtruth')
    axs[1,0].imshow(pred/pred.max())
    axs[1,0].set_title('pred: predicted result')
    #plt.show()
    axs[2,0].imshow(gcms/gcms.max())
    axs[2,0].set_title('base: average of bi-linear interpolation')
    
    diff_target_pred = torch.abs(target[0,:,:]-pred)
    diff_target_base = torch.abs(target[0,:,:]-gcms)
    diff1 = axs[1,1].imshow(diff_target_pred/diff_target_pred.max())
    axs[1,1].set_title('abs(GT-pred)')
    diff2 = axs[2,1].imshow(diff_target_base/diff_target_base.max())
    axs[2,1].set_title('abs(GT-base)')
    
    fig.colorbar(diff1,ax=axs[1,1])
    fig.colorbar(diff2,ax=axs[2,1])
    
    ## hide x labels and  tick labels for top plots and y ticks for right plots
    for ax in axs.flat:
        ax.label_outer()
    plt.savefig(savepath+'pred_vs_base_vs_GT.png',dpi=1200,bbox_inches='tight')
    plt.show()
    
    # fig, axs = plt.subplots(6,3)
    # for i in range(6):
    #     for j in range(3):
    #         axs[i,j].imshow(gcms[:,:,i*3+j])
    #         axs[i,j].set_title('{}'.format(i*3+j+1))
    # ## hide x labels and  tick labels for top plots and y ticks for right plots
    # for ax in axs.flat:
    #     ax.label_outer()
    #plt.show()

plot()