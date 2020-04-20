#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:28:43 2019

@author: yumin
"""

import numpy as np
from skimage.transform import resize

#%% plot pdf and CDF to validate bias corrected values
def _plot_bc_results(gcms_input,gcms_bc,obs_LR,nbins=100):
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig1, ax = plt.subplots(2)
    sns.kdeplot(gcms_input.reshape((-1,)),color='b',linewidth=2,ax=ax[0],label='GCM')
    sns.kdeplot(gcms_bc.reshape((-1,)),color='r',linewidth=2,ax=ax[0],label='GCM_bc')
    sns.kdeplot(obs_LR.reshape((-1,)),linestyle='dashed',color='k',linewidth=2,ax=ax[0],label='obs')
    ax[0].set_ylabel('PDF')
    ax[0].set_title('Bias Correction Results')
    ax[1].hist(gcms_input.reshape((-1,)),bins=nbins,density=True,histtype='step',cumulative=True,color='b',linewidth=2,label='model')  
    ax[1].hist(gcms_bc.reshape((-1,)),bins=nbins,density=True,histtype='step',cumulative=True,color='r',linewidth=2,label='model_bc')
    ax[1].hist(obs_LR.reshape((-1,)),bins=nbins,density=True,histtype='step',cumulative=True,linestyle='dashed',color='k',linewidth=2,label='obs')
    ax[1].set_ylabel('CDF')
    ax[1].set_xlabel('value')
    ax[1].legend(loc='lower right')    
    plt.show()
        
    #fig2,axs = plt.subplots(4)
    #axs[0].imshow(gcms_input[0,:,:])
    #axs[0].set_title('gcm')
    #axs[1].imshow(gcms_bc[0,:,:])
    #axs[1].set_title('gcm_bc')
    #axs[2].imshow(obs_LR[0,:,:])
    #axs[2].set_title('obs_LR')
    #axs[3].imshow(obs[0,:,:])
    #axs[3].set_title('obs')
    #for ax in axs.flat:
    #    ax.label_outer()
    plt.show()
        
#%% masked out oceans
#def mask(obs,gcms):
#    '''
#    obs: observed variables, [Ntrain,Nlat_HR,Nlon_HR]
#    gcms: model simulated variables, [Nmon,Nlat,Nlon]
#    '''
#    USAmask_HR = np.sum(obs,axis=0)
#    USAmask_LR = resize(USAmask_HR,gcms.shape[1:],order=1,preserve_range=True)
#    obs_LR = np.transpose(resize(np.transpose(obs,axes=(1,2,0)),USAmask_LR.shape,order=1,preserve_range=True),axes=(2,0,1))
#    for mon in range(len(gcms)):
#        gcms[mon,:,:][USAmask_LR==0] = 0.0
#    for mon in range(len(obs_LR)):    
#        obs_LR[mon,:,:][USAmask_LR==0] = 0.0
#    return gcms,obs,obs_LR,USAmask_HR,USAmask_LR
#gcms,obs,obs_LR,USAmask_HR,USAmask_LR = mask(obs,gcms) 
 
def mask(obs,gcms_train,gcms_test):
    '''
    obs: observed variables, [Ntrain,Nlat_HR,Nlon_HR]
    gcms_train: train model simulated variables, [Ntrain,Nlat,Nlon]
    gcms_test: test model simulated variables, [Ntest,Nlat,Nlon]
    '''
    USAmask_HR = np.sum(obs,axis=0) # [Nlat, Nlon]
    USAmask_LR = resize(USAmask_HR,gcms_train.shape[1:],order=1,preserve_range=True)
    obs_LR = np.transpose(resize(np.transpose(obs,axes=(1,2,0)),USAmask_LR.shape,order=1,preserve_range=True),axes=(2,0,1))
    for mon in range(len(gcms_train)):
        gcms_train[mon,:,:][USAmask_LR==0] = 0
    for mon in range(len(gcms_test)):
        gcms_test[mon,:,:][USAmask_LR==0] = 0
    for mon in range(len(obs_LR)):    
        obs_LR[mon,:,:][USAmask_LR==0] = 0
        
    return gcms_train,gcms_test,obs_LR,USAmask_LR,USAmask_HR    
    
    
    
#%% bias correction (BC)    
def _bc(vals,cdfobs,cdfmodel,xbins):
    """ CDF mapping for bias correction """
    """ note that values exceeding the range of the training set"""
    """ are set to -999 at the moment - possibly could leave unchanged?"""
    # calculate exact CDF values using linear interpolation
    #cdfmodely = np.interp(vals,xbins,cdfmodel,left=0.0, right=None)
    cdfmodely = np.interp(vals,xbins,cdfmodel,left=None, right=None)
    # now use interpol again to invert the obsCDF, hence reversed x,y
    #corrected = np.interp(cdfmodely,cdfobs,xbins,left=0.0,right=None)
    corrected = np.interp(cdfmodely,cdfobs,xbins,left=None,right=None)
    
    return corrected

def _cal_cdfs(obs,model,nbin=100):
    '''
    obs: observed values, 1d array, [Ntrain,]
    model: model values, 1d array, [Ntrain,]
    output: cdfs of observed and model variables, and bins, 1d arrays
    '''
    if len(model)>len(obs):
        model = model[:len(obs)] # should just use training data
    max_value = max(np.amax(obs),np.amax(model))
    #width = max_value/nbin
    #xbins = np.arange(0.0,max_value+width,width)
    
    
    min_value = min(np.amin(obs),np.amin(model))
    width = (max_value-min_value)/nbin
    xbins = np.arange(min_value,max_value+width,width)
    
    
    # create PDF
    pdfobs, _ = np.histogram(obs,bins=xbins)
    pdfmodel, _ = np.histogram(model,bins=xbins)
    # create CDF with zero in first entry.
    cdfobs = np.insert(np.cumsum(pdfobs),0,0.0)
    cdfmodel = np.insert(np.cumsum(pdfmodel),0,0.0)
    
    return cdfobs,cdfmodel,xbins    

def _bias_correction(model,cdfobs,cdfmodel,xbins):
    '''
    model: model values, 1d array, [Nmon,]
    cdfobs: cdf of (training) observed variables, 1d array, [len(xbins),]
    cdfmodels: cdf of (training) model simulated variables, 1d array, [len(xbins),]
    xbins: bin edges of cdfs, 1d array
    output: corrected_valuse: bias corrected values, 1d array
    '''
    corrected_values = np.zeros(model.shape)
    for mon in range(len(model)):
        corrected_values[mon] = _bc(model[mon],cdfobs,cdfmodel,xbins)

    return corrected_values

#def bias_correction(gcms,obs_LR,USAmask_LR,nbin=100,verbose=False):
#    '''
#    gcms: [Nmon,Nlat,Nlon]
#    obs_LR: [Ntrain,Nlat,Nlon]
#    USAmask_LR: [Nlat,Nlon]
#    '''
#    #nbin = 100 # number of bins
#    Ntrain = len(obs_LR)
#    Nmon,Nlat,Nlon = gcms.shape    
#    gcms_bc = np.zeros(gcms.shape)
#    for nlat in range(Nlat):
#        for nlon in range(Nlon):
#            if USAmask_LR[nlat,nlon]==0:
#                continue
#            obs, model = obs_LR[:,nlat,nlon], gcms[:,nlat,nlon]
#            cdfobs,cdfmodel,xbins = _cal_cdfs(obs,model[:Ntrain],nbin)
#            gcms_bc[:,nlat,nlon] = _bias_correction(model,cdfobs,cdfmodel,xbins)
#    if verbose:
#        _plot_bc_results(gcms,gcms_bc,obs_LR)
#    return gcms_bc

#gcms_bc = bias_correction(gcms,obs_LR,USAmask_LR,nbin=100,verbose=False)

def bias_correction(gcms_train,gcms_test,obs_LR,USAmask_LR,nbin=100,verbose=False):
    '''
    gcms_train: [Ntrain,Nlat,Nlon]
    gcms_test: [Ntest,Nlat,Nlon]
    obs_LR: [Ntrain,Nlat,Nlon]
    USAmask_LR: [Nlat,Nlon]
    return: gcms_bc: bias corrected gcms_test, [Ntest,Nlat,Nlon]
    '''
    #nbin = 100 # number of bins
    #Ntrain = len(obs_LR)
    #Nmon,Nlat,Nlon = gcms.shape 
    Ntest,Nlat,Nlon = gcms_test.shape
    gcms_bc = np.zeros(gcms_test.shape)
    for nlat in range(Nlat):
        for nlon in range(Nlon):
            if USAmask_LR[nlat,nlon]==0:
                continue
            obs, model_train, model_test = obs_LR[:,nlat,nlon], gcms_train[:,nlat,nlon], gcms_test[:,nlat,nlon]
            cdfobs,cdfmodel,xbins = _cal_cdfs(obs,model_train,nbin)
            gcms_bc[:,nlat,nlon] = _bias_correction(model_test,cdfobs,cdfmodel,xbins)
    if verbose:
        _plot_bc_results(gcms_test,gcms_bc,obs_LR,nbins=nbin)
        
    return gcms_bc

#%% plot pdf and CDF to validate bias corrected values
#_plot_bc_results(gcms,gcms_bc,obs_LR)

#%% Spatial Disaggregation
#%% plot SD results 
def _plot_SD_results(gcms_bc,gcms_bc_sd,climatology_obs,climatology_obs_LR,scaling_factor_LR,scaling_factor_HR):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(gcms_bc[0,:,:])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('gcms_bc[0,:,:]')
    fig = plt.figure()
    plt.imshow(gcms_bc_sd[0,:,:])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('gcms_bc_sd[0,:,:]')
    fig = plt.figure()
    plt.imshow(climatology_obs[0,:,:])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('climatology_obs[0,:,:]')
    fig = plt.figure()
    plt.imshow(climatology_obs_LR[0,:,:])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('climatology_obs_LR[0,:,:]')
    fig = plt.figure()
    plt.imshow(scaling_factor_LR[0,:,:])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('scaling_factor_LR[0,:,:]')
    fig = plt.figure()
    plt.imshow(scaling_factor_HR[0,:,:])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('scaling_factor_HR[0,:,:]')
    plt.show()
#%% cal climatology
def _cal_climatology(obs,shape_LR):
    '''
    obs: observed variables, [Ntrain,Nlat_HR,Nlon_HR]
    shape_LR: LR shape, (Nlat,Nlon)
    '''
    ####%% cal monthly mean avg (climatology)
    climatology_obs = np.zeros((12,obs.shape[1],obs.shape[2]))
    climatology_obs_LR = np.zeros((12,shape_LR[0],shape_LR[1]))
    for mon in range(12):
        climatology_obs[mon,:,:] = np.mean(obs[mon::12,:,:],axis=0)
        climatology_obs_LR[mon,:,:] = resize(climatology_obs[mon,:,:],shape_LR,order=1,preserve_range=True)
        
    return climatology_obs, climatology_obs_LR

#climatology_obs, climatology_obs_LR = _cal_climatology(obs[:Ntrain,:,:],obs_LR.shape[1:])

#%% cal scaling factor
def _cal_scaling_factor(gcms_bc,climatology_obs_LR,is_precipitation=True):
    '''
    gcms_bc: bias corrected model simulated variables, [Nmon,Nlat,Nlon], must start from January
    climatology_obs_LR: long-term average of LR observed variables, [12,Nlat,Nlon]
    '''
    #### calculate (LR) scaling factors from LR obs and model
    scaling_factor_LR = np.zeros((gcms_bc.shape))
    
    for mon in range(gcms_bc.shape[0]):
        if is_precipitation:
            anomaly = gcms_bc[mon,:,:]-climatology_obs_LR[mon%12,:,:]
            #scaling_factor_LR[mon,:,:] = np.divide(anomaly,(climatology_obs_LR[mon%12,:,:]+1.0),where=(climatology_obs_LR[mon%12,:,:]!=0))
            np.divide(anomaly,(climatology_obs_LR[mon%12,:,:]+1.0),out=scaling_factor_LR[mon,:,:],where=(climatology_obs_LR[mon%12,:,:]!=0))
        else: # temperature
            scaling_factor_LR[mon,:,:] = climatology_obs_LR[mon%12,:,:] - gcms_bc[mon,:,:]
    
    return scaling_factor_LR

#scaling_factor_LR = cal_scaling_factor(gcms_bc,climatology_obs_LR)

#%% spatial disaggregation
def _spatial_disaggregation(scaling_factor_LR,climatology_obs,is_precipitation=True):
    '''
    scaling_factor_LR: [Nmon,Nlat,Nlon]
    climatology_obs: [Nmon,Nlat_HR,Nlon_HR]
    is_precipitation: precipitation or temperature, bool
    '''
    #### interpolate scaling factor to HR shape using bi-linear interpolation
    scaling_factor_HR = np.transpose(resize(np.transpose(scaling_factor_LR,axes=(1,2,0)),climatology_obs.shape[1:],order=1,preserve_range=True),axes=(2,0,1))
    #### multiply (or add) back to climatology HR
    gcms_bc_sd = np.zeros((scaling_factor_LR.shape[0],climatology_obs.shape[1],climatology_obs.shape[2]))
    for mon in range(scaling_factor_LR.shape[0]):
        if is_precipitation:
            gcms_bc_sd[mon,:,:] = np.multiply(1.0+scaling_factor_HR[mon,:,:],climatology_obs[mon%12,:,:])
        else:
            gcms_bc_sd[mon,:,:] = scaling_factor_HR[mon,:,:]+climatology_obs[mon%12,:,:]              
           
    if is_precipitation:
        gcms_bc_sd[gcms_bc_sd<0] = 0.0
        
    return gcms_bc_sd, scaling_factor_HR

#gcms_bc_sd, scaling_factor_HR = _spatial_disaggregation(scaling_factor_LR,climatology_obs,is_precipitation=True)


def spatial_disaggregation(obs,gcms_bc,is_precipitation=True,verbose=False):
    '''
    obs: observed variable, [Ntrain,Nlat_HR,Nlon_HR]
    gcms_bc: bias corrected model simulated variables, [Nmon,Nlat,Nlon]
    '''
#    import matplotlib.pyplot as plt
    
    climatology_obs, climatology_obs_LR = _cal_climatology(obs,gcms_bc.shape[1:]) # [12,Nlat,Nlon], [12,Nlat,Nlon]
    
#    fig = plt.figure()
#    plt.imshow(np.mean(obs,axis=0))
#    plt.title('np.mean(obs)')
#    fig = plt.figure()
#    plt.imshow(np.mean(gcms_bc,axis=0))
#    plt.title('np.mean(gcms_bc)')
#    fig = plt.figure()
#    plt.imshow(np.mean(climatology_obs,axis=0))
#    plt.title('np.mean(climatology_obs)')
#    fig = plt.figure()
#    plt.imshow(np.mean(climatology_obs_LR,axis=0))
#    plt.title('np.mean(climatology_obs_LR)')
    
    
    scaling_factor_LR = _cal_scaling_factor(gcms_bc,climatology_obs_LR,is_precipitation=is_precipitation) #[Nmon,Nlat,Nlon]
    
#    fig = plt.figure()
#    plt.imshow(np.mean(scaling_factor_LR,axis=0))
#    plt.title('np.mean(scaling_factor_LR)')
    
    # [Nmon,Nlat,Nlon], [Nmon,Nlat,Nlon]
    gcms_bc_sd, scaling_factor_HR = _spatial_disaggregation(scaling_factor_LR,climatology_obs,is_precipitation=is_precipitation)

#    fig = plt.figure()
#    plt.imshow(np.mean(gcms_bc_sd,axis=0))
#    plt.title('np.mean(gcms_bc_sd)')
#    fig = plt.figure()
#    plt.imshow(np.mean(scaling_factor_HR,axis=0))
#    plt.title('np.mean(scaling_factor_HR)')

    if verbose:
        _plot_SD_results(gcms_bc,gcms_bc_sd,climatology_obs,climatology_obs_LR,scaling_factor_LR,scaling_factor_HR)

    return gcms_bc_sd

#%%
if __name__ == '__main__':
    import os
    from tqdm import tqdm
    import numpy as np
    #Ntrain = 600 # 1<Ntrain<Nmon
    nbin = 100
    verbose = True
    variable =  'ppt' # 'tmax' #'tmin' # 
    resolution = 0.5
    is_ensamble = True #False
    if is_ensamble:
        savepath = '../results/Climate/PRISM_GCM/BCSD_ensamble/{}/{}by{}/'.format(variable,resolution,resolution)
    else:
        savepath = '../results/Climate/PRISM_GCM/BCSD/{}/{}by{}/'.format(variable,resolution,resolution)
    #savepath = None
    if variable=='ppt' or variable=='pr':
        is_precipitation = True
    elif variable=='tmax' or variable=='tmin':
        is_precipitation = False
    #cpc = np.load('../data/Climate/cpcdata/processeddata/cpc_prmean_monthly_0.25by0.25/cpc_prmean_monthly_0.25by0.25_195001-200512_USA.npy')
    #gcms = np.load('../data/Climate/GCMdata/processeddata_30by75_points/18gcms_prmean_monthly_1by1_195001-200512_USA.npy') #[Ngcm,Nmon,Nlat,Nlon]
    #cpc_train = cpc[:Ntrain,:,:] #[Ntrain,Nlat,Nlon]
    #%% read in data
    #datapath = '../data/Climate/PRISM_GCMdata/{}by{}/'.format(resolution,resolution)
    datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/'.format(variable,resolution,resolution)
    train_filenames = [f for f in os.listdir(datapath+'train/') if f.endswith('.npz')]
    train_filenames = sorted(train_filenames)
    test_filenames = [f for f in os.listdir(datapath+'test/') if f.endswith('.npz')]
    test_filenames = sorted(test_filenames)
    
#    fn = 'prism_gcm_log1p_prmean_monthly_{}to1.0_195001-200512_USA_month'.format(resolution)
#    train_filenames = [] # month from 1 to 600
#    Ntrain = len(os.listdir(datapath+'train/'))
#    for month in range(1,Ntrain+1):
#        train_filenames.append(fn+str(month)+'.npz')
#    
#    test_filenames = [] # month from 1 to 600
#    Ntest = len(os.listdir(datapath+'test/'))
#    for month in range(637,Ntest+637):
#        test_filenames.append(fn+str(month)+'.npz')

    gcms_train = []
    prism_train = []
    for filename in train_filenames:
        data = np.load(datapath+'train/'+filename)
        gcms_train.append(data['gcms']) # [Ngcm,Nlat,Nlon]
        prism_train.append(np.squeeze(data['prism'])) # [1,Nlat,Nlon] --> [Nlat,Nlon]
    gcms_train = np.stack(gcms_train,axis=1) #[Ngcm,Nlat,Nlon] --> [Ngcm,Nmon,Nlat,Nlon]
    prism_train = np.stack(prism_train,axis=0) # [Nlat,Nlon] --> [Nmon,Nlat,Nlon]    
    print('gcms_train.shape={}\nprism_train.shape={}'.format(gcms_train.shape,prism_train.shape))

#    climatology_prism_train = np.zeros((12,prism_train.shape[1],prism_train.shape[2]))
#    for mon in range(12):
#        climatology_prism_train[mon,:,:] = np.mean(prism_train[mon::12,:,:],axis=0)
#    savename = '../data/Climate/PRISM_GCMdata/prism_gcm_log1p_prmean_monthly_{}by{}_195001-200512_USA_month_1to600.npy'.format(resolution,resolution)
#    np.save(savename,climatology_prism_train)    
 
    gcms_test = []
    prism_test = []
    for filename in test_filenames:
        data = np.load(datapath+'test/'+filename)
        gcms_test.append(data['gcms']) # [Ngcm,Nlat,Nlon]
        prism_test.append(np.squeeze(data['prism'])) # [1,Nlat,Nlon] --> [Nlat,Nlon] 
    gcms_test = np.stack(gcms_test,axis=1) #[Ngcm,Nlat,Nlon] --> [Ngcm,Nmon,Nlat,Nlon]
    prism_test = np.stack(prism_test,axis=0) # [Nlat,Nlon] --> [Nmon,Nlat,Nlon]   
    print('gcms_test.shape={}\nprism_test.shape={}'.format(gcms_test.shape,prism_test.shape))
    
    if variable=='pr' or variable=='ppt':
        gcms_train = np.expm1(gcms_train) # unit: mm/day
        prism_train = np.expm1(prism_train) # unit: mm/day
        gcms_test = np.expm1(gcms_test) # unit: mm/day
        prism_test = np.expm1(prism_test) # unit: mm/day
    elif variable=='tmax' or variable=='tmin':
        gcms_train = gcms_train*50.0 # unit: Celsius
        gcms_test = gcms_test*50.0 # unit: Celsius
        prism_train = prism_train*50.0 # unit: Celsius
        prism_test = prism_test*50.0 # unit: Celsius

#    climatology_prism_train = np.zeros((12,prism_train.shape[1],prism_train.shape[2]))
#    for mon in range(12):
#        climatology_prism_train[mon,:,:] = np.mean(prism_train[mon::12,:,:],axis=0)
#    savename = '../data/Climate/PRISM_GCMdata/prism_gcm_prmean_monthly_{}by{}_195001-200512_USA_month_1to600.npy'.format(resolution,resolution)
#    np.save(savename,climatology_prism_train)   

    #%% do bcsd first and then take average (ensamble)
    if is_ensamble:
        #gcms_bcsd = np.zeros((gcms.shape[0],gcms.shape[1],prism_train.shape[1],prism_train.shape[2])) #[Ngcm,Nmon,Nlat_HR,Nlon_HR]
        gcms_bcsd = np.zeros((gcms_test.shape[0],gcms_test.shape[1],prism_test.shape[1],prism_test.shape[2])) #[Ngcm,Nmon,Nlat_HR,Nlon_HR]
        for ngcm in tqdm(range(gcms_test.shape[0])):
            #gcmsn = gcms[ngcm,:,:,:] #[Nmon,Nlat,Nlon]    
            #gcmsn,prism_train,prism_train_LR,USAmask_HR,USAmask_LR,USAmask_HR = mask(prism_train,gcmsn)    
            #gcms_bc = bias_correction(gcmsn,prism_train_LR,USAmask_LR,nbin=nbin,verbose=False)
            gcmsn_train,gcmsn_test = gcms_train[ngcm,:,:,:],gcms_test[ngcm,:,:,:] #[Nmon,Nlat,Nlon]
            gcmsn_train,gcmsn_test,prism_train_LR,USAmask_LR,USAmask_HR = mask(prism_train,gcmsn_train,gcmsn_test)
            
            gcms_bc = bias_correction(gcmsn_train,gcmsn_test,prism_train_LR,USAmask_LR,nbin=nbin,verbose=False)
            #gcms_bcsd[ngcm,:,:,:] = spatial_disaggregation(prism_train,gcms_bc,is_precipitation=True,verbose=False)
            gcms_bcsd[ngcm,:,:,:] = spatial_disaggregation(prism_train,gcms_bc,is_precipitation=is_precipitation,verbose=False)
    
        #gcms_bcsd = np.expm1(gcms_bcsd) # unit: mm/day, # [Ngcm,Ntest,Nlat_HR,Nlon_HR]
        #prism_test = np.expm1(prism_test)  # unit: mm/day, [Ntest,Nlat,Nlon]    
        #gcms_bcsd_ = gcms_bcsd
        gcms_bcsd = np.mean(gcms_bcsd,axis=0) # [Ngcm,Ntest,Nlat_HR,Nlon_HR] --> [Ntest,Nlat_HR,Nlon_HR]     
    #%% take average first and then do bcsd
    else:
        gcmsn_train,gcmsn_test = np.mean(gcms_train,axis=0),np.mean(gcms_test,axis=0) #[Nmon,Nlat,Nlon]
        gcmsn_train,gcmsn_test,prism_train_LR,USAmask_LR,USAmask_HR = mask(prism_train,gcmsn_train,gcmsn_test)
        
        ##np.savez(datapath+'prism_USAmask_{}by{}.npz'.format(resolution,resolution),USAmask_HR=USAmask_HR,USAmask_LR=USAmask_LR)
              
        gcms_bc = bias_correction(gcmsn_train,gcmsn_test,prism_train_LR,USAmask_LR,nbin=nbin,verbose=True)
        #gcms_bcsd = spatial_disaggregation(prism_train,gcms_bc,is_precipitation=True,verbose=False)#[Nmon,Nlat,Nlon]
        gcms_bcsd = spatial_disaggregation(prism_train,gcms_bc,is_precipitation=is_precipitation,verbose=False)#[Nmon,Nlat,Nlon]
    
        #gcms_bcsd = np.expm1(gcms_bcsd) # unit: mm/day, # [Ngcm,Ntest,Nlat_HR,Nlon_HR]
        #prism_test = np.expm1(prism_test)  # unit: mm/day, [Ntest,Nlat,Nlon]    
    
    
#    import matplotlib.pyplot as plt
#    fig = plt.figure()
#    plt.imshow(np.mean(gcms_bc,axis=0))
#    plt.title('np.mean(gcms_bc)')
    
    #%% addjust out of boundary issue
    gcms_bcsd_masked = []
    for mon in range(gcms_bcsd.shape[0]):
        gcms_bcsd_mon = np.copy(gcms_bcsd[mon,:,:]) #[Nlat,Nlon]
        gcms_bcsd_mon[USAmask_HR==0] = 0
        gcms_bcsd_masked.append(gcms_bcsd_mon)
    
    gcms_bcsd_masked = np.stack(gcms_bcsd_masked,axis=0) #[Nmon,Nlat,Nlon]

#    import matplotlib.pyplot as plt
#    fig = plt.figure()
#    plt.imshow(USAmask_HR)
#    plt.title('USAmask_HR')
#    fig = plt.figure()
#    plt.imshow(np.mean(gcms_bcsd_masked,axis=0))
#    plt.title('np.mean(gcms_bcsd_masked)')
#    fig = plt.figure()
#    plt.imshow(np.mean(prism_test,axis=0))
#    plt.title('np.mean(prism_test)')
    
    #%%        
    ## calculate MSE
    MSE = np.mean((gcms_bcsd-prism_test)**2)
    print('BCSD MSE: {}'.format(MSE))
    
    MSE_masked = np.mean((gcms_bcsd_masked-prism_test)**2)
    print('Masked BCSD MSE: {}'.format(MSE_masked))
    
    
    diff_target_bcsd = np.abs(gcms_bcsd[-1,:,:]-prism_test[-1,:,:]) #[Nlat,Nlon]
    
    if savepath:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        np.savez(savepath+'gcms_bcsd_pred_MSE{}.npz'.format(MSE),gcms_bcsd=gcms_bcsd,MSE=MSE)
        np.save(savepath+'gcms_bcsd.npy',gcms_bcsd)
        np.save(savepath+'gcms_bcsd_MSE{}.npy'.format(MSE),MSE)
    
    if verbose:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        #plt.imshow(np.mean(np.expm1(gcms_test[:,-1,:,:]),axis=0))
        plt.imshow(np.mean(gcms_test[:,-1,:,:],axis=0))
        plt.xlabel('longitide')
        plt.xticks([], [])
        plt.ylabel('latitude')
        plt.yticks([], [])
        plt.title('input mean(gcms_test)')
        if savepath:
            plt.savefig(savepath+'input_gcm.png',dpi=1200,bbox_inches='tight')
        plt.show()
        
        fig = plt.figure()
        plt.imshow(gcms_bcsd[-1,:,:])
        plt.xlabel('longitide')
        plt.xticks([], [])
        plt.ylabel('latitude')
        plt.yticks([], [])
        plt.title('gcms_bcsd result')
        if savepath:
            plt.savefig(savepath+'pred_result.png',dpi=1200,bbox_inches='tight')
        plt.show()
        
        fig = plt.figure()
        plt.imshow(prism_test[-1,:,:])
        plt.title('groundtruth (GT)')
        plt.xticks([], [])
        plt.xlabel('logitude')
        plt.yticks([], [])
        plt.ylabel('latitude')
        if savepath:
            plt.savefig(savepath+'groundtruth.png',dpi=1200,bbox_inches='tight')
        plt.show()
        
        fig = plt.figure()
        plt.imshow(diff_target_bcsd)
        plt.title('abs(GT-pred)')
        plt.xticks([], [])
        plt.xlabel('logitude')
        plt.yticks([], [])
        plt.ylabel('latitude')
        plt.colorbar(fraction=0.02)
        if savepath:
            plt.savefig(savepath+'abs_diff.png',dpi=1200,bbox_inches='tight')
        plt.show()
        
        
        
    








