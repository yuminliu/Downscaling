#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:52:52 2020

@author: yumin
"""



def plot_elevation():
    import numpy as np
    from plots import plot_map
    elevation = np.load('/home/yumin/DS/DATA/PRISM/PRISMdata/elevation/processeddata/prism_elevation_0.125by0.125_USA.npy')
    data = np.load('../data/Climate/PRISM_GCMdata/ppt/0.125by0.125/prism_USAmask_0.125by0.125.npz')
    USAmask_HR = data['USAmask_HR']
    elevation[USAmask_HR==0] = np.nan
    img = np.flipud(elevation)
    title = 'Elevation of CONUS'
    savepath = '../results/Climate/PRISM_GCM/Figures/' #None
    savename = 'elevation_of_CONUS_0.125by0.125'
    plot_map(img,title=title,savepath=savepath,savename=savename,cmap='Reds',clim=None)


def _plot_pdf_temperature(trainpath):
    import os
    import numpy as np
    from skimage.transform import resize
    #import seaborn as sns
    #trainpath = '../data/Climate/PRISM_GCMdata/tmax/0.125by0.125/train/'
    trainnames = [f for f in os.listdir(trainpath) if f.endswith('.npz')]
    trainnames = sorted(trainnames)
    prism_train = []
    gcm_train = []
    for name in trainnames:
        data = np.load(trainpath+name)
        prism_train.append(np.squeeze(data['prism']))
        gcm_train.append(data['gcms'])
    prism_train = np.stack(prism_train,axis=0)
    gcm_train = np.stack(gcm_train,axis=1)
    prism_train_ori = prism_train*50.0
    gcm_train_ori = gcm_train*50.0
    USAmask_HR = np.sum(prism_train,axis=0)
    USAmask_LR = resize(USAmask_HR,gcm_train.shape[2:],order=1,preserve_range=True)
    
    #USAmask_HR[USAmask_HR<1.0] = 0
    #USAmask_LR[USAmask_LR<1.0] = 0
    
    for i in range(len(prism_train)):
        prism_train[i,:,:][USAmask_HR==0] = np.nan
        prism_train_ori[i,:,:][USAmask_HR==0] = np.nan
        for j in range(len(gcm_train)):
            gcm_train[j,i,:,:][USAmask_LR==0] = np.nan
            gcm_train_ori[j,i,:,:][USAmask_LR==0] = np.nan
    
    prism_train_ori_flat = prism_train_ori.flatten()
    prism_train_flat = prism_train.flatten()
    gcm_train_ori_flat = gcm_train_ori.flatten()
    gcm_train_flat = gcm_train.flatten()
    
    prism_train_ori_flat = prism_train_ori_flat[~np.isnan(prism_train_ori_flat)]
    prism_train_flat = prism_train_flat[~np.isnan(prism_train_flat)]
    gcm_train_ori_flat = gcm_train_ori_flat[~np.isnan(gcm_train_ori_flat)]
    gcm_train_flat = gcm_train_flat[~np.isnan(gcm_train_flat)]
        
    return prism_train_ori_flat,gcm_train_ori_flat
    
    
    
    
    
    #%%  
def plot_pdf_temperature():
    import matplotlib.pyplot as plt
    
    trainpath_tmax = '../data/Climate/PRISM_GCMdata/tmax/0.125by0.125/train/'
    trainpath_tmin = '../data/Climate/PRISM_GCMdata/tmin/0.125by0.125/train/'
    prism_train_ori_flat_tmax,gcm_train_ori_flat_tmax = _plot_pdf_temperature(trainpath_tmax)
    prism_train_ori_flat_tmin,gcm_train_ori_flat_tmin = _plot_pdf_temperature(trainpath_tmin)

    bins = 1000
    #fig = plt.figure()
            
    plt.subplot(211)
    plt.hist(prism_train_ori_flat_tmax,bins=bins,density=True,facecolor='r',alpha=0.8)
    plt.hist(prism_train_ori_flat_tmin,bins=bins,density=True,facecolor='b',alpha=0.6)
    plt.legend(['tmax','tmin'])
    plt.xlim(-30,50)
    plt.ylim(0,0.08)
    plt.title('PRISM temperature',fontsize=5)
    plt.ylabel('Density')
    plt.grid(True)
    #plt.show()
    
    plt.subplot(212)
    plt.hist(gcm_train_ori_flat_tmax,bins=bins,density=True,facecolor='r',alpha=0.8)
    plt.hist(gcm_train_ori_flat_tmin,bins=bins,density=True,facecolor='b',alpha=0.6)
    plt.legend(['tmax','tmin'])
    plt.xlim(-30,50)
    plt.ylim(0,0.08)
    plt.title('GCM temperature',fontsize=5)
    plt.ylabel('Density')
    plt.grid(True)
    
    savepath = '../data/Climate/PRISM_GCMdata/'
    savename = 'prism_gcm_pdf_train_temperature'
    plt.savefig(savepath+savename+'.jpg',dpi=1200,bbox_inches='tight')
    plt.show()
    
    #sns.distplot(gcm_train_ori_flat,bins=bins,hist=False,kde=True,color='red')
    ##plt.subplot(212)
    #sns.distplot(gcm_train_flat,bins=bins,hist=False,kde=True,color='blue')
    #plt.legend(['original','after log1p transform'])
    #plt.show()

#plot_pdf_temperature()

def plot_pdf_precipitation():
    import os
    import numpy as np
    from skimage.transform import resize
    import seaborn as sns
    is_precipitation = True
    if is_precipitation:
        trainpath = '../data/Climate/PRISM_GCMdata/ppt/0.125by0.125/train/'
    else:
        trainpath = '../data/Climate/PRISM_GCMdata/tmax/0.125by0.125/train/'
    trainnames = [f for f in os.listdir(trainpath) if f.endswith('.npz')]
    trainnames = sorted(trainnames)
    prism_train = []
    gcm_train = []
    for name in trainnames:
        data = np.load(trainpath+name)
        prism_train.append(np.squeeze(data['prism']))
        gcm_train.append(data['gcms'])
    prism_train = np.stack(prism_train,axis=0)
    gcm_train = np.stack(gcm_train,axis=1)
    if is_precipitation:
        prism_train_ori = np.expm1(prism_train)
        gcm_train_ori = np.expm1(gcm_train)
    else:
        prism_train_ori = prism_train*50.0
        gcm_train_ori = gcm_train*50.0
    USAmask_HR = np.sum(prism_train,axis=0)
    USAmask_LR = resize(USAmask_HR,gcm_train.shape[2:],order=1,preserve_range=True)
    
    #USAmask_HR[USAmask_HR<1.0] = 0
    #USAmask_LR[USAmask_LR<1.0] = 0
    
    for i in range(len(prism_train)):
        prism_train[i,:,:][USAmask_HR==0] = np.nan
        prism_train_ori[i,:,:][USAmask_HR==0] = np.nan
        for j in range(len(gcm_train)):
            gcm_train[j,i,:,:][USAmask_LR==0] = np.nan
            gcm_train_ori[j,i,:,:][USAmask_LR==0] = np.nan
    
    prism_train_ori_flat = prism_train_ori.flatten()
    prism_train_flat = prism_train.flatten()
    gcm_train_ori_flat = gcm_train_ori.flatten()
    gcm_train_flat = gcm_train.flatten()
    
    prism_train_ori_flat = prism_train_ori_flat[~np.isnan(prism_train_ori_flat)]
    prism_train_flat = prism_train_flat[~np.isnan(prism_train_flat)]
    gcm_train_ori_flat = gcm_train_ori_flat[~np.isnan(gcm_train_ori_flat)]
    gcm_train_flat = gcm_train_flat[~np.isnan(gcm_train_flat)]
    
    
    prism_train_ori_flat = prism_train_ori_flat[prism_train_ori_flat>0.0]
    prism_train_flat = prism_train_flat[prism_train_flat>0.0]
    gcm_train_ori_flat = gcm_train_ori_flat[gcm_train_ori_flat>0.0]
    gcm_train_flat = gcm_train_flat[gcm_train_flat>0.0]
    
    
    #%%    
    import matplotlib.pyplot as plt
    if is_precipitation:
        #fig = plt.subplots(nrows=2,ncols=1)
        #bins = 500
        bins = 1000
        fig = plt.figure()
                
        plt.subplot(211)
        plt.hist(prism_train_ori_flat,bins=bins,density=True,facecolor='r',alpha=0.8)
        plt.hist(prism_train_flat,bins=bins,density=True,facecolor='b',alpha=0.6)
        plt.legend(['original','after log1p transform'])
        plt.xlim(-1,12)
        plt.ylim(0,0.8)
        plt.title('PRISM ppt',fontsize=5)
        plt.ylabel('Density')
        plt.grid(True)
        #plt.show()
        
        plt.subplot(212)
        plt.hist(gcm_train_ori_flat,bins=bins,density=True,facecolor='r',alpha=0.8)
        plt.hist(gcm_train_flat,bins=bins,density=True,facecolor='b',alpha=0.6)
        plt.legend(['original','after log1p transform'])
        plt.xlim(-1,12)
        plt.ylim(0,0.8)
        plt.title('GCM ppt',fontsize=5)
        plt.ylabel('Density')
        plt.grid(True)
        savepath = '../data/Climate/PRISM_GCMdata/ppt/'
        savename = 'prism_gcm_pdf_train_ppt'
        #plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
        plt.show()
        
        #sns.distplot(gcm_train_ori_flat,bins=bins,hist=False,kde=True,color='red')
        ##plt.subplot(212)
        #sns.distplot(gcm_train_flat,bins=bins,hist=False,kde=True,color='blue')
        #plt.legend(['original','after log1p transform'])
        #plt.show()


#%%
def plot_ynet():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from plots import plot_map
    from paths import folders, prednames
    
    variable = 'tmin' # 'tmax' # 'ppt' # 
    scale = 2
    
    modelname = 'YNet'
    month = 0 # month to be plot
    ngcm = 0 # gcm number to be plot
    clim = None #[0,25]
    clim_diff = [0,5.0] #[0,3.5] # None #[0,18]
    
    resolution = 1/scale
    if variable=='ppt':
        is_precipitation = True # False # 
    elif variable=='tmax' or variable=='tmin':
        is_precipitation = False # 
    #%% predict result
    folder = folders[variable]['YNet'][scale]
    predname = prednames[variable]['YNet'][scale]
    
    #folder = '2020-01-08_11.03.16.421380_debug'
    #predname = 'pred_results_MSE1.4542109436459012' #'gcms_bcsd.npz'
    predpath = '../results/Climate/PRISM_GCM/YNet30/{}/scale{}/{}/'.format(variable,scale,folder)
    preds = np.load(predpath+predname+'.npy')
    savepath = '../results/Climate/PRISM_GCM/Figures/abs_diff/' # None
    #names = {8:'pred_results_MSE1.5432003852393892'}
    #folders = {8:'2020-01-09_12.22.56.720490'}
    #%% USA mask
    datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/'.format(variable,resolution,resolution)
    maskdatapath = datapath+'prism_USAmask_{}by{}.npz'.format(resolution,resolution)
    maskdata = np.load(maskdatapath)
    USAmask_HR,USAmask_LR = maskdata['USAmask_HR'],maskdata['USAmask_LR']
    USAmask_HR,USAmask_LR = abs(USAmask_HR),abs(USAmask_LR)
    USAmask_HR[USAmask_HR<1] = 0.0
    USAmask_LR[USAmask_LR<1] = 0.0
    
    #%% test data
    test_filenames = [f for f in os.listdir(datapath+'test/') if f.endswith('.npz')]
    test_filenames = sorted(test_filenames)
    gcms_test = []
    prism_test = []
    for filename in test_filenames:
        data = np.load(datapath+'test/'+filename)
        gcms_test.append(data['gcms']) # [Ngcm,Nlat,Nlon]
        prism_test.append(np.squeeze(data['prism'])) # [1,Nlat,Nlon] --> [Nlat,Nlon] 
    gcms_test = np.stack(gcms_test,axis=1) #[Ngcm,Nlat,Nlon] --> [Ngcm,Nmon,Nlat,Nlon]
    prism_test = np.stack(prism_test,axis=0) # [Nlat,Nlon] --> [Nmon,Nlat,Nlon]   
    #print('gcms_test.shape={}\nprism_test.shape={}'.format(gcms_test.shape,prism_test.shape))
    #print('gcms_test.max={}\nprism_test.max={}\npreds.max={}'.format(np.amax(gcms_test),np.amax(prism_test),np.amax(preds)))
    
    if is_precipitation:
        gcms_test = np.expm1(gcms_test)
        prism_test = np.expm1(prism_test)
    else:
        gcms_test = gcms_test*50.0
        prism_test = prism_test*50.0   
    print('YNet:\ngcms_test=[{},{}]\nprism_test=[{},{}]\npreds=[{},{}]\n\n'.format(np.amin(gcms_test),np.amax(gcms_test),
          np.amin(prism_test),np.amax(prism_test),np.amin(preds),np.amax(preds)))
    
# =============================================================================
#     MSE_DJF = np.mean((prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])**2)
#     MSE_MAM = np.mean((prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])**2)
#     MSE_JJA = np.mean((prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])**2)
#     MSE_SON = np.mean((prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])**2)
#     print('MSE_DJF={}\nMSE_MAM={}\nMSE_JJA={}\nMSE_SON={}\n'.format(MSE_DJF,MSE_MAM,MSE_JJA,MSE_SON))
#     np.savez(predpath+'MSE_seasonal.npz',MSE_DJF=MSE_DJF,MSE_MAM=MSE_MAM,MSE_JJA=MSE_JJA,MSE_SON=MSE_SON)
# =============================================================================
    
    bias = np.mean(preds-prism_test)
    corr = np.corrcoef(preds.flatten(),prism_test.flatten())
    print('bias={}'.format(bias))
    print('corr={}'.format(corr))
    
# =============================================================================
#     bias_DJF = np.mean(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
#     bias_MAM = np.mean(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
#     bias_JJA = np.mean(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
#     bias_SON = np.mean(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
#     print('bias_DJF={}\nbias_MAM={}\nbias_JJA={}\nbias_SON={}\n'.format(bias_DJF,bias_MAM,bias_JJA,bias_SON))
#     #np.savez(predpath+'bias_seasonal.npz',bias_DJF=bias_DJF,bias_MAM=bias_MAM,bias_JJA=bias_JJA,bias_SON=bias_SON)
#     corr_DJF = np.corrcoef(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
#     corr_MAM = np.corrcoef(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
#     corr_JJA = np.corrcoef(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
#     corr_SON = np.corrcoef(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
#     print('corr_DJF={}\ncorr_MAM={}\ncorr_JJA={}\ncorr_SON={}\n'.format(corr_DJF,corr_MAM,corr_JJA,corr_SON))
#     #np.savez(predpath+'corr_seasonal.npz',corr_DJF=corr_DJF,corr_MAM=corr_MAM,corr_JJA=corr_JJA,corr_SON=corr_SON)
# =============================================================================
    
    pred = preds[month,:,:]
    pred[USAmask_HR==0] = np.nan
    #pred[abs(pred)<=0.0001] = np.nan
    
    #gcm = gcms_test[ngcm,month,:,:]
    gcm = np.mean(gcms_test,axis=0)
    gcm = gcm[month,:,:]
    gcm[USAmask_LR==0] = np.nan
    #gcm[abs(gcm)<=0.0001] = np.nan
    
    prism = prism_test[month,:,:]
    prism[USAmask_HR==0] = np.nan
    #prism[abs(prism)<=0.0001] = np.nan
    
    absdiff = abs(preds[month,:,:]-prism_test[month,:,:])
    absdiff[USAmask_HR==0] = np.nan
    #absdiff[absdiff<=0.0001] = np.nan
    
    absdiff_avg = np.mean(abs(preds-prism_test),axis=0)
    absdiff_avg[USAmask_HR==0] = np.nan
    #absdiff_avg[absdiff_avg<=0.0001] = np.nan
    
    diff_avg = np.mean(preds-prism_test,axis=0)
    diff_avg[USAmask_HR==0] = np.nan
    #diff_avg[diff_avg<=0.0001] = np.nan
    
    
    
    #%% plot figures
    img = np.flipud(prism)
    title = 'ground truth {} '.format(variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
    savename = 'groundtruth_{}_{}by{}_month{}'.format(variable,resolution,resolution,month+1)
    plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim)
    
    img = np.flipud(gcm)
    #title = 'input GCM mean ppt '+str(1)+'$^{\circ}$x'+str(1)+'$^{\circ}$'
    title = 'input GCM mean {}'.format(variable)
    savename = 'input_gcm_mean_{}_month{}'.format(variable,month+1)
    plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim)
    
    img = np.flipud(pred)
    title = '{} pred {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
    savename = '{}_pred_result_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
    plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim)
    #plot_map(img,title=None,savepath=None,savename=None,cmap='YlOrRd',
    #             lonlat=[235,24.125,293.458,49.917],resolution='i',area_thresh=10000)
    
    img = np.flipud(absdiff)
    title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
    savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
    plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)
    
    img = np.flipud(absdiff_avg)
    #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
    #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
    title = '{} GT mean absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
    savename = '{}_avg_abs_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
    plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)
    
    img = np.flipud(diff_avg)
    #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
    #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
    title = '{} GT mean error {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
    savename = '{}_avg_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
    plot_map(img,title=title,savepath=savepath,savename=savename,cmap='cool',clim=None)

#plot_ynet()

#%%
def plot_rednet():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from plots import plot_map
    from paths import folders, prednames
    
    variable = 'tmin' # 'tmax' # 'ppt' # 
    scale = 2
    #resolution = 0.125
    #is_precipitation = True # False # 
    modelname = 'REDNet'
    month = 0 # month to be plot
    ngcm = 0 # gcm number to be plot
    clim_diff = [0,5.0] #[0,3.5] # None #[0,18]
    
    resolution = 1/scale
    if variable=='ppt':
        is_precipitation = True # False # 
    elif variable=='tmax' or variable=='tmin':
        is_precipitation = False # 
    #%% predict result
    folder = folders[variable]['REDNet'][scale]
    predname = prednames[variable]['REDNet'][scale]
    #folder = '2020-01-09_12.22.56.720490'
    #predname = 'pred_results_MSE1.5432003852393892' #'gcms_bcsd.npz'
    predpath = '../results/Climate/PRISM_GCM/REDNet30/{}/scale{}/{}/'.format(variable,scale,folder)
    preds = np.load(predpath+predname+'.npy')
    savepath = '../results/Climate/PRISM_GCM/Figures/abs_diff/'
    #names = {8:'pred_results_MSE1.5432003852393892'}
    #folders = {8:'2020-01-09_12.22.56.720490'}
    #%% USA mask
    datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/'.format(variable,resolution,resolution)
    maskdatapath = datapath+'prism_USAmask_{}by{}.npz'.format(resolution,resolution)
    maskdata = np.load(maskdatapath)
    USAmask_HR,USAmask_LR = maskdata['USAmask_HR'],maskdata['USAmask_LR']
    USAmask_HR,USAmask_LR = abs(USAmask_HR),abs(USAmask_LR)
    USAmask_HR[USAmask_HR<1] = 0.0
    USAmask_LR[USAmask_LR<1] = 0.0
    
    #%% test data
    test_filenames = [f for f in os.listdir(datapath+'test/') if f.endswith('.npz')]
    test_filenames = sorted(test_filenames)
    gcms_test = []
    prism_test = []
    for filename in test_filenames:
        data = np.load(datapath+'test/'+filename)
        gcms_test.append(data['gcms']) # [Ngcm,Nlat,Nlon]
        prism_test.append(np.squeeze(data['prism'])) # [1,Nlat,Nlon] --> [Nlat,Nlon] 
    gcms_test = np.stack(gcms_test,axis=1) #[Ngcm,Nlat,Nlon] --> [Ngcm,Nmon,Nlat,Nlon]
    prism_test = np.stack(prism_test,axis=0) # [Nlat,Nlon] --> [Nmon,Nlat,Nlon]   
    #print('gcms_test.shape={}\nprism_test.shape={}'.format(gcms_test.shape,prism_test.shape))
    #print('gcms_test.max={}\nprism_test.max={}\npreds.max={}'.format(np.amax(gcms_test),np.amax(prism_test),np.amax(preds)))
    
    if is_precipitation:
        gcms_test = np.expm1(gcms_test)
        prism_test = np.expm1(prism_test)
    else:
        gcms_test = gcms_test*50.0
        prism_test = prism_test*50.0   
    print('REDNet:\ngcms_test=[{},{}]\nprism_test=[{},{}]\npreds=[{},{}]\n\n'.format(np.amin(gcms_test),np.amax(gcms_test),
          np.amin(prism_test),np.amax(prism_test),np.amin(preds),np.amax(preds)))
    
# =============================================================================
#     MSE_DJF = np.mean((prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])**2)
#     MSE_MAM = np.mean((prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])**2)
#     MSE_JJA = np.mean((prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])**2)
#     MSE_SON = np.mean((prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])**2)
#     print('MSE_DJF={}\nMSE_MAM={}\nMSE_JJA={}\nMSE_SON={}\n'.format(MSE_DJF,MSE_MAM,MSE_JJA,MSE_SON))
#     np.savez(predpath+'MSE_seasonal.npz',MSE_DJF=MSE_DJF,MSE_MAM=MSE_MAM,MSE_JJA=MSE_JJA,MSE_SON=MSE_SON)
# =============================================================================
    
    bias = np.mean(preds-prism_test)
    corr = np.corrcoef(preds.flatten(),prism_test.flatten())
    print('bias={}'.format(bias))
    print('corr={}'.format(corr))
    
# =============================================================================
#     bias_DJF = np.mean(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
#     bias_MAM = np.mean(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
#     bias_JJA = np.mean(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
#     bias_SON = np.mean(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
#     print('bias_DJF={}\nbias_MAM={}\nbias_JJA={}\nbias_SON={}\n'.format(bias_DJF,bias_MAM,bias_JJA,bias_SON))
#     #np.savez(predpath+'bias_seasonal.npz',bias_DJF=bias_DJF,bias_MAM=bias_MAM,bias_JJA=bias_JJA,bias_SON=bias_SON)
#     corr_DJF = np.corrcoef(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
#     corr_MAM = np.corrcoef(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
#     corr_JJA = np.corrcoef(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
#     corr_SON = np.corrcoef(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
#     print('corr_DJF={}\ncorr_MAM={}\ncorr_JJA={}\ncorr_SON={}\n'.format(corr_DJF,corr_MAM,corr_JJA,corr_SON))
#     #np.savez(predpath+'corr_seasonal.npz',corr_DJF=corr_DJF,corr_MAM=corr_MAM,corr_JJA=corr_JJA,corr_SON=corr_SON)
# =============================================================================
    
    pred = preds[month,:,:]
    pred[USAmask_HR==0] = np.nan
    #pred[abs(pred)<=0.0001] = np.nan
    
    #gcm = gcms_test[ngcm,month,:,:]
    gcm = np.mean(gcms_test,axis=0)
    gcm = gcm[month,:,:]
    gcm[USAmask_LR==0] = np.nan
    #gcm[abs(gcm)<=0.0001] = np.nan
    
    prism = prism_test[month,:,:]
    prism[USAmask_HR==0] = np.nan
    #prism[abs(prism)<=0.0001] = np.nan
    
    absdiff = abs(preds[month,:,:]-prism_test[month,:,:])
    absdiff[USAmask_HR==0] = np.nan
    #absdiff[absdiff<=0.0001] = np.nan
    
    absdiff_avg = np.mean(abs(preds-prism_test),axis=0)
    absdiff_avg[USAmask_HR==0] = np.nan
    #absdiff_avg[absdiff_avg<=0.0001] = np.nan
    
    diff_avg = np.mean(preds-prism_test,axis=0)
    diff_avg[USAmask_HR==0] = np.nan
    #diff_avg[diff_avg<=0.0001] = np.nan
    
    #%% plot figures
# =============================================================================
#     img = np.flipud(prism)
#     title = 'ground truth {} '.format(variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = 'groundtruth_{}_{}by{}_month{}'.format(variable,resolution,resolution,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename)
#     
#     img = np.flipud(gcm)
#     #title = 'input GCM mean ppt '+str(1)+'$^{\circ}$x'+str(1)+'$^{\circ}$'
#     title = 'input GCM mean {}'.format(variable)
#     savename = 'input_gcm_mean_{}_month{}'.format(variable,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename)
#     
#     img = np.flipud(pred)
#     title = '{} pred {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_pred_result_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename)
#     #plot_map(img,title=None,savepath=None,savename=None,cmap='YlOrRd',
#     #             lonlat=[235,24.125,293.458,49.917],resolution='i',area_thresh=10000)
#     
#     img = np.flipud(absdiff)
#     title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)
#     
#     img = np.flipud(absdiff_avg)
#     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     title = '{} GT mean absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_avg_abs_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
#     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)
#     
#     img = np.flipud(diff_avg)
#     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     title = '{} GT mean error {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_avg_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
#     plot_map(img,title=title,savepath=savepath,savename=savename,cmap='cool',clim=None)
# =============================================================================

#plot_rednet()

def plot_espcn():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from plots import plot_map
    from paths import folders, prednames
    variable = 'tmin' # 'tmax' # 'ppt' # 
    scale = 2
    #resolution = 0.125
    #is_precipitation = True # False # 
    modelname = 'ESPCN'
    month = 0 # month to be plot
    ngcm = 0 # gcm number to be plot
    clim_diff = [0,5.0] #[0,3.5] #None # [0,15] # 
    
    resolution = 1/scale
    if variable=='ppt':
        is_precipitation = True # False # 
    elif variable=='tmax' or variable=='tmin':
        is_precipitation = False # 
    #%% predict result
    folder = folders[variable]['ESPCN'][scale]
    predname = prednames[variable]['ESPCN'][scale]
    #folder = '2020-01-08_18.18.38.602588'
    #predname = 'pred_results_MSE2.4984338548448353' #'gcms_bcsd.npz'
    predpath = '../results/Climate/PRISM_GCM/ESPCN/{}/scale{}/{}/'.format(variable,scale,folder)
    preds = np.load(predpath+predname+'.npy')
    savepath = '../results/Climate/PRISM_GCM/Figures/abs_diff/' # None #
    
    #%% USA mask
    datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/'.format(variable,resolution,resolution)
    maskdatapath = datapath+'prism_USAmask_{}by{}.npz'.format(resolution,resolution)
    maskdata = np.load(maskdatapath)
    USAmask_HR,USAmask_LR = maskdata['USAmask_HR'],maskdata['USAmask_LR']
    USAmask_HR,USAmask_LR = abs(USAmask_HR),abs(USAmask_LR)
    
    USAmask_HR[USAmask_HR<1] = 0.0
    USAmask_LR[USAmask_LR<1] = 0.0
    #%% test data
    test_filenames = [f for f in os.listdir(datapath+'test/') if f.endswith('.npz')]
    test_filenames = sorted(test_filenames)
    gcms_test = []
    prism_test = []
    for filename in test_filenames:
        data = np.load(datapath+'test/'+filename)
        gcms_test.append(data['gcms']) # [Ngcm,Nlat,Nlon]
        prism_test.append(np.squeeze(data['prism'])) # [1,Nlat,Nlon] --> [Nlat,Nlon] 
    gcms_test = np.stack(gcms_test,axis=1) #[Ngcm,Nlat,Nlon] --> [Ngcm,Nmon,Nlat,Nlon]
    prism_test = np.stack(prism_test,axis=0) # [Nlat,Nlon] --> [Nmon,Nlat,Nlon]   
    #print('gcms_test.shape={}\nprism_test.shape={}'.format(gcms_test.shape,prism_test.shape))
    #print('gcms_test.max={}\nprism_test.max={}\npreds.max={}\n'.format(np.amax(gcms_test),np.amax(prism_test),np.amax(preds)))
    
    if is_precipitation:
        gcms_test = np.expm1(gcms_test)
        prism_test = np.expm1(prism_test)
    else:
        gcms_test = gcms_test*50.0
        prism_test = prism_test*50.0   
    print('ESPCN:\ngcms_test=[{},{}]\nprism_test=[{},{}]\npreds=[{},{}]\n\n'.format(np.amin(gcms_test),np.amax(gcms_test),
          np.amin(prism_test),np.amax(prism_test),np.amin(preds),np.amax(preds)))
    
# =============================================================================
#     MSE_DJF = np.mean((prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])**2)
#     MSE_MAM = np.mean((prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])**2)
#     MSE_JJA = np.mean((prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])**2)
#     MSE_SON = np.mean((prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])**2)
#     print('MSE_DJF={}\nMSE_MAM={}\nMSE_JJA={}\nMSE_SON={}\n'.format(MSE_DJF,MSE_MAM,MSE_JJA,MSE_SON))
#     np.savez(predpath+'MSE_seasonal.npz',MSE_DJF=MSE_DJF,MSE_MAM=MSE_MAM,MSE_JJA=MSE_JJA,MSE_SON=MSE_SON)
# =============================================================================
    
    bias = np.mean(preds-prism_test)
    corr = np.corrcoef(preds.flatten(),prism_test.flatten())
    print('bias={}'.format(bias))
    print('corr={}'.format(corr))
    
# =============================================================================
#     bias_DJF = np.mean(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
#     bias_MAM = np.mean(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
#     bias_JJA = np.mean(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
#     bias_SON = np.mean(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
#     print('bias_DJF={}\nbias_MAM={}\nbias_JJA={}\nbias_SON={}\n'.format(bias_DJF,bias_MAM,bias_JJA,bias_SON))
#     #np.savez(predpath+'bias_seasonal.npz',bias_DJF=bias_DJF,bias_MAM=bias_MAM,bias_JJA=bias_JJA,bias_SON=bias_SON)
#     corr_DJF = np.corrcoef(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
#     corr_MAM = np.corrcoef(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
#     corr_JJA = np.corrcoef(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
#     corr_SON = np.corrcoef(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
#     print('corr_DJF={}\ncorr_MAM={}\ncorr_JJA={}\ncorr_SON={}\n'.format(corr_DJF,corr_MAM,corr_JJA,corr_SON))
#     #np.savez(predpath+'corr_seasonal.npz',corr_DJF=corr_DJF,corr_MAM=corr_MAM,corr_JJA=corr_JJA,corr_SON=corr_SON)
# =============================================================================
    
    pred = preds[month,:,:]
    pred[USAmask_HR==0] = np.nan
    #pred[abs(pred)<=0.0001] = np.nan
    
    #gcm = gcms_test[ngcm,month,:,:]
    gcm = np.mean(gcms_test,axis=0)
    gcm = gcm[month,:,:]
    gcm[USAmask_LR==0] = np.nan
    #gcm[abs(gcm)<=0.0001] = np.nan
    
    prism = prism_test[month,:,:]
    prism[USAmask_HR==0] = np.nan
    #prism[abs(prism)<=0.0001] = np.nan
    
    absdiff = abs(preds[month,:,:]-prism_test[month,:,:])
    absdiff[USAmask_HR==0] = np.nan
    #absdiff[absdiff<=0.0001] = np.nan
    
    absdiff_avg = np.mean(abs(preds-prism_test),axis=0)
    absdiff_avg[USAmask_HR==0] = np.nan
    #absdiff_avg[absdiff_avg<=0.0001] = np.nan
    
    diff_avg = np.mean(preds-prism_test,axis=0)
    diff_avg[USAmask_HR==0] = np.nan
    #diff_avg[diff_avg<=0.0001] = np.nan
    
    #%% plot figures
# =============================================================================
#     img = np.flipud(prism)
#     title = 'ground truth {} '.format(variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = 'groundtruth_{}_{}by{}_month{}'.format(variable,resolution,resolution,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename)
#     
#     img = np.flipud(gcm)
#     #title = 'input GCM mean ppt '+str(1)+'$^{\circ}$x'+str(1)+'$^{\circ}$'
#     title = 'input GCM mean {}'.format(variable)
#     savename = 'input_gcm_mean_{}_month{}'.format(variable,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename)
#     
#     img = np.flipud(pred)
#     title = '{} pred {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_pred_result_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename)
#     #plot_map(img,title=title,savepath=savepath,savename=savename,cmap='YlOrRd')
#     #plot_map(img,title=None,savepath=None,savename=None,cmap='YlOrRd',
#     #             lonlat=[235,24.125,293.458,49.917],resolution='i',area_thresh=10000)
#     
#     img = np.flipud(absdiff)
#     title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)
#     
#     img = np.flipud(absdiff_avg)
#     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     title = '{} GT mean absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_avg_abs_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
#     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)
#     
#     img = np.flipud(diff_avg)
#     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     title = '{} GT mean error {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_avg_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
#     plot_map(img,title=title,savepath=savepath,savename=savename,cmap='cool',clim=None)
# 
# =============================================================================
#plot_espcn()

#%%
def plot_deepsd():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from plots import plot_map
    from paths import folders, prednames
    variable = 'tmin' # 'tmax' # 'ppt' # 
    scale = 2
    #resolution = 0.125
    #is_precipitation = True # False # 
    modelname = 'DeepSD'
    month = 0 # month to be plot
    ngcm = 0 # gcm number to be plot
    clim_diff = [0,5.0] #[0,3.5] #None #[0,18]
    
    resolution = 1/scale
    if variable=='ppt':
        is_precipitation = True # False # 
    elif variable=='tmax' or variable=='tmin':
        is_precipitation = False # 
    folder = folders[variable]['DeepSD'][scale]
    predname = prednames[variable]['DeepSD'][scale]
    #%% predict result
    #folder = '2020-01-08_16.06.29.813876'
    #predname = 'pred_results_MSE1.7008076906204224' #'gcms_bcsd.npz'
    predpath = '../results/Climate/PRISM_GCM/DeepSD/{}/train_together/{}by{}/{}/'.format(variable,resolution,resolution,folder)
    preds = np.load(predpath+predname+'.npy')
    savepath = '../results/Climate/PRISM_GCM/Figures/abs_diff/'
    
    #%% USA mask
    datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/'.format(variable,resolution,resolution)
    maskdatapath = datapath+'prism_USAmask_{}by{}.npz'.format(resolution,resolution)
    maskdata = np.load(maskdatapath)
    USAmask_HR,USAmask_LR = maskdata['USAmask_HR'],maskdata['USAmask_LR']
    USAmask_HR,USAmask_LR = abs(USAmask_HR),abs(USAmask_LR)
    
    USAmask_HR[USAmask_HR<1] = 0.0
    USAmask_LR[USAmask_LR<1] = 0.0
    #%% test data
    test_filenames = [f for f in os.listdir(datapath+'test/') if f.endswith('.npz')]
    test_filenames = sorted(test_filenames)
    gcms_test = []
    prism_test = []
    for filename in test_filenames:
        data = np.load(datapath+'test/'+filename)
        gcms_test.append(data['gcms']) # [Ngcm,Nlat,Nlon]
        prism_test.append(np.squeeze(data['prism'])) # [1,Nlat,Nlon] --> [Nlat,Nlon] 
    gcms_test = np.stack(gcms_test,axis=1) #[Ngcm,Nlat,Nlon] --> [Ngcm,Nmon,Nlat,Nlon]
    prism_test = np.stack(prism_test,axis=0) # [Nlat,Nlon] --> [Nmon,Nlat,Nlon]   
    #print('gcms_test.shape={}\nprism_test.shape={}'.format(gcms_test.shape,prism_test.shape))
    #print('gcms_test.max={}\n,prism_test.max={}'.format(np.amax(gcms_test),np.amax(prism_test)))
    
    if is_precipitation:
        gcms_test = np.expm1(gcms_test)
        prism_test = np.expm1(prism_test)
    else:
        gcms_test = gcms_test*50.0
        prism_test = prism_test*50.0   
    print('DeepSD:\ngcms_test=[{},{}]\nprism_test=[{},{}]\npreds=[{},{}]\n\n'.format(np.amin(gcms_test),np.amax(gcms_test),
          np.amin(prism_test),np.amax(prism_test),np.amin(preds),np.amax(preds)))
    
# =============================================================================
#     MSE_DJF = np.mean((prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])**2)
#     MSE_MAM = np.mean((prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])**2)
#     MSE_JJA = np.mean((prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])**2)
#     MSE_SON = np.mean((prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])**2)
#     print('MSE_DJF={}\nMSE_MAM={}\nMSE_JJA={}\nMSE_SON={}\n'.format(MSE_DJF,MSE_MAM,MSE_JJA,MSE_SON))
#     np.savez(predpath+'MSE_seasonal.npz',MSE_DJF=MSE_DJF,MSE_MAM=MSE_MAM,MSE_JJA=MSE_JJA,MSE_SON=MSE_SON)
# =============================================================================

    bias = np.mean(preds-prism_test)
    corr = np.corrcoef(preds.flatten(),prism_test.flatten())
    print('bias={}'.format(bias))
    print('corr={}'.format(corr))
    
# =============================================================================
#     bias_DJF = np.mean(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
#     bias_MAM = np.mean(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
#     bias_JJA = np.mean(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
#     bias_SON = np.mean(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
#     print('bias_DJF={}\nbias_MAM={}\nbias_JJA={}\nbias_SON={}\n'.format(bias_DJF,bias_MAM,bias_JJA,bias_SON))
#     #np.savez(predpath+'bias_seasonal.npz',bias_DJF=bias_DJF,bias_MAM=bias_MAM,bias_JJA=bias_JJA,bias_SON=bias_SON)
#     corr_DJF = np.corrcoef(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
#     corr_MAM = np.corrcoef(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
#     corr_JJA = np.corrcoef(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
#     corr_SON = np.corrcoef(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
#     print('corr_DJF={}\ncorr_MAM={}\ncorr_JJA={}\ncorr_SON={}\n'.format(corr_DJF,corr_MAM,corr_JJA,corr_SON))
#     #np.savez(predpath+'corr_seasonal.npz',corr_DJF=corr_DJF,corr_MAM=corr_MAM,corr_JJA=corr_JJA,corr_SON=corr_SON)
# =============================================================================
    
    pred = preds[month,:,:]
    pred[USAmask_HR==0] = np.nan
    #pred[abs(pred)<=0.0001] = np.nan
    
    #gcm = gcms_test[ngcm,month,:,:]
    gcm = np.mean(gcms_test,axis=0)
    gcm = gcm[month,:,:]
    gcm[USAmask_LR==0] = np.nan
    #gcm[abs(gcm)<=0.0001] = np.nan
    
    prism = prism_test[month,:,:]
    prism[USAmask_HR==0] = np.nan
    #prism[abs(prism)<=0.0001] = np.nan
    
    absdiff = abs(preds[month,:,:]-prism_test[month,:,:])
    absdiff[USAmask_HR==0] = np.nan
    #absdiff[absdiff<=0.0001] = np.nan
    
    absdiff_avg = np.mean(abs(preds-prism_test),axis=0)
    absdiff_avg[USAmask_HR==0] = np.nan
    #absdiff_avg[absdiff_avg<=0.0001] = np.nan
    
    diff_avg = np.mean(preds-prism_test,axis=0)
    diff_avg[USAmask_HR==0] = np.nan
    #diff_avg[diff_avg<=0.0001] = np.nan
    
    #%% plot figures
# =============================================================================
#     img = np.flipud(prism)
#     title = 'ground truth {} '.format(variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = 'groundtruth_{}_{}by{}_month{}'.format(variable,resolution,resolution,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename)
#     
#     img = np.flipud(gcm)
#     #title = 'input GCM mean ppt '+str(1)+'$^{\circ}$x'+str(1)+'$^{\circ}$'
#     title = 'input GCM mean {}'.format(variable)
#     savename = 'input_gcm_mean_{}_month{}'.format(variable,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename)
#     
#     img = np.flipud(pred)
#     title = '{} pred {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_pred_result_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename)
#     #plot_map(img,title=None,savepath=None,savename=None,cmap='YlOrRd',
#     #             lonlat=[235,24.125,293.458,49.917],resolution='i',area_thresh=10000)
#     
#     img = np.flipud(absdiff)
#     title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)
# 
#     
#     img = np.flipud(absdiff_avg)
#     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     title = '{} GT mean absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_avg_abs_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
#     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)
#     
#     img = np.flipud(diff_avg)
#     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     title = '{} GT mean error {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_avg_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
#     plot_map(img,title=title,savepath=savepath,savename=savename,cmap='cool',clim=None)
# =============================================================================
    
#plot_deepsd()

#%%
def plot_bcsd():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from plots import plot_map
    from paths import folders, prednames
    variable = 'ppt' # 'tmin' # 'tmax' # 
    scale = 8
    #resolution = 0.125
    #is_precipitation = True # False # 
    modelname = 'BCSD'
    month = 0 # month to be plot
    ngcm = 0 # gcm number to be plot
    clim_diff = [0,5.0] #[0,3.5] # None #[0,18]
    
    resolution = 1/scale
    if variable=='ppt':
        is_precipitation = True # False # 
    elif variable=='tmax' or variable=='tmin':
        is_precipitation = False # 
    #%% predict result
    #folder = folders[variable]['BCSD'][scale]
    predname = prednames[variable]['BCSD'][scale]
    #predname = 'gcms_bcsd_pred_MSE1.7627197220364208' #'gcms_bcsd.npz'
    predpath = '../results/Climate/PRISM_GCM/BCSD/{}/{}by{}/'.format(variable,resolution,resolution)
    preddata = np.load(predpath+predname+'.npz')
    preds = preddata['gcms_bcsd']
    #MSE = preddata['MSE']
    savepath = '../results/Climate/PRISM_GCM/Figures/abs_diff/'
    
    #%% USA mask
    datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/'.format(variable,resolution,resolution)
    maskdatapath = datapath+'prism_USAmask_{}by{}.npz'.format(resolution,resolution)
    maskdata = np.load(maskdatapath)
    USAmask_HR,USAmask_LR = maskdata['USAmask_HR'],maskdata['USAmask_LR']
    USAmask_HR,USAmask_LR = abs(USAmask_HR),abs(USAmask_LR)
    
    USAmask_HR[USAmask_HR<1] = 0.0
    USAmask_LR[USAmask_LR<1] = 0.0
    #%% test data
    test_filenames = [f for f in os.listdir(datapath+'test/') if f.endswith('.npz')]
    test_filenames = sorted(test_filenames)
    gcms_test = []
    prism_test = []
    for filename in test_filenames:
        data = np.load(datapath+'test/'+filename)
        gcms_test.append(data['gcms']) # [Ngcm,Nlat,Nlon]
        prism_test.append(np.squeeze(data['prism'])) # [1,Nlat,Nlon] --> [Nlat,Nlon] 
    gcms_test = np.stack(gcms_test,axis=1) #[Ngcm,Nlat,Nlon] --> [Ngcm,Nmon,Nlat,Nlon]
    prism_test = np.stack(prism_test,axis=0) # [Nlat,Nlon] --> [Nmon,Nlat,Nlon]   
    #print('gcms_test.shape={}\nprism_test.shape={}'.format(gcms_test.shape,prism_test.shape))    
    #print('gcms_test.max={}\nprism_test.max={}\n'.format(np.amax(gcms_test),np.amax(prism_test)))
      
    if is_precipitation:
        gcms_test = np.expm1(gcms_test)
        prism_test = np.expm1(prism_test)
    else:
        gcms_test = gcms_test*50.0
        prism_test = prism_test*50.0      
    print('BCSD:\ngcms_test=[{},{}]\nprism_test=[{},{}]\npreds=[{},{}]\n\n'.format(np.amin(gcms_test),np.amax(gcms_test),
          np.amin(prism_test),np.amax(prism_test),np.amin(preds),np.amax(preds)))
    # =============================================================================
    #     MSE_DJF = np.mean((prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])**2)
    #     MSE_MAM = np.mean((prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])**2)
    #     MSE_JJA = np.mean((prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])**2)
    #     MSE_SON = np.mean((prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])**2)
    #     print('MSE_DJF={}\nMSE_MAM={}\nMSE_JJA={}\nMSE_SON={}\n'.format(MSE_DJF,MSE_MAM,MSE_JJA,MSE_SON))
    #     np.savez(predpath+'MSE_seasonal.npz',MSE_DJF=MSE_DJF,MSE_MAM=MSE_MAM,MSE_JJA=MSE_JJA,MSE_SON=MSE_SON)
    # =============================================================================
    
    bias = np.mean(preds-prism_test)
    corr = np.corrcoef(preds.flatten(),prism_test.flatten())
    print('bias={}'.format(bias))
    print('corr={}'.format(corr))
    
# =============================================================================
#     bias_DJF = np.mean(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
#     bias_MAM = np.mean(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
#     bias_JJA = np.mean(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
#     bias_SON = np.mean(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
#     print('bias_DJF={}\nbias_MAM={}\nbias_JJA={}\nbias_SON={}\n'.format(bias_DJF,bias_MAM,bias_JJA,bias_SON))
#     #np.savez(predpath+'bias_seasonal.npz',bias_DJF=bias_DJF,bias_MAM=bias_MAM,bias_JJA=bias_JJA,bias_SON=bias_SON)
#     corr_DJF = np.corrcoef(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
#     corr_MAM = np.corrcoef(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
#     corr_JJA = np.corrcoef(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
#     corr_SON = np.corrcoef(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
#     print('corr_DJF={}\ncorr_MAM={}\ncorr_JJA={}\ncorr_SON={}\n'.format(corr_DJF,corr_MAM,corr_JJA,corr_SON))
#     #np.savez(predpath+'corr_seasonal.npz',corr_DJF=corr_DJF,corr_MAM=corr_MAM,corr_JJA=corr_JJA,corr_SON=corr_SON)
# =============================================================================
    
    pred = preds[month,:,:]
    pred[USAmask_HR==0] = np.nan
    #pred[abs(pred)<=0.0001] = np.nan
    
    #gcm = gcms_test[ngcm,month,:,:]
    gcm = np.mean(gcms_test,axis=0)
    gcm = gcm[month,:,:]
    gcm[USAmask_LR==0] = np.nan
    #gcm[abs(gcm)<=0.0001] = np.nan
    
    prism = prism_test[month,:,:]
    prism[USAmask_HR==0] = np.nan
    #prism[abs(prism)<=0.0001] = np.nan
    
    absdiff = abs(preds[month,:,:]-prism_test[month,:,:])
    absdiff[USAmask_HR==0] = np.nan
    #absdiff[absdiff<=0.0001] = np.nan
    
    absdiff_avg = np.mean(abs(preds-prism_test),axis=0)
    print('[absdiff_avg.min,absdiff_avg.max]=[{},{}]'.format(np.amin(absdiff_avg),np.amax(absdiff_avg)))
    absdiff_avg[USAmask_HR==0] = np.nan
    #absdiff_avg[absdiff_avg<=0.0001] = np.nan
    
    diff_avg = np.mean(preds-prism_test,axis=0)
    diff_avg[USAmask_HR==0] = np.nan
    #diff_avg[diff_avg<=0.0001] = np.nan
    
    
    prism_avg = np.mean(prism_test,axis=0)
    prism_avg[USAmask_HR==0] = np.nan
    
    
    # =============================================================================
    #     winter = np.sum(prism_test[[-1,0,1,11,12,13,23,24,25],:,:],axis=0)
    #     spring = np.sum(prism_test[[2,3,4,14,15,16,26,27,28],:,:],axis=0)
    #     summer = np.sum(prism_test[[5,6,7,17,18,19,29,30,31],:,:],axis=0)
    #     autumn = np.sum(prism_test[[8,9,10,20,21,22,32,33,34],:,:],axis=0)
    #     
    #     winter[USAmask_HR==0] = np.nan
    #     spring[USAmask_HR==0] = np.nan
    #     summer[USAmask_HR==0] = np.nan
    #     autumn[USAmask_HR==0] = np.nan
    #     
    #     
    #     import matplotlib.pyplot as plt
    #     fig = plt.figure()
    #     plt.imshow(winter,cmap='YlOrRd')
    #     plt.title('winter ppt')
    #     plt.colorbar()
    #     plt.show()
    #     
    #     fig = plt.figure()
    #     plt.imshow(spring,cmap='YlOrRd')
    #     plt.title('spring ppt')
    #     plt.colorbar()
    #     plt.show()
    #     
    #     fig = plt.figure()
    #     plt.imshow(summer,cmap='YlOrRd')
    #     plt.title('summer ppt')
    #     plt.colorbar()
    #     plt.show()
    #     
    #     fig = plt.figure()
    #     plt.imshow(autumn,cmap='YlOrRd')
    #     plt.title('autumn ppt')
    #     plt.colorbar()
    #     plt.show()
    # =============================================================================
    
    #%% plot figures
    img = np.flipud(prism)
    title = 'ground truth {} '.format(variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
    savename = 'groundtruth_{}_{}by{}_month{}'.format(variable,resolution,resolution,month+1)
    plot_map(img,title=title,savepath=savepath,savename=savename)
    
# =============================================================================
#     img = np.flipud(gcm)
#     #title = 'input GCM mean ppt '+str(1)+'$^{\circ}$x'+str(1)+'$^{\circ}$'
#     title = 'input GCM mean {}'.format(variable)
#     savename = 'input_gcm_mean_{}_month{}'.format(variable,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename)
# =============================================================================
    
# =============================================================================
#     img = np.flipud(pred)
#     title = '{} pred {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_pred_result_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename)
#     #plot_map(img,title=None,savepath=None,savename=None,cmap='YlOrRd',
#     #             lonlat=[235,24.125,293.458,49.917],resolution='i',area_thresh=10000)
#     
#     img = np.flipud(absdiff)
#     title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)
# =============================================================================
    
# =============================================================================
#     img = np.flipud(absdiff_avg)
#     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     title = '{} GT mean absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_avg_abs_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
#     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)
#     
#     img = np.flipud(diff_avg)
#     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     title = '{} GT mean error {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_avg_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
#     plot_map(img,title=title,savepath=savepath,savename=savename,cmap='cool',clim=None)
#     
#     img = np.flipud(prism_avg)
#     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     title = 'GT mean {} over 3 years '.format(variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = 'GT_avg_{}_{}by{}_2'.format(variable,resolution,resolution)
#     plot_map(img,title=title,savepath=savepath,savename=savename)
# =============================================================================

plot_bcsd()




















#%%
#def plot_map(img,title=None,savepath=None,savename=None,cmap='YlOrRd',
#             lonlat=[235,24.125,293.458,49.917],resolution='i',area_thresh=10000):
#    import numpy as np
#    from mpl_toolkits.basemap import Basemap
#    #import matplotlib.pyplot as plt
#    #lonlat=[235,24.125,293.458,49.917]
#    #area_thresh=10000
#    lats = np.arange(20.0,51.0,5.0)
#    lons = np.arange(235.0,300.0,10.0)
#    fig = plt.figure()
#    m = Basemap(llcrnrlon=lonlat[0],llcrnrlat=lonlat[1],urcrnrlon=lonlat[2],urcrnrlat=lonlat[3],
#                projection='cyl',resolution='i',area_thresh=area_thresh)
#    m.drawcoastlines(linewidth=1.0)
#    m.drawcountries(linewidth=1.0)
#    m.drawstates()
#    
#    m.drawparallels(lats,labels=[True,False,False,False],dashes=[1,2])
#    m.drawmeridians(lons,labels=[False,False,False,True],dashes=[1,2])
#    #m.imshow(np.flipud(np.sqrt(pred)),alpha=1.0)
#    m.imshow(img,cmap=cmap,alpha=1.0)
#    plt.colorbar(fraction=0.02)
#    plt.show()
#    if title:
#        plt.title(title)
#    if savepath and savename:
#        plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')

