#%%
def readGCM():
    #### generate whole USA map, saved on 10/22/2019
    #### read and process NASA GCM data and aglined
    import numpy as np
    from netCDF4 import Dataset
    from ncdump import ncdump
    from os import listdir
    from os.path import isfile, join
    import os
    
    variablename = 'pr' #['pr' 'tas' 'tasmax' 'tasmin'] 'pr_37models'
    filepath = '../data/Climate/GCMdata/rawdata/'+variablename+'/'
    savepath0 = None #'../data/Climate/GCMdata/processeddata/'
    savepath = None #savepath0+variablename+'/'
    # if not os.path.exists(savepath0):
    #     os.makedirs(savepath0)
    if savepath and not os.path.exists(savepath):
        os.makedirs(savepath)
    
    filenames = [f for f in listdir(filepath) if isfile(join(filepath,f))]
    
    kk = 0
    values_gcms = []
    for filename in filenames:
        #filename = 'regridded_1deg_pr_amon_inmcm4_historical_r1i1p1_195001-200512.nc'
        dataset = Dataset(filepath+filename,mode='r')
        #dataset = Dataset(filepath+filename,mode='r',format="NETCDF3")
        #dataset = Dataset(filename,mode='r')
        if kk==0:
            nc_attrs, nc_dims, nc_vars = ncdump(dataset)
            kk += 1
       
        # original longitude is from 0.5 to 359.5 by 1, 360 points 
        # original latitude is from -89.5 to 89.5 by 1, 180 points 
        # whole USA longitude from [230.5E,304.5E] by 1, 75 points
        # whole USA latitude from [20.5N, 49.5N] by 1, 30 points
        # whole USA longitude from [234.5E,295.5E] by 1, 62 points
        # whole USA latitude from [24.5N, 49.5N] by 1, 26 points
        
        # retangular USA longitude from [245.5 to 277.5] by 1, 33 points
        # retangular USA latitude from 33.5 to 49.5 by 1, 17 points
        # original month from 195001 to 200512, 672 points
        # month from 200001 to 200412, 60 months
        
        time = dataset.variables['time'][:] # 195001 - 200512
        lats = dataset.variables['latitude'][110:140] # [20.5N, 49.5N]
        lons = dataset.variables['longitude'][230:305] # [230.5E, 304.5E]
        #lats = dataset.variables['latitude'][114:140] # [24.5N, 49.5N]
        #lons = dataset.variables['longitude'][234:296] # [234.5E, 295.5E]
        #### whole USA
        ## monthly mean precipitation, unit: mm/day
        value1_gcm = dataset.variables[variablename][:,110:140,230:305]#[month,lat,lon] 195001-200512, totally 
        #value1_gcm = dataset.variables[variablename][:,114:140,234:296]#[month,lat,lon] 195001-200512, totally 
        #value2_gcm = np.ma.filled(value1_gcm,-1.0e-8)
        value2_gcm = np.ma.filled(value1_gcm,0)
        (Nmon,Nlat,Nlon) = value2_gcm.shape # [672,30,75]
        value_gcm = np.zeros((Nmon,Nlat,Nlon)) # [672,30,75]
        for t in range(Nmon):
            value_gcm[t,:,:] = np.flipud(value2_gcm[t,:,:]) # lats from [49.5N,25.5N]  
        
        #### retangular USA
        ## monthly mean precipitation, unit: mm/day
        #precipitation = dataset.variables['pr'][600:660,123:140,245:278]#[month,lat,lon]    
        #prmean_month_gcm = np.ma.filled(precipitation,np.nan)   
    
        if np.isnan(np.sum(value_gcm)):
            print(filename + '\n')
            break
        savename = filename.replace('.nc','_USA.npy') 
        np.save(savepath+savename,value_gcm)
        values_gcms.append(value_gcm) 

    values_gcms = np.stack(values_gcms,axis=0) # [Ngcm,Nmon,Nlat,Nlon]
    print('values_gcms.shape={}'.format(values_gcms.shape))
    
    time = np.array(time)
    #### whole USA
    # latitude from [20.5N, 49.5N] by 1, 30 points
    # latitude from [24.5N, 49.5N] by 1, 26 points
    lats_gcm1 = dataset.variables['latitude'][110:140]
    #lats_gcm1 = dataset.variables['latitude'][114:140] 
    lats_gcm = np.flipud(lats_gcm1) # lats from [49.5N,24.5N]
    # longitude from [230.5E, 304.5E] by 1, 75 points
    # longitude from [234.5E, 295.5E] by 1, 62 points
    lons_gcm = dataset.variables['longitude'][230:305]
    #lons_gcm = dataset.variables['longitude'][234:296] 
    lats_gcm = np.array(lats_gcm)
    lons_gcm = np.array(lons_gcm)
    #np.save(savepath0+'time_gcm.npy',time)
    #np.save(savepath0+'lats_gcm.npy',lats_gcm)
    #np.save(savepath0+'lons_gcm.npy',lons_gcm)
    #np.save(savepath0+'18gcms_prmean_monthly_1by1_195001-200512_USA.npy',values_gcms)
    
    #### retangular USA
    ### latitude from 33.5 to 49.5 by 1, 17 points
    #lats_gcm = dataset.variables['latitude'][123:140] 
    ## longitude from 245.5 to 277.5 by 1, 33 points
    #lons_gcm = dataset.variables['longitude'][245:278] 
    #np.save(savepath+'lats_gcm.npy',lats_gcm)
    #np.save(savepath+'lons_gcm.npy',lons_gcm)
    
    
    # test = dataset.variables['pr'][:,:,:]
    # test = test[:,110:140,230:305] # usa area
    # for t in range(test.shape[0]):
    #     test[t,:,:] = np.flipud(test[t,:,:])
    
    # img = np.sum(test,axis=0)
    # img = img/abs(img).max()
    # import matplotlib.pyplot as plt
    # plt.figure()
    # #plt.imshow(img)
    # #plt.imshow(value_gcm[0,:,:])
    # plt.show()


    ## save every month data
    for mon in range(values_gcms.shape[0]):
        values_gcms_per_month = np.transpose(values_gcms[:,mon,:,:],axes=(1,2,0)) # [Nlat,Nlon,Ngcm]
        if mon==0:
            print('values_gcms_per_month.shape={}'.format(values_gcms_per_month.shape))
        



#readGCM()


#%%
def readCPC():
    #### read the CPC data and aglined
    import numpy as np
    from netCDF4 import Dataset
    import os
    from ncdump import ncdump

    filepath = '../data/Climate/CPCdata/rawdata/precip.V1.0.mon.mean.nc'
    savepath = '../data/Climate/CPCdata/processeddata/cpc_prmean_monthly_0.25by0.25/'
    
    
    dataset = Dataset(filepath,mode='r')
    nc_attrs, nc_dims, nc_vars = ncdump(dataset)
    
    
    # original latitude from 20.125 to 49.875 by 0.25, 120 points
    # latitude from 20.375N to 49.625N by 0.25, 118 points
    ## latitude from 24.375N to 49.625N by 0.25, 102 points
    lats_2 = dataset.variables['lat'][:]
    #lats_2 = dataset.variables['lat'][1:-1]
    #lats_2 = dataset.variables['lat'][17:-1]  
    # original longitude from 230.125E to 304.875E by 0.25, 300 points
    # longitude from 230.375E to 304.625E by 0.25, 298 points
    # longitude from 234.375E to 295.625E by 0.25, 246 points
    lons_2 = dataset.variables['lon'][:] 
    #lons_2 = dataset.variables['lon'][1:-1]
    #lons_2 = dataset.variables['lon'][17:263] 
    # time from 195001 to 200512 by 1, 672 points
    time = dataset.variables['time'][24:696]
    
    precipitation_2 = dataset.variables['precip'][24:696,:,:]#[month,lat,lon]
    #precipitation_2 = dataset.variables['precip'][24:696,1:-1,1:-1]#[month,lat,lon]    
    #precipitation_2 = dataset.variables['precip'][24:696,17:-1,17:263]#[month,lat,lon]
    #prmean_month = np.ma.filled(precipitation_2, -1.0e-8) # daily mean precipitation, unite: mm/day
    prmean_month = np.ma.filled(precipitation_2,0) # daily mean precipitation, unite: mm/day
    
    # flip precipitation upside down regarding to latitude
    for i in range(prmean_month.shape[0]):
        prmean_month[i,:,:] = np.flipud(prmean_month[i,:,:]) 
        
    #prmean_month = prmean_month/abs(prmean_month[:]).max()
    #import matplotlib.pyplot as plt
    #for i in range(1):
    #    a = prmean_month[i,:,:]
    #    #a = np.flipud(a)
    #    plt.imshow(a)
    
    # aligned with GCM longtitude and latitude
    # average longitude from [230.375E,304.625E], 297 points
    # average latitude from [20.375N,49.625N], 117 points
    ## average longitude from [234.5E,295.5E], 245 points
    ## average latitude from [24.5N,49.5N], 101 points
    lats_2 = np.array(lats_2[::-1]) # flip latitude upside down
    lons_2 = np.array(lons_2)
    '''
    lats = np.zeros(len(lats_2)-1,)
    lons = np.zeros(len(lons_2)-1,)
    for i in range(len(lats_2)-1):
        lats[i] = 0.5*(lats_2[i]+lats_2[i+1])
    for i in range(len(lons_2)-1):
        lons[i] = 0.5*(lons_2[i]+lons_2[i+1])
    (Nmon,Nlat,Nlon) = prmean_month.shape
    prmean_month_cpc = np.zeros((Nmon,Nlat-1,Nlon-1)) # [672, 117, 297],[672, 101, 245]
    for t in range(Nmon):
        for i in range(Nlat-1):
            for j in range(Nlon-1):
                prmean_month_cpc[t,i,j] = 0.25*(prmean_month[t,i,j]+prmean_month[t,i+1,j]
                                          +prmean_month[t,i,j+1]+prmean_month[t,i+1,j+1])
    #prmean_month_cpc[prmean_month_cpc<0] = -1.0e-8
    prmean_month_cpc[prmean_month_cpc<0] = 0
    '''

    lats = lats_2
    lons = lons_2
    prmean_month_cpc = np.array(prmean_month)

    if np.isnan(np.sum(prmean_month_cpc)):
        print("Error! nan found!\n")            
        
    #    img = np.sum(prmean_month_cpc,axis=0)
    #    img = img/abs(img).max()    
    #    import matplotlib.pyplot as plt
    #    plt.figure()
    #    plt.imshow(img)
    
    time = np.array(time)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    #print('lons={}'.format(lons))
    #print('lats={}'.format(lats))
    #print('prmena_month_cpc.shape={},type(prmean_month_cpc)={}'.format(prmean_month_cpc.shape,type(prmean_month_cpc)))
    #np.save(savepath+'lons_cpc.npy',lons)
    #np.save(savepath+'lats_cpc.npy',lats) 
    #np.save(savepath+'time_cpc.npy',time)    
    #np.save(savepath+'cpc_prmean_monthly_195001-200512_USA.npy',prmean_month_cpc)
    ##import matplotlib.pyplot as plt
    ##plt.imshow(prmean_month_cpc[0,:,:])
    ##plt.show()

#readCPC()

#%%
def mergexy():
    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from skimage.transform import resize


    savepath = '../data/Climate/CPC_GCMdata/'
    gcmpath = '../data/Climate/GCMdata/processeddata/18gcms_prmean_monthly_1by1_195001-200512_USA.npy'
    cpcpath = '../data/Climate/CPCdata/processeddata/cpc_prmean_monthly_0.25by0.25/cpc_prmean_monthly_0.25by0.25_195001-200512_USA.npy'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    gcmdata = np.load(gcmpath)
    cpcdata = np.load(cpcpath)

    print('gcmdata.shape={}'.format(gcmdata.shape))
    print('cpcdata.shape={}'.format(cpcdata.shape))

    MaskMapUSA = np.sum(cpcdata,axis=0) #[Nlat,Nlon]
    MaskMapUSAsmall = resize(image=MaskMapUSA,output_shape=gcmdata.shape[2:],preserve_range=True)
    print('MaskMapUSA.shape={}'.format(MaskMapUSA.shape))
    print('MaskMapUSAsmall.shape={}'.format(MaskMapUSAsmall.shape))

    for mon in range(cpcdata.shape[0]):
        X = gcmdata[:,mon,:,:] #[Ngcm,Nlat,Nlon], mm/day
        #y = cpcdata[[mon],:,:] #[1,Nlat,Nlon], mm/day
        y = cpcdata[mon,:,:] #[Nlat,Nlon], mm/day
        
        for ngcm in range(X.shape[0]):
            X[ngcm,:,:][MaskMapUSAsmall<=0] = 0
        y[MaskMapUSA<=0] = 0
        y = y[np.newaxis,...] #[1,Nlat,Nlon], mm/day
        #X = gcmdata
        #y = cpcdata

        X = np.log(1.0+X) ##[Ngcm,Nlat,Nlon], [0,3.737655630962239]
        y = np.log(1.0+y) ##[1,Nlat,Nlon],[0,4.159132957458496]        
        #print('X.max={}'.format(max(X.flatten())))
        #print('y.max={}'.format(max(y.flatten())))
        savename = 'cpc_gcm_log1p_prmean_monthly_0.25to1.0_195001-200512_USA_month'+str(mon+1)
        np.savez(savepath+savename+'.npz',gcms=X,cpc=y)
        # if mon==0:
        #     fig = plt.figure()
        #     sns.distplot(X.flatten())
        #     plt.title('GCM')
        #     plt.show()
        #     fig = plt.figure()
        #     sns.distplot(y.flatten())
        #     plt.title('CPC')
        #     plt.show()

#mergexy()





#%%
