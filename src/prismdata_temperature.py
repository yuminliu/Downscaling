#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:13:52 2019

@author: yumin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:08:39 2019

@author: yumin
"""
def collect_GCM():
    import os
    import glob
    import shutil
    variable = 'tasmin' #'tasmax'
    datapath = '/home/yumin/Downloads/CMIP5/CommonGrid/'
    savepath = '/home/yumin/myProgramFiles/DATA/GCM/GCMdata/'+variable+'/raw/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    pathlist = glob.glob(datapath+'**/*.nc', recursive=True)
    filepaths = [f for f in pathlist if 'historical' in f and variable in f and f.endswith('.nc')]
    filepaths = sorted(filepaths)
    
    for f in filepaths:
        newf = savepath+f.split('/')[-1]
        shutil.copyfile(f,newf)


#%%
def readGCM():
    '''
    generate whole USA map, saved on 12/11/2019
    read and process NASA GCM data and aglined
    '''
    import numpy as np
    from netCDF4 import Dataset
    from ncdump import ncdump
    from os import listdir
    from os.path import isfile, join
    import os
    
    variable = 'tasmin' #'tasmax' #['pr' 'tas' 'tasmax' 'tasmin'] 'pr_37models'
    filepath = '/home/yumin/myProgramFiles/DATA/GCM/GCMdata/'+variable+'/raw/'
    savepath0 = '/home/yumin/myProgramFiles/DATA/GCM/GCMdata/'+variable+'/processeddata_26by59_points/'
    savepath = savepath0+variable+'/'
    
    filenames = [f for f in listdir(filepath) if isfile(join(filepath,f))]
    #filenames = [filenames[0]]
    
    values_gcms = []
    # if not os.path.exists(savepath0):
    #     os.makedirs(savepath0)
    if savepath and not os.path.exists(savepath):
        os.makedirs(savepath)
    for kk,filename in enumerate(filenames):
        #filename = 'regridded_1deg_pr_amon_inmcm4_historical_r1i1p1_195001-200512.nc'
        dataset = Dataset(filepath+filename,mode='r')
        #dataset = Dataset(filepath+filename,mode='r',format="NETCDF3")
        #dataset = Dataset(filename,mode='r')
        if kk==0:
            nc_attrs, nc_dims, nc_vars = ncdump(dataset)
       
        # original longitude is from 0.5 to 359.5 by 1, 360 points 
        # original latitude is from -89.5 to 89.5 by 1, 180 points 
        # whole USA longitude from [230.5E,304.5E] by 1, 75 points
        # whole USA latitude from [20.5N, 49.5N] by 1, 30 points
        # whole USA longitude from [235.5E,293.5E] by 1, 59 points
        # whole USA latitude from [24.5N, 49.5N] by 1, 26 points
    
        # original month from 195001 to 200512, 672 points
        # month from 200001 to 200412, 60 months
        
        time = dataset.variables['time'][:] # 195001 - 200512
        #lats = dataset.variables['latitude'][110:140] # [20.5N, 49.5N]
        #lons = dataset.variables['longitude'][230:305] # [230.5E, 304.5E]
        lats = dataset.variables['latitude'][114:140] # [24.5N, 49.5N]
        lons = dataset.variables['longitude'][235:294] # [235.5E, 293.5E]
        #### whole USA
        ## monthly mean precipitation, unit: mm/day
        #value1_gcm = dataset.variables[variable][:,110:140,230:305]#[month,lat,lon] 195001-200512, totally 
        value1_gcm = dataset.variables[variable][:,114:140,235:294]#[month,lat,lon] 195001-200512, totally 
        
        #value2_gcm = np.ma.filled(value1_gcm,-1.0e-8)
        value2_gcm = np.ma.filled(value1_gcm,0)
        (Nmon,Nlat,Nlon) = value2_gcm.shape # [672,26,59]
        value_gcm = np.zeros((Nmon,Nlat,Nlon)) # [672,26,59]
        for t in range(Nmon):
            value_gcm[t,:,:] = np.flipud(value2_gcm[t,:,:]) # lats from [49.5N,24.5N]  
    
        #### retangular USA
        ## monthly mean precipitation, unit: mm/day
        #precipitation = dataset.variables['pr'][600:660,123:140,245:278]#[month,lat,lon]    
        #prmean_month_gcm = np.ma.filled(precipitation,np.nan)   
    
        if np.isnan(np.sum(value_gcm)):
            print(filename + 'has NAN!\n')
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
    #lats_gcm1 = dataset.variables['latitude'][110:140]
    lats_gcm1 = dataset.variables['latitude'][114:140] 
    lats_gcm = np.flipud(lats_gcm1) # lats from [49.5N,24.5N]
    # longitude from [230.5E, 304.5E] by 1, 75 points
    # longitude from [234.5E, 295.5E] by 1, 62 points
    #lons_gcm = dataset.variables['longitude'][230:305]
    lons_gcm = dataset.variables['longitude'][235:294] 
    lats_gcm = np.array(lats_gcm)
    lons_gcm = np.array(lons_gcm)
    np.save(savepath0+'time_gcm.npy',time)
    np.save(savepath0+'lats_gcm.npy',lats_gcm)
    np.save(savepath0+'lons_gcm.npy',lons_gcm)
    np.save(savepath0+'{}gcms_{}_monthly_1by1_195001-200512_USA.npy'.format(len(filenames),variable),values_gcms)
    print('time=\n{}'.format(time))
    print('lats_gcm=\n{}'.format(lats_gcm))
    print('lons_gcm=\n{}'.format(lons_gcm))

#readGCM()



#%%
def readElevation():
    #### read the PRISM elevation data and aglined with GCM
    import numpy as np
    from netCDF4 import Dataset
    import os
    from ncdump import ncdump
    
    #%%
    #prefix = '/home/yumin/Desktop/DS/'
    #prefix = '/scratch/wang.zife/YuminLiu/'
    #datapath = prefix+'myPythonFiles/Downscaling/data/Climate/PRISMdata/raw/wcs_4km_prism.nc'
    datapath = '/home/yumin/myProgramFiles/DATA/PRISM/PRISMdata/elevation/raw/wcs_4km_prism.nc'
    savepath = '/home/yumin/myProgramFiles/DATA/PRISM/PRISMdata/elevation/processeddata/' #None
    dataset = Dataset(datapath,mode='r')
    nc_attrs, nc_dims, nc_vars = ncdump(dataset)
    
    ## orginal latitude from [49.9375N, 24.1041N], by 1/24 (~0.04), 621 points
    ## orginal longitude from [-125.021, -66.521], by 1/24 (~0.04), 1405 points
    ## orginal longitude from [234.979E, 293.479E], by 1/24 (~0.04), 1405 points
    lats_o = dataset.variables['lat'][:]
    lons_o = dataset.variables['lon'][:]+360
    elevation_o  = dataset.variables['Band1'][:]
    print('elevation_o.min()={}, elevation_o.max()={}'.format(elevation_o.min(),elevation_o.max()))
    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #plt.imshow(elevation_o)
    #plt.show()
    
    #%% align with GCM grid
    ## latitude from [49.9167N,24.125N] by 1/24 (~0.04), 620 points
    ## longitude from [235E,293.458E] by 1/24 (~0.04), 1404 points
    lats = np.zeros(len(lats_o)-1,)
    lons = np.zeros(len(lons_o)-1,)
    for i in range(len(lats_o)-1):
        lats[i] = 0.5*(lats_o[i]+lats_o[i+1])
    for i in range(len(lons_o)-1):
        lons[i] = 0.5*(lons_o[i]+lons_o[i+1])
    (Nlat,Nlon) = elevation_o.shape
    elevation = np.zeros((Nlat-1,Nlon-1)) # [620,1404]
    for i in range(Nlat-1):
        for j in range(Nlon-1):
            elevation[i,j] = 0.25*(elevation_o[i,j]+elevation_o[i+1,j]
                                      +elevation_o[i,j+1]+elevation_o[i+1,j+1])    
    if np.isnan(np.sum(elevation)):
        print("Error! nan found!\n")            
    print('elevation.min()={}, elevation.max()={}'.format(elevation.min(),elevation.max()))
    #%%
    #if savepath and not os.path.exists(savepath):
    #    os.makedirs(savepath)
    #np.save(savepath+'lons_prism.npy',lons)
    #np.save(savepath+'lats_prism.npy',lats)    
    #np.save(savepath+'prism_elevation_USA.npy',elevation)
    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #plt.imshow(elevation)
    #plt.show()
        
    #%%
    def savedata(savepath,savename,lats,lons,data):
        if savepath and not os.path.exists(savepath):
            os.makedirs(savepath)    
        #np.save(savepath+'lons_prism.npy',lons)
        #np.save(savepath+'lats_prism.npy',lats) 
        np.save(savepath+'prism_elevation_{}_USA.npy'.format(savename),data)
    ## latitude from [49.5N,24.5N]
    ## longitude from [235.5E,293.5E]
    lats_24 = lats[10:611]
    lons_24 = lons[12:]
    elevation_24 = elevation[10:611,12:] # [lat,lon]
    #elevation_24[elevation_24<0] = 0.0
    #savedata(savepath+'prism_prmean_monthly_0.04by0.04/','0.04by0.04',lats_24,lons_24,elevation_24)
    savedata(savepath,'0.04by0.04',lats_24,lons_24,elevation_24)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(elevation_24)
    #plt.savefig(savepath+'prism_prmean_monthly_0.04by0.04/elevation_prism_24.png',dpi=1200,bbox_inches='tight')
    plt.savefig(savepath+'elevation_prism_24.png',dpi=1200,bbox_inches='tight')
    #plt.show()
    
    ## GCM is 26 by 59
    ## 8x is 208 by 472
    ## 4x is 104 by 236
    ## 2x is 52 by 118
    from skimage.transform import resize
    lats_8 = np.linspace(lats_24[0],lats_24[-1],num=208)
    lons_8 = np.linspace(lons_24[0],lons_24[-1],num=472)
    elevation_8 = resize(elevation_24,(208,472),order=1,preserve_range=True)
    #elevation_8[elevation_8<0] = 0.0
    #savedata(savepath+'prism_prmean_monthly_0.125by0.125/','0.125by0.125',lats_8,lons_8,elevation_8)
    savedata(savepath,'0.125by0.125',lats_8,lons_8,elevation_8)
    #import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(elevation_8)
    #plt.savefig(savepath+'prism_prmean_monthly_0.125by0.125/elevation_prism_8.png',dpi=1200,bbox_inches='tight')
    plt.savefig(savepath+'elevation_prism_8.png',dpi=1200,bbox_inches='tight')
    #plt.show()
    
    lats_4 = np.linspace(lats_24[0],lats_24[-1],num=104)
    lons_4 = np.linspace(lons_24[0],lons_24[-1],num=236)
    elevation_4 = resize(elevation_24,(104,236),order=1,preserve_range=True)
    #elevation_4[elevation_4<0] = 0.0
    #savedata(savepath+'prism_prmean_monthly_0.25by0.25/','0.25by0.25',lats_4,lons_4,elevation_4)
    savedata(savepath,'0.25by0.25',lats_4,lons_4,elevation_4)
    #import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(elevation_4)
    #plt.savefig(savepath+'prism_prmean_monthly_0.25by0.25/elevation_prism_4.png',dpi=1200,bbox_inches='tight')
    plt.savefig(savepath+'elevation_prism_4.png',dpi=1200,bbox_inches='tight')
    #plt.show()
    
    lats_2 = np.linspace(lats_24[0],lats_24[-1],num=52)
    lons_2 = np.linspace(lons_24[0],lons_24[-1],num=118)
    elevation_2 = resize(elevation_24,(52,118),order=1,preserve_range=True)
    #elevation_2[elevation_2<0] = 0.0
    #savedata(savepath+'prism_prmean_monthly_0.5by0.5/','0.5by0.5',lats_2,lons_2,elevation_2)
    savedata(savepath,'0.5by0.5',lats_2,lons_2,elevation_2)
    #import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(elevation_2)
    #plt.savefig(savepath+'prism_prmean_monthly_0.5by0.5/elevation_prism_2.png',dpi=1200,bbox_inches='tight')
    plt.savefig(savepath+'elevation_prism_2.png',dpi=1200,bbox_inches='tight')
    #plt.show()
    
    
    lats_1 = np.linspace(lats_24[0],lats_24[-1],num=26)
    lons_1 = np.linspace(lons_24[0],lons_24[-1],num=59)
    elevation_1 = resize(elevation_24,(26,59),order=1,preserve_range=True)
    #elevation_1[elevation_1<0] = 0.0
    #savedata(savepath+'prism_prmean_monthly_1.0by1.0/','1.0by1.0',lats_1,lons_1,elevation_1)
    savedata(savepath,'1.0by1.0',lats_1,lons_1,elevation_1)
    #import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(elevation_1)
    #plt.savefig(savepath+'prism_prmean_monthly_1.0by1.0/elevation_prism_1.png',dpi=1200,bbox_inches='tight')
    plt.savefig(savepath+'elevation_prism_1.png',dpi=1200,bbox_inches='tight')
    #plt.show()

#readElevation()


#%%
def zip_to_nc(prefix='/home/yumin/Desktop/DS/DATA/',path='PRISM/monthly/',variable='tmin',start_year=1950,end_year=2005):
    '''
    read the original .zip file, unpack to .bil file, and then save to .nc file
    '''
    import os
    import zipfile
    import xarray as xr
    from readbilfile import BilFile
    #prefix = '/home/yumin/Desktop/DS/DATA/'
    #root_path = prefix+'PRISM/monthly/ppt/'
    #prefix,path,variable,start_year,end_year = '/home/yumin/Desktop/DS/DATA/','PRISM/monthly/','tmax',1950,2005
    root_path = prefix+path+variable+'/'
    
    datapaths= []
    for i, (root,dirs,files) in enumerate(os.walk(root_path)):
        #if i>=60: break
        for file in files:
            if file.endswith('all_bil.zip'):
                datapaths.append([root,file])
    datapaths = sorted(datapaths)
    #datapaths = datapaths[:1]
    #print('datapaths=\n{}'.format(datapaths))
    xray_data_all = []
    for i, (datapath,filename) in enumerate(datapaths):
        #year = 1950+i
        year = start_year+i
        if not str(year) in filename:
            print('Error: {} year not in order!'.format(year))
            
        temppath = root_path+'temppath/'+datapath.split('/')[-1]
        if not os.path.exists(temppath):
            os.makedirs(temppath)    
        with zipfile.ZipFile(datapath+'/'+filename, 'r') as zf:
            zf.extractall(temppath+'/'+filename.replace('.zip',''))    
        filepath = temppath+'/'+filename.replace('.zip','')
        
        bilhdrfiles = [f for f in os.listdir(filepath) if len(f.split('_')[-2])==6 and f.endswith('.bil')]
        bilhdrfiles = sorted(bilhdrfiles)
        #bilhdrfiles = bilhdrfiles[:1]
        #print('bilhdrfiles=\n{}'.format(bilhdrfiles))
        xray_data = []
        for bilfile in bilhdrfiles:
            b = BilFile(filepath+'/'+bilfile)
            #nodatavalue, data = b.get_array()
            
            #print('nodatavalue={}, data={}'.format(nodatavalue, data))
            
            dr = b.bil_to_xray()
            if dr.shape[1:]!=(621,1405):
                print('error: {} dr.shape not the same!'.format(dr.shape))
            #print('dr.shape={}'.format(dr.shape))
            #print('dr={}'.format(dr))
            xray_data.append(dr)
            xray_data_all.append(dr)
        if len(xray_data)!=12:
            print('Error: {} less than 12 months!'.format(bilfile))
    
        savepath = root_path+'processed/'+datapath.split('/')[-1]
        if not os.path.exists(savepath):
            os.makedirs(savepath)    
        #ds = xr.Dataset({'ppt': xr.concat(xray_data, dim='time')})
        ds = xr.Dataset({variable: xr.concat(xray_data, dim='time')})
        save_nc_file = savepath+'/'+filename.replace('_all_bil.zip','_monthly.nc')
        ds.to_netcdf(save_nc_file, format='NETCDF4')
    
    ds_all = xr.Dataset({variable: xr.concat(xray_data_all, dim='time')})
    ###save_nc_file_all = root_path+'/'+filename.replace('_2005_all_bil.zip','_1950-2005_monthly_total.nc')
    save_nc_file_all = root_path+'/'+filename.replace('_{}_all_bil.zip'.format(end_year),'_{}-{}_monthly_total.nc'.format(start_year,end_year))
    ds_all.to_netcdf(save_nc_file_all, format='NETCDF4')
                
#zip_to_nc()  
    
    
#%%
def readPRISM():
    '''
    read the .nc format PRISM data produced by zip_to_nc() function and aglined with GCM
    '''
    import numpy as np
    from netCDF4 import Dataset
    import os
    from ncdump import ncdump
    
    variable = 'tmin' #'tmax'
    #filepath = '/home/yumin/Desktop/DS/DATA/PRISM/monthly/ppt/PRISM_ppt_stable_4kmM3_1950-2005_monthly_total.nc'
    #savepath = '../data/Climate/PRISMdata/processeddata/prism_prmean_monthly_0.04by0.04_original_grid/'
    filepath = '/home/yumin/Desktop/DS/DATA/PRISM/monthly/{}/PRISM_{}_stable_4kmM3_1950-2005_monthly_total.nc'.format(variable,variable)
    savepath = '/home/yumin/Desktop/DS/DATA/PRISM/PRISMdata/{}/processeddata/prism_{}_monthly_0.04by0.04_original_grid/'.format(variable,variable)
    ##%%
    dataset = Dataset(filepath,mode='r')
    nc_attrs, nc_dims, nc_vars = ncdump(dataset)
    
    # original latitude from 49.9375N to 24.1042N by 1/24 (~0.04), 621 points
    # latitude from N to N by ,  points
    lats_o = dataset.variables['lat'][:]
    # original longitude from -125.021 to -66.5208 by 1/24 (~0.04), 1405 points
    # original longitude from 234.979E to 293.479E by 1/24 (~0.04), 1405 points
    lons_o = dataset.variables['lon'][:]+360 
    # time from 195001 to 200512 by 1, 672 points
    time = dataset.variables['time']#
    time = list(time)
    
    variable_o = dataset.variables[variable][:] #maximum temperature, [month,lat,lon], unit: degree Celsius?
    variable_month = np.ma.filled(variable_o,0)#/30.0 # daily mean precipitation, unite: mm/day
    #variable_month = variable_month[0:5,:,:]
    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #plt.imshow(variable_month[1,:,:])
    
    ##%% align with GCM grid
    ## latitude from [49.9167N,24.125N] by 1/24 (~0.04), 620 points
    ## longitude from [235E,293.458E] by 1/24 (~0.04), 1404 points
    lats = np.zeros(len(lats_o)-1,)
    lons = np.zeros(len(lons_o)-1,)
    for i in range(len(lats_o)-1):
        lats[i] = 0.5*(lats_o[i]+lats_o[i+1])
    for i in range(len(lons_o)-1):
        lons[i] = 0.5*(lons_o[i]+lons_o[i+1])
    (Nmon,Nlat,Nlon) = variable_month.shape
    variable_month_prism = np.zeros((Nmon,Nlat-1,Nlon-1)) # [672, 620, 1404]
    for t in range(Nmon):
        for i in range(Nlat-1):
            for j in range(Nlon-1):
                variable_month_prism[t,i,j] = 0.25*(variable_month[t,i,j]+variable_month[t,i+1,j]
                                          +variable_month[t,i,j+1]+variable_month[t,i+1,j+1])
    
    variable_month_prism[variable_month_prism==0] = 0
    #if np.isnan(np.sum(prmean_month_prism)):
    #    print("Error! nan found!\n")            
    
    ##%%
    if savepath and not os.path.exists(savepath):
        os.makedirs(savepath)
    np.save(savepath+'lons_prism.npy',lons)
    np.save(savepath+'lats_prism.npy',lats) 
    np.save(savepath+'time_prism.npy',time)    
    np.save(savepath+'prism_{}_monthly_195001-200512_USA.npy'.format(variable),variable_month_prism)
    print('time=\n{}'.format(time))
    print('lats=\n{}'.format(lats))
    print('lons=\n{}'.format(lons))
#    import matplotlib.pyplot as plt
#    fig = plt.figure()
#    plt.imshow(variable_month_prism[7,:,:])
#    plt.show()

#readPRISM()

#%% downsample to 8x, 4x and 2x resolutions with respect to GCM resolution
def downsamplePRISM():
    import numpy as np
    import os
    
    variable = 'tmin' #'tmax'
    def savedata(savepath,savename,lats,lons,time,data,variable):
        if savepath and not os.path.exists(savepath):
            os.makedirs(savepath)    
        np.save(savepath+'lons_prism.npy',lons)
        np.save(savepath+'lats_prism.npy',lats) 
        np.save(savepath+'time_prism.npy',time)    
        np.save(savepath+'prism_{}_monthly_{}_195001-200512_USA.npy'.format(variable,savename),data)
    
    prefix = '/home/yumin/Desktop/DS/'
    #prefix = '/scratch/wang.zife/YuminLiu/'
    #datapath = prefix+'myPythonFiles/Downscaling/data/Climate/PRISMdata/processeddata/prism_prmean_monthly_0.04by0.04_original_grid/'
    datapath = prefix+'DATA/PRISM/PRISMdata/{}/processeddata/prism_{}_monthly_0.04by0.04_original_grid/'.format(variable,variable)
    savepath = prefix+'DATA/PRISM/PRISMdata/{}/processeddata/'.format(variable)
    lats = np.load(datapath+'lats_prism.npy')
    lons = np.load(datapath+'lons_prism.npy')
    time = np.load(datapath+'time_prism.npy')
    variable_month_prism = np.load(datapath+'prism_{}_monthly_195001-200512_USA.npy'.format(variable))
    
    ## latitude from [49.5N,24.5N]
    ## longitude from [235.5E,293.5E]
    ### longitude from [235.5E,292.5E]
    lats_24 = lats[10:611]
    lons_24 = lons[12:]
    print('lats_24 in [{},{}]'.format(min(lats_24),max(lats_24)))
    print('lons_24 in [{},{}]'.format(min(lons_24),max(lons_24)))
    variable_month_prism_24 = np.transpose(variable_month_prism[:,10:611,12:],axes=(1,2,0)) # [lat,lon,month]
    #lons_8 = lons[12:1381]
    #variable_month_prism_24 = variable_month_prism[:,10:611,12:1381]
    del lats, lons, variable_month_prism
     
    savedata(savepath+'prism_{}_monthly_0.04by0.04/'.format(variable),'0.04by0.04',lats_24,lons_24,time,
             np.transpose(variable_month_prism_24,axes=(2,0,1)),variable=variable) #[month,lat,lon]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(np.sum(variable_month_prism_24,axis=2))
    plt.savefig(savepath+'prism_{}_monthly_0.04by0.04/{}_month_prism_24.png'.format(variable,variable),dpi=1200,bbox_inches='tight')
    #plt.show()
    
    ## GCM is 26 by 59
    ## 8x is 208 by 472
    ## 4x is 104 by 236
    ## 2x is 52 by 118
    from skimage.transform import resize
    lats_8 = np.linspace(lats_24[0],lats_24[-1],num=208)
    lons_8 = np.linspace(lons_24[0],lons_24[-1],num=472)
    variable_month_prism_8 = resize(variable_month_prism_24,(208,472),order=1,preserve_range=True) #[lat,lon,month]
    variable_month_prism_8 = np.transpose(variable_month_prism_8,axes=(2,0,1)) #[month,lat,lon]
    savedata(savepath+'prism_{}_monthly_0.125by0.125/'.format(variable),'0.125by0.125',lats_8,lons_8,time,variable_month_prism_8,variable=variable)
    #import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(np.sum(variable_month_prism_8,axis=0))
    plt.savefig(savepath+'prism_{}_monthly_0.125by0.125/{}_month_prism_8.png'.format(variable,variable),dpi=1200,bbox_inches='tight')
    #plt.show()
    del variable_month_prism_8
    
    lats_4 = np.linspace(lats_24[0],lats_24[-1],num=104)
    lons_4 = np.linspace(lons_24[0],lons_24[-1],num=236)
    variable_month_prism_4 = resize(variable_month_prism_24,(104,236),order=1,preserve_range=True) #[lat,lon,month]
    variable_month_prism_4 = np.transpose(variable_month_prism_4,axes=(2,0,1)) #[month,lat,lon]
    savedata(savepath+'prism_{}_monthly_0.25by0.25/'.format(variable),'0.25by0.25',lats_4,lons_4,time,variable_month_prism_4,variable=variable)
    #import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(np.sum(variable_month_prism_4,axis=0))
    plt.savefig(savepath+'prism_{}_monthly_0.25by0.25/{}_month_prism_4.png'.format(variable,variable),dpi=1200,bbox_inches='tight')
    #plt.show()
    del variable_month_prism_4
    
    lats_2 = np.linspace(lats_24[0],lats_24[-1],num=52)
    lons_2 = np.linspace(lons_24[0],lons_24[-1],num=118)
    variable_month_prism_2 = resize(variable_month_prism_24,(52,118),order=1,preserve_range=True) #[lat,lon,month]
    variable_month_prism_2 = np.transpose(variable_month_prism_2,axes=(2,0,1)) #[month,lat,lon]
    savedata(savepath+'prism_{}_monthly_0.5by0.5/'.format(variable),'0.5by0.5',lats_2,lons_2,time,variable_month_prism_2,variable=variable)
    #import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(np.sum(variable_month_prism_2,axis=0))
    plt.savefig(savepath+'prism_{}_monthly_0.5by0.5/{}_month_prism_2.png'.format(variable,variable),dpi=1200,bbox_inches='tight')
    #plt.show()
    del variable_month_prism_2
    
    lats_1 = np.linspace(lats_24[0],lats_24[-1],num=26)
    lons_1 = np.linspace(lons_24[0],lons_24[-1],num=59)
    variable_month_prism_1 = resize(variable_month_prism_24,(26,59),order=1,preserve_range=True) #[lat,lon,month]
    variable_month_prism_1 = np.transpose(variable_month_prism_1,axes=(2,0,1)) #[month,lat,lon]
    savedata(savepath+'prism_{}_monthly_1.0by1.0/'.format(variable),'1.0by1.0',lats_1,lons_1,time,variable_month_prism_1,variable=variable)
    #import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(np.sum(variable_month_prism_1,axis=0))
    plt.savefig(savepath+'prism_{}_monthly_1.0by1.0/{}_month_prism_1.png'.format(variable,variable),dpi=1200,bbox_inches='tight')
    #plt.show()
    del variable_month_prism_1

#downsamplePRISM()

#%%
def get_climatology():
    import numpy as np
    
    variable = 'tmin' #'tmax'
    resolution = '0.125'
    prefix = '/home/yumin/Desktop/DS/'
    datapath = prefix+'DATA/PRISM/PRISMdata/{}/processeddata/prism_{}_monthly_{}by{}/'.format(variable,variable,resolution,resolution)
    filename = 'prism_{}_monthly_{}by{}_195001-200512_USA.npy'.format(variable,resolution,resolution)
    savepath = datapath
    savename = filename.replace('200512','199912_month_1to600_climatology')
    data = np.load(datapath+filename) # [Nmonth,Nlat,Nlon]
    Ntrain = 600
    train_data = data[:Ntrain,:,:]
    climatology = []
    for mon in range(12):
        monthdata = train_data[mon::12,:,:]
        mean = np.mean(monthdata,axis=0) #[Nlat,Nlon]
        climatology.append(mean)
        
    climatology = np.stack(climatology,axis=0)
    np.save(savepath+savename,climatology)  

#get_climatology()
             
#%%
def mergexy(resolution='0.125'):
    import os
    import numpy as np
    #import seaborn as sns
    #import matplotlib.pyplot as plt
    from skimage.transform import resize
    
    variable = 'tmin' #'tmax'
    #variable2 = 'tasmax'
    #resolution = '0.5'
    prefix = '/home/yumin/Desktop/DS/DATA/'
    #prefix = '/scratch/wang.zife/YuminLiu/'
    #prefix = '/home/yumin/myProgramFiles/DATA/'
    gcmpathname = prefix+'GCM/GCMdata/tasmin/processeddata_26by59_points/33gcms_tasmin_monthly_1by1_195001-200512_USA.npy'
    
    climatologypath = prefix+'PRISM/PRISMdata/{}/processeddata/prism_{}_monthly_{}by{}/'.format(variable,variable,resolution,resolution)
    climatologyname = 'prism_{}_monthly_{}by{}_195001-199912_month_1to600_climatology_USA.npy'.format(variable,resolution,resolution)
    prismpath = prefix+'PRISM/PRISMdata/{}/processeddata/prism_{}_monthly_{}by{}/'.format(variable,variable,resolution,resolution)
    prismname = 'prism_{}_monthly_{}by{}_195001-200512_USA.npy'.format(variable,resolution,resolution)
    elevationpath = prefix+'PRISM/PRISMdata/elevation/processeddata/'
    elevationname = 'prism_elevation_{}by{}_USA.npy'.format(resolution,resolution)
    savepath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/'.format(variable,resolution,resolution)
    
    gcmdata = np.load(gcmpathname) # [Ngcm,Nmon,Nlat,Nlon], [-29.03,49.94], #[0.0, 32.99746322631836]
    climatologydata = np.load(climatologypath+climatologyname) # [12,Nlat,Nlon], [-12.287,45.407]
    prismdata = np.load(prismpath+prismname) # [Nmon,Nlat,Nlon] [-22.17,48.25], [0.0, 67.89530826556293]
    elevation = np.load(elevationpath+elevationname) #[Nlat,Nlon] [-74.09817504882812, 3677.848941389869]
    
    print('gcmdata.shape={}'.format(gcmdata.shape))
    print('prismdata.shape={}'.format(prismdata.shape))
    print('elevation.shape={}'.format(elevation.shape))
    print('gcmdata.min()={}, gcmdata.max()={}'.format(gcmdata.min(), gcmdata.max()))
    print('prismdata.min()={}, prismdata.max()={}'.format(prismdata.min(), prismdata.max()))
    print('elevation.min()={}, elevation.max()={}'.format(elevation.min(), elevation.max()))
    
    MaskMapUSA = np.sum(abs(prismdata),axis=0) #[Nlat,Nlon]
    MaskMapUSA[MaskMapUSA==0] = 0.0
    MaskMapUSA[MaskMapUSA!=0] = 1.0 #[0.0, 1.0]
    MaskMapUSAsmall = resize(image=MaskMapUSA,output_shape=gcmdata.shape[2:],preserve_range=True)
    MaskMapUSAsmall[MaskMapUSAsmall==0] = 0.0
    MaskMapUSAsmall[MaskMapUSAsmall!=0] = 1.0 #[0.0, 1.0]
    print('MaskMapUSA.shape={}'.format(MaskMapUSA.shape))
    print('MaskMapUSAsmall.shape={}'.format(MaskMapUSAsmall.shape))
    print('MaskMapUSA.min()={}, max()={}'.format(MaskMapUSA.min(),MaskMapUSA.max()))
    print('MaskMapUSAsmall.min()={}, max()={}'.format(MaskMapUSAsmall.min(),MaskMapUSAsmall.max()))
    
    elevation[elevation<=0] = 0.0 #[Nlat,Nlon] [0.0,3674.64], [0.0, 3677.848941389869]
    elevation[MaskMapUSA==0] = 0.0 # [0.0, 3677.848941389869]
    elevation = elevation[np.newaxis,...] #[1,Nlat,Nlon]
    elevation = np.log(1.0+elevation)/10.0 # [0.0, 8.210] --> [0.0,0.82]
    
    if savepath and not os.path.exists(savepath):
        os.makedirs(savepath)
    #np.savez(savepath+'MaskMapUSA_0.125by0.125_MaskMapUSA_1.0by1.0_elevation_0.125by0.125.npz',
    #         MaskMapUSA=MaskMapUSA,MaskMapUSAsmall=MaskMapUSAsmall,elevation=elevation)
    for mon in range(prismdata.shape[0]):
        gcms = gcmdata[:,mon,:,:] #[Ngcm,Nlat,Nlon], unit: Celsius
        prism = prismdata[mon,:,:] #[Nlat,Nlon], unit: Celsius    
        for ngcm in range(gcms.shape[0]):
            gcms[ngcm,:,:][MaskMapUSAsmall==0] = 0.0
        prism[MaskMapUSA==0] = 0.0
        prism = prism[np.newaxis,...] #[1,Nlat,Nlon], Celsius
        climatology = climatologydata[[mon%12],:,:] #[1,Nlat,Nlon]
        #gcms = np.log(1.0+gcms) ##[Ngcm,Nlat,Nlon], [0,3.737655630962239]
        #prism = np.log(1.0+prism) ##[1,Nlat,Nlon],[0,4.159132957458496]    
        gcms = gcms/50.0 ##[Ngcm,Nlat,Nlon], [-0.5806386947631836, 0.9989141845703124]
        prism = prism/50.0 ##[1,Nlat,Nlon],[-0.4434135270901671, 0.9651596902285366] 
        climatology = climatology/50.0 #[1,Nlat,Nlon],[-0.24574000000000001, 0.90814]
        #print('gcms.max={}'.format(max(gcms.flatten())))
        #print('y.max={}'.format(max(y.flatten())))
     
        if mon<9:
            savename = 'prism_gcm_divide50_{}_monthly_{}to1.0_195001-200512_USA_month00{}'.format(variable,resolution,mon+1)
        elif mon<99:
            savename = 'prism_gcm_divide50_{}_monthly_{}to1.0_195001-200512_USA_month0{}'.format(variable,resolution,mon+1)
        else:
            savename = 'prism_gcm_divide50_{}_monthly_{}to1.0_195001-200512_USA_month{}'.format(variable,resolution,mon+1)
        np.savez(savepath+savename+'.npz',gcms=gcms,prism=prism,elevation=elevation,climatology=climatology)
        
        # if mon==0:
        #     fig = plt.figure()
        #     sns.distplot(gcms.flatten())
        #     plt.title('GCM')
        #     plt.show()
        #     fig = plt.figure()
        #     sns.distplot(prism.flatten())
        #     plt.title('PRISM')
        #     plt.show()

#mergexy()        
        

#%% main
if __name__=='__main__':
    ##zip_to_nc()
    ##readPRISM()
    #print('processing downsamplePRISM...')
    #downsamplePRISM()
    #print('downsamplePRISM Done!')
    #print('processing readElevation...')
    #readElevation()
    #print('readElevation Done!')
    ##readGCM()
    
    resolutions = ['0.125','0.25','0.5']
    #resolutions = ['0.5']
    for res in resolutions:
        print('processing mergexy(resolution={})'.format(res))
        mergexy(resolution=res)
        print('mergexy(resolution={}) Done!'.format(res))
    
    print('All Jobs Done!')
