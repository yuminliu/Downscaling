#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:18:44 2019

@author: yumin
"""

import gdal
import gdalconst
import numpy as np
import xarray as xr
import pandas as pd
class BilFile(object):

    def __init__(self, bil_file):
        self.bil_file = bil_file
        self.hdr_file = bil_file.split('.')[0]+'.hdr'
   
        self.time = bil_file.split('_')[-2]
        #self.time = pd.to_datetime(bil_file.split('_')[-2], format='%Y%m')
        #self.nodatavalue, self.data = None, None
        #gdal.GetDriverByName('EHdr').Register()
        img = gdal.Open(self.bil_file, gdalconst.GA_ReadOnly)
        band = img.GetRasterBand(1)
        self.nodatavalue = band.GetNoDataValue()
        self.ncol = img.RasterXSize
        self.nrow = img.RasterYSize
        geotransform = img.GetGeoTransform()
        self.originX = geotransform[0]
        self.originY = geotransform[3]
        self.pixelWidth = geotransform[1]
        self.pixelHeight = geotransform[5]
        self.data = band.ReadAsArray()
        self.data = np.ma.masked_where(self.data==self.nodatavalue, self.data)
            
        #print('self.time={}\nself.nodatavalue={}\nself.ncol={}\nself.nrow={}\nself.originX={}\nself.originY={}\nself.pixelWidth={}\nself.pixelHeight={}\nself.data={}\n'
        #      .format(self.time,self.nodatavalue,self.ncol,self.nrow,self.originX,self.originY,self.pixelWidth,self.pixelHeight,self.data))    
            
  
    def get_array(self, mask=None):         
        if mask is not None:
            self.data = np.ma.masked_where(mask==True, self.data)            
        return self.nodatavalue, self.data
   
    
    
    def bil_to_xray(self):
       lats = np.linspace(self.originY, self.originY + self.pixelHeight *(self.nrow - 1),self.nrow)
       lons = np.linspace(self.originX, self.originX + self.pixelWidth * (self.ncol -1),self.ncol)
       dr = xr.DataArray(self.data[np.newaxis, :, :],coords=dict(time=[self.time], lat=lats, lon=lons),dims=['time', 'lat', 'lon'])          
       return dr




#import pandas as pd
#a = pd.to_datetime('195001', format='%Y%m')
#print('a={}'.format(a))


   