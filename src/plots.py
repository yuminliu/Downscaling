#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:06:31 2020

@author: yumin
"""

def plot_map(img,title=None,savepath=None,savename=None,cmap='YlOrRd',
             lonlat=[235.5,24.5,293.5,49.5],resolution='i',area_thresh=10000,clim=None):
    import numpy as np
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    #lonlat=[235,24.125,293.458,49.917]
    #lonlat=[235.5,24.5,293.5,49.5]
    #area_thresh=10000
    lats = np.arange(20.0,51.0,5.0)
    lons = np.arange(235.0,300.0,10.0)
    fig = plt.figure()
    m = Basemap(llcrnrlon=lonlat[0],llcrnrlat=lonlat[1],urcrnrlon=lonlat[2],urcrnrlat=lonlat[3],
                projection='cyl',resolution='i',area_thresh=area_thresh)
    m.drawcoastlines(linewidth=1.0)
    m.drawcountries(linewidth=1.0)
    m.drawstates()
    
    m.drawparallels(lats,labels=[True,False,False,False],dashes=[1,2])
    m.drawmeridians(lons,labels=[False,False,False,True],dashes=[1,2])
    #m.imshow(np.flipud(np.sqrt(pred)),alpha=1.0)
    #m.imshow(img,alpha=1.0)
    m.imshow(img,cmap=cmap,alpha=1.0)
    plt.colorbar(fraction=0.02)
    if clim:
        plt.clim(clim)
    
    if title:
        plt.title(title)
    if savepath and savename:
        plt.savefig(savepath+savename+'.jpg',dpi=1200,bbox_inches='tight')
    plt.show()