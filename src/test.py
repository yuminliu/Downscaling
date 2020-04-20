



import numpy as np
datapath = '../data/Climate/PRISM_GCMdata/ppt/0.5by0.5/test/prism_gcm_log1p_prmean_monthly_0.5to1.0_195001-200512_USA_month637.npz'
datapath = '/home/yumin/DS//DATA/GCM/GCMdata/ppt/35channels/processeddata_26by59_points/lons_gcm.npy'
data = np.load(datapath)
#gcm = data['gcms']

















# =============================================================================
# import os
# import numpy as np
# datapath = '../data/Climate/PRISM_GCMdata/tmin/0.5by0.5/train/'
# train_filenames = [f for f in os.listdir(datapath) if f.endswith('.npz')]
# train_filenames = sorted(train_filenames)
# 
# gcms_train = []
# prism_train = []
# for filename in train_filenames:
#     data = np.load(datapath+filename)
#     gcms_train.append(data['gcms']) # [Ngcm,Nlat,Nlon]
#     prism_train.append(np.squeeze(data['prism'])) # [1,Nlat,Nlon] --> [Nlat,Nlon]
# gcms_train = np.stack(gcms_train,axis=1) #[Ngcm,Nlat,Nlon] --> [Ngcm,Nmon,Nlat,Nlon]
# prism_train = np.stack(prism_train,axis=0) # [Nlat,Nlon] --> [Nmon,Nlat,Nlon]    
# print('gcms_train.shape={}\nprism_train.shape={}'.format(gcms_train.shape,prism_train.shape))
# mask = np.sum(prism_train,axis=0)
# import matplotlib.pyplot as plt
# fig = plt.figure()
# plt.imshow(mask)
# 
# =============================================================================












# =============================================================================
# '''
# obs: observed values, 1d array, [Ntrain,]
# model: model values, 1d array, [Ntrain,]
# output: cdfs of observed and model variables, and bins, 1d arrays
# '''
# import numpy as np
# obs = np.arange(-50,100)
# model = np.arange(0,150)
# nbin = 20
# if len(model)>len(obs):
#     model = model[:len(obs)] # should just use training data
# max_value = max(np.amax(obs),np.amax(model))
# width = max_value/nbin
# xbins = np.arange(0.0,max_value+width,width)
# # create PDF
# pdfobs, _ = np.histogram(obs,bins=xbins)
# pdfmodel, _ = np.histogram(model,bins=xbins)
# # create CDF with zero in first entry.
# cdfobs = np.insert(np.cumsum(pdfobs),0,0.0)
# cdfmodel = np.insert(np.cumsum(pdfmodel),0,0.0)
#     
#     
#     
#     
#     
# import numpy as np
# obs2 = np.arange(-50,100)
# model2 = np.arange(0,150)
# nbin2 = 20
# if len(model2)>len(obs2):
#     model2 = model2[:len(obs2)] # should just use training data
# max_value2 = max(np.amax(obs2),np.amax(model2))
# width2 = max_value2/nbin2
# xbins2 = np.arange(-50,max_value2+width2,width2)
# # create PDF
# pdfobs2, _ = np.histogram(obs2,bins=xbins2)
# pdfmodel2, _ = np.histogram(model2,bins=xbins2)
# # create CDF with zero in first entry.
# cdfobs2 = np.insert(np.cumsum(pdfobs2),0,0.0)
# cdfmodel2 = np.insert(np.cumsum(pdfmodel2),0,0.0)    
#     
#     
#     
#     
#     
# import matplotlib.pyplot as plt
# #rng = np.random.RandomState(10)  # deterministic random data
# #a = np.hstack((rng.normal(size=1000),rng.normal(loc=5, scale=2, size=1000)))
# #_ = plt.hist(a, bins='auto')  # arguments are passed to np.histogram
# #plt.title("Histogram with 'auto' bins")
# #Text(0.5, 1.0, "Histogram with 'auto' bins")
# fig = plt.figure()
# plt.hist(obs,bins=xbins)
# plt.hist(model,bins=xbins)
# fig = plt.figure()
# plt.hist(obs2,bins=xbins2)
# plt.hist(model2,bins=xbins2)
# plt.show()    
# =============================================================================
    
    
    
    
    
# =============================================================================
# import numpy as np
# 
# path = '/home/yumin/DS/DATA/GCM/GCMdata/ppt/35channels/processeddata_26by59_points/'
# latsname = 'lats_gcm.npy'
# lonsname = 'lons_gcm.npy'
# 
# lats = np.load(path+latsname)
# lons = np.load(path+lonsname)
# 
# =============================================================================


# =============================================================================
# # Draw the locations of cities on a map of the US
# import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
# from geopy.geocoders import Nominatim
# import math
# 
# cities = [["Chicago",10],
#           ["Boston",10],
#           ["New York",5],
#           ["San Francisco",25]]
# scale = 5
# 
# map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
#         projection='lcc',lat_1=32,lat_2=45,lon_0=-95)
# 
# # load the shapefile, use the name 'states'
# map.readshapefile('../data/Climate/map/st99_d00', name='states', drawbounds=True)
# 
# # Get the location of each city and plot it
# #geolocator = Nominatim()
# #for (city,count) in cities:
# #    loc = geolocator.geocode(city)
# #    x, y = map(loc.longitude, loc.latitude)
# #    map.plot(x,y,marker='o',color='Red',markersize=int(math.sqrt(count))*scale)
# #plt.show()
# =============================================================================


#%%
# =============================================================================
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap as Basemap
# from matplotlib.colors import rgb2hex
# from matplotlib.patches import Polygon
# # Lambert Conformal map of lower 48 states.
# m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
#         projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
# # draw state boundaries.
# # data from U.S Census Bureau
# # http://www.census.gov/geo/www/cob/st2000.html
# shp_info = m.readshapefile('../data/Climate/map/st99_d00','states',drawbounds=True)
# # population density by state from
# # http://en.wikipedia.org/wiki/List_of_U.S._states_by_population_density
# popdensity = np.load('../data/Climate/map/state_popdensity.npy',allow_pickle=True).item()
# 
# # choose a color for each state based on population density.
# colors={}
# statenames=[]
# cmap = plt.cm.hot # use 'hot' colormap
# vmin = 0; vmax = 450 # set range.
# for shapedict in m.states_info:
#     statename = shapedict['NAME']
#     # skip DC and Puerto Rico.
#     if statename not in ['District of Columbia','Puerto Rico']:
#         pop = popdensity[statename]
#         # calling colormap with value between 0 and 1 returns
#         # rgba value.  Invert color range (hot colors are high
#         # population), take sqrt root to spread out colors more.
#         colors[statename] = cmap(1.-np.sqrt((pop-vmin)/(vmax-vmin)))[:3]
#     statenames.append(statename)
# # cycle through state names, color each one.
# ax = plt.gca() # get current axes instance
# for nshape,seg in enumerate(m.states):
#     # skip DC and Puerto Rico.
#     if statenames[nshape] not in ['District of Columbia','Puerto Rico']:
#         #color = rgb2hex(colors[statenames[nshape]]) 
#         #poly = Polygon(seg,facecolor=color,edgecolor=color)
#         poly = Polygon(seg,facecolor='w',edgecolor='k')
#         ax.add_patch(poly)
# plt.title('Filling State Polygons by Population Density')
# plt.show()
# =============================================================================
