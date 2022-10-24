#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:04:08 2019

@author: tbrivoal
"""
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import cartopy.feature as cfeature


import matplotlib.pyplot as plt
import netCDF4 as nc
from matplotlib import pyplot
from matplotlib import colors as mcolors
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkblue","blue","white","red","firebrick"])
darkred = plt.cm.Reds(np.linspace(0.99, 1, 2))
darkblue = plt.cm.YlGnBu(np.linspace(0.99, 1, 2))
bwr_cmap1 = cmap(np.linspace(0, 0.5, 252))
bwr_cmap2 = cmap(np.linspace(0.5, 1, 252))
w_cmap = cmap(np.linspace(0.5, 0.5, 78))
# bwr_cmap2 = plt.cm.(np.linspace(0.5, 1, 126))
# combine them and build a new colormap
colors = np.vstack((darkblue,bwr_cmap1,w_cmap,bwr_cmap2,darkred))
mymap =mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


Ufile_NEMO_4_0 = xr.open_dataset('FLDR/CFG_NM_1d_curl.nc', chunks={'time_counter': 50})
Coordfile = xr.open_dataset('FLDR/domain_cfg.nc', drop_variables={"x", "y",})

Curl_U=Ufile_NEMO_4_0.socurl.squeeze().sel(time_counter=slice('DSTART', 'DEND')).mean(dim='time_counter').values

matplotlib.rcParams.update({'font.size': 10})
proj=ccrs.Mercator()
plt.figure(figsize=(8,8))
ax = plt.subplot(111, projection=proj)
lat=Coordfile.gphit.squeeze().values
lon=Coordfile.glamt.squeeze().values

ax.coastlines(resolution='50m')
lon_formatter = LongitudeFormatter(degree_symbol='° ')
lat_formatter = LatitudeFormatter(degree_symbol='° ')
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
# ax.stock_img()
land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['land'])
ax.add_feature(land_50m)
axes=ax.gridlines( draw_labels=False, linewidth=0)
axes.ylabels_right = False
axes.xlabels_top = False

im1 = plt.contourf(lon, lat, Curl_U ,levels=np.linspace(-1e-4,1e-4,51), cmap=mymap,transform=ccrs.PlateCarree(),extend='both')
cbar=plt.colorbar(ax=ax,ticks=np.linspace(-1e-4,1e-4,11),orientation="vertical") #ticks=[-40,20,0,20,40]) #ticks=[0,1,2,3]) #
cbar.set_label('Vorticity (1/s)', rotation=270, labelpad=12, fontsize=12,fontweight="bold")
########
#

plt.savefig('CFG_NM_VOR_moyenne_VER_DSTART_DEND.png')

                                                   
    
     
