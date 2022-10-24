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

import matplotlib.pyplot as plt
import netCDF4 as nc
from matplotlib import pyplot
from matplotlib import colors as mcolors
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


Tfile_VER1 = xr.open_dataset('FLDR_VERSION1/CFG_NM_1d_gridT_OSTIA.nc', chunks={'time_counter': 50})
Tfile_VER2 = xr.open_dataset('FLDR_VERSION2/CFG_NM_1d_gridT_OSTIA.nc', chunks={'time_counter': 50})
Tfile_OSTIA = xr.open_dataset('/scratch/work/brivoalt/DATA/SST/OSTIA_SST_ORIGIN_y2017-2018_eNEATL36_1m.nc')

T_VER1=Tfile_VER1.votemper.squeeze().sel(time_counter=slice('DSTART', 'DEND')).mean(dim='time_counter')
T_VER2=Tfile_VER2.votemper.squeeze().sel(time_counter=slice('DSTART', 'DEND')).mean(dim='time_counter')
T_OSTIA=Tfile_OSTIA.analysed_sst.squeeze().sel(time=slice('DSTART', 'DEND')).mean(dim='time') - 273.15

matplotlib.rcParams.update({'font.size': 10})
proj=ccrs.Mercator()
plt.figure(figsize=(16,8))
ax = plt.subplot(133, projection=proj)
lat_tmp=Tfile_VER1.lat
lon_tmp=Tfile_VER1.lon
lon,lat = np.meshgrid(lon_tmp,lat_tmp)

ax.coastlines(resolution='50m')
ax.set_extent((-20.0, 16.5, 25.0, 63.5))
lon_formatter = LongitudeFormatter(degree_symbol='° ')
lat_formatter = LatitudeFormatter(degree_symbol='° ')
ax.set_xticks([ -20 , -10 , 0 , 10  ], crs=ccrs.PlateCarree())
ax.set_yticks([ 30, 40, 50, 60], crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

axes=ax.gridlines( draw_labels=False, linewidth=0)
axes.ylabels_right = False
axes.xlabels_top = False

im1 = plt.contourf(lon, lat, T_VER2 - T_OSTIA ,levels=np.linspace(-0.7,0.7,51), cmap=mymap,transform=ccrs.PlateCarree(),extend='both')
cbar=plt.colorbar(ax=ax,ticks=[np.linspace(-0.7,0.7,11)],orientation="vertical",fraction=0.046, pad=0.04) #ticks=[-40,20,0,20,40]) #ticks=[0,1,2,3]) #
cbar.set_label('SST difference (°C)', rotation=270, labelpad=12, fontsize=12,fontweight="bold")
ax.set_title('VER2 - OSTIA')

ax = plt.subplot(131, projection=proj)

ax.coastlines(resolution='50m')
ax.set_extent((-20.0, 16.5, 25.0, 63.5))
lon_formatter = LongitudeFormatter(degree_symbol='° ')
lat_formatter = LatitudeFormatter(degree_symbol='° ')
ax.set_xticks([ -20 , -10 , 0 , 10  ], crs=ccrs.PlateCarree())
ax.set_yticks([ 30, 40, 50, 60], crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

axes=ax.gridlines( draw_labels=False, linewidth=0)
axes.ylabels_right = False
axes.xlabels_top = False

im1 = plt.contourf(lon, lat, T_OSTIA  ,levels=np.linspace(5,23,51), cmap="jet",transform=ccrs.PlateCarree(),extend='both')
cbar=plt.colorbar(ax=ax,ticks=[np.linspace(5,23,11)],orientation="vertical",fraction=0.046, pad=0.04) #ticks=[-40,20,0,20,40]) #ticks=[0,1,2,3]) #
cbar.set_label('SST OSTIA (°C)', rotation=270, labelpad=12, fontsize=12,fontweight="bold")
ax.set_title('VER2')

ax = plt.subplot(132, projection=proj)

ax.coastlines(resolution='50m')
ax.set_extent((-20.0, 16.5, 25.0, 63.5))
lon_formatter = LongitudeFormatter(degree_symbol='° ')
lat_formatter = LatitudeFormatter(degree_symbol='° ')
ax.set_xticks([ -20 , -10 , 0 , 10  ], crs=ccrs.PlateCarree())
ax.set_yticks([ 30, 40, 50, 60], crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

axes=ax.gridlines( draw_labels=False, linewidth=0)
axes.ylabels_right = False
axes.xlabels_top = False
im1 = plt.contourf(lon, lat, T_VER1 - T_OSTIA,levels=np.linspace(-0.7,0.7,51), cmap=mymap,transform=ccrs.PlateCarree(),extend='both')
cbar=plt.colorbar(ax=ax,ticks=[np.linspace(-0.7,0.7,11)],orientation="vertical",fraction=0.046, pad=0.04) #ticks=[-40,20,0,20,40]) #ticks=[0,1,2,3]) #
cbar.set_label('SST (°C)', rotation=270, labelpad=12, fontsize=12,fontweight="bold")
ax.set_title('VER1 - OSTIA')
#plt.tight_layout()
plt.savefig('map_SST_moyenne_OSTIA_VER2_m_VER1_DSTART_DEND.png')
plt.close()
    
     
