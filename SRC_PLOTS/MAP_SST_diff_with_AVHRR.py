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
import cmocean
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
print("imports OK")

#Tfile_VER = xr.open_dataset('FLDR/CFG_NM_1d_grid2D_sst.nc', chunks={'time_counter': 50})
obs_file=xr.open_mfdataset('/scratch/work/legalloudeco/PERMANENT_SPACE/OBSERVATIONS/SST/AVHRR/ORIGIN_GRID/dt/avhrr-only-v2.MONTH_ana*').sel(time=slice('DSTART', 'DEND'))
mod_file=xr.open_dataset('FLDR/CFG_NM_1d_gridT_AVHRR.nc')
sst_mod=mod_file.votemper.sel(time_counter=slice('DSTART', 'DEND')).mean(dim='time_counter').squeeze()
sst_obs=obs_file.sst.squeeze()[:,:].mean(dim='time').squeeze()
sst_obs=sst_obs.where(abs(sst_obs.values) <100, np.nan ) 
diff = sst_mod - sst_obs
rms = xr.ufuncs.sqrt((sst_mod - sst_obs)**2)
print('compute OK')
matplotlib.rcParams.update({'font.size': 10})
proj=ccrs.Mercator()
plt.figure(figsize=(8,8))
lat1D=obs_file.lat.squeeze().values
lon1D=obs_file.lon.squeeze().values
lon,lat = np.meshgrid(lon1D,lat1D,indexing='ij')
ax = plt.subplot(111, projection=proj)

lon_formatter = LongitudeFormatter(degree_symbol='° ')
lat_formatter = LatitudeFormatter(degree_symbol='° ')
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

axes=ax.gridlines( draw_labels=False, linewidth=0)
axes.ylabels_right = False
axes.xlabels_top = False
ax.set_extent((-21.0, 16.0, 25.0, 63.5))
mymap.set_bad('grey')
im1 = plt.contourf(lon, lat, diff.T ,levels=np.linspace(-1,1,51), cmap=mymap,transform=ccrs.PlateCarree(),extend='both')
cbar=plt.colorbar(ax=ax,ticks=np.linspace(-1,1,11),orientation="vertical",fraction=0.046, pad=0.04) #ticks=[-40,20,0,20,40]) #ticks=[0,1,2,3]) #
cbar.set_label('SST (°C)', rotation=270, labelpad=12, fontsize=12,fontweight="bold")
im1 = plt.contour(lon, lat, diff.T ,levels=np.linspace(-1,1,11), colors='k',linewidths=0.5,transform=ccrs.PlateCarree())
ax.coastlines(resolution='50m')

ax.set_title('VER - OBS')
plt.tight_layout()
plt.savefig('SST_moyenne_VER_vs_OBS_DIFF_AVHRR_DSTART_DEND.png',dpi=400)
plt.close()
    

plt.figure(figsize=(8,8))
lat1D=obs_file.lat.squeeze().values
lon1D=obs_file.lon.squeeze().values
lon,lat = np.meshgrid(lon1D,lat1D,indexing='ij')
ax = plt.subplot(111, projection=proj)

lon_formatter = LongitudeFormatter(degree_symbol='° ')
lat_formatter = LatitudeFormatter(degree_symbol='° ')
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

axes=ax.gridlines( draw_labels=False, linewidth=0)
axes.ylabels_right = False
axes.xlabels_top = False
ax.set_extent((-21.0, 16.0, 25.0, 63.5))
mymap.set_bad('grey')
im1 = plt.contourf(lon, lat, rms.T ,levels=np.linspace(0,1.5,51), cmap=mymap,transform=ccrs.PlateCarree(),extend='both')
cbar=plt.colorbar(ax=ax,ticks=np.linspace(0,1.5,11),orientation="vertical",fraction=0.046, pad=0.04) #ticks=[-40,20,0,20,40]) #ticks=[0,1,2,3]) #
cbar.set_label('SST (°C)', rotation=270, labelpad=12, fontsize=12,fontweight="bold")
im1 = plt.contour(lon, lat, rms.T ,levels=np.linspace(0,1.5,11), colors='k',linewidths=0.5,transform=ccrs.PlateCarree())
ax.coastlines(resolution='50m')

ax.set_title('RMS(VER, OBS)')
plt.tight_layout()
plt.savefig('SST_moyenne_VER_vs_OBS_RMS_AVHRR_DSTART_DEND.png',dpi=400)
plt.close()

