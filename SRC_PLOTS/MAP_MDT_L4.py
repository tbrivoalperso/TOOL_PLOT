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


#Tfile_VER = xr.open_dataset('FLDR/CFG_NM_1d_grid2D_ssh.nc', chunks={'time_counter': 50})
obs_file=xr.open_mfdataset('/scratch/work/brivoalt/DATA_eNEATL36/VALIDATION_REGIONAL/eNEATL36/OBS/OBS_DATA/SSH/cnes_obs-sl_glo_phy-mdt_my_0.125deg_P20Y_1648474891667.nc')
mdt_ref_file = xr.open_dataset('/scratch/work/brivoalt/RUNS_NEMO/eNEATL36_AGRIF_vvl/EXP02_AGRIF_finaldomain_bathycorrected_qco_boost2_noslip/eNEATL36_1d_grid2D_ssh_rmpMDT.nc')
mdt_model=mdt_ref_file.sossheig.squeeze().mean(dim='time_counter')
ssh_obs=obs_file.mdt.squeeze()[0,:,:]
ssh_obs=ssh_obs.where(abs(ssh_obs.values) <100, np.nan ) 
ssh_obs=ssh_obs.where(~np.isnan(mdt_model.values),np.nan)

ssh_obs_m= ssh_obs.mean()
mdt_model_m = mdt_model.mean()
bias = ssh_obs_m - mdt_model_m

ssh_obs=ssh_obs - bias
matplotlib.rcParams.update({'font.size': 10})
proj=ccrs.Mercator()
plt.figure(figsize=(8,8))
lat1D=obs_file.latitude.squeeze().values
lon1D=obs_file.longitude.squeeze().values
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
im1 = plt.contourf(lon, lat, ssh_obs.T ,levels=np.linspace(-0.3,0.3,51), cmap=mymap,transform=ccrs.PlateCarree(),extend='both')
cbar=plt.colorbar(ax=ax,ticks=np.linspace(-0.3,0.3,11),orientation="vertical",fraction=0.046, pad=0.04) #ticks=[-40,20,0,20,40]) #ticks=[0,1,2,3]) #
cbar.set_label('MDT (m)', rotation=270, labelpad=12, fontsize=12,fontweight="bold")
im1 = plt.contour(lon, lat, ssh_obs.T ,levels=np.linspace(-0.3,0.3,11), colors='k',linewidths=0.5,transform=ccrs.PlateCarree())
ax.coastlines(resolution='50m')

ax.set_title('OBS')
plt.tight_layout()
plt.savefig('MDT_moyenne_OBS.png',dpi=400)
plt.close()
    
     
