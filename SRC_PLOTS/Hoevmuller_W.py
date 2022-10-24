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

def preprocess(ds):
    y_sec=950
    idepth=26
    return ds.isel(y=y_sec).isel(depthw=idepth)

Tfile_VER = xr.open_mfdataset('FLDR/CFG_NM_1h_gridW3D_MONTH_ana??-2017????.nc',preprocess=preprocess, concat_dim='time_counter')
T_VER=Tfile_VER.woce.squeeze()
Coordfile = xr.open_dataset('FLDR/domain_cfg.nc', drop_variables={"x", "y",})
print(T_VER.shape)
y_sec=950
idepth=26

T_VER=np.array(T_VER[:,150:600].values.squeeze())

matplotlib.rcParams.update({'font.size': 10})
proj=ccrs.Mercator()
plt.figure(figsize=(8,8))
lon=np.array(Coordfile.glamt.squeeze()[y_sec,150:600])

time=np.array(Tfile_VER.time_counter.squeeze())
print(time.shape,lon.shape)

print(lon)
lon2D,time2D=np.meshgrid(lon,time,indexing='ij')

ax = plt.subplot(111)
print(time2D.shape)
print(lon2D.shape)
print(T_VER.shape)
print(T_VER.T.shape)
im1 = plt.contourf(lon2D, time2D, T_VER.T ,levels=np.linspace(-0.005,0.005,51), cmap=mymap,extend='both')
cbar=plt.colorbar(ax=ax,ticks=np.linspace(-0.005,0.005,11),orientation="vertical",fraction=0.046, pad=0.04) #ticks=[-40,20,0,20,40]) #ticks=[0,1,2,3]) #
cbar.set_label('W (m/s)', rotation=270, labelpad=12, fontsize=12,fontweight="bold")
im1 = plt.contour(lon2D, time2D, T_VER.T ,levels=np.linspace(-0.005,0.005,5), colors='k',linewidths=0.2)
plt.axvline(Coordfile.glamt.squeeze()[y_sec,IMIN],color='k',linewidth=1.5)
print(Coordfile.glamt.squeeze()[y_sec,IMIN])
ax.set_title('VER' + ' W at depth=' + str(int(Tfile_VER.depthw)))
ax.set_xlabel('longitude (°)')
plt.tight_layout()
plt.savefig('CFG_NM_Hoevmoller_W_moyenne_VER_DSTART_DEND.png',dpi=400)
plt.close()
    
     
