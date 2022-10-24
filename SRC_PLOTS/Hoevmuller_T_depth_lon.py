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
    y_sec=620
    return ds.isel(y=np.arange(y_sec-36,y_sec+36))

Tfile_VER = xr.open_mfdataset('FLDR/CFG_NM_1d_gridT25h_MONTH_ana??-2017????.nc',preprocess=preprocess, concat_dim='time_counter').sel(time_counter=slice('DSTART', 'DEND'))
T_VER=Tfile_VER.votemper.mean(dim='time_counter').mean(dim='y').squeeze()
Coordfile = xr.open_dataset('FLDR/domain_cfg.nc', drop_variables={"x", "y",})
print(T_VER.shape)
y_sec=620

T_VER=np.array(T_VER[:,250:450].values.squeeze())

matplotlib.rcParams.update({'font.size': 10})
proj=ccrs.Mercator()
plt.figure(figsize=(8,8))
lon=np.array(Coordfile.glamt.squeeze()[y_sec,250:450])

deptht=np.array(Tfile_VER.deptht.squeeze())



lon2D,deptht2D=np.meshgrid(lon,deptht,indexing='ij')

ax = plt.subplot(111)
print(lon2D.shape)
print(T_VER.shape)
im1 = plt.contourf(lon2D, deptht2D, T_VER.T ,levels=np.linspace(7,25,51), cmap=cmocean.cm.thermal,extend='both')
cbar=plt.colorbar(ax=ax,ticks=np.linspace(7,25,11),orientation="vertical",fraction=0.046, pad=0.04) #ticks=[-40,20,0,20,40]) #ticks=[0,1,2,3]) #
cbar.set_label('T° (°C)', rotation=270, labelpad=12, fontsize=12,fontweight="bold")
im1 = plt.contour(lon2D, deptht2D, T_VER.T ,levels=np.linspace(7,25,11), colors='k',linewidths=0.5)
ax.set_ylim(0,500)
ax.set_title('VER')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('CFG_NM_moyenne_Hoevmoeller_T_UPWELL_VER_DSTART_DEND.png')
plt.close()
    
     
