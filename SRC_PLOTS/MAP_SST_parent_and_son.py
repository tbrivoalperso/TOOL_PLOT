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


Tfile_VER = xr.open_dataset('FLDR/eNEATL36_1d_gridT.nc', chunks={'time_counter': 50})
Tfile_VER_son = xr.open_dataset('FLDR/1_AGRIF_1d_gridT.nc', chunks={'time_counter': 50})

Coordfile = xr.open_dataset('FLDR/domain_cfg.nc', drop_variables={"x", "y",})
Coordfile_son = xr.open_dataset('FLDR/1_domain_cfg.nc', drop_variables={"x", "y",})

T_VER=Tfile_VER.votemper.squeeze().sel(time_counter=slice('DSTART', 'DEND')).mean(dim='time_counter').values
T_VER_son=Tfile_VER_son.votemper.squeeze().sel(time_counter=slice('DSTART', 'DEND')).mean(dim='time_counter').values
var_IBI_AGRIF=np.zeros(T_VER.squeeze().shape)
valeur_cadre=1.

matplotlib.rcParams.update({'font.size': 10})
proj=ccrs.Mercator()
plt.figure(figsize=(8,8))
lat=Coordfile.gphit.squeeze().values
lon=Coordfile.glamt.squeeze().values
lat_son=Coordfile_son.gphit.squeeze().values
lon_son=Coordfile_son.glamt.squeeze().values

ax = plt.subplot(111, projection=proj)

ax.coastlines(resolution='50m')
lon_formatter = LongitudeFormatter(degree_symbol='° ')
lat_formatter = LatitudeFormatter(degree_symbol='° ')
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

axes=ax.gridlines( draw_labels=False, linewidth=0)
axes.ylabels_right = False
axes.xlabels_top = False
imin_IBI_AGRIF=IMIN
imax_IBI_AGRIF=IMAX
jmin_IBI_AGRIF=JMIN
jmax_IBI_AGRIF=JMAX
T_VER[jmin_IBI_AGRIF:jmax_IBI_AGRIF,imin_IBI_AGRIF:imax_IBI_AGRIF]=T_VER_son[5:-3:3,5:-3:3]
print(lat[jmin_IBI_AGRIF,imin_IBI_AGRIF])
print(lat_son[5,5])
im1 = plt.contourf(lon, lat, T_VER ,levels=np.linspace(9,29,51), cmap=cmocean.cm.thermal,transform=ccrs.PlateCarree(),extend='both')
cbar=plt.colorbar(ax=ax,ticks=np.linspace(9,29,11),orientation="vertical",fraction=0.046, pad=0.04)
cbar.set_label('SST (°C)', rotation=270, labelpad=12, fontsize=12,fontweight="bold")
im1 = plt.contour(lon, lat, T_VER ,levels=np.linspace(9,29,11), colors='k',linewidths=0.5,transform=ccrs.PlateCarree())

######## DOMAINE : AGRIF  ###############
var_IBI_AGRIF[jmin_IBI_AGRIF,imin_IBI_AGRIF:imax_IBI_AGRIF]=valeur_cadre
var_IBI_AGRIF[jmax_IBI_AGRIF,imin_IBI_AGRIF:imax_IBI_AGRIF]=valeur_cadre
var_IBI_AGRIF[jmin_IBI_AGRIF:jmax_IBI_AGRIF,imin_IBI_AGRIF]=valeur_cadre
var_IBI_AGRIF[jmin_IBI_AGRIF:jmax_IBI_AGRIF,imax_IBI_AGRIF]=valeur_cadre
########
im1 = plt.contour(lon, lat, var_IBI_AGRIF[:,:], cmap=None,levels=np.linspace(-1,1,3), transform=ccrs.PlateCarree(),linewidths=0.5,colors=('k','k','k','k','k','k'))
ax.set_title('VER')
plt.tight_layout()
plt.savefig('CFG_NM_SST_moyenne_VER_DSTART_DEND_parent_and_son.png', dpi=300)
plt.close()
    
     
