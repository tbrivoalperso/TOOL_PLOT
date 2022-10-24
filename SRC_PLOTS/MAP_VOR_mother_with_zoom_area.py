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


Ufile_NEMO_4_0 = xr.open_dataset('/scratch/work/brivoalt/RUNS_NEMO/eNEATL36_AGRIF_vvl/EXP01_AGRIF/eNEATL36_1h_curl.nc', chunks={'time_counter': 50})

#Ufile_NEMO_4_0 = xr.open_dataset('/scratch/work/brivoalt/MAKE_DOMAINcfg/eNEATL36/coordinates.nc')
Curl_U=Ufile_NEMO_4_0.socurl.squeeze().isel(time_counter=911) #.sel(time_counter='2018-01-01')

print(np.nanmin(Curl_U.values))
print(np.nanmax(Curl_U.values))

print(Curl_U.shape)

matplotlib.rcParams.update({'font.size': 10})
proj=ccrs.Mercator()
plt.figure(figsize=(16,8))
ax = plt.subplot(111, projection=proj)
lat=Ufile_NEMO_4_0.nav_lat.squeeze()
lon=Ufile_NEMO_4_0.nav_lon.squeeze()

ax.coastlines(resolution='50m')
ax.set_extent((-21.0, 16.0, 25.0, 63.5))
lon_formatter = LongitudeFormatter(degree_symbol='° ')
lat_formatter = LatitudeFormatter(degree_symbol='° ')
ax.set_xticks([ -20 , -10 , 0 , 10  ], crs=ccrs.PlateCarree())
ax.set_yticks([ 30, 40, 50, 60], crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

axes=ax.gridlines( draw_labels=False, linewidth=0)
axes.ylabels_right = False
axes.xlabels_top = False

im1 = plt.contourf(lon, lat, Curl_U ,levels=np.linspace(-1e-4,1e-4,51), cmap=mymap,transform=ccrs.PlateCarree(),extend='both')
cbar=plt.colorbar(ax=ax,ticks=np.linspace(-1e-4,1e-4,11),orientation="vertical") #ticks=[-40,20,0,20,40]) #ticks=[0,1,2,3]) #
cbar.set_label('Vorticity (1/s)', rotation=270, labelpad=12, fontsize=12,fontweight="bold")
########
valeur_cadre=1.
var_IBI_AGRIF=np.zeros(Curl_U.shape)
print('IBI : ')
print(np.min(lon[:,0].values))
print(np.min(lat[0,:].values))
print(np.max(lon[:,-1].values))
print(np.max(lat[-1,:].values))


######## DOMAINE : IBI SERVICE  ###############
# Imin = 0 ; jmin=0 ; Imax=900 ; Jmax = 1450 
imin_IBI_AGRIF=312
imax_IBI_AGRIF=752
jmin_IBI_AGRIF=751
jmax_IBI_AGRIF=1176
var_IBI_AGRIF[jmin_IBI_AGRIF-1:jmin_IBI_AGRIF+1,imin_IBI_AGRIF:imax_IBI_AGRIF]=valeur_cadre
var_IBI_AGRIF[jmax_IBI_AGRIF-1:jmax_IBI_AGRIF+1,imin_IBI_AGRIF:imax_IBI_AGRIF]=valeur_cadre
var_IBI_AGRIF[jmin_IBI_AGRIF:jmax_IBI_AGRIF,imin_IBI_AGRIF-1:imin_IBI_AGRIF+1]=valeur_cadre
var_IBI_AGRIF[jmin_IBI_AGRIF:jmax_IBI_AGRIF,imax_IBI_AGRIF-1:imax_IBI_AGRIF+1]=valeur_cadre
print('IBI AGRIF : ')
print(np.min(lon[:, imin_IBI_AGRIF].values))
print(np.min(lat[jmin_IBI_AGRIF, :].values))
print(np.max(lon[:, imax_IBI_AGRIF].values))
print(np.max(lat[jmax_IBI_AGRIF, :].values))

########

########
im1 = plt.contour(lon, lat, var_IBI_AGRIF, cmap=None,levels=np.linspace(-1,1,5), transform=ccrs.PlateCarree(),linewidths=1.5,colors=('k','k','k','b','b','b'))



plt.savefig('VORTICITY_mother_with_zoom_area.png')
plt.close()
    
     
