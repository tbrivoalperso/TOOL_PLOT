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


Ufile_NEMO_4_0 = xr.open_dataset('FLDR/res_harm_ssh.nc')
Ufile_FES = xr.open_dataset('/scratch/work/brivoalt/DATA_eNEATL36/tide_FES2014_grid_T_NEMO4_2.nc')
Coordfile = xr.open_dataset('FLDR/domain_cfg.nc', drop_variables={"x", "y",})

zx=Ufile_FES.N2_z1.squeeze().values
zy=-Ufile_FES.N2_z2.squeeze().values
zx = np.where(zx>1.e4, np.nan, zx)
zy = np.where(zy>1.e4, np.nan, zy)
zx = np.where(zx==0., np.nan, zx)
zy = np.where(zy==0., np.nan, zy)

amp_FES=np.sqrt(zx*zx+zy*zy)
pha_FES=np.arctan2(-zy,zx)*180./np.pi
top_level=Coordfile.top_level.squeeze().values
amp_FES=np.where(top_level>0,amp_FES,np.nan)
pha_FES=np.where(top_level>0,pha_FES,np.nan)

zx=Ufile_NEMO_4_0.N2_x_elev.squeeze().values
zy=Ufile_NEMO_4_0.N2_y_elev.squeeze().values
zx = np.where(zx>1.e4, np.nan, zx)
zy = np.where(zy>1.e4, np.nan, zy)
zx = np.where(zx==0., np.nan, zx)
zy = np.where(zy==0., np.nan, zy)

amp=np.sqrt(zx*zx+zy*zy)
pha=np.arctan2(-zy,zx)*180./np.pi

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

im1 = plt.contourf(lon, lat, amp - amp_FES ,levels=np.linspace(-0.15,0.15,31), cmap=mymap,transform=ccrs.PlateCarree(),extend='both')
cbar=plt.colorbar(ax=ax,ticks=np.linspace(-0.15,0.15,11),orientation="vertical") #ticks=[-40,20,0,20,40]) #ticks=[0,1,2,3]) #
cbar.set_label('N2 amplitude difference (m)', rotation=270, labelpad=12, fontsize=12,fontweight="bold")


print("Bias :", np.nanmean(amp[JMIN:JMAX,IMIN:IMAX] - amp_FES[JMIN:JMAX,IMIN:IMAX]))
print("RMS :", np.nanmean(np.sqrt((amp[JMIN:JMAX,IMIN:IMAX] - amp_FES[JMIN:JMAX,IMIN:IMAX])**2)))

########
#
#levels=np.arange(-180.,180.,20.)
#im2 = plt.contour(lon, lat, pha ,levels = levels,transform=ccrs.PlateCarree(), colors='k')
#plt.clabel(im2,levels[1::2], fontsize=9, inline=1)
plt.savefig('CFG_NM_tide_N2_diff_FES_VER_DSTART_DEND.png')

                                                   
    
     
