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
import datetime
import matplotlib.pyplot as plt
import netCDF4 as nc
from matplotlib import pyplot
from matplotlib import colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
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


Ufile_VER = xr.open_dataset('FLDR/UFILENM')
Vfile_VER = xr.open_dataset('FLDR/VFILENM')

Coordfile = xr.open_dataset('FLDR/domain_cfg.nc', drop_variables={"x", "y",})
time_counter=Ufile_VER.time_counter.squeeze()
var_IBI_AGRIF=np.zeros(Ufile_VER.nav_lon.squeeze().shape)
valeur_cadre=1.

for t in range(len(time_counter)):
    time_new=time_counter[t].dt.strftime("%Y-%m-%d_%Hh%M")
    print(str(time_new.values))
    U_VER=Ufile_VER.sozocrtx.squeeze().isel(time_counter=t)[::3,::3] *100 
    V_VER=Vfile_VER.somecrty.squeeze().isel(time_counter=t)[::3,::3] *100 
    
    T_VER=0.5*(U_VER.values**2 + V_VER.values**2)
    
    matplotlib.rcParams.update({'font.size': 10})
    proj=ccrs.PlateCarree()
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111, projection=proj)
    lat=Coordfile.gphit.squeeze().values[::3,::3]
    lon=Coordfile.glamt.squeeze().values[::3,::3]
    ax.set_extent((-5.5, 5., 45., 53.))
    
    ax.coastlines(resolution='50m')
    lon_formatter = LongitudeFormatter(degree_symbol='° ')
    lat_formatter = LatitudeFormatter(degree_symbol='° ')
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    axes=ax.gridlines( draw_labels=False, linewidth=0)
    axes.ylabels_right = False
    axes.xlabels_top = False

    im1 = plt.contourf(lon, lat, T_VER ,levels=np.linspace(0,1000,51), cmap="jet",transform=ccrs.PlateCarree(),extend='both')
    cbar=plt.colorbar(im1,ticks=[np.linspace(0,1000,11)],orientation="vertical",fraction=0.046, pad=0.04) #ticks=[-40,20,0,20,40]) #ticks=[0,1,2,3]) #
    cbar.set_label('KE (m²/s²)', rotation=270, labelpad=12, fontsize=12,fontweight="bold")
    ax.set_title('VER, ' +str(time_counter[t].values))
    plt.tight_layout()
    AGRIF=False
    if AGRIF:
        ######## DOMAINE : AGRIF  ###############
        imin_IBI_AGRIF=IMIN
        imax_IBI_AGRIF=IMAX
        jmin_IBI_AGRIF=JMIN
        jmax_IBI_AGRIF=JMAX
        var_IBI_AGRIF[jmin_IBI_AGRIF-2:jmin_IBI_AGRIF+1,imin_IBI_AGRIF:imax_IBI_AGRIF]=valeur_cadre
        var_IBI_AGRIF[jmax_IBI_AGRIF-2:jmax_IBI_AGRIF+1,imin_IBI_AGRIF:imax_IBI_AGRIF]=valeur_cadre
        var_IBI_AGRIF[jmin_IBI_AGRIF:jmax_IBI_AGRIF,imin_IBI_AGRIF-2:imin_IBI_AGRIF+1]=valeur_cadre
        var_IBI_AGRIF[jmin_IBI_AGRIF:jmax_IBI_AGRIF,imax_IBI_AGRIF-2:imax_IBI_AGRIF+1]=valeur_cadre
        ########
        im1 = plt.contour(lon, lat, var_IBI_AGRIF[::3,::3], cmap=None,levels=np.linspace(-1,1,5), transform=ccrs.PlateCarree(),linewidths=0.5,colors=('k','k','k','k','k','k'))
            
    plt.savefig('SCRIPTID_CFG_NM_1h_VER_'+str(time_new.values)+'.png', dpi=400)
    plt.close()
        
         
