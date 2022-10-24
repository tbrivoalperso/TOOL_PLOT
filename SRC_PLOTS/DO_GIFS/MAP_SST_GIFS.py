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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmocean
import matplotlib.pyplot as plt
import netCDF4 as nc
from matplotlib import pyplot
from matplotlib import colors as mcolors
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import colorsys
def man_cmap(cmap, value=1.):
    colors = cmap(np.arange(cmap.N))
    hls = np.array([colorsys.rgb_to_hls(*c) for c in colors[:,:3]])
    hls[:,1] *= value
    rgb = np.clip(np.array([colorsys.hls_to_rgb(*c) for c in hls]), 0,1)
    return mcolors.LinearSegmentedColormap.from_list("", rgb)


Tfile_VER = xr.open_dataset('FLDR/TFILENM')

Coordfile = xr.open_dataset('FLDR/domain_cfg.nc', drop_variables={"x", "y",})
time_counter=Tfile_VER.time_counter.squeeze()
var_IBI_AGRIF=np.zeros(Tfile_VER.nav_lon.squeeze().shape)
valeur_cadre=1.
cmap = plt.cm.get_cmap("jet")

for t in range(len(time_counter)):
    time_new=time_counter[t].dt.strftime("%Y-%m-%d_%Hh%M")
    print(str(time_new.values))

    T_VER=Tfile_VER.sosstmod.squeeze().isel(time_counter=t)[:,:]
    
    matplotlib.rcParams.update({'font.size': 10})
    proj=ccrs.Mercator()
    plt.figure(figsize=(8,8))
    lat=Coordfile.gphit.squeeze().values[:,:]
    lon=Coordfile.glamt.squeeze().values[:,:]
    
    ax = plt.subplot(111, projection=proj)
    
    ax.coastlines(resolution='50m')
    lon_formatter = LongitudeFormatter(degree_symbol='° ')
    lat_formatter = LatitudeFormatter(degree_symbol='° ')
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    axes=ax.gridlines( draw_labels=False, linewidth=0)
    axes.ylabels_right = False
    axes.xlabels_top = False
    im1 = plt.contourf(lon, lat, T_VER ,levels=np.linspace(9,29,25), cmap=cmocean.cm.thermal,transform=ccrs.PlateCarree(),extend='both')
    cbar=plt.colorbar(ax=ax,ticks=[np.linspace(9,29,11)],orientation="vertical",fraction=0.046, pad=0.04) #ticks=[-40,20,0,20,40]) #ticks=[0,1,2,3]) #
    cbar.set_label('SST (°C)', rotation=270, labelpad=12, fontsize=12,fontweight="bold")
    im1 = plt.contour(lon, lat, T_VER ,levels=np.linspace(9,29,11), colors='k',linewidths=0.5,transform=ccrs.PlateCarree())

    ax.set_title('VER, ' +str(time_counter[t].values))
    plt.tight_layout()
    AGRIF=AGRIF_bool

    if AGRIF:
        ######## DOMAINE : AGRIF  ###############
        imin_IBI_AGRIF=IMIN
        imax_IBI_AGRIF=IMAX
        jmin_IBI_AGRIF=JMIN
        jmax_IBI_AGRIF=JMAX
        var_IBI_AGRIF[jmin_IBI_AGRIF,imin_IBI_AGRIF:imax_IBI_AGRIF]=valeur_cadre
        var_IBI_AGRIF[jmax_IBI_AGRIF,imin_IBI_AGRIF:imax_IBI_AGRIF]=valeur_cadre
        var_IBI_AGRIF[jmin_IBI_AGRIF:jmax_IBI_AGRIF,imin_IBI_AGRIF]=valeur_cadre
        var_IBI_AGRIF[jmin_IBI_AGRIF:jmax_IBI_AGRIF,imax_IBI_AGRIF]=valeur_cadre
        
        ########
        im1 = plt.contour(lon, lat, var_IBI_AGRIF[:,:], cmap=None,levels=np.linspace(-1,1,2), transform=ccrs.PlateCarree(),linewidths=0.5,colors=('k','k','k','k','k','k'))
        
    ax.set_extent((-8, 4., 45., 57.))        
    plt.savefig('SCRIPTID_CFG_NM_1h_VER_'+str(time_new.values)+'.png', dpi=400)
    plt.close()
        
         
