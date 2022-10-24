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
    y_min=380
    y_max=445
    x_min=508
    x_max=530
    
    return ds.isel(y=np.arange(y_min,y_max)).isel(x=np.arange(x_min,x_max))

def preprocess_agrif(ds):
    y_min=380
    y_max=445
    x_min=508
    x_max=530
    imin_IBI_AGRIF=IMIN
    imax_IBI_AGRIF=IMAX
    jmin_IBI_AGRIF=JMIN
    jmax_IBI_AGRIF=JMAX

    y_min_agrif=(y_min - jmin_IBI_AGRIF)*3 + 4 
    y_max_agrif=(y_max - jmin_IBI_AGRIF)*3 + 4
    x_min_agrif=(x_min - imin_IBI_AGRIF)*3 + 4
    x_max_agrif=(x_max - imin_IBI_AGRIF)*3 + 4

    return ds.isel(y=np.arange(y_min_agrif,y_max_agrif)).isel(x=np.arange(x_min_agrif,x_max_agrif))


Tfile_VER1 = xr.open_mfdataset('FLDR_VERSION1/CFG_NAME_1d_gridU25h_????????-2017????.nc',preprocess=preprocess, concat_dim='time_counter').sel(time_counter=slice('DSTART', 'DEND'))
Tfile_AGRIF_VER1=xr.open_mfdataset('FLDR_VERSION1/AGRIF_CFG_NAME_1d_gridU25h_????????-2017????.nc',preprocess=preprocess_agrif, concat_dim='time_counter').sel(time_counter=slice('DSTART', 'DEND'))
Tfile_VER2=xr.open_mfdataset('FLDR_VERSION2/CFG_NAME_1d_gridU25h_????????-2017????.nc',preprocess=preprocess, concat_dim='time_counter').sel(time_counter=slice('DSTART', 'DEND'))

Coordfile_VER1 = xr.open_mfdataset('FLDR_VERSION1/domain_cfg.nc', drop_variables={"x", "y",},preprocess=preprocess).squeeze()
Coordfile_AGRIF_VER1 = xr.open_mfdataset('FLDR_VERSION1/1_domain_cfg.nc', drop_variables={"x", "y",},preprocess=preprocess_agrif).squeeze()
Coordfile_VER2 = xr.open_mfdataset('FLDR_VERSION2/domain_cfg.nc', drop_variables={"x", "y",},preprocess=preprocess).squeeze()

U_tot_VER1 = Tfile_VER1.vozocrtx*Coordfile_VER1.e1t.values *Coordfile_VER1.e2t.values * Coordfile_VER1.e3t_0.values   #.mean(dim='y').mean(dim='x')
U_tot_AGRIF_VER1 = Tfile_AGRIF_VER1.vozocrtx*Coordfile_AGRIF_VER1.e1t.values *Coordfile_AGRIF_VER1.e2t.values * Coordfile_AGRIF_VER1.e3t_0.values    #.mean(dim='y').mean(dim='x')
U_tot_VER2 = Tfile_VER2.vozocrtx*Coordfile_VER2.e1t.values *Coordfile_VER2.e2t.values * Coordfile_VER2.e3t_0.values     #.mean(dim='y').mean(dim='x')
U_tot_VER1_sect=U_tot_VER1.mean(dim='x')/Coordfile_VER1.e1t.mean(dim='x').values 
U_tot_AGRIF_VER1_sect=U_tot_AGRIF_VER1.mean(dim='x')/Coordfile_AGRIF_VER1.e1t.mean(dim='x').values
U_tot_VER2_sect=U_tot_VER2.mean(dim='x')/Coordfile_VER2.e1t.mean(dim='x').values

U_tot_VER1_sect_out=U_tot_VER1_sect.where(U_tot_VER1_sect < 0, np.nan) 
for i in range(len(U_tot_VER1_sect_out[0,:,0])):
    print(U_tot_VER1_sect_out[0,i,:].values)

#plt.figure((10,8))
#ax = plt.subplot(121)
#plt.pl

