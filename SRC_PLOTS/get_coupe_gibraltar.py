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
import os
import glob
from datetime import datetime, date, timedelta


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
    x_min=516
    
    return ds.isel(y=np.arange(y_min,y_max)).isel(x=np.arange(x_min-20,x_min+20))

def preprocess_agrif(ds):
    y_min=380
    y_max=445
    x_min=516
    imin_IBI_AGRIF=IMIN
    imax_IBI_AGRIF=IMAX
    jmin_IBI_AGRIF=JMIN
    jmax_IBI_AGRIF=JMAX

    y_min_agrif=(y_min - jmin_IBI_AGRIF)*3 + 4 
    y_max_agrif=(y_max - jmin_IBI_AGRIF)*3 + 4
    x_min_agrif=(x_min - imin_IBI_AGRIF)*3 + 4

    return ds.isel(y=np.arange(y_min_agrif,y_max_agrif)).isel(x=np.arange(x_min_agrif-20,x_min_agrif+20))

y_min=380
y_max=445
x_min=516
imin_IBI_AGRIF=IMIN
imax_IBI_AGRIF=IMAX
jmin_IBI_AGRIF=JMIN
jmax_IBI_AGRIF=JMAX

y_min_agrif=(y_min - jmin_IBI_AGRIF)*3 + 4
y_max_agrif=(y_max - jmin_IBI_AGRIF)*3 + 4
x_min_agrif=(x_min - imin_IBI_AGRIF)*3 + 4
i=0
start_date_str='DSTART'
end_date_str='DEND'
start_date = datetime.strptime(start_date_str,"%Y-%m-%d")
end_date = datetime.strptime(end_date_str,  "%Y-%m-%d")
delta = timedelta(days=1)

current_date=start_date
while current_date <= end_date:
   tdate=current_date.strftime("%Y%m%d")
   current_date=current_date + delta
   filepath='FLDR_VERSION1/DATA_IMMERSE/CFG_NAME_1d_gridU25h_'+tdate+'-'+tdate+'.nc'
   Tfile_VER1 = xr.open_dataset(filepath).sel(time_counter=slice('DSTART', 'DEND')).isel(y=np.arange(y_min,y_max)).isel(x=np.arange(x_min-20,x_min+20))
   U_tmp=Tfile_VER1.vozocrtx.values      
   try:
      U_tot_VER1=np.concatenate((U_tot_VER1,U_tmp),axis=0)
   except:
      U_tot_VER1=U_tmp

Tfile_VER1_tmp = xr.open_mfdataset('FLDR_VERSION1/DATA_IMMERSE/CFG_NAME_1d_gridU25h_????????-????????.nc',parallel=True).sel(time_counter=slice('DSTART', 'DEND'))
U_tmp = Tfile_VER1_tmp.vozocrtx.sel(time_counter=slice('DSTART', 'DEND')).isel(y=np.arange(y_min,y_max)).isel(x=np.arange(x_min-20,x_min+20)).squeeze()
U_tot_VER1_da=xr.DataArray(U_tot_VER1,coords=U_tmp.coords,dims=U_tmp.dims, name=U_tmp.name, attrs=U_tmp.attrs)
U_tot_VER1_da.to_netcdf('FLDR_VERSION1/CFG_NAME_1d_gridU25h_DSTART_DEND_extract_gibraltar.nc')


current_date=start_date
while current_date <= end_date:
   tdate=current_date.strftime("%Y%m%d")
   current_date=current_date + delta
   filepath='FLDR_VERSION1/DATA_IMMERSE/AGRIF_CFG_NAME_1d_gridU25h_'+tdate+'-'+tdate+'.nc'
   Tfile_AGRIF_VER1 = xr.open_dataset(filepath).sel(time_counter=slice('DSTART', 'DEND')).isel(y=np.arange(y_min_agrif,y_max_agrif)).isel(x=np.arange(x_min_agrif-60,x_min_agrif+60))
   U_tmp=Tfile_AGRIF_VER1.vozocrtx.values
   try:
      U_tot_AGRIF_VER1=np.concatenate((U_tot_AGRIF_VER1,U_tmp),axis=0)
   except:
      U_tot_AGRIF_VER1=U_tmp

Tfile_AGRIF_VER1_tmp = xr.open_mfdataset('FLDR_VERSION1/DATA_IMMERSE/AGRIF_CFG_NAME_1d_gridU25h_????????-????????.nc',parallel=True).sel(time_counter=slice('DSTART', 'DEND')) 
U_tmp = Tfile_AGRIF_VER1_tmp.vozocrtx.sel(time_counter=slice('DSTART', 'DEND')).isel(y=np.arange(y_min_agrif,y_max_agrif)).isel(x=np.arange(x_min_agrif-60,x_min_agrif+60)).squeeze()
U_tot_AGRIF_VER1_da=xr.DataArray(U_tot_AGRIF_VER1,coords=U_tmp.coords,dims=U_tmp.dims, name=U_tmp.name, attrs=U_tmp.attrs)
U_tot_AGRIF_VER1_da.to_netcdf('FLDR_VERSION1/AGRIF_CFG_NAME_1d_gridU25h_DSTART_DEND_extract_gibraltar.nc')

#
current_date=start_date
while current_date <= end_date:
   tdate=current_date.strftime("%Y%m%d")
   current_date=current_date + delta
   filepath='FLDR_VERSION2/DATA_IMMERSE/CFG_NAME_1d_gridU25h_'+tdate+'-'+tdate+'.nc'
   Tfile_VER2 = xr.open_dataset(filepath).sel(time_counter=slice('DSTART', 'DEND')).isel(y=np.arange(y_min,y_max)).isel(x=np.arange(x_min-20,x_min+20))
   U_tmp = Tfile_VER2.vozocrtx.values
   try:
      U_tot_VER2=np.concatenate((U_tot_VER2,U_tmp),axis=0)
   except:
      U_tot_VER2=U_tmp
   print(U_tot_VER2.shape)

Tfile_VER2_tmp = xr.open_mfdataset('FLDR_VERSION2/DATA_IMMERSE/CFG_NAME_1d_gridU25h_????????-????????.nc',parallel=True).sel(time_counter=slice('DSTART', 'DEND'))
U_tmp = Tfile_VER2_tmp.vozocrtx.sel(time_counter=slice('DSTART', 'DEND')).isel(y=np.arange(y_min,y_max)).isel(x=np.arange(x_min-20,x_min+20))
U_tot_VER2_da=xr.DataArray(U_tot_VER2,coords=U_tmp.coords,dims=U_tmp.dims, name=U_tmp.name, attrs=U_tmp.attrs)
U_tot_VER2_da.to_netcdf('FLDR_VERSION2/CFG_NAME_1d_gridU25h_DSTART_DEND_extract_gibraltar.nc')
#

