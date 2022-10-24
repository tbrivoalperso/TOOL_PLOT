#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:04:08 2019

@author: tbrivoal
"""

import matplotlib as mpl
mpl.use('Agg')
import cartopy.crs as ccrs
import os, sys
import xarray as xr
import numpy as np
import scipy.stats as st
from scipy.stats import linregress
from numpy.polynomial.polynomial import polyfit
from scipy import signal
from scipy.stats import norm
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import xrft
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from scipy import stats
import netCDF4 as nc
import matplotlib.ticker as mticker
from matplotlib import colors
import k_omega_functions as kw

###################################################################################################################
ISAGRIF1=False
ISAGRIF2=False
ndeg=2.5

U_1_file = xr.open_dataset('FLDR_VERSION1/CFG_NAME_1h_gridU.nc') #, chunks={'space': 10})
V_1_file = xr.open_dataset('FLDR_VERSION1/CFG_NAME_1h_gridV.nc') #, chunks={'space': 10})

Coordfile_1 = xr.open_dataset('FLDR_VERSION1/domain_cfg.nc', drop_variables={"x", "y",})
res_1=1/36
npts_ndeg_1=int(ndeg*(1/res_1))

U_2_file = xr.open_dataset('FLDR_VERSION2/CFG_NAME_1h_gridU.nc') #, chunks={'space': 10})
V_2_file = xr.open_dataset('FLDR_VERSION2/CFG_NAME_1h_gridV.nc') #, chunks={'space': 10})

Coordfile_2 = xr.open_dataset('FLDR_VERSION2/domain_cfg.nc', drop_variables={"x", "y",})
res_2=1/36
npts_ndeg_2=int(ndeg*(1/res_2))

U_1=U_1_file.sozocrtx.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:].rename('KE') #
V_1=V_1_file.somecrty.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:].rename('KE')#

U_2=U_2_file.sozocrtx.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:].rename('KE')#
V_2=V_2_file.somecrty.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:].rename('KE')#

#areanames=['ouessant', 'ATL_46N','WMED','MANCHE','GIBR','GLION']
#colors=['b','r','g','orange','brown','lightgreen']
#listlat_center=np.array([48,46,40,50,36,42.5])
#listlon_center=np.array([-6,-7,4,0, -7.5,4.5])

areanames=['ATL_46N']
colors=['r']
listlat_center=np.array([46])
listlon_center=np.array([-7])

indexy_center_1=np.zeros(listlat_center.shape).astype(int)
indexx_center_1=np.zeros(listlon_center.shape).astype(int)
indexy_center_11=np.zeros(listlat_center.shape).astype(int)
indexx_center_11=np.zeros(listlon_center.shape).astype(int)

nav_lat_1=Coordfile_1.gphit.squeeze().values
nav_lon_1=Coordfile_1.glamt.squeeze().values
lat_found=0
lon_found=0
    #
print_dx_dy = True
extract_area = False 
for narea in range(len(listlat_center)):
    # Selecting SSH area
    tmplat=abs(nav_lat_1 - listlat_center[narea])
    tmplon=abs(nav_lon_1 - listlon_center[narea])
    tmp=tmplat + tmplon
    indexy_center_1[narea], indexx_center_1[narea] = int(np.argwhere(tmp == np.nanmin(tmp))[0,0]), int(np.argwhere(tmp == np.nanmin(tmp))[0,1])
    print("CFG_NAME, selecting point i, j :",indexy_center_1[narea], indexx_center_1[narea])
    print("at latitude, longitude :", nav_lat_1[indexy_center_1[narea], indexx_center_1[narea]], nav_lon_1[indexy_center_1[narea], indexx_center_1[narea]])

    file_KE_1 = 'FLDR_VERSION1/CFG_NAME_1h_gridKE_'+str(areanames[narea])+'.nc'
    file_KE_2 = 'FLDR_VERSION2/CFG_NAME_1h_gridKE_'+str(areanames[narea])+'.nc'
    file_U_1 = 'FLDR_VERSION1/CFG_NAME_1h_gridU_'+str(areanames[narea])+'.nc'
    file_U_2 = 'FLDR_VERSION2/CFG_NAME_1h_gridU_'+str(areanames[narea])+'.nc'
    file_V_1 = 'FLDR_VERSION1/CFG_NAME_1h_gridV_'+str(areanames[narea])+'.nc'
    file_V_2 = 'FLDR_VERSION2/CFG_NAME_1h_gridV_'+str(areanames[narea])+'.nc'
    if print_dx_dy:
        ## SIZE PARAMETERS
        dx_1 = Coordfile_1.e1t.squeeze()[indexy_center_1[narea]- npts_ndeg_1:indexy_center_1[narea]+ npts_ndeg_1,\
                          indexx_center_1[narea]- npts_ndeg_1:indexx_center_1[narea]+ npts_ndeg_1].mean(skipna=True)
        dy_1 = Coordfile_1.e2t.squeeze()[indexy_center_1[narea]- npts_ndeg_1:indexy_center_1[narea]+ npts_ndeg_1,\
                          indexx_center_1[narea]- npts_ndeg_1:indexx_center_1[narea]+ npts_ndeg_1].mean(skipna=True)
        print('DX, DY for area : '+str(areanames[narea])+ '=', dx_1.values, dy_1.values)
    if extract_area:
        # extract area & compute KE
        U_1_area=U_1[:,indexy_center_1[narea]- npts_ndeg_1:indexy_center_1[narea]+ npts_ndeg_1,\
                           indexx_center_1[narea]- npts_ndeg_1:indexx_center_1[narea]+ npts_ndeg_1]
        V_1_area=V_1[:,indexy_center_1[narea]- npts_ndeg_1:indexy_center_1[narea]+ npts_ndeg_1,\
                           indexx_center_1[narea]- npts_ndeg_1:indexx_center_1[narea]+ npts_ndeg_1]
        KE_1_area = 0.5 * (U_1_area**2 + V_1_area**2)
        U_2_area=U_2[:,indexy_center_1[narea]- npts_ndeg_2:indexy_center_1[narea]+ npts_ndeg_2,\
                           indexx_center_1[narea]- npts_ndeg_2:indexx_center_1[narea]+ npts_ndeg_2]
        V_2_area=V_2[:,indexy_center_1[narea]- npts_ndeg_2:indexy_center_1[narea]+ npts_ndeg_2,\
                           indexx_center_1[narea]- npts_ndeg_2:indexx_center_1[narea]+ npts_ndeg_2]
        KE_2_area = 0.5 * (U_2_area**2 + V_2_area**2)
        # save files
        KE_1_area.to_netcdf(file_KE_1)
        KE_2_area.to_netcdf(file_KE_2)
        U_1_area.to_netcdf(file_U_1)
        U_2_area.to_netcdf(file_U_2)
        V_1_area.to_netcdf(file_V_1)
        V_2_area.to_netcdf(file_V_2)
    
        print('Saving file : ', file_KE_1)
        print('Saving file : ', file_KE_2)
     
