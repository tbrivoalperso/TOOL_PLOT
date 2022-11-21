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

# GENERAL PARAMETERS

ndeg=2.5 # number of lon / lat degrees to select / 2 => to select a 5° x 5° box, put ndeg=2.5
fldr_1 ='/scratch/work/brivoalt/RUNS_NEMO/eNEATL36_AGRIF_vvl/EXP02_AGRIF_finaldomain_bathycorrected_qco_boost2_noslip/'
Coordfile_1=fldr_1 +'domain_cfg.nc'
res_1=1/36 # Resolution of files 

fldr_2='/scratch/work/brivoalt/RUNS_NEMO/eNEATL36/eNEATL36_trunk_r4_2-RC_r15113_IFS_EXP02_2017_2018_AGRIFTWIN_TRUE_BFR/'
Coordfile_2=fldr_2 +'domain_cfg.nc'
res_2=1/36
DSTART='2017-06-01'
DEND='2018-07-01'

print_dx_dy = True
extract_area = True
str_start_date = 'DSTART'
str_end_date = 'DEND'
format = '%Y-%m-%d'
sdate = datetime.datetime.strptime(str_start_date, format)   # start date
edate = datetime.datetime.strptime(str_end_date, format)

list_dates=pandas.date_range(sdate,edate-timedelta(days=1),freq='d')

def extract_area(date ):
    
    areanames=['ATL_46N'] # name(s) of area(s) to extract
    listlat_center=np.array([46]) # lat of the center of the area(s) to extract
    listlon_center=np.array([-7]) # lon of the center of the area(s) to extract
    

    ###################################################################################################################
    # Read U & V data 

    Ufile_1=fldr_1 +'eNEATL36_1h_gridU_'+str(date)+'-'+str(date)+'.nc'
    Vfile_1=fldr_1 +'eNEATL36_1h_gridV_'+str(date)+'-'+str(date)+'.nc'
    Ufile_2=fldr_2 +'eNEATL36_1h_gridU_'+str(date)+'-'+str(date)+'.nc'
    Vfile_2=fldr_2 +'eNEATL36_1h_gridV_'+str(date)+'-'+str(date)+'.nc'

    U_1_file = xr.open_dataset(Ufile_1) 
    V_1_file = xr.open_dataset(Vfile_1) 
    ds_coord1 = xr.open_dataset(Coordfile_1, drop_variables={"x", "y",})
    
    U_1_file = xr.open_dataset(Ufile_1) 
    V_1_file = xr.open_dataset(Vfile_1) 
    ds_coord1 = xr.open_dataset(Coordfile_1, drop_variables={"x", "y",})
    
    U_1=U_1_file.sozocrtx.squeeze()
    V_1=V_1_file.somecrty.squeeze()
    
    U_2=U_2_file.sozocrtx.squeeze()
    V_2=V_2_file.somecrty.squeeze()
    
    
    # Convert ndeg in number of points
    npts_ndeg_1=int(ndeg*(1/res_1))
    npts_ndeg_2=int(ndeg*(1/res_2))
    
    # Initialise index list
    indexy_center_1=np.zeros(listlat_center.shape).astype(int)
    indexx_center_1=np.zeros(listlon_center.shape).astype(int)
    indexy_center_11=np.zeros(listlat_center.shape).astype(int)
    indexx_center_11=np.zeros(listlon_center.shape).astype(int)
    
    nav_lat_1=ds_coord1.gphit.squeeze().values
    nav_lon_1=ds_coord1.glamt.squeeze().values
    lat_found=0
    lon_found=0
        #
    for narea in range(len(listlat_center)):
        # Selecting SSH area
        tmplat=abs(nav_lat_1 - listlat_center[narea])
        tmplon=abs(nav_lon_1 - listlon_center[narea])
        tmp=tmplat + tmplon
        indexy_center_1[narea], indexx_center_1[narea] = int(np.argwhere(tmp == np.nanmin(tmp))[0,0]), int(np.argwhere(tmp == np.nanmin(tmp))[0,1])
        print("eNEATL36, selecting point i, j :",indexy_center_1[narea], indexx_center_1[narea])
        print("at latitude, longitude :", nav_lat_1[indexy_center_1[narea], indexx_center_1[narea]], nav_lon_1[indexy_center_1[narea], indexx_center_1[narea]])
    
        file_U_1 = fldr_1 +'eNEATL36_1h_gridU_'+str(areanames[narea])+'.nc'
        file_V_1 = fldr_1 +'eNEATL36_1h_gridV_'+str(areanames[narea])+'.nc'
        file_KE_1 = fldr_1 +'eNEATL36_1h_gridKE_'+str(areanames[narea])+'.nc'
    
        file_U_2 = fldr_2 +'eNEATL36_2h_gridU_'+str(areanames[narea])+'.nc'
        file_V_2 = fldr_2 +'eNEATL36_2h_gridV_'+str(areanames[narea])+'.nc'
        file_KE_2 = fldr_2 +'eNEATL36_2h_gridKE_'+str(areanames[narea])+'.nc'
    
        if print_dx_dy:
            ## SIZE PARAMETERS
            dx_1 = ds_coord1.e1t.squeeze()[indexy_center_1[narea]- npts_ndeg_1:indexy_center_1[narea]+ npts_ndeg_1,\
                              indexx_center_1[narea]- npts_ndeg_1:indexx_center_1[narea]+ npts_ndeg_1].mean(skipna=True)
            dy_1 = ds_coord1.e2t.squeeze()[indexy_center_1[narea]- npts_ndeg_1:indexy_center_1[narea]+ npts_ndeg_1,\
                              indexx_center_1[narea]- npts_ndeg_1:indexx_center_1[narea]+ npts_ndeg_1].mean(skipna=True)
            print('DX, DY for area : '+str(areanames[narea])+ '=', dx_1.values, dy_1.values)
        if extract_area:
            # extract area & compute KE
            U_1_area=U_1[:,indexy_center_1[narea]- npts_ndeg_1:indexy_center_1[narea]+ npts_ndeg_1,\
                               indexx_center_1[narea]- npts_ndeg_1:indexx_center_1[narea]+ npts_ndeg_1]
            V_1_area=V_1[:,indexy_center_1[narea]- npts_ndeg_1:indexy_center_1[narea]+ npts_ndeg_1,\
                               indexx_center_1[narea]- npts_ndeg_1:indexx_center_1[narea]+ npts_ndeg_1]
            KE_1_area = 0.5 * (U_1_area.rename('KE')**2 + V_1_area.rename('KE')**2)
    
            U_2_area=U_2[:,indexy_center_1[narea]- npts_ndeg_2:indexy_center_1[narea]+ npts_ndeg_2,\
                               indexx_center_1[narea]- npts_ndeg_2:indexx_center_1[narea]+ npts_ndeg_2]
            V_2_area=V_2[:,indexy_center_1[narea]- npts_ndeg_2:indexy_center_1[narea]+ npts_ndeg_2,\
                               indexx_center_1[narea]- npts_ndeg_2:indexx_center_1[narea]+ npts_ndeg_2]
            KE_2_area = 0.5 * (U_2_area.rename('KE')**2 + V_2_area.rename('KE')**2)
            # save files
            KE_1_area.to_netcdf(file_KE_1)
            KE_2_area.to_netcdf(file_KE_2)
            U_1_area.to_netcdf(file_U_1)
            U_2_area.to_netcdf(file_U_2)
            V_1_area.to_netcdf(file_V_1)
            V_2_area.to_netcdf(file_V_2)
        
            print('Saving file : ', file_KE_1)
            print('Saving file : ', file_KE_2)
     
