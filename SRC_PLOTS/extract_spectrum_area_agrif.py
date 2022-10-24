#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:04:08 2019

@author: tbrivoal
"""
import matplotlib as mpl
mpl.use('Agg')
import cartopy.crs as ccrs
import sys
import xarray as xr
import numpy as np
import scipy.stats as st
from scipy.stats import linregress
from numpy.polynomial.polynomial import polyfit
from scipy import signal
from scipy.stats import norm
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import matplotlib.pyplot as plt
from scipy import stats
import netCDF4 as nc
import matplotlib.ticker as mticker
from matplotlib import colors
def add_second_axis(ax1):
     """ Add a x-axis at the top of the spectra figures """
     ax2 = ax1.twiny()
     ax2.set_xscale('log')
     ax2.set_xlim(ax1.axis()[0], ax1.axis()[1])
     kp = 1./np.array([50., 30.,15.,10.,5.,3.])
     lp=np.array([50., 30.,15.,10.,5.,3.])
     ax2.set_xticks(kp)
     ax2.set_xticklabels(lp)
     ax2.grid()
     plt.xlabel('Wavelength (km)')

def plot_pdf(var,color,label,linestyle='solid'):
    minrng_inp2  =  0
    maxrng_inp2  =  250
    binwdth_inp2 =  0.5*2
    minrng_inp1  =  0
    maxrng_inp1  =  250
    binwdth_inp1 =  0.5*2



    bin_edges_inp1 = np.arange( minrng_inp1, maxrng_inp1, binwdth_inp1)
    bin_edges_inp2 = np.arange( minrng_inp2, maxrng_inp2, binwdth_inp2)
    bin_center_inp1 = (bin_edges_inp1[1:] + bin_edges_inp1[:-1]) * .5
    bin_widths_inp1 =  bin_edges_inp1[1:] - bin_edges_inp1[:-1]
    bin_center_inp2 = (bin_edges_inp2[1:] + bin_edges_inp2[:-1]) * .5
    bin_widths_inp2 =  bin_edges_inp2[1:] - bin_edges_inp2[:-1]

    buf_n = var
    buf_n = np.array(sorted(buf_n[~np.isnan(buf_n)]))
    mu, std = norm.fit(buf_n)
    p = norm.pdf(buf_n, mu, std)
    plt.plot(buf_n, p, color,linewidth=2,label=label, linestyle=linestyle)
    plt.axvline(x=mu,linewidth=2,color=color, linestyle=linestyle)
    plt.xlim(0,maxrng_inp1)
    plt.xlabel('KE (m²/s²)') #,fontweight="bold")
    plt.ylabel('Probabiliy density') #,fontweight="bold")


###################################################################################################################
ISAGRIF1=False
ISAGRIF2=False
ndeg=2.5

U_1_file = xr.open_dataset('FLDR_VERSION1/CFG_NAME_1h_gridU.nc') #, chunks={'space': 10})
V_1_file = xr.open_dataset('FLDR_VERSION1/CFG_NAME_1h_gridV.nc') #, chunks={'space': 10})

Coordfile_1 = xr.open_dataset('FLDR_VERSION1/domain_cfg.nc', drop_variables={"x", "y",})
res_1=1/36
npts_ndeg_1=int(ndeg*(1/res_1))

if ISAGRIF1:
    U_11_file = xr.open_dataset('FLDR_VERSION1/AGRIF_CFG_NAME_1h_gridU.nc') #, chunks={'space': 10})
    V_11_file = xr.open_dataset('FLDR_VERSION1/AGRIF_CFG_NAME_1h_gridV.nc') #, chunks={'space': 10})

    Coordfile_11 = xr.open_dataset('FLDR_VERSION1/1_domain_cfg.nc', drop_variables={"x", "y",})
    res_11=1/108
    npts_ndeg_11=int(ndeg*(1/res_11))

U_2_file = xr.open_dataset('FLDR_VERSION2/CFG_NAME_1h_gridU.nc') #, chunks={'space': 10})
V_2_file = xr.open_dataset('FLDR_VERSION2/CFG_NAME_1h_gridV.nc') #, chunks={'space': 10})

Coordfile_2 = xr.open_dataset('FLDR_VERSION2/domain_cfg.nc', drop_variables={"x", "y",})
res_2=1/36
npts_ndeg_2=int(ndeg*(1/res_2))

if ISAGRIF2:
    U_21_file = xr.open_dataset('FLDR_VERSION2/AGRIF_CFG_NAME_1h_gridU.nc') #, chunks={'space': 10})
    V_21_file = xr.open_dataset('FLDR_VERSION2/AGRIF_CFG_NAME_1h_gridV.nc') #, chunks={'space': 10})

    Coordfile_21 = xr.open_dataset('FLDR_VERSION2/1_domain_cfg.nc', drop_variables={"x", "y",})
    res_21=1/108
    npts_ndeg_21=int(ndeg*(1/res_21))


U_1=U_1_file.sozocrtx.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:]#
V_1=V_1_file.somecrty.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:]#
if ISAGRIF1: U_11=U_11_file.sozocrtx.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:]#
if ISAGRIF1: V_11=V_11_file.somecrty.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:]#

U_2=U_2_file.sozocrtx.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:]#
V_2=V_2_file.somecrty.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:]#

if ISAGRIF2: U_21=U_21_file.sozocrtx.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:]#
if ISAGRIF2: V_21=V_21_file.somecrty.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:]#

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
if ISAGRIF1: nav_lat_11=Coordfile_11.gphit.squeeze().values
if ISAGRIF1: nav_lon_11=Coordfile_11.glamt.squeeze().values
lat_found=0
lon_found=0
    #
dx_1=1/(3600)
dx_11=1/(3600)
for narea in range(len(listlat_center)):
    file_KE_1 = 'FLDR_VERSION1/CFG_NAME_1h_gridKE_'+str(areanames[narea])+'.nc'
    file_U_1 = 'FLDR_VERSION1/CFG_NAME_1h_gridU_'+str(areanames[narea])+'.nc'
    file_V_1 = 'FLDR_VERSION1/CFG_NAME_1h_gridV_'+str(areanames[narea])+'.nc'
    file_KE_2 = 'FLDR_VERSION2/CFG_NAME_1h_gridKE_'+str(areanames[narea])+'.nc'
    file_U_2 = 'FLDR_VERSION2/CFG_NAME_1h_gridU_'+str(areanames[narea])+'.nc'
    file_V_2 = 'FLDR_VERSION2/CFG_NAME_1h_gridV_'+str(areanames[narea])+'.nc'
    if ISAGRIF1:
        file_KE_11 = 'FLDR_VERSION1/AGRIF_CFG_NAME_1h_gridKE_'+str(areanames[narea])+'.nc'
        file_U_11 = 'FLDR_VERSION1/AGRIF_CFG_NAME_1h_gridU_'+str(areanames[narea])+'.nc'
        file_V_11 = 'FLDR_VERSION1/AGRIF_CFG_NAME_1h_gridV_'+str(areanames[narea])+'.nc'
    if ISAGRIF2:
        file_KE_21 = 'FLDR_VERSION2/AGRIF_CFG_NAME_1h_gridKE_'+str(areanames[narea])+'.nc'
        file_U_21 = 'FLDR_VERSION2/AGRIF_CFG_NAME_1h_gridU_'+str(areanames[narea])+'.nc'
        file_V_21 = 'FLDR_VERSION2/AGRIF_CFG_NAME_1h_gridV_'+str(areanames[narea])+'.nc'

    # Selecting SSH area
    tmplat=abs(nav_lat_1 - listlat_center[narea])
    tmplon=abs(nav_lon_1 - listlon_center[narea])
    tmp=tmplat + tmplon
    print(np.argwhere(tmp == np.nanmin(tmp)).shape)
    indexy_center_1[narea], indexx_center_1[narea] = int(np.argwhere(tmp == np.nanmin(tmp))[0,0]), int(np.argwhere(tmp == np.nanmin(tmp))[0,1])
    print("CFG_NAME, selecting point i, j :",indexy_center_1[narea], indexx_center_1[narea])
    print("at latitude, longitude :", nav_lat_1[indexy_center_1[narea], indexx_center_1[narea]], nav_lon_1[indexy_center_1[narea], indexx_center_1[narea]]) 
    U_1_area=U_1[:,indexy_center_1[narea]- npts_ndeg_1:indexy_center_1[narea]+ npts_ndeg_1,\
                       indexx_center_1[narea]- npts_ndeg_1:indexx_center_1[narea]+ npts_ndeg_1]
    print(U_1_area.shape)
    V_1_area=V_1[:,indexy_center_1[narea]- npts_ndeg_1:indexy_center_1[narea]+ npts_ndeg_1,\
                       indexx_center_1[narea]- npts_ndeg_1:indexx_center_1[narea]+ npts_ndeg_1]
    KE_1_area = 0.5 * (U_1_area.rename('KE')**2 + V_1_area.rename('KE')**2)
    print(KE_1_area.shape)
    U_2_area=U_2[:,indexy_center_1[narea]- npts_ndeg_2:indexy_center_1[narea]+ npts_ndeg_2,\
                       indexx_center_1[narea]- npts_ndeg_2:indexx_center_1[narea]+ npts_ndeg_2]
    V_2_area=V_2[:,indexy_center_1[narea]- npts_ndeg_2:indexy_center_1[narea]+ npts_ndeg_2,\
                       indexx_center_1[narea]- npts_ndeg_2:indexx_center_1[narea]+ npts_ndeg_2]
    KE_2_area = 0.5 * (U_2_area.rename('KE')**2 + V_2_area.rename('KE')**2)

    if ISAGRIF1: 
        tmplat=abs(nav_lat_11 - listlat_center[narea])
        tmplon=abs(nav_lon_11 - listlon_center[narea])
        tmp=tmplat + tmplon
        indexy_center_11[narea], indexx_center_11[narea] = int(np.argwhere(tmp == np.nanmin(tmp))[0,0]), int(np.argwhere(tmp == np.nanmin(tmp))[0,1])
        print("AGRIF_CFG_NAME, selecting point i, j :",indexy_center_11[narea], indexx_center_11[narea])
        print("at latitude, longitude :", nav_lat_11[indexy_center_11[narea], indexx_center_11[narea]], nav_lon_11[indexy_center_11[narea], indexx_center_11[narea]])
        U_11_area=U_11[:,indexy_center_11[narea]- npts_ndeg_11:indexy_center_11[narea]+ npts_ndeg_11,\
                                              indexx_center_11[narea]- npts_ndeg_11:indexx_center_11[narea]+ npts_ndeg_11]
        V_11_area=V_11[:,indexy_center_11[narea]- npts_ndeg_11:indexy_center_11[narea]+ npts_ndeg_11,\
                                              indexx_center_11[narea]- npts_ndeg_11:indexx_center_11[narea]+ npts_ndeg_11]
        KE_11_area = 0.5 * (U_11_area.rename('KE')**2 + V_11_area.rename('KE')**2)

    if ISAGRIF2:
        U_21_area=U_21[:,indexy_center_11[narea]- npts_ndeg_21:indexy_center_11[narea]+ npts_ndeg_21,\
                             indexx_center_11[narea]- npts_ndeg_21:indexx_center_11[narea]+ npts_ndeg_21]
        V_21_area=V_21[:,indexy_center_11[narea]- npts_ndeg_21:indexy_center_11[narea]+ npts_ndeg_21,\
                             indexx_center_11[narea]- npts_ndeg_21:indexx_center_11[narea]+ npts_ndeg_21]
        KE_21_area = 0.5 * (U_21_area.rename('KE')**2 + V_21_area.rename('KE')**2)
    
    KE_1_area.to_netcdf(file_KE_1)
    U_1_area.to_netcdf(file_U_1)
    V_1_area.to_netcdf(file_V_1)

    KE_2_area.to_netcdf(file_KE_2)
    U_2_area.to_netcdf(file_U_2)
    V_2_area.to_netcdf(file_V_2)

    if ISAGRIF1:
        KE_11_area.to_netcdf(file_KE_11)
        U_11_area.to_netcdf(file_U_11)
        V_11_area.to_netcdf(file_V_11)

    if ISAGRIF2:
        KE_21_area.to_netcdf(file_KE_21)
        U_21_area.to_netcdf(file_U_21)
        V_21_area.to_netcdf(file_V_21)

    #
    #
    #
