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
ndeg=1.5

KE_file1 = xr.open_dataset('FLDR_VERSION1/eNEATL36_1h_KE4spectrum.nc') #, chunks={'space': 10})
KE_file2 = xr.open_dataset('FLDR_VERSION2/eNEATL36_1h_KE4spectrum.nc') #, chunks={'space': 10})

Coordfile_1 = xr.open_dataset('FLDR_VERSION1/domain_cfg.nc', drop_variables={"x", "y",})
Coordfile_2 = xr.open_dataset('FLDR_VERSION2/domain_cfg.nc', drop_variables={"x", "y",})

res_1=1/36
npts_ndeg_1=int(ndeg*(1/res_1))
res_2=1/36
npts_ndeg_2=int(ndeg*(1/res_2))


KE_1=KE_file1.KE.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:]#
KE_2=KE_file2.KE.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:]#

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
indexy_center_2=np.zeros(listlat_center.shape).astype(int)
indexx_center_2=np.zeros(listlon_center.shape).astype(int)

nav_lat_1=Coordfile_1.gphit.squeeze().values
nav_lon_1=Coordfile_1.glamt.squeeze().values
nav_lat_2=Coordfile_2.gphit.squeeze().values
nav_lon_2=Coordfile_2.glamt.squeeze().values

lat_found=0
lon_found=0
    #
dx_1=1/(3600)
dx_2=1/(3600)

for narea in range(len(listlat_center)):
    tmplat=abs(nav_lat_1 - listlat_center[narea])
    tmplon=abs(nav_lon_1 - listlon_center[narea])
    tmp=tmplat + tmplon
    print(np.argwhere(tmp == np.nanmin(tmp)).shape)
    indexy_center_1[narea], indexx_center_1[narea] = int(np.argwhere(tmp == np.nanmin(tmp))[0,0]), int(np.argwhere(tmp == np.nanmin(tmp))[0,1])
    indexy_center_2[narea], indexx_center_2[narea] = int(np.argwhere(tmp == np.nanmin(tmp))[0,0]), int(np.argwhere(tmp == np.nanmin(tmp))[0,1])

    print("CFG_NAME, selecting point i, j :",indexy_center_1[narea], indexx_center_1[narea])
    print("at latitude, longitude :", nav_lat_1[indexy_center_1[narea], indexx_center_1[narea]], nav_lon_1[indexy_center_1[narea], indexx_center_1[narea]]) 

    ## SIZE PARAMETERS
    dx_1 = Coordfile_1.e1t.squeeze()[indexy_center_1[narea]- npts_ndeg_1:indexy_center_1[narea]+ npts_ndeg_1,\
                      indexx_center_1[narea]- npts_ndeg_1:indexx_center_1[narea]+ npts_ndeg_1].mean(skipna=True)
    dx_2 = Coordfile_2.e1t.squeeze()[indexy_center_2[narea]- npts_ndeg_2:indexy_center_2[narea]+ npts_ndeg_2,\
                       indexx_center_2[narea]- npts_ndeg_2:indexx_center_2[narea]+ npts_ndeg_2].mean(skipna=True)
    dy_1 = Coordfile_1.e2t.squeeze()[indexy_center_1[narea]- npts_ndeg_1:indexy_center_1[narea]+ npts_ndeg_1,\
                      indexx_center_1[narea]- npts_ndeg_1:indexx_center_1[narea]+ npts_ndeg_1].mean(skipna=True)
    dy_2 = Coordfile_2.e2t.squeeze()[indexy_center_2[narea]- npts_ndeg_2:indexy_center_2[narea]+ npts_ndeg_2,\
                       indexx_center_2[narea]- npts_ndeg_2:indexx_center_2[narea]+ npts_ndeg_2].mean(skipna=True)

#    KE_1_area=KE_1[:,indexy_center_1[narea]- npts_ndeg_1:indexy_center_1[narea]+ npts_ndeg_1,\
#                       indexx_center_1[narea]- npts_ndeg_1:indexx_center_1[narea]+ npts_ndeg_1]
#    print(KE_1)
#    print(indexy_center_1[narea]- npts_ndeg_1,indexy_center_1[narea]+ npts_ndeg_1,\
#                       indexx_center_1[narea]- npts_ndeg_1,indexx_center_1[narea]+ npts_ndeg_1)
#    KE_2_area=KE_2[:,indexy_center_2[narea]- npts_ndeg_2:indexy_center_2[narea]+ npts_ndeg_2,\
#                       indexx_center_2[narea]- npts_ndeg_2:indexx_center_2[narea]+ npts_ndeg_2]

#    print(KE_1_area.values)


## CAREFULL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SHOULD BE DELETED IF AREA IS SELECTED !

    KE_1_area=KE_1.fillna(0.)
    KE_2_area=KE_2.fillna(0.)
## END


    Ny_1 = len(KE_1_area.y)
    Nx_1 = len(KE_1_area.x)
    Ny_2 = len(KE_2_area.y)
    Nx_2 = len(KE_2_area.x)
    print(Ny_1)
    print(dy_1)

    KE_1_area = xr.DataArray(KE_1_area.values, dims=['time_counter','YC','XC'],
                              coords={'time_counter': np.arange(len(KE_1_area.time_counter))*3600 * 24, 
                                      'YC':np.arange(0,Ny_1*dy_1,dy_1),
                                      'XC':np.arange(0,Nx_1*dx_1,dx_1)})
    KE_2_area = xr.DataArray(KE_2_area.values, dims=['time_counter','YC','XC'],
                              coords={'time_counter': np.arange(len(KE_2_area.time_counter))*3600 * 24,
                                      'YC':np.arange(0,Ny_2*dy_2,dy_2),
                                      'XC':np.arange(0,Nx_2*dx_2,dx_2)})


    FKE_1_area = xrft.fft(xrft.fft(KE_1_area.fillna(0.), 
                                     dim=['YC', 'XC'], window='hann', detrend='constant', 
                                     true_phase=True, true_amplitude=True 
                                    ), 
                            dim=['time_counter'], window='hann', detrend='constant', 
                            true_phase=True, true_amplitude=True
                           )
    FKE_2_area = xrft.fft(xrft.fft(KE_2_area.fillna(0.),
                                     dim=['YC', 'XC'], window='hann', detrend='constant',
                                     true_phase=True, true_amplitude=True
                                    ),
                            dim=['time_counter'], window='hann', detrend='constant',
                            true_phase=True, true_amplitude=True
                           )

    FKE_1_area = FKE_1_area.isel(freq_time_counter=slice(len(FKE_1_area.freq_time_counter)//2,None)) * 2
    FKE_2_area = FKE_2_area.isel(freq_time_counter=slice(len(FKE_2_area.freq_time_counter)//2,None)) * 2
 
    isoFKE_1_area = kw.isotropize(kw.density(np.abs(FKE_1_area)**2, ['freq_time_counter','freq_YC','freq_XC']), ['freq_YC','freq_XC'], nfactor=4, kwargs={'truncate':True}).compute() 
    isoFKE_2_area = kw.isotropize(kw.density(np.abs(FKE_2_area)**2, ['freq_time_counter','freq_YC','freq_XC']), ['freq_YC','freq_XC'], nfactor=4, kwargs={'truncate':True}).compute()

    fig, axes = plt.subplots(figsize=(13,5), nrows=1, ncols=2)
    fig.set_tight_layout(True)
    ax00 = axes[0]
    ax01 = axes[1]
    print(np.nanmin(isoFKE_1_area.isel(freq_r=slice(1,None),freq_time_counter=slice(1,None)).values))
    print(np.nanmax(isoFKE_1_area.isel(freq_r=slice(1,None),freq_time_counter=slice(1,None)).values))
 
    im0 = ax00.pcolormesh(isoFKE_1_area.freq_r.isel(freq_r=slice(1,None)) * 1e3, 
                          isoFKE_1_area.freq_time_counter.isel(freq_time_counter=slice(1,None)) *3600 ,
                          (isoFKE_1_area.isel(freq_r=slice(1,None),freq_time_counter=slice(1,None)).values), 
                          cmap='Spectral_r', 
                          norm=clr.LogNorm(vmin=1, vmax=1e10), 
                          shading='auto', rasterized=True)
   
    im1 = ax01.pcolormesh(isoFKE_2_area.freq_r.isel(freq_r=slice(1,None)) * 1e3,
                          isoFKE_2_area.freq_time_counter.isel(freq_time_counter=slice(1,None)) *3600,
                          (isoFKE_2_area.isel(freq_r=slice(1,None),freq_time_counter=slice(1,None))),
                          cmap='Spectral_r',
                          norm=clr.LogNorm(vmin=1e-6, vmax=1e6),
                          shading='auto', rasterized=True)

 
    ax00.set_xscale('log')
    ax00.set_yscale('log')
    ax00.set_ylim([1e-3,None])
    ax00.set_xlim([1e-3,None])
    ax00.set_xlabel('wavenumber (cpkm)')
    ax00.set_ylabel('frequency (cph)')

    ax01.set_xscale('log')
    ax01.set_yscale('log')
    ax01.set_ylim([1e-3,None])
    ax01.set_xlim([1e-3,None])
    ax01.set_xlabel('wavenumber (cpkm)')
    ax01.set_ylabel('frequency (cph)')

    fig.colorbar(im0, ax=ax00)
    fig.colorbar(im1, ax=ax01)
    
    plt.savefig('test_k_omega')
