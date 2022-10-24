#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:04:08 2019

@author: tbrivoal
"""
import matplotlib as mpl
mpl.use('Agg')
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy.signal import wiener, filtfilt, butter, gaussian, freqz

from matplotlib import pyplot, dates, ticker
import datetime         
mpl.rcParams.update({'font.size': 15})

locator1 = dates.MonthLocator(interval=1)
locator2 = dates.MonthLocator(interval=2)
formatter = dates.DateFormatter('%b %Y')

Tfile_VER1 = xr.open_dataset('FLDR_VERSION1/CFG_NM_1d_gridT_OSTIA.nc', chunks={'time_counter': 50})
Tfile_VER2 = xr.open_dataset('FLDR_VERSION2/CFG_NM_1d_gridT_OSTIA.nc', chunks={'time_counter': 50})
Tfile_OSTIA = xr.open_dataset('/scratch/work/brivoalt/DATA/SST/OSTIA_SST_ORIGIN_y2017-2018_eNEATL36.nc')

T_VER1=Tfile_VER1.votemper.squeeze().sel(time_counter=slice('DSTART_TS', 'DEND'))
T_VER2=Tfile_VER2.votemper.squeeze().sel(time_counter=slice('DSTART_TS', 'DEND'))
T_OSTIA=Tfile_OSTIA.analysed_sst.squeeze().sel(time=slice('DSTART_TS', 'DEND')) - 273.15
TIME=Tfile_VER1.time_counter.sel(time_counter=slice('DSTART_TS', 'DEND'))

# mask data -----------------------------------------------
T_VER1 = T_VER1.where(~np.isnan(T_OSTIA.values) , np.nan)
T_VER2 = T_VER2.where(~np.isnan(T_OSTIA.values) , np.nan)
T_OSTIA = T_OSTIA.where(~np.isnan(T_VER2.values) , np.nan)
print(T_OSTIA.values)
# ---------------------------------------------------------
RMS_T_VER1 = np.nanmean(xr.ufuncs.sqrt((T_VER1.values - T_OSTIA.values)**2),axis=(1,2))
RMS_T_VER2 = np.nanmean(xr.ufuncs.sqrt((T_VER2.values - T_OSTIA.values)**2),axis=(1,2))

T_VER1 = T_VER1.mean(dim='lat').mean(dim='lon')
T_VER2 = T_VER2.mean(dim='lat').mean(dim='lon')
T_OSTIA = T_OSTIA.mean(dim='lat').mean(dim='lon')
plt.figure(figsize=(10,10)) #taille en inch/pouce
ax = plt.subplot(211)
b = gaussian(30, 7)


ax.plot(TIME,T_VER1,'b', linewidth=0.5)
ax.plot(TIME,T_VER2,'r', linewidth=0.5)
ax.plot(TIME,T_OSTIA,'k', linewidth=0.5)


ax.plot(TIME,filters.convolve1d(T_VER1, b/b.sum()),'b',label='VER1', linewidth=3)
ax.plot(TIME,filters.convolve1d(T_VER2, b/b.sum()),'r',label='VER2', linewidth=3)
ax.plot(TIME,filters.convolve1d(T_OSTIA, b/b.sum()),'k',label='OSTIA', linewidth=3)

ax.xaxis.set_major_locator(locator2) 
ax.xaxis.set_minor_locator(locator1) 
ax.xaxis.set_major_formatter(formatter)

plt.title('Sea surface temperature',fontsize=20)
plt.xlabel('',fontsize=15)
plt.ylabel('SST (°C)',fontsize=15)
plt.legend()
plt.xticks(fontsize=12)

ax = plt.subplot(212)
b = gaussian(30, 7)


ax.plot(TIME,RMS_T_VER1 ,'b', linewidth=0.5)
ax.plot(TIME,RMS_T_VER2,'r', linewidth=0.5)


ax.plot(TIME,filters.convolve1d(RMS_T_VER1, b/b.sum()),'b',label='VER1', linewidth=3)
ax.plot(TIME,filters.convolve1d(RMS_T_VER2, b/b.sum()),'r',label='VER2', linewidth=3)

ax.xaxis.set_major_locator(locator2)
ax.xaxis.set_minor_locator(locator1)
ax.xaxis.set_major_formatter(formatter)
plt.title('RMSE with OSTIA',fontsize=20)
plt.xlabel('',fontsize=15)
plt.ylabel('RMSE SST (°C)',fontsize=15)
plt.legend()

plt.xticks(fontsize=12)
plt.savefig('TS_CFG_NM_SST_moyenne_VER_DSTART_TS_DEND.png', dpi=300)
