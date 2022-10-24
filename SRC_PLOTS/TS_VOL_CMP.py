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

Tfile_VER1 = xr.open_dataset('FLDR_VERSION1/CFG_NM_1h_grid2D.nc', chunks={'time_counter': 50})
Tfile_VER2 = xr.open_dataset('FLDR_VERSION2/CFG_NM_1h_grid2D.nc', chunks={'time_counter': 50})

T_VER1=Tfile_VER1.somxlavt.squeeze().sel(time_counter=slice('DSTART_TS', 'DEND')).mean(dim='x').mean(dim='y')
T_VER2=Tfile_VER2.somxlavt.squeeze().sel(time_counter=slice('DSTART_TS', 'DEND')).mean(dim='x').mean(dim='y')
TIME=Tfile_VER1.time_counter



plt.figure(figsize=(10,7)) #taille en inch/pouce
ax = plt.subplot(111)
b = gaussian(30, 7)


ax.plot(TIME,T_VER1,'b', linewidth=0.5)
ax.plot(TIME,T_VER2,'r', linewidth=0.5)


ax.plot(TIME,filters.convolve1d(T_VER1, b/b.sum()),'b',label='VER1', linewidth=3)
ax.plot(TIME,filters.convolve1d(T_VER2, b/b.sum()),'r',label='VER2', linewidth=3)

ax.xaxis.set_major_locator(locator2) 
ax.xaxis.set_minor_locator(locator1) 
ax.xaxis.set_major_formatter(formatter)

plt.xticks(fontsize=12)

plt.title('Mixed Layer depth',fontsize=20)
plt.xlabel('',fontsize=15)
plt.ylabel('MLD (m)',fontsize=15)
plt.legend()
plt.savefig('TS_CFG_NM_KE_moyenne_VER_DSTART_TS_DEND.png', dpi=300)
