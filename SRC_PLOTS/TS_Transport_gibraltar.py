#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:04:08 2019

@author: tbrivoal
"""
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib import colors as mcolors
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import wiener, filtfilt, butter, gaussian, freqz
from scipy.ndimage import filters

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

Tfile_VER1 = xr.open_dataset('FLDR_VERSION1/CFG_NAME_1d_gridU25h_DSTART_DEND_extract_gibraltar.nc').sel(time_counter=slice('DSTART', 'DEND'))
Tfile_AGRIF_VER1 = xr.open_dataset('FLDR_VERSION1/AGRIF_CFG_NAME_1d_gridU25h_DSTART_DEND_extract_gibraltar.nc').sel(time_counter=slice('DSTART', 'DEND'))
Tfile_VER2 = xr.open_dataset('FLDR_VERSION2/CFG_NAME_1d_gridU25h_DSTART_DEND_extract_gibraltar.nc').sel(time_counter=slice('DSTART', 'DEND'))

Coordfile_VER1 = xr.open_mfdataset('FLDR_VERSION1/domain_cfg.nc', drop_variables={"x", "y",}).squeeze().isel(y=np.arange(y_min,y_max)).isel(x=np.arange(x_min-20,x_min+20))
Coordfile_AGRIF_VER1 = xr.open_mfdataset('FLDR_VERSION1/1_domain_cfg.nc', drop_variables={"x", "y",}).squeeze().isel(y=np.arange(y_min_agrif,y_max_agrif)).isel(x=np.arange(x_min_agrif-20,x_min_agrif+20))
Coordfile_VER2 = xr.open_mfdataset('FLDR_VERSION2/domain_cfg.nc', drop_variables={"x", "y",}).squeeze().isel(y=np.arange(y_min,y_max)).isel(x=np.arange(x_min-20,x_min+20))


print(Coordfile_AGRIF_VER1.glamt.values)#
print(Coordfile_VER1.glamt.values)
#print('dataset loaded')
#
area_sec_VER1=Coordfile_VER1.e2t.values[:,20] * Coordfile_VER1.e3t_0.values[:,:,20]
area_sec_AGRIF_VER1=Coordfile_AGRIF_VER1.e2t.values[:,20] * Coordfile_AGRIF_VER1.e3t_0.values[:,:,20]
area_sec_VER2=Coordfile_VER2.e2t.values[:,20] * Coordfile_VER2.e3t_0.values[:,:,20]
print(area_sec_VER1.shape)
print(area_sec_AGRIF_VER1.shape)
U_tot_VER1 = Tfile_VER1.vozocrtx[:,:,:,20]
U_tot_AGRIF_VER1 = Tfile_AGRIF_VER1.vozocrtx[:,:,:,20-1:20+1].mean(dim='x')
U_tot_VER2 = Tfile_VER2.vozocrtx[:,:,:,20]
#
U_tot_VER1_out=U_tot_VER1.where(U_tot_VER1 < 0, np.nan).values 
U_tot_AGRIF_VER1_out=U_tot_AGRIF_VER1.where(U_tot_AGRIF_VER1 < 0, np.nan).values
U_tot_VER2_out=U_tot_VER2.where(U_tot_VER2 < 0, np.nan).values
#
#print('sum')
U_tot_VER1_out = np.nansum(U_tot_VER1_out * area_sec_VER1, axis=(1,2)) / 10e6
U_tot_AGRIF_VER1_out = np.nansum(U_tot_AGRIF_VER1_out * area_sec_AGRIF_VER1,axis=(1,2)) / 10e6
U_tot_VER2_out = np.nansum(U_tot_VER2_out * area_sec_VER2,axis=(1,2)) / 10e6
#
#print('inflow')
#print('mask_data')
#
U_tot_VER1_in=U_tot_VER1.where(U_tot_VER1 > 0, np.nan).values
U_tot_AGRIF_VER1_in=U_tot_AGRIF_VER1.where(U_tot_AGRIF_VER1 > 0, np.nan).values
U_tot_VER2_in=U_tot_VER2.where(U_tot_VER2 > 0, np.nan).values
#
#print('sum')
U_tot_VER1_in = np.nansum(U_tot_VER1_in * area_sec_VER1,axis=(1,2)) / 10e6
U_tot_AGRIF_VER1_in = np.nansum(U_tot_AGRIF_VER1_in * area_sec_AGRIF_VER1,axis=(1,2)) / 10e6
U_tot_VER2_in = np.nansum(U_tot_VER2_in * area_sec_VER2,axis=(1,2)) / 10e6

U_tot_VER1_net = np.nansum(U_tot_VER1 * area_sec_VER1,axis=(1,2)) / 10e6
U_tot_AGRIF_VER1_net = np.nansum(U_tot_AGRIF_VER1 * area_sec_AGRIF_VER1,axis=(1,2)) / 10e6
U_tot_VER2_net = np.nansum(U_tot_VER2 * area_sec_VER2,axis=(1,2)) / 10e6

b = gaussian(30, 7)


#
print('plotting ...')
plt.figure(figsize=(10,15))
ax = plt.subplot(311)
ax.plot(Tfile_VER1.time_counter,filters.convolve1d(U_tot_VER1_out, b/b.sum()), color='b',label='VER1')
ax.plot(Tfile_AGRIF_VER1.time_counter, filters.convolve1d(U_tot_AGRIF_VER1_out, b/b.sum()), color='b',ls='dotted', label='VER1 (zoom)')
ax.plot(Tfile_VER2.time_counter, filters.convolve1d(U_tot_VER2_out, b/b.sum()), color='r',label='VER2')
ax.set_xlabel('date')
ax.set_ylabel('Flow (Sv)')
ax.set_title('outflow (Sw)')
plt.legend()
ax = plt.subplot(312)

ax.plot(Tfile_VER1.time_counter,filters.convolve1d(U_tot_VER1_in, b/b.sum()), color='b',label='VER1')
ax.plot(Tfile_AGRIF_VER1.time_counter, filters.convolve1d(U_tot_AGRIF_VER1_in, b/b.sum()), color='b',ls='dotted', label='VER1 (zoom)')
ax.plot(Tfile_VER2.time_counter, filters.convolve1d(U_tot_VER2_in, b/b.sum()), color='r',label='VER2')
ax.set_xlabel('date')
ax.set_ylabel('Flow (Sv)')
ax.set_title('inflow (Sw)')
plt.legend()

ax = plt.subplot(313)

ax.plot(Tfile_VER1.time_counter,filters.convolve1d(U_tot_VER1_net, b/b.sum()), color='b',label='VER1')
ax.plot(Tfile_AGRIF_VER1.time_counter, filters.convolve1d(U_tot_AGRIF_VER1_net, b/b.sum()), color='b',ls='dotted', label='VER1 (zoom)')
ax.plot(Tfile_VER2.time_counter, filters.convolve1d(U_tot_VER2_net, b/b.sum()), color='r',label='VER2')
ax.set_xlabel('date')
ax.set_ylabel('Flow (Sv)')
ax.set_title('netflow (Sw)')


plt.savefig('TS_Transports_gibraltar_VER2_m_VER1_DSTART_DEND.png',dpi=400)


