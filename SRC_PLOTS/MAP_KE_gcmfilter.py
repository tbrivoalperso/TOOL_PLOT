#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:04:08 2019

@author: tbrivoal
"""
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import scipy.stats as st
from scipy.stats import linregress
from numpy.polynomial.polynomial import polyfit
from scipy import signal
from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import netCDF4 as nc
from matplotlib import pyplot
from matplotlib import colors as mcolors
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gcm_filters


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

Tfile_VER = xr.open_dataset('FLDR/CFG_NM_1d_KE2.nc', chunks={'time_counter': 50})
Coordfile = xr.open_dataset('FLDR/domain_cfg.nc', drop_variables={"x", "y",})

T_VER=Tfile_VER.KE2.squeeze().sel(time_counter=slice('DSTART', 'DEND')) *100 *100 *0.5# .mean(dim='time_counter') *100 *100 * 0.5 
wet_mask = xr.where(~xr.ufuncs.isnan(T_VER[:,:,:].squeeze()), 1, 0)
print(T_VER.shape)
print(wet_mask)
dxe = Coordfile.e1t.squeeze()
dye = Coordfile.e2t.squeeze()
dxn = Coordfile.e1v.squeeze()
dyn = Coordfile.e2v.squeeze()
dx_min = min(dxe.min(),dye.min(),dxn.min(),dyn.min())
print(dx_min)
area = dxe * dye
print('filter')
#kappa_w = xr.ones_like(dxe)
#kappa_s = xr.ones_like(dxe)
#specs = {
#    'filter_scale': 10000,
#    'dx_min': dx_min.values,
#    'filter_shape': gcm_filters.FilterShape.GAUSSIAN
#}
#
#filter_tripolar_regular_with_land = gcm_filters.Filter(
#    **specs,
#    grid_type=gcm_filters.GridType.IRREGULAR_WITH_LAND,
#    grid_vars={'area': area,'dxw': dxe, 'dyw': dye, 'dxs': dxn, 'dys': dyn, 'wet_mask': wet_mask,'kappa_w': kappa_w, 'kappa_s': kappa_s}
#)
specs = {
    'filter_scale': 20,
    'dx_min': 1,
    'filter_shape': gcm_filters.FilterShape.GAUSSIAN
}

filter_tripolar_regular_with_land = gcm_filters.Filter(
    **specs,
    grid_type=gcm_filters.GridType.TRIPOLAR_REGULAR_WITH_LAND_AREA_WEIGHTED,
    grid_vars={'area': area, 'wet_mask': wet_mask}
)

KE_LS=filter_tripolar_regular_with_land.apply(T_VER.squeeze(), dims=['y', 'x'])

# plots spectrums
KE_LS_mean=KE_LS[0,:,:] #.mean(dim="time_counter")
KE_spectre=KE_LS.where(~np.isnan(KE_LS_mean.values),0)

for j in range(len(KE_spectre[0,0,:])):
       ff_KE_spectre[:,j], ff_KE_spectreki0[:,j] = signal.welch(KE_spectre[:,j], fs=dx,nperseg=int(len(KE_spectre[:,0])), window='hanning', noverlap=no,nfft=2*int(len(KE_spectre[:,0])-1),  detrend='linear', return_onesided=True, scaling='spectrum')

mean_f0_KE_spectre = np.nanmean(ff_KE_spectre,axis=1)
mean_fi0_KE_spectre = np.nanmean(ff_KE_spectreki0,axis=1)

ax = plt.subplot(111)
ax.plot(mean_f0_KE_spectre,mean_fi0_KE_spectre, 'k', lw=2, label ='VER')
ax.set_yscale('log')
ax.set_xscale('log')

ax.set_xlim(wnmin,wnmax)
ax.legend()
print(mean_fi0_KE_spectre)
ax.set_ylabel('power') # regex: ($10log10$)
ax.set_xlabel('Wavenumber (1/km)')

plt.savefig('spectre_KE_filt_gcmfilter_DSTART_DEND.png')


#KE_LS=filter_tripolar_regular_with_land.apply(T_VER.squeeze(), dims=['y', 'x'])
#
#KE_filt=(T_VER - KE_LS).mean(dim='time_counter')
#print('filter OK')
##print(KE_filt.shape)
#matplotlib.rcParams.update({'font.size': 10})
#proj=ccrs.PlateCarree()
#plt.figure(figsize=(8,8))
#ax = plt.subplot(111, projection=proj)
#lat=Coordfile.gphit.squeeze().values
#lon=Coordfile.glamt.squeeze().values
#
#ax.coastlines(resolution='50m')
#lon_formatter = LongitudeFormatter(degree_symbol='° ')
#lat_formatter = LatitudeFormatter(degree_symbol='° ')
#ax.xaxis.set_major_formatter(lon_formatter)
#ax.yaxis.set_major_formatter(lat_formatter)
#
#axes=ax.gridlines( draw_labels=False, linewidth=0)
#axes.ylabels_right = False
#axes.xlabels_top = False
#im1 = plt.contourf(lon, lat, KE_filt ,levels=np.linspace(0,600,51), cmap="jet",transform=ccrs.PlateCarree(),extend='both')
#cbar=plt.colorbar(im1,ticks=np.linspace(0,600,11),orientation="vertical",fraction=0.046, pad=0.04) #ticks=[-40,20,0,20,40]) #ticks=[0,1,2,3]) #
#cbar.set_label('KE (m²/s²)', rotation=270, labelpad=12, fontsize=12,fontweight="bold")
#ax.set_title('VER')
#plt.tight_layout()
#
#plt.savefig('CFG_NM_KE_moyenne_VER_DSTART_DEND')
#plt.close()
#    
#     
