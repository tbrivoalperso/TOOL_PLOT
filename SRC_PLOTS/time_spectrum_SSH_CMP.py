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
import scipy.stats as st
from scipy.stats import linregress
from numpy.polynomial.polynomial import polyfit
from scipy import signal
from scipy.stats import norm

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

ssh_1_file = xr.open_dataset('FLDR_VERSION1/CFG_NM_1h_grid2D.nc') #, chunks={'time': 10})
ssh_1=ssh_1_file.sossheig.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,::9,::9].stack(xy=("x", "y"))
ssh_1=ssh_1.dropna(dim="xy")
ssh_1=ssh_1.values

ssh_2_file = xr.open_dataset('FLDR_VERSION2/CFG_NM_1h_grid2D.nc') #, chunks={'time': 10})
ssh_2=ssh_2_file.sossheig.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,::9,::9].stack(xy=("x", "y"))
ssh_2=ssh_2.dropna(dim="xy")
ssh_2=ssh_2.values

ff_ssh_1=np.zeros((int(len(ssh_1[:,0])),int(len(ssh_1[0,:]))))
ff_ssh_1ki0=np.zeros((int(len(ssh_1[:,0])),int(len(ssh_1[0,:]))))

ff_ssh_2=np.zeros((int(len(ssh_2[:,0])),int(len(ssh_2[0,:]))))
ff_ssh_2ki0=np.zeros((int(len(ssh_2[:,0])),int(len(ssh_2[0,:]))))
print(ssh_1.shape)
print(ssh_2.shape)

no = 0  # Nombres d'overlapping points
dx=1/3600 # 1/ largeur de maille moyenne en km
freqmin_hz=1/(31*24*3600)
freqmax_hz=dx
periodmax_h=24*31 # 1mois #(1/freqmax_hz)/3600
print(ff_ssh_1.shape)
for i in range(len(ssh_1[0,:])):
#    print(i)
    ff_ssh_1[:,i], ff_ssh_1ki0[:,i] = signal.welch(ssh_1[:,i], fs=dx,nperseg=int(len(ssh_1[:,0])), window='hanning', noverlap=no,nfft=2*int(len(ssh_1[:,0])-1),  detrend='linear', return_onesided=True, scaling='spectrum')

for i in range(len(ssh_2[0,:])):
    ff_ssh_2[:,i], ff_ssh_2ki0[:,i] = signal.welch(ssh_2[:,i], fs=dx,nperseg=int(len(ssh_2[:,0])), window='hanning', noverlap=no,nfft=2*int(len(ssh_2[:,0])-1),  detrend='linear', return_onesided=True, scaling='spectrum')

mean_f0_ssh_1 = np.nanmean(ff_ssh_1,axis=1)
mean_fi0_ssh_1 = np.nanmean(ff_ssh_1ki0,axis=1)
mean_f0_ssh_2 = np.nanmean(ff_ssh_2,axis=1)
mean_fi0_ssh_2 = np.nanmean(ff_ssh_2ki0,axis=1)

ax = plt.subplot(111)
ax.plot(mean_f0_ssh_1,mean_fi0_ssh_1, 'b-', lw=2, label ='VER1')
ax.plot(mean_f0_ssh_2,mean_fi0_ssh_2, 'r-', lw=2, label ='VER2')

ax.set_yscale('log')
ax.set_xscale('log')

ax.set_xlim(freqmin_hz,freqmax_hz)
ax.legend()
print(mean_fi0_ssh_1)
ax.set_ylabel('power') # regex: ($10log10$)
ax.set_xlabel('Frequency (Hz)')

plt.savefig('CFG_NM_time_spectrum_freq_VER1_VER2_DSTART_DEND.png')
plt.close()



ax = plt.subplot(111)
ax.plot((1/mean_f0_ssh_1)/3600,mean_fi0_ssh_1, 'b-', lw=2, label ='VER1')
ax.plot((1/mean_f0_ssh_2)/3600,mean_fi0_ssh_2, 'r-', lw=2, label ='VER2')

ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
ax.set_xlim(0,periodmax_h)
print(mean_fi0_ssh_1)
ax.set_ylabel('power') # regex: ($10log10$)
ax.set_xlabel('Period (h)')

plt.savefig('CFG_NM_time_spectrum_period_VER1_VER2_DSTART_DEND.png')
plt.close()

