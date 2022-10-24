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
ssh_1=ssh_1_file.sossheig.squeeze()[1:357,150:330,8:120] * 100 

ssh_2_file = xr.open_dataset('FLDR_VERSION2/CFG_NM_1h_grid2D.nc') #, chunks={'time': 10})
ssh_2=ssh_2_file.sossheig.squeeze()[1:357,150:330,8:120] * 100 

ff_ssh_1=np.zeros((int(len(ssh_1[:,0,0])),int(len(ssh_1[0,0,:])),int(len(ssh_1[0,0,:])+1)))
ff_ssh_1ki0=np.zeros((int(len(ssh_1[:,0,0])),int(len(ssh_1[0,0,:])),int(len(ssh_1[0,0,:])+1)))

ff_ssh_2=np.zeros((int(len(ssh_2[:,0,0])),int(len(ssh_2[0,0,:])),int(len(ssh_2[0,0,:])+1)))
ff_ssh_2ki0=np.zeros((int(len(ssh_2[:,0,0])),int(len(ssh_2[0,0,:])),int(len(ssh_2[0,0,:])+1)))

no = 0  # Nombres d'overlapping points
dx=1/7 # 1/ largeur de maille moyenne en km

for i in range(len(ssh_1[:,0,0])):
    print(i)
    for j in range(len(ssh_1[0,0,:])):
        ff_ssh_1[i,j,:], ff_ssh_1ki0[i,j,:] = signal.welch(ssh_1[i,j,:], fs=dx, window='hanning', nperseg=int(len(ssh_1[0,0,:])), noverlap=no, nfft=2*len(ssh_1[0,0,:]), detrend='linear', return_onesided=True, scaling='spectrum')

        ff_ssh_2[i,j,:], ff_ssh_2ki0[i,j,:] = signal.welch(ssh_2[i,j,:], fs=dx, window='hanning', nperseg=int(len(ssh_2[0,0,:])), noverlap=no, nfft=2*len(ssh_2[0,0,:]), detrend='linear', return_onesided=True, scaling='spectrum')

mean_f0_ssh_1 = np.nanmean(ff_ssh_1,axis=(0,1))
mean_fi0_ssh_1 = np.nanmean(ff_ssh_1ki0,axis=(0,1))

mean_f0_ssh_2 = np.nanmean(ff_ssh_2,axis=(0,1))
mean_fi0_ssh_2 = np.nanmean(ff_ssh_2ki0,axis=(0,1))

ax = plt.subplot(111)
ax.plot(1/mean_f0_ssh_1,mean_fi0_ssh_1, 'g-', lw=2, label ='ABL REL')
ax.plot(1/mean_f0_ssh_2,mean_fi0_ssh_2, 'g', lw=2, label ='ABL ABS', linestyle='--')
ax.set_yscale('log')
ax.set_xlim(0,500)
ax.legend()

ax.set_ylabel('power') # regex: ($10log10$)
ax.set_xlabel('wavelenght (km)')


plt.savefig('space_spectrum_KE_ABL.png')
plt.close()

