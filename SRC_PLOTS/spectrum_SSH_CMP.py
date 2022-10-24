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

vozocrtx_file = xr.open_dataset('/scratch/work/brivoalt/run_ABL_nocfbfrc/NEATL12_1d_ABL_gridU.nc') #, chunks={'time': 10})
vomecrty_file = xr.open_dataset('/scratch/work/brivoalt/run_ABL_nocfbfrc/NEATL12_1d_ABL_gridV.nc') #, chunks={'time': 10})
vozocrtx=vozocrtx_file.vozocrtx.squeeze()[1:357,150:330,8:120] * 100 
vomecrty=vomecrty_file.vomecrty.squeeze()[1:357,150:330,8:120] * 100 


maskfile = xr.open_dataset('/scratch/work/brivoalt/PYTHON_PLOTS/COUPLING_COEFFICIENTS/mask_file_1m_IBI12.nc')
tmp=np.zeros(vozocrtx.shape)
tmp[::-1,:,:]=maskfile.wndm[0,150:330,8:120].values * 100
KE_abl=0.5*(vozocrtx**2 + vomecrty**2)
#KE_abl=np.where(~np.isnan(tmp[:,:,:]),(KE_abl.values),np.nan)



vozocrtx_file = xr.open_dataset('/scratch/work/brivoalt/run_ABL_nocfbfrc_ABS/NEATL12_1d_ABL_gridU.nc') #, chunks={'time': 10})
vomecrty_file = xr.open_dataset('/scratch/work/brivoalt/run_ABL_nocfbfrc_ABS/NEATL12_1d_ABL_gridV.nc') #, chunks={'time': 10})
vozocrtx=vozocrtx_file.vozocrtx.squeeze()[1:357,150:330,8:120] * 100 
vomecrty=vomecrty_file.vomecrty.squeeze()[1:357,150:330,8:120] * 100

KE_abl_abs=0.5*(vozocrtx**2 + vomecrty**2) 
#KE_abl_abs=np.where(~np.isnan(tmp[:,:,:]),(KE_abl_abs.values),np.nan)





###################################################################################################################

u_mnh_file = xr.open_dataset('/scratch/work/brivoalt/DATA_MNH/SCRIPT/NEMO_VARS/NEATL12_1d_MNH_gridU.nc') #, chunks={'time': 10})
v_mnh_file = xr.open_dataset('/scratch/work/brivoalt/DATA_MNH/SCRIPT/NEMO_VARS/NEATL12_1d_MNH_gridV.nc') #, chunks={'time': 10})
#WW_mnh_file = xr.open_dataset('/scratch/work/brivoalt/run_MNH_ABS_APR_2/NEATL12_1d_ABL_WW_10m.nc') #, chunks={'time': 10})
vozocrtx=u_mnh_file.vozocrtx.squeeze()[1:357,150:330,8:120] * 100
vomecrty=v_mnh_file.vomecrty.squeeze()[1:357,150:330,8:120] * 100

KE_mnh=0.5*(vozocrtx**2 + vomecrty**2) 
#KE_mnh=np.where(~np.isnan(tmp[:,:,:]),(KE_mnh.values),np.nan)



u_mnh_abs_file = xr.open_dataset('/scratch/work/brivoalt/DATA_MNH/SCRIPT/NEMO_VARS_NOCFB/NEATL12_1d_MNH_gridU.nc') #, chunks={'time': 10})
v_mnh_abs_file = xr.open_dataset('/scratch/work/brivoalt/DATA_MNH/SCRIPT/NEMO_VARS_NOCFB/NEATL12_1d_MNH_gridV.nc') #, chunks={'time': 10})
vozocrtx=u_mnh_abs_file.vozocrtx.squeeze()[1:357,150:330,8:120] * 100
vomecrty=v_mnh_abs_file.vomecrty.squeeze()[1:357,150:330,8:120] * 100
KE_mnh_abs=0.5*(vozocrtx**2 + vomecrty**2) 
#KE_mnh_abs=np.where(~np.isnan(tmp[:,:,:]),(KE_mnh_abs.values),np.nan)


###################################################################################################################

vozocrtx_file = xr.open_dataset('/scratch/work/brivoalt/run_ABL_nocfbfrc_FRC_ABS/NEATL12_1d_ABL_gridU.nc') #, chunks={'time': 10})
vomecrty_file = xr.open_dataset('/scratch/work/brivoalt/run_ABL_nocfbfrc_FRC_ABS/NEATL12_1d_ABL_gridV.nc') #, chunks={'time': 10})
vozocrtx=vozocrtx_file.vozocrtx.squeeze()[1:357,150:330,8:120] * 100
vomecrty=vomecrty_file.vomecrty.squeeze()[1:357,150:330,8:120] * 100
sozotaux=vozocrtx_file.sozotaux.squeeze()[1:357,150:330,8:120] * 100
sometauy=vomecrty_file.sometauy.squeeze()[1:357,150:330,8:120] * 100


WW_frc_abs = (1/1025)*(vozocrtx*sozotaux + vomecrty*sometauy).rolling(time_counter=30, center=True).mean()
KE_frc_abs=0.5*(vozocrtx**2 + vomecrty**2)



vozocrtx_file = xr.open_dataset('/scratch/work/brivoalt/run_ABL_nocfbfrc_FRC_REL/NEATL12_1d_ABL_gridU.nc') #, chunks={'time': 10})
vomecrty_file = xr.open_dataset('/scratch/work/brivoalt/run_ABL_nocfbfrc_FRC_REL/NEATL12_1d_ABL_gridV.nc') #, chunks={'time': 10})
vozocrtx=vozocrtx_file.vozocrtx.squeeze()[1:357,150:330,8:120] * 100
vomecrty=vomecrty_file.vomecrty.squeeze()[1:357,150:330,8:120] * 100
sozotaux=vozocrtx_file.sozotaux.squeeze()[1:357,150:330,8:120] * 100
sometauy=vomecrty_file.sometauy.squeeze()[1:357,150:330,8:120] * 100

WW_frc_rel = (1/1025)*(vozocrtx*sozotaux + vomecrty*sometauy).rolling(time_counter=30, center=True).mean()
KE_frc_rel=0.5*(vozocrtx**2 + vomecrty**2)

###################################################################################################################

vozocrtx_file = xr.open_dataset('/scratch/work/brivoalt/run_ABL_nocfbfrc_TRC100_ABS/NEATL12_1d_ABL_gridU.nc') #, chunks={'time': 10})
vomecrty_file = xr.open_dataset('/scratch/work/brivoalt/run_ABL_nocfbfrc_TRC100_ABS/NEATL12_1d_ABL_gridV.nc') #, chunks={'time': 10})
vozocrtx=vozocrtx_file.vozocrtx.squeeze()[1:357,150:330,8:120] * 100
vomecrty=vomecrty_file.vomecrty.squeeze()[1:357,150:330,8:120] * 100
WW_trc100_abs = (1/1025)*(vozocrtx*sozotaux + vomecrty*sometauy).rolling(time_counter=30, center=True).mean()
KE_trc100_abs=0.5*(vozocrtx**2 + vomecrty**2)



vozocrtx_file = xr.open_dataset('/scratch/work/brivoalt/run_ABL_nocfbfrc_TRC100/NEATL12_1d_ABL_gridU.nc') #, chunks={'time': 10})
vomecrty_file = xr.open_dataset('/scratch/work/brivoalt/run_ABL_nocfbfrc_TRC100/NEATL12_1d_ABL_gridV.nc') #, chunks={'time': 10})
vozocrtx=vozocrtx_file.vozocrtx.squeeze()[1:357,150:330,8:120] * 100
vomecrty=vomecrty_file.vomecrty.squeeze()[1:357,150:330,8:120] * 100
sozotaux=vozocrtx_file.sozotaux.squeeze()[1:357,150:330,8:120] * 100
sometauy=vomecrty_file.sometauy.squeeze()[1:357,150:330,8:120] * 100

WW_trc100_rel = (1/1025)*(vozocrtx*sozotaux + vomecrty*sometauy).rolling(time_counter=30, center=True).mean()
KE_trc100_rel=0.5*(vozocrtx**2 + vomecrty**2)


vozocrtx_file = xr.open_dataset('/scratch/work/brivoalt/run_OA_RETRO/NEATL12_1d_ABL_gridU.nc') #, chunks={'time': 10})
vomecrty_file = xr.open_dataset('/scratch/work/brivoalt/run_OA_RETRO/NEATL12_1d_ABL_gridV.nc') #, chunks={'time': 10})
vozocrtx=vozocrtx_file.vozocrtx.squeeze()[1:357,150:330,8:120] * 100
vomecrty=vomecrty_file.vomecrty.squeeze()[1:357,150:330,8:120] * 100
sozotaux=vozocrtx_file.sozotaux.squeeze()[1:357,150:330,8:120] * 100
sometauy=vomecrty_file.sometauy.squeeze()[1:357,150:330,8:120] * 100

WW_oa_retro = (1/1025)*(vozocrtx*sozotaux + vomecrty*sometauy).rolling(time_counter=30, center=True).mean()
KE_oa_retro=0.5*(vozocrtx**2 + vomecrty**2)


ff_KE_abl=np.zeros((int(len(KE_abl[:,0,0])),int(len(KE_abl[0,0,:])),int(len(KE_abl[0,0,:])+1)))
ff_KE_ablki0=np.zeros((int(len(KE_abl[:,0,0])),int(len(KE_abl[0,0,:])),int(len(KE_abl[0,0,:])+1)))

ff_KE_abl_abs=np.zeros((int(len(KE_abl_abs[:,0,0])),int(len(KE_abl_abs[0,0,:])),int(len(KE_abl_abs[0,0,:])+1)))
ff_KE_abl_abski0=np.zeros((int(len(KE_abl_abs[:,0,0])),int(len(KE_abl_abs[0,0,:])),int(len(KE_abl_abs[0,0,:])+1)))

ff_KE_mnh=np.zeros((int(len(KE_mnh[:,0,0])),int(len(KE_mnh[0,0,:])),int(len(KE_mnh[0,0,:])+1)))
ff_KE_mnhki0=np.zeros((int(len(KE_mnh[:,0,0])),int(len(KE_mnh[0,0,:])),int(len(KE_mnh[0,0,:])+1)))

ff_KE_mnh_abs=np.zeros((int(len(KE_mnh_abs[:,0,0])),int(len(KE_mnh_abs[0,0,:])),int(len(KE_mnh_abs[0,0,:])+1)))
ff_KE_mnh_abski0=np.zeros((int(len(KE_mnh_abs[:,0,0])),int(len(KE_mnh_abs[0,0,:])),int(len(KE_mnh_abs[0,0,:])+1)))

ff_KE_frc_rel=np.zeros((int(len(KE_frc_rel[:,0,0])),int(len(KE_frc_rel[0,0,:])),int(len(KE_frc_rel[0,0,:])+1)))
ff_KE_frc_relki0=np.zeros((int(len(KE_frc_rel[:,0,0])),int(len(KE_frc_rel[0,0,:])),int(len(KE_frc_rel[0,0,:])+1)))

ff_KE_frc_abs=np.zeros((int(len(KE_frc_abs[:,0,0])),int(len(KE_frc_abs[0,0,:])),int(len(KE_frc_abs[0,0,:])+1)))
ff_KE_frc_abski0=np.zeros((int(len(KE_frc_abs[:,0,0])),int(len(KE_frc_abs[0,0,:])),int(len(KE_frc_abs[0,0,:])+1)))

ff_KE_trc100_rel=np.zeros((int(len(KE_trc100_rel[:,0,0])),int(len(KE_trc100_rel[0,0,:])),int(len(KE_trc100_rel[0,0,:])+1)))
ff_KE_trc100_relki0=np.zeros((int(len(KE_trc100_rel[:,0,0])),int(len(KE_trc100_rel[0,0,:])),int(len(KE_trc100_rel[0,0,:])+1)))

ff_KE_trc100_abs=np.zeros((int(len(KE_trc100_abs[:,0,0])),int(len(KE_trc100_abs[0,0,:])),int(len(KE_trc100_abs[0,0,:])+1)))
ff_KE_trc100_abski0=np.zeros((int(len(KE_trc100_abs[:,0,0])),int(len(KE_trc100_abs[0,0,:])),int(len(KE_trc100_abs[0,0,:])+1)))


ff_KE_oa_retro=np.zeros((int(len(KE_oa_retro[:,0,0])),int(len(KE_oa_retro[0,0,:])),int(len(KE_oa_retro[0,0,:])+1)))
ff_KE_oa_retroki0=np.zeros((int(len(KE_oa_retro[:,0,0])),int(len(KE_oa_retro[0,0,:])),int(len(KE_oa_retro[0,0,:])+1)))

no = 0  # Nombres d'overlapping points
dx=1/7 # 1/ largeur de maille moyenne en km

for i in range(len(KE_abl[:,0,0])):
    print(i)
    for j in range(len(KE_abl[0,0,:])):
        ff_KE_abl[i,j,:], ff_KE_ablki0[i,j,:] = signal.welch(KE_abl[i,j,:], fs=dx, window='hanning', nperseg=int(len(KE_abl[0,0,:])), noverlap=no, nfft=2*len(KE_abl[0,0,:]), detrend='linear', return_onesided=True, scaling='spectrum')

        ff_KE_abl_abs[i,j,:], ff_KE_abl_abski0[i,j,:] = signal.welch(KE_abl_abs[i,j,:], fs=dx, window='hanning', nperseg=int(len(KE_abl_abs[0,0,:])), noverlap=no, nfft=2*len(KE_abl_abs[0,0,:]), detrend='linear', return_onesided=True, scaling='spectrum')

        ff_KE_mnh[i,j,:], ff_KE_mnhki0[i,j,:] = signal.welch(KE_mnh[i,j,:], fs=dx, window='hanning', nperseg=int(len(KE_mnh[0,0,:])), noverlap=no, nfft=2*len(KE_mnh[0,0,:]), detrend='linear', return_onesided=True, scaling='spectrum')

        ff_KE_mnh_abs[i,j,:], ff_KE_mnh_abski0[i,j,:] = signal.welch(KE_mnh_abs[i,j,:], fs=dx, window='hanning', nperseg=int(len(KE_mnh_abs[0,0,:])), noverlap=no, nfft=2*len(KE_mnh_abs[0,0,:]), detrend='linear', return_onesided=True, scaling='spectrum')

        ff_KE_frc_abs[i,j,:], ff_KE_frc_abski0[i,j,:] = signal.welch(KE_frc_abs[i,j,:], fs=dx, window='hanning', nperseg=int(len(KE_frc_abs[0,0,:])), noverlap=no, nfft=2*len(KE_frc_abs[0,0,:]), detrend='linear', return_onesided=True, scaling='spectrum')

        ff_KE_frc_rel[i,j,:], ff_KE_frc_relki0[i,j,:] = signal.welch(KE_frc_rel[i,j,:], fs=dx, window='hanning', nperseg=int(len(KE_frc_rel[0,0,:])), noverlap=no, nfft=2*len(KE_frc_rel[0,0,:]), detrend='linear', return_onesided=True, scaling='spectrum')

        ff_KE_trc100_abs[i,j,:], ff_KE_trc100_abski0[i,j,:] = signal.welch(KE_trc100_abs[i,j,:], fs=dx, window='hanning', nperseg=int(len(KE_trc100_abs[0,0,:])), noverlap=no, nfft=2*len(KE_trc100_abs[0,0,:]), detrend='linear', return_onesided=True, scaling='spectrum')

        ff_KE_trc100_rel[i,j,:], ff_KE_trc100_relki0[i,j,:] = signal.welch(KE_trc100_rel[i,j,:], fs=dx, window='hanning', nperseg=int(len(KE_trc100_rel[0,0,:])), noverlap=no, nfft=2*len(KE_trc100_rel[0,0,:]), detrend='linear', return_onesided=True, scaling='spectrum')

        ff_KE_oa_retro[i,j,:], ff_KE_oa_retroki0[i,j,:] = signal.welch(KE_oa_retro[i,j,:], fs=dx, window='hanning', nperseg=int(len(KE_oa_retro[0,0,:])), noverlap=no, nfft=2*len(KE_oa_retro[0,0,:]), detrend='linear', return_onesided=True, scaling='spectrum')

mean_f0_KE_abl = np.nanmean(ff_KE_abl,axis=(0,1))
mean_fi0_KE_abl = np.nanmean(ff_KE_ablki0,axis=(0,1))

mean_f0_KE_abl_abs = np.nanmean(ff_KE_abl_abs,axis=(0,1))
mean_fi0_KE_abl_abs = np.nanmean(ff_KE_abl_abski0,axis=(0,1))

mean_f0_KE_mnh = np.nanmean(ff_KE_mnh,axis=(0,1))
mean_fi0_KE_mnh = np.nanmean(ff_KE_mnhki0,axis=(0,1))

mean_f0_KE_mnh_abs = np.nanmean(ff_KE_mnh_abs,axis=(0,1))
mean_fi0_KE_mnh_abs = np.nanmean(ff_KE_mnh_abski0,axis=(0,1))

mean_f0_KE_frc_rel = np.nanmean(ff_KE_frc_rel,axis=(0,1))
mean_fi0_KE_frc_rel = np.nanmean(ff_KE_frc_relki0,axis=(0,1))

mean_f0_KE_frc_abs = np.nanmean(ff_KE_frc_abs,axis=(0,1))
mean_fi0_KE_frc_abs = np.nanmean(ff_KE_frc_abski0,axis=(0,1))

mean_f0_KE_trc100_rel = np.nanmean(ff_KE_trc100_rel,axis=(0,1))
mean_fi0_KE_trc100_rel = np.nanmean(ff_KE_trc100_relki0,axis=(0,1))

mean_f0_KE_trc100_abs = np.nanmean(ff_KE_trc100_abs,axis=(0,1))
mean_fi0_KE_trc100_abs = np.nanmean(ff_KE_trc100_abski0,axis=(0,1))

mean_f0_KE_oa_retro = np.nanmean(ff_KE_oa_retro,axis=(0,1))
mean_fi0_KE_oa_retro = np.nanmean(ff_KE_oa_retroki0,axis=(0,1))


ax = plt.subplot(111)
ax.plot(1/mean_f0_KE_abl,mean_fi0_KE_abl, 'g-', lw=2, label ='ABL REL')
ax.plot(1/mean_f0_KE_abl_abs,mean_fi0_KE_abl_abs, 'g', lw=2, label ='ABL ABS', linestyle='--')
ax.plot(1/mean_f0_KE_mnh,mean_fi0_KE_mnh, 'k-', lw=2, label ='MNH REL')
ax.plot(1/mean_f0_KE_mnh_abs,mean_fi0_KE_mnh_abs, 'k', lw=2, label ='MNH ABS', linestyle='--')
ax.plot(1/mean_f0_KE_frc_rel,mean_fi0_KE_frc_rel, 'b-', lw=2, label ='FRC REL')
ax.plot(1/mean_f0_KE_frc_abs,mean_fi0_KE_frc_abs, 'b', lw=2, label ='FRC ABS', linestyle='--')
ax.plot(1/mean_f0_KE_trc100_rel,mean_fi0_KE_trc100_rel, 'lightgreen', lw=2, label ='ABL TRC100')
ax.plot(1/mean_f0_KE_trc100_abs,mean_fi0_KE_trc100_abs, 'lightgreen', lw=2, label ='ABL TRC100 ABS', linestyle='--')
ax.plot(1/mean_f0_KE_oa_retro,mean_fi0_KE_oa_retro, 'orange', lw=2, label ='OA RETRO')


ax.set_yscale('log')
ax.set_xlim(0,500)
#ax.set_ylim(10e-2,35)
#ax.set_xticks([(1e-7),(1e-6),(1.e-5)])
ax.legend()

ax.set_ylabel('power') # regex: ($10log10$)
ax.set_xlabel('wavelenght (km)')


plt.savefig('space_spectrum_KE_ABL.png')
plt.close()

