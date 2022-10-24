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
ndeg=1.5

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


U_1=U_1_file.sozocrtx.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:]#.stack(xy=("x", "y"))
V_1=V_1_file.somecrty.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:]#.stack(xy=("x", "y"))
if ISAGRIF1: U_11=U_11_file.sozocrtx.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:]#.stack(xy=("x", "y"))
if ISAGRIF1: V_11=V_11_file.somecrty.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:]#.stack(xy=("x", "y"))

U_2=U_2_file.sozocrtx.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:]#.stack(xy=("x", "y"))
V_2=V_2_file.somecrty.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:]#.stack(xy=("x", "y"))

if ISAGRIF2: U_21=U_21_file.sozocrtx.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:]#.stack(xy=("x", "y"))
if ISAGRIF2: V_21=V_21_file.somecrty.squeeze().sel(time_counter=slice('DSTART', 'DEND'))[:,:,:]#.stack(xy=("x", "y"))

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

    # Selecting SSH area
    tmplat=abs(nav_lat_1 - listlat_center[narea])
    tmplon=abs(nav_lon_1 - listlon_center[narea])
    tmp=tmplat + tmplon
    print(np.argwhere(tmp == np.nanmin(tmp)).shape)
    indexy_center_1[narea], indexx_center_1[narea] = int(np.argwhere(tmp == np.nanmin(tmp))[0,0]), int(np.argwhere(tmp == np.nanmin(tmp))[0,1])
    print("CFG_NAME, selecting point i, j :",indexy_center_1[narea], indexx_center_1[narea])
    print("at latitude, longitude :", nav_lat_1[indexy_center_1[narea], indexx_center_1[narea]], nav_lon_1[indexy_center_1[narea], indexx_center_1[narea]]) 
    U_1_area=U_1[:,indexy_center_1[narea]- npts_ndeg_1:indexy_center_1[narea]+ npts_ndeg_1,\
                       indexx_center_1[narea]- npts_ndeg_1:indexx_center_1[narea]+ npts_ndeg_1].stack(xy=("x", "y"))
    print(U_1_area.shape)
    V_1_area=V_1[:,indexy_center_1[narea]- npts_ndeg_1:indexy_center_1[narea]+ npts_ndeg_1,\
                       indexx_center_1[narea]- npts_ndeg_1:indexx_center_1[narea]+ npts_ndeg_1].stack(xy=("x", "y"))
    KE_1_area = 0.5 * (U_1_area**2 + V_1_area**2)
    print(KE_1_area.shape)
    U_2_area=U_2[:,indexy_center_1[narea]- npts_ndeg_2:indexy_center_1[narea]+ npts_ndeg_2,\
                       indexx_center_1[narea]- npts_ndeg_2:indexx_center_1[narea]+ npts_ndeg_2].stack(xy=("x", "y"))
    V_2_area=V_2[:,indexy_center_1[narea]- npts_ndeg_2:indexy_center_1[narea]+ npts_ndeg_2,\
                       indexx_center_1[narea]- npts_ndeg_2:indexx_center_1[narea]+ npts_ndeg_2].stack(xy=("x", "y"))
    KE_2_area = 0.5 * (U_2_area**2 + V_2_area**2)

    if ISAGRIF1: 
        tmplat=abs(nav_lat_11 - listlat_center[narea])
        tmplon=abs(nav_lon_11 - listlon_center[narea])
        tmp=tmplat + tmplon
        indexy_center_11[narea], indexx_center_11[narea] = int(np.argwhere(tmp == np.nanmin(tmp))[0,0]), int(np.argwhere(tmp == np.nanmin(tmp))[0,1])
        print("AGRIF_CFG_NAME, selecting point i, j :",indexy_center_11[narea], indexx_center_11[narea])
        print("at latitude, longitude :", nav_lat_11[indexy_center_11[narea], indexx_center_11[narea]], nav_lon_11[indexy_center_11[narea], indexx_center_11[narea]])
        U_11_area=U_11[:,indexy_center_11[narea]- npts_ndeg_11:indexy_center_11[narea]+ npts_ndeg_11,\
                                              indexx_center_11[narea]- npts_ndeg_11:indexx_center_11[narea]+ npts_ndeg_11].stack(xy=("x", "y"))
        V_11_area=V_11[:,indexy_center_11[narea]- npts_ndeg_11:indexy_center_11[narea]+ npts_ndeg_11,\
                                              indexx_center_11[narea]- npts_ndeg_11:indexx_center_11[narea]+ npts_ndeg_11].stack(xy=("x", "y"))
        KE_11_area = 0.5 * (U_11_area**2 + V_11_area**2)

    if ISAGRIF2:
        U_21_area=U_21[:,indexy_center_11[narea]- npts_ndeg_21:indexy_center_11[narea]+ npts_ndeg_21,\
                             indexx_center_11[narea]- npts_ndeg_21:indexx_center_11[narea]+ npts_ndeg_21].stack(xy=("x", "y"))
        V_21_area=V_21[:,indexy_center_11[narea]- npts_ndeg_21:indexy_center_11[narea]+ npts_ndeg_21,\
                             indexx_center_11[narea]- npts_ndeg_21:indexx_center_11[narea]+ npts_ndeg_21].stack(xy=("x", "y"))
        KE_21_area = 0.5 * (U_21_area**2 + V_21_area**2)
    
    KE_1_area=KE_1_area.fillna(0.)
    if ISAGRIF1: KE_11_area=KE_11_area.fillna(0.)
    KE_2_area=KE_2_area.fillna(0.)
    if ISAGRIF2: KE_21_area=KE_21_area.fillna(0.)
 
    
    ff_KE_1_area=np.zeros(KE_1_area.shape)
    ff_KE_1_areaki0=np.zeros(KE_1_area.shape)
    if ISAGRIF1: 
        ff_KE_11_area=np.zeros(KE_11_area.shape)
        ff_KE_11_areaki0=np.zeros(KE_11_area.shape)


    ff_KE_2_area=np.zeros(KE_2_area.shape)
    ff_KE_2_areaki0=np.zeros(KE_2_area.shape)
    if ISAGRIF2:
        ff_KE_21_area=np.zeros(KE_21_area.shape)
        ff_KE_21_areaki0=np.zeros(KE_21_area.shape)

    
    no = 0  # Nombres d'overlapping points
    freqmin_hz=1/(2*31*24*3600)
    freqmax_hz=1/3600
    periodmax_h=2*24*31*3600 # 1mois #(1/freqmax_hz)/3600
 
    #
    print("Compute spectrum for model data")
    for i in range(len(KE_1_area[0,:])):
       ff_KE_1_area[:,i], ff_KE_1_areaki0[:,i] = signal.welch(KE_1_area[:,i], fs=dx_1,nperseg=int(len(KE_1_area[:,0])),\
                                                                    window='hanning', noverlap=no,nfft=2*int(len(KE_1_area[:,0])-1),\
                                                                    detrend='linear', return_onesided=True, scaling='spectrum')
       ff_KE_2_area[:,i], ff_KE_2_areaki0[:,i] = signal.welch(KE_2_area[:,i], fs=dx_1,nperseg=int(len(KE_2_area[:,0])),\
                                                                        window='hanning', noverlap=no,nfft=2*int(len(KE_2_area[:,0])-1),\
                                                                        detrend='linear', return_onesided=True, scaling='spectrum')



    for i in range(len(KE_11_area[0,:])):
       if ISAGRIF1: ff_KE_11_area[:,i], ff_KE_11_areaki0[:,i] = signal.welch(KE_11_area[:,i], fs=dx_11,nperseg=int(len(KE_11_area[:,0])),\
                                                                          window='hanning', noverlap=no,nfft=2*int(len(KE_11_area[:,0])-1),\
                                                                          detrend='linear', return_onesided=True, scaling='spectrum')
       

       if ISAGRIF2: ff_KE_21_area[:,i], ff_KE_21_areaki0[:,i] = signal.welch(KE_21_area[:,i], fs=dx_11,nperseg=int(len(KE_21_area[:,0])),\
                                                                      window='hanning', noverlap=no,nfft=2*int(len(KE_21_area[:,0])-1),\
                                                                      detrend='linear', return_onesided=True, scaling='spectrum')

    mean_f0_KE_1_area = np.nanmean(ff_KE_1_area,axis=1)
    mean_fi0_KE_1_area = np.nanmean(ff_KE_1_areaki0,axis=1)

    if ISAGRIF1:
        mean_f0_KE_11_area = np.nanmean(ff_KE_11_area,axis=1)
        mean_fi0_KE_11_area = np.nanmean(ff_KE_11_areaki0,axis=1)
    
    mean_f0_KE_2_area = np.nanmean(ff_KE_2_area,axis=1)
    mean_fi0_KE_2_area = np.nanmean(ff_KE_2_areaki0,axis=1)
    if ISAGRIF2:
        mean_f0_KE_21_area = np.nanmean(ff_KE_21_area,axis=1)
        mean_fi0_KE_21_area = np.nanmean(ff_KE_21_areaki0,axis=1)
    
    #
    ax = plt.subplot(111)


    ax.loglog(mean_f0_KE_1_area,mean_fi0_KE_1_area, 'b', lw=2, label ='CFG_NAME VER1')
    if ISAGRIF1: ax.loglog(mean_f0_KE_11_area,mean_fi0_KE_11_area, 'b', lw=2,ls='dotted', label ='AGRIF_CFG_NAME VER1')

    ax.loglog(mean_f0_KE_2_area,mean_fi0_KE_2_area, 'r', lw=2, label ='CFG_NAME VER2')
    if ISAGRIF2: ax.loglog(mean_f0_KE_21_area,mean_fi0_KE_21_area, 'r', lw=2,ls='dotted', label ='AGRIF_CFG_NAME VER2')
    
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    #
    ax.set_xlim(freqmin_hz,freqmax_hz)

    ax.legend()
    ax.set_ylabel('power') # regex: ($10log10$)
    ax.set_xlabel('Frequency (Hz)')

    #
    plt.grid(True, which="both", linestyle='-', color='0.65')

    plt.savefig('CFG_NM_time_spectrum_frequency_VER1_VER2_DSTART_DEND'+str(areanames[narea])+'.png')
    plt.close()
    ax = plt.subplot(111)


    ax.loglog(1/mean_f0_KE_1_area,mean_fi0_KE_1_area, 'b', lw=2, label ='CFG_NAME VER1')
    if ISAGRIF1: ax.loglog(1/mean_f0_KE_11_area,mean_fi0_KE_11_area, 'b', lw=2,ls='dotted', label ='AGRIF_CFG_NAME VER1')

    ax.loglog(1/mean_f0_KE_2_area,mean_fi0_KE_2_area, 'r', lw=2, label ='CFG_NAME VER2')
    if ISAGRIF2: ax.loglog(1/mean_f0_KE_21_area,mean_fi0_KE_21_area, 'r', lw=2,ls='dotted', label ='AGRIF_CFG_NAME VER2')

    #ax.set_yscale('log')
    #ax.set_xscale('log')
    #
    ax.set_xlim(0,periodmax_h)

    ax.legend()
    ax.set_ylabel('power') # regex: ($10log10$)
    ax.set_xlabel('Period (s)')
    plt.axvline(12 * 3600)
    plt.axvline(18 * 3600)
    plt.axvline(24 * 3600)

    #
    plt.grid(True, which="both", linestyle='-', color='0.65')

    plt.savefig('CFG_NM_time_spectrum_period_VER1_VER2_DSTART_DEND'+str(areanames[narea])+'.png')
    plt.close()
    

valeur_cadre=1.

proj=ccrs.Mercator()
plt.figure(figsize=(8,8))
ax = plt.subplot(111, projection=proj)

ax.coastlines(resolution='50m')
ax.set_extent((-21.0, 16.0, 25.0, 63.5))
lon_formatter = LongitudeFormatter(degree_symbol='° ')
lat_formatter = LatitudeFormatter(degree_symbol='° ')
ax.set_xticks([ -20 , -10 , 0 , 10  ], crs=ccrs.PlateCarree())
ax.set_yticks([ 30, 40, 50, 60], crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

axes=ax.gridlines( draw_labels=False, linewidth=0)
axes.ylabels_right = False
axes.xlabels_top = False

for narea in range(len(listlat_center)):

    var_IBI_AGRIF=np.zeros(U_1[0,:,:].shape)
    imin_IBI_AGRIF=indexx_center_1[narea] - npts_ndeg_1
    imax_IBI_AGRIF=indexx_center_1[narea] + npts_ndeg_1
    jmin_IBI_AGRIF=indexy_center_1[narea] - npts_ndeg_1
    jmax_IBI_AGRIF=indexy_center_1[narea] + npts_ndeg_1
    var_IBI_AGRIF[jmin_IBI_AGRIF:jmax_IBI_AGRIF,imin_IBI_AGRIF:imax_IBI_AGRIF]=valeur_cadre
    print('IBI AGRIF : ')
    
    ########
    im1 = plt.contour(Coordfile_1.glamt.squeeze(), Coordfile_1.gphit.squeeze(), var_IBI_AGRIF, cmap=None,levels=np.linspace(-1,1,2), transform=ccrs.PlateCarree(),linewidths=1.5,colors=(colors[narea],colors[narea]))

    ########


#    var_IBI_AGRIF=np.zeros(U_11[0,:,:].shape)
#    imin_IBI_AGRIF=indexx_center_11[narea] - npts_ndeg_11
#    imax_IBI_AGRIF=indexx_center_11[narea] + npts_ndeg_11
#    jmin_IBI_AGRIF=indexy_center_11[narea] - npts_ndeg_11
#    jmax_IBI_AGRIF=indexy_center_11[narea] + npts_ndeg_11
#    var_IBI_AGRIF[jmin_IBI_AGRIF:jmax_IBI_AGRIF,imin_IBI_AGRIF:imax_IBI_AGRIF]=valeur_cadre
#    var_IBI_AGRIF[jmax_IBI_AGRIF:jmax_IBI_AGRIF,imin_IBI_AGRIF:imax_IBI_AGRIF]=valeur_cadre
#    print('IBI AGRIF : ')
#
#    ########
#    im1 = plt.contour(Coordfile_11.glamt.squeeze(), Coordfile_11.gphit.squeeze(), var_IBI_AGRIF, cmap=None,levels=np.linspace(-1,1,2),linestyles=["dashed","dashed"], transform=ccrs.PlateCarree(),linewidths=1.5,colors=(colors[narea]),colors[narea]))
#
    plt.savefig('MAP_POSITION_SPECTRES.png',dpi=400)

    #
    #
    #
