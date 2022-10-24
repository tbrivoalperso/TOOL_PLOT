#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:41:00 2020

@author: mhamon
"""
import datetime
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
from plot_ola_oov2.READ_NML import SYSid, CYCLE, TYPE2DO, level, DIR_OLA_OFF1, DIR_OLA_OFF2, samout1, samout2, type_ccl1, type_ccl2, \
                                   RUNid1, RUNid2, RES, SET1, SET2, PREC, do_comp, DIR_PLOT_EXP, PARAM_PLOT, \
                                   PARAM_PLOT_POLARNorth, PARAM_PLOT_POLARSouth
from plot_ola_oov2.LOAD_OLA_OOV2 import load_ola, load_olasam_bin
from lib_pyt.Sort_list import FFcompare_list
from lib_pyt.Griddata_custom import grid_data_NUMBA

import matplotlib.pyplot as plt
from lib_pyt.Class_plot_cartopy import Plot_map_cartopy as map
from lib_pyt.cmaps import cmaps
import scipy.stats as stats
print(RUNid1,RUNid2)
print(DIR_OLA_OFF1)
print(CYCLE)
tmp=0

def interp_on_regular_latorlon_array(VAR_IN,LATorLON_IN,LATorLON_ARRAY_OUT):
   bin_means = np.zeros((len(LATorLON_ARRAY_OUT)-1,len(VAR_IN[0,:])))
   bin_count = np.zeros((len(LATorLON_ARRAY_OUT)-1,len(VAR_IN[0,:])))
   bin_edges = np.zeros((len(LATorLON_ARRAY_OUT),len(VAR_IN[0,:])))
   VAR_IN_count=np.where(np.isnan(VAR_IN),0,1)
   #print(bin_means.shape)
   for d in range(len(VAR_IN[0,:])):
       bin_means[:,d], bin_edges[:,d], binnumber = stats.binned_statistic(LATorLON_IN,VAR_IN[:,d], statistic='nanmean', bins=LATorLON_ARRAY_OUT)
       bin_count[:,d], bin_edges[:,d], binnumber = stats.binned_statistic(LATorLON_IN,VAR_IN_count[:,d], statistic='sum', bins=LATorLON_ARRAY_OUT)
   return bin_means, bin_count, bin_edges


LATINTERVAL=0.02
LONINTERVAL=0.02

#######################################################################
########### POSITION DES BOITES ######################################
AREANAMES=['GIBR', 'UPWELL']

LATMIN={}
LATMAX={}
LONMIN={}
LONMAX={}
axis={} # 0 = along latitude axis, 1 = along longitude axis
#---------------------
LATMIN['GIBR']=35.8
LATMAX['GIBR']=36.1
LONMIN['GIBR']=-5.8
LONMAX['GIBR']=-5.2
axis['GIBR']=0

#---------------------
LATMIN['UPWELL']=39.7
LATMAX['UPWELL']=40.3
LONMIN['UPWELL']=-11
LONMAX['UPWELL']=-6.5
axis['UPWELL']=1


for ic, cyc in enumerate(CYCLE):
    print(cyc,ic)
    OLAfile_1 = xr.open_dataset(DIR_OLA_OFF1 + 'OLA_VP_T_R' + cyc +'.nc')
    OBS1=OLAfile_1.TEMP_INTERP.values
    MOD1=OLAfile_1.TEMP_EQUIVALENT_MODELE.values
    LAT=OLAfile_1.LATITUDE.values
    LON=OLAfile_1.LONGITUDE.values
    if do_comp:
        OLAfile_2 = xr.open_mfdataset(DIR_OLA_OFF2 + 'OLA_VP_T_R' + cyc +'.nc')
        OBS2=OLAfile_2.TEMP_INTERP.values
        MOD2=OLAfile_2.TEMP_EQUIVALENT_MODELE.values
    print(OBS1.shape)
    print(OBS2.shape)
    try:
        LAT_TOT = np.concatenate((LAT_TOT,LAT), axis=0)
        LON_TOT = np.concatenate((LON_TOT,LON), axis=0)

        OBS1_TOT = np.concatenate((OBS1_TOT, OBS1), axis=0)
        MOD1_TOT = np.concatenate((MOD1_TOT, MOD1), axis=0)
    except NameError:
        OBS1_TOT = OBS1
        MOD1_TOT = MOD1
        LAT_TOT=LAT
        LON_TOT=LON

    if do_comp:
        try:
            OBS2_TOT = np.concatenate((OBS2_TOT, OBS2), axis=0)
            MOD2_TOT = np.concatenate((MOD2_TOT, MOD2), axis=0)
        except NameError:
            OBS2_TOT = OBS2
            MOD2_TOT = MOD2
    print(OBS1_TOT.shape)
    print(OBS2_TOT.shape)

DEPTH1=OLAfile_1.DEPTH.values    
DEPTH2=OLAfile_2.DEPTH.values
# Check depths and size
if do_comp:
    if (DEPTH1==DEPTH2).all():
        print("DEPTHS OK")
    else:
        print("error, depths are different in ARR1 and ARR2")
        sys.exit()
    
    if (OBS1_TOT.shape==OBS2_TOT.shape):
        print("SIZE OK")
    else:
        print("error, ARR1 and ARR2 have different shape: ", OBS1_TOT.shape, OBS2_TOT.shape)
        sys.exit()


for areanm in AREANAMES:
    print(areanm)
    testlat=(LAT_TOT>=LATMIN[areanm]) & (LAT_TOT<=LATMAX[areanm])
    testlon=(LON_TOT>=LONMIN[areanm]) & (LON_TOT<=LONMAX[areanm])
    indexok=np.where((testlat) & (testlon) )
    OBS1_SEL=OBS1_TOT[indexok,:].squeeze()
    MOD1_SEL=MOD1_TOT[indexok,:].squeeze()
   # MOD1_SEL=np.where(np.isnan(OBS1_SEL),np.nan,MOD1_SEL)
    LAT_SEL=LAT_TOT[indexok]
    LON_SEL=LON_TOT[indexok] 
    RMS_1=np.sqrt((OBS1_SEL - MOD1_SEL)**2)
    INNOV_1=OBS1_SEL - MOD1_SEL
    
    if do_comp:
        OBS2_SEL=OBS2_TOT[indexok,:].squeeze()
        MOD2_SEL=MOD2_TOT[indexok,:].squeeze()
    #    MOD2_SEL=np.where(np.isnan(OBS2_SEL),np.nan,MOD2_SEL)

        RMS_2=np.sqrt((OBS2_SEL - MOD2_SEL)**2)
        INNOV_2=OBS2_SEL - MOD2_SEL
    print(OBS1_SEL.shape)
    print(MOD1_SEL.shape) 
    print(OBS1_SEL[0,:])
    print(MOD1_SEL[0,:])

    ####### PLOTS ###############
    ###############################
    plt.figure(figsize=(8,10))
    ax = plt.subplot(111)
    if axis[areanm] == 0 : 
        LAT_regular=np.arange(LATMIN[areanm],LATMAX[areanm],LATINTERVAL)
        OBS1_SEL_mean_on_x_regular, OBS1_SEL_count_on_x_regular , x_edges=  interp_on_regular_latorlon_array(OBS1_SEL,LAT_SEL,LAT_regular)
    else:
        LON_regular=np.arange(LONMIN[areanm],LONMAX[areanm],LONINTERVAL)
        OBS1_SEL_mean_on_x_regular, OBS1_SEL_count_on_x_regular , x_edges=  interp_on_regular_latorlon_array(OBS1_SEL,LON_SEL,LON_regular)

    x_center=((x_edges[1:,:] + x_edges[:-1,:])/2)[:,0]
    x_center_2D, depth2D= np.meshgrid(x_center,DEPTH1,indexing='ij')
    plt.contourf(x_center_2D, depth2D,OBS1_SEL_mean_on_x_regular,levels=np.linspace(12,19,51), cmap=cmocean.cm.thermal,extend='both')
    plt.colorbar()
    ax.set_ylim(0,800)
    plt.gca().invert_yaxis()
    plt.savefig(DIR_PLOT_EXP+areanm+"_HOEVMOLLER_mean_TEMP_OBS_NOOBS"+RUNid1+"_"+RUNid2+".png")
    plt.close() 

    ###############################
    plt.figure(figsize=(8,10))
    ax = plt.subplot(111)
    plt.contourf(x_center_2D, depth2D,np.where(OBS1_SEL_count_on_x_regular == 0, np.nan,OBS1_SEL_count_on_x_regular) ,levels=np.linspace(0,np.nanmax(OBS1_SEL_count_on_x_regular),51), cmap='jet',extend='both')
    plt.colorbar()
    ax.set_ylim(0,800)
    plt.gca().invert_yaxis()
    plt.savefig(DIR_PLOT_EXP+areanm+"_HOEVMOLLER_count_TEMP_OBS_NOOBS"+RUNid1+"_"+RUNid2+".png")
    plt.close()


    #######################################

    plt.figure(figsize=(8,10))
    ax = plt.subplot(111)
    if axis[areanm] == 0 :
        LAT_regular=np.arange(LATMIN[areanm],LATMAX[areanm],LATINTERVAL)
        MOD1_SEL_mean_on_x_regular, MOD1_SEL_count_on_x_regular , x_edges=  interp_on_regular_latorlon_array(MOD1_SEL,LAT_SEL,LAT_regular)
    else:
        LON_regular=np.arange(LONMIN[areanm],LONMAX[areanm],LONINTERVAL)
        MOD1_SEL_mean_on_x_regular, MOD1_SEL_count_on_x_regular , x_edges=  interp_on_regular_latorlon_array(MOD1_SEL,LON_SEL,LON_regular)

    x_center=((x_edges[1:,:] + x_edges[:-1,:])/2)[:,0]
    x_center_2D, depth2D= np.meshgrid(x_center,DEPTH1,indexing='ij')
    plt.contourf(x_center_2D, depth2D,MOD1_SEL_mean_on_x_regular,levels=np.linspace(12,19,51), cmap=cmocean.cm.thermal,extend='both')
    plt.colorbar()
    ax.set_ylim(0,800)
    plt.gca().invert_yaxis()
    plt.savefig(DIR_PLOT_EXP+areanm+"_HOEVMOLLER_mean_TEMP_MOD_NOOBS"+RUNid1+".png")
    plt.close() 

    #######################################
    if do_comp:
        plt.figure(figsize=(8,10))
        ax = plt.subplot(111)
        if axis[areanm] == 0 :
            LAT_regular=np.arange(LATMIN[areanm],LATMAX[areanm],LATINTERVAL)
            MOD2_SEL_mean_on_x_regular, MOD2_SEL_count_on_x_regular , x_edges=  interp_on_regular_latorlon_array(MOD2_SEL,LAT_SEL,LAT_regular)
        else:
            LON_regular=np.arange(LONMIN[areanm],LONMAX[areanm],LONINTERVAL)
            MOD2_SEL_mean_on_x_regular, MOD2_SEL_count_on_x_regular , x_edges=  interp_on_regular_latorlon_array(MOD2_SEL,LON_SEL,LON_regular)
    
        x_center=((x_edges[1:,:] + x_edges[:-1,:])/2)[:,0]
        x_center_2D, depth2D= np.meshgrid(x_center,DEPTH1,indexing='ij')
        plt.contourf(x_center_2D, depth2D,MOD2_SEL_mean_on_x_regular,levels=np.linspace(12,19,51), cmap=cmocean.cm.thermal,extend='both')

        plt.colorbar()
        ax.set_ylim(0,800)
        plt.gca().invert_yaxis()
        plt.savefig(DIR_PLOT_EXP+areanm+"_HOEVMOLLER_mean_TEMP_MOD_NOOBS"+RUNid2+".png")
        plt.close() 

    ############## RMSE ######################

    plt.figure(figsize=(8,10))
    ax = plt.subplot(111)
    if axis[areanm] == 0 :
        LAT_regular=np.arange(LATMIN[areanm],LATMAX[areanm],LATINTERVAL)
        RMS_1_mean_on_x_regular, RMS_1_count_on_x_regular , x_edges=  interp_on_regular_latorlon_array(RMS_1,LAT_SEL,LAT_regular)
    else:
        LON_regular=np.arange(LONMIN[areanm],LONMAX[areanm],LONINTERVAL)
        RMS_1_mean_on_x_regular, RMS_1_count_on_x_regular , x_edges=  interp_on_regular_latorlon_array(RMS_1,LON_SEL,LON_regular)

    x_center=((x_edges[1:,:] + x_edges[:-1,:])/2)[:,0]
    x_center_2D, depth2D= np.meshgrid(x_center,DEPTH1,indexing='ij')
    plt.contourf(x_center_2D, depth2D,RMS_1_mean_on_x_regular,levels=np.linspace(12,19,51), cmap=cmocean.cm.thermal,extend='both')

    plt.colorbar()
    ax.set_ylim(0,800)
    plt.gca().invert_yaxis()
    plt.savefig(DIR_PLOT_EXP+areanm+"_HOEVMOLLER_mean_RMS_TEMP_MOD_NOOBS"+RUNid1+".png")
    plt.close()



    #######################################
    if do_comp:
        plt.figure(figsize=(8,10))
        ax = plt.subplot(111)
        if axis[areanm] == 0 :
            LAT_regular=np.arange(LATMIN[areanm],LATMAX[areanm],LATINTERVAL)
            RMS_2_mean_on_x_regular, RMS_2_count_on_x_regular , x_edges=  interp_on_regular_latorlon_array(RMS_2,LAT_SEL,LAT_regular)
        else:
            LON_regular=np.arange(LONMIN[areanm],LONMAX[areanm],LONINTERVAL)
            RMS_2_mean_on_x_regular, RMS_2_count_on_x_regular , x_edges=  interp_on_regular_latorlon_array(RMS_2,LON_SEL,LON_regular)
    
        x_center=((x_edges[1:,:] + x_edges[:-1,:])/2)[:,0]
        x_center_2D, depth2D= np.meshgrid(x_center,DEPTH1,indexing='ij')
        plt.contourf(x_center_2D, depth2D,RMS_2_mean_on_x_regular,levels=np.linspace(12,19,51), cmap=cmocean.cm.thermal,extend='both')

        plt.colorbar()
        ax.set_ylim(0,800)
        plt.gca().invert_yaxis()
        plt.savefig(DIR_PLOT_EXP+areanm+"_HOEVMOLLER_mean_RMS_TEMP_MOD_NOOBS"+RUNid2+".png")
        plt.close()

    ############## RMSE and INNOV ######################

    plt.figure(figsize=(8,10))
    ax = plt.subplot(111)
    if axis[areanm] == 0 :
        LAT_regular=np.arange(LATMIN[areanm],LATMAX[areanm],LATINTERVAL)
        INNOV_1_mean_on_x_regular, INNOV_1_count_on_x_regular , x_edges=  interp_on_regular_latorlon_array(INNOV_1,LAT_SEL,LAT_regular)
    else:
        LON_regular=np.arange(LONMIN[areanm],LONMAX[areanm],LONINTERVAL)
        INNOV_1_mean_on_x_regular, INNOV_1_count_on_x_regular , x_edges=  interp_on_regular_latorlon_array(INNOV_1,LON_SEL,LON_regular)

    x_center=((x_edges[1:,:] + x_edges[:-1,:])/2)[:,0]
    x_center_2D, depth2D= np.meshgrid(x_center,DEPTH1,indexing='ij')
    plt.contourf(x_center_2D, depth2D,INNOV_1_mean_on_x_regular,levels=np.linspace(12,19,51), cmap=cmocean.cm.thermal,extend='both')
    plt.colorbar()
    ax.set_ylim(0,800)
    plt.gca().invert_yaxis()
    plt.savefig(DIR_PLOT_EXP+areanm+"_HOEVMOLLER_mean_INNOV_TEMP_MOD_NOOBS"+RUNid1+".png")
    plt.close()



    #######################################
    if do_comp:
        plt.figure(figsize=(8,10))
        ax = plt.subplot(111)
        if axis[areanm] == 0 :
            LAT_regular=np.arange(LATMIN[areanm],LATMAX[areanm],LATINTERVAL)
            INNOV_2_mean_on_x_regular, INNOV_2_count_on_x_regular , x_edges=  interp_on_regular_latorlon_array(INNOV_2,LAT_SEL,LAT_regular)
        else:
            LON_regular=np.arange(LONMIN[areanm],LONMAX[areanm],LONINTERVAL)
            INNOV_2_mean_on_x_regular, INNOV_2_count_on_x_regular , x_edges=  interp_on_regular_latorlon_array(INNOV_2,LON_SEL,LON_regular)
    
        x_center=((x_edges[1:,:] + x_edges[:-1,:])/2)[:,0]
        x_center_2D, depth2D= np.meshgrid(x_center,DEPTH1,indexing='ij')
        plt.contourf(x_center_2D, depth2D,INNOV_2_mean_on_x_regular,levels=np.linspace(12,19,51), cmap=cmocean.cm.thermal,extend='both')

        plt.colorbar()
        ax.set_ylim(0,800)
        plt.gca().invert_yaxis()
        plt.savefig(DIR_PLOT_EXP+areanm+"_HOEVMOLLER_mean_INNOV_TEMP_MOD_NOOBS"+RUNid2+".png")
        plt.close()

