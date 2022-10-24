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

from plot_ola_oov2.READ_NML import SYSid, CYCLE, TYPE2DO, level, DIR_OLA_OFF1, DIR_OLA_OFF2, samout1, samout2, type_ccl1, type_ccl2, \
                                   RUNid1, RUNid2, RES, SET1, SET2, PREC, do_comp, DIR_PLOT_EXP, PARAM_PLOT, \
                                   PARAM_PLOT_POLARNorth, PARAM_PLOT_POLARSouth
from plot_ola_oov2.LOAD_OLA_OOV2 import load_ola, load_olasam_bin
from lib_pyt.Sort_list import FFcompare_list
from lib_pyt.Griddata_custom import grid_data_NUMBA
import matplotlib

import matplotlib.pyplot as plt
from lib_pyt.Class_plot_cartopy import Plot_map_cartopy as map
from lib_pyt.cmaps import cmaps
matplotlib.rcParams.update({'font.size': 13})
print(RUNid1,RUNid2)
print(DIR_OLA_OFF1)
print(CYCLE)
tmp=0
for ic, cyc in enumerate(CYCLE):
    print(cyc,ic)
    OLAfile_1 = xr.open_dataset(DIR_OLA_OFF1 + 'OLA_VP_S_R' + cyc +'.nc')
    OBS1=OLAfile_1.PSAL_INTERP.values
    MOD1=OLAfile_1.PSAL_EQUIVALENT_MODELE.values
    if do_comp:
        OLAfile_2 = xr.open_mfdataset(DIR_OLA_OFF2 + 'OLA_VP_S_R' + cyc +'.nc')
        OBS2=OLAfile_2.PSAL_INTERP.values
        MOD2=OLAfile_2.PSAL_EQUIVALENT_MODELE.values
    print(OBS1.shape)
    try:
        OBS1_TOT = np.concatenate((OBS1_TOT, OBS1), axis=0)
        MOD1_TOT = np.concatenate((MOD1_TOT, MOD1), axis=0)
    except NameError:
        OBS1_TOT = OBS1
        MOD1_TOT = MOD1
    if do_comp:
        try:
            OBS2_TOT = np.concatenate((OBS2_TOT, OBS2), axis=0)
            MOD2_TOT = np.concatenate((MOD2_TOT, MOD2), axis=0)
        except NameError:
            OBS2_TOT = OBS2
            MOD2_TOT = MOD2

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
        print("error, ARR1 and ARR2 have difference shape")
        sys.exit()

RMS_1=np.sqrt((OBS1_TOT - MOD1_TOT)**2)
RMS_1_d=np.nanmean(RMS_1,axis=0)
RMS_1_d = RMS_1_d[1::]

BIAS_1=MOD1_TOT - OBS1_TOT
BIAS_1_d=np.nanmean(MOD1_TOT - OBS1_TOT,axis=0)
BIAS_1_d = BIAS_1_d[1::]
if do_comp:
    RMS_2=np.sqrt((OBS2_TOT - MOD2_TOT)**2)
    RMS_2_d=np.nanmean(RMS_2,axis=0)
    RMS_2_d = RMS_2_d[1::]

    BIAS_2=MOD2_TOT - OBS2_TOT
    BIAS_2_d=np.nanmean(MOD2_TOT - OBS2_TOT,axis=0)
    BIAS_2_d = BIAS_2_d[1::]

MOD1_TOT = np.where(np.isnan(OBS1_TOT),np.nan, MOD1_TOT)
MOD2_TOT = np.where(np.isnan(OBS2_TOT),np.nan, MOD2_TOT)

OBS1_d=np.nanmean(OBS1_TOT,axis=0)
MOD1_d=np.nanmean(MOD1_TOT,axis=0)
OBS1_d=OBS1_d[1::]
MOD1_d=MOD1_d[1::]
if do_comp:
    OBS2_d=np.nanmean(OBS2_TOT,axis=0)
    MOD2_d=np.nanmean(MOD2_TOT,axis=0)
    OBS2_d=OBS2_d[1::]
    MOD2_d=MOD2_d[1::]

####### BAR profiles ###########"
####### RMSE ###############
DEPTH1 = DEPTH1[1::]

plt.figure(figsize=(8,10))
ax = plt.subplot(111)
ax.barh(np.array(range(int(len(OBS1_d)))), OBS1_d, color = 'k', height=0.165  ,label='Obs')
ax.barh(np.array(range(int(len(MOD1_d)))) + 0.165, MOD1_d, color = 'b', height=0.165  ,label=RUNid1)

if do_comp:
    ax.barh(np.array(range(int(len(MOD2_d))))+0.33, MOD2_d, color = 'r', height=0.165,label=RUNid2)

    plt.legend()
    ax.set_yticks(range(int(len(RMS_1_d))))
    ax.set_yticklabels(np.round(DEPTH1, decimals=1))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    ax.set_xlabel("S (psu)")
    ax.set_ylabel("DEPTH (m)")
    ax.set_title("Salinity")
    plt.savefig(DIR_PLOT_EXP+"PSAL_NOOBS"+RUNid1+"_"+RUNid2+".png")
    print("fig saved here :", DIR_PLOT_EXP+"PSAL_NOOBS"+RUNid1+"_"+RUNid2+".png")
    plt.close()


####### RMSE ###############

plt.figure(figsize=(8,10))
ax = plt.subplot(111)
ax.barh(np.array(range(int(len(RMS_1_d)))), RMS_1_d, color = 'b', height=0.25  ,label=RUNid1)
if do_comp:
    ax.barh(np.array(range(int(len(RMS_2_d))))+0.25, RMS_2_d, color = 'r', height=0.25,label=RUNid2)

    plt.legend()
    ax.set_yticks(range(int(len(RMS_1_d))))
    ax.set_yticklabels(np.round(DEPTH1, decimals=1))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    ax.set_xlabel("RMS (psu)")
    ax.set_ylabel("DEPTH (m)")
    ax.set_title("RMSE on Salinity")
    plt.savefig(DIR_PLOT_EXP+"PSAL_RMS_NOOBS"+RUNid1+"_"+RUNid2+".png")
    print("fig saved here :", DIR_PLOT_EXP+"PSAL_RMS_NOOBS"+RUNid1+"_"+RUNid2+".png")
    plt.close()
else:

    plt.legend()
    ax.set_yticks(range(int(len(RMS_1_d))))
    ax.set_yticklabels(np.round(DEPTH1, decimals=1))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    ax.set_xlabel("RMS (psu)")
    ax.set_ylabel("DEPTH (m)")
    ax.set_title("RMSE on Salinity")
    plt.savefig(DIR_PLOT_EXP+"PSAL_RMS_NOOBS"+RUNid1+".png")
    print("fig saved here :", DIR_PLOT_EXP+"PSAL_RMS_NOOBS"+RUNid1+".png")
    plt.close()


####### BIAS ###############
plt.figure(figsize=(8,10))
ax = plt.subplot(111)
ax.barh(np.array(range(int(len(BIAS_1_d)))), BIAS_1_d, color = 'b', height=0.25  ,label=RUNid1)
if do_comp:
    ax.barh(np.array(range(int(len(BIAS_2_d))))+0.25, BIAS_2_d, color = 'r', height=0.25,label=RUNid2)

    plt.legend()
    ax.set_yticks(range(int(len(BIAS_1_d))))
    ax.set_yticklabels(np.round(DEPTH1, decimals=1))
    ax.set_xlabel("BIAS (psu)")
    ax.set_ylabel("DEPTH (m)")
    ax.set_title("BIAS on Salinity")
    plt.tight_layout()

    plt.gca().invert_yaxis()
    plt.savefig(DIR_PLOT_EXP+"PSAL_BIAS_NOOBS"+RUNid1+"_"+RUNid2+".png")
    print("fig saved here :", DIR_PLOT_EXP+"PSAL_BIAS_NOOBS"+RUNid1+"_"+RUNid2+".png")
    plt.close()
else:

    plt.legend()
    ax.set_yticks(range(int(len(BIAS_1_d))))
    ax.set_yticklabels(np.round(DEPTH1, decimals=1))
    ax.set_xlabel("BIAS (psu)")
    ax.set_ylabel("DEPTH (m)")
    ax.set_title("BIAS on Salinity")
    plt.tight_layout()

    plt.gca().invert_yaxis()
    plt.savefig(DIR_PLOT_EXP+"PSAL_BIAS_NOOBS"+RUNid1+".png")
    print("fig saved here :", DIR_PLOT_EXP+"PSAL_BIAS_NOOBS"+RUNid1+".png")
    plt.close()


######## RMSE DIFF ############
if do_comp:
    plt.figure(figsize=(8,10))
    ax = plt.subplot(111)
    label=RUNid2 + "-" + RUNid1
    ax.barh(np.array(range(int(len(RMS_1_d)))), RMS_2_d - RMS_1_d, color = 'b', height=0.25  ,label=label)
    
    plt.legend()
    ax.set_yticks(range(int(len(RMS_1_d))))
    ax.set_yticklabels(np.round(DEPTH1, decimals=1))
    ax.set_xlabel("RMS (psu)")
    ax.set_ylabel("DEPTH (m)")
    ax.set_title("RMSE difference ("+RUNid1+"-"+RUNid2+"on Salinity")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(DIR_PLOT_EXP+"PSAL_RMS_diff_NOOBS"+RUNid1+"_"+RUNid2+".png")
    print("fig saved here :", DIR_PLOT_EXP+"PSAL_RMS_diff_NOOBS"+RUNid1+"_"+RUNid2+".png")
    plt.close()
 
######## BIAS DIFF ###########
if do_comp:
    plt.figure(figsize=(8,10))
    ax = plt.subplot(111)
    label=RUNid2 + "-" + RUNid1
    ax.barh(np.array(range(int(len(BIAS_1_d)))), BIAS_2_d - BIAS_1_d, color = 'b', height=0.25  ,label=label)

    plt.legend()
    ax.set_yticks(range(int(len(BIAS_1_d))))
    ax.set_yticklabels(np.round(DEPTH1, decimals=1))
    ax.set_xlabel("BIAS (psu)")
    ax.set_ylabel("DEPTH (m)")
    ax.set_title("Mean difference ("+RUNid1+"-"+RUNid2+"on Salinity")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(DIR_PLOT_EXP+"PSAL_BIAS_diff_NOOBS"+RUNid1+"_"+RUNid2+".png")
    print("fig saved here :", DIR_PLOT_EXP+"PSAL_BIAS_diff_NOOBS"+RUNid1+"_"+RUNid2+".png")
    plt.close()


