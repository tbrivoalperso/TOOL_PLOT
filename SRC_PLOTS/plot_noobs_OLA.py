#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:41:00 2020

@author: mhamon
"""
import datetime

import numpy as np
import pandas as pd
import xarray as xr
from plot_ola_oov2.READ_NML import SYSid, CYCLE, TYPE2DO, level, DIR_OLA_OFF1, DIR_OLA_OFF2, samout1, samout2, type_ccl1, type_ccl2, \
                                   RUNid1, RUNid2, RES, SET1, SET2, PREC, do_comp, DIR_PLOT_EXP, PARAM_PLOT, \
                                   PARAM_PLOT_POLARNorth, PARAM_PLOT_POLARSouth
from plot_ola_oov2.LOAD_OLA_OOV2 import load_ola, load_olasam_bin
from lib_pyt.Sort_list import FFcompare_list
from lib_pyt.Griddata_custom import grid_data_NUMBA

import matplotlib.pyplot as plt
from lib_pyt.Class_plot_cartopy import Plot_map_cartopy as map
from lib_pyt.cmaps import cmaps
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

AUTOSHIFT_SLA=True

map2d = map()
map2d.params=PARAM_PLOT

map2dpolarn = map()
map2dpolarn.params=PARAM_PLOT_POLARNorth

map2dpolars = map()
map2dpolars.params=PARAM_PLOT_POLARSouth

offset_time=(datetime.datetime(1970,1,1)-datetime.datetime(1950,1,1)).days
kwargs = {'format': '%.1f'}
for ii, type_ola in enumerate(TYPE2DO):

    res = RES['res_'+type_ola]
    setid1 = SET1['setid_'+type_ola]
    prec = PREC['prec_'+type_ola]
    print(setid1)
    try :
        del olavectot1, olavectot2
    except NameError:
        pass

    for ic, cyc in enumerate(CYCLE):
        if samout1:
            olavec1 = load_olasam_bin(DIR_OLA_OFF1, cyc, type_ccl1, type_ola, setid=setid1, level=level)
        else:
            olavec1 = load_ola(DIR_OLA_OFF1, cyc, type_ola, setid=setid1, level=level)

        if do_comp:
            setid2 = SET2['setid_'+type_ola]
            if samout2:
                olavec2 = load_olasam_bin(DIR_OLA_OFF2, cyc, type_ccl2, type_ola, setid=setid2, level=level)
            else:
                olavec2 = load_ola(DIR_OLA_OFF2, cyc, type_ola, setid=setid2, level=level)
            # On reorganise les vecteur pour s'assurer d'avoir les memes points d'observations.
            olavec1 = np.round(olavec1,prec)
            olavec2 = np.round(olavec2,prec)

            arg1,arg2 = FFcompare_list(olavec1[:,:3],olavec2[:,:3])

            olavec1 = olavec1[arg1,:]
            olavec2 = olavec2[arg2,:]

        try:
            olavectot1 = np.concatenate((olavectot1, olavec1), axis=0)
        except NameError:
            olavectot1 = olavec1
        if do_comp:
            try:
                olavectot2 = np.concatenate((olavectot2, olavec2), axis=0)
            except NameError:
                olavectot2 = olavec2

    indok = np.where(np.isfinite(olavectot1[:,3]))[0]

    olavectot1 = olavectot1[indok,:]
    if do_comp:
        olavectot2 = olavectot2[indok,:]
    
    OBS1 = olavectot1[:,3]
    EQUIV1 = olavectot1[:,4]
    if type_ola == 'IS_SLA' and AUTOSHIFT_SLA:
        print("remove mean bias from SLA")
        mean_bias1=np.nanmean(EQUIV1) - np.nanmean(OBS1)
        EQUIV1=EQUIV1 - mean_bias1
        print("Bias removed on EQUIV1 = ", mean_bias1)
    BIAS1 = EQUIV1 - OBS1
    ERR1 = olavectot1[:,5]
    RMSE1 = np.sqrt((EQUIV1 - OBS1)**2)   #abs(BIAS1)
    NORM_ERROR1 = RMSE1/ERR1*100
    if do_comp:
        OBS2 = olavectot2[:,3]
        EQUIV2 = olavectot2[:,4]
        if type_ola == 'IS_SLA' and AUTOSHIFT_SLA:
            print("remove mean bias from SLA")
            mean_bias2=np.nanmean(EQUIV2) - np.nanmean(OBS2)
            EQUIV2=EQUIV2 - mean_bias2
            print("Bias removed on EQUIV2 = ", mean_bias2)
        BIAS2 = EQUIV2 - OBS2
        ERR2 = olavectot2[:,5]
        RMSE2 = np.sqrt((EQUIV2 - OBS2)**2)   #abs(BIAS2)
        NORM_ERROR2 = RMSE2/ERR2*100
    
    LON = olavectot1[:,0]
    LAT = olavectot1[:,1]
    TIME = olavectot1[:,2]
    
    # OBS
    list1 = np.column_stack((LAT, LON, OBS1))
    nbdata1, xlon, ylat, obsgrid1 = grid_data_NUMBA(list1,res,res,1)
    if do_comp:
        list2 = np.column_stack((LAT, LON, OBS2))
        nbdata2, xlon, ylat, obsgrid2 = grid_data_NUMBA(list2,res,res,1)
    # EQUIV MODEL
    list1 = np.column_stack((LAT, LON, EQUIV1))
    nbdata1, xlon, ylat, equivgrid1 = grid_data_NUMBA(list1,res,res,1)
    if do_comp:
        list2 = np.column_stack((LAT, LON, EQUIV2))
        nbdata2, xlon, ylat, equivgrid2 = grid_data_NUMBA(list2,res,res,1)

    #BIAS
    INNXR1 = xr.Dataset()
    INNXR1['BIAS'] = ('t', BIAS1)
    INNXR1 = INNXR1.assign_coords(t=TIME)
    if type_ola == 'DS_SIC':
        INNXR1N = INNXR1.where(LAT > 0)
        INNXR1S = INNXR1.where(LAT < 0)
        TSINN1N = INNXR1N.groupby(np.trunc(INNXR1N.t)).sum()
        TSINN1S = INNXR1S.groupby(np.trunc(INNXR1S.t)).sum()
    else:
        TSINN1 = INNXR1.groupby(np.trunc(INNXR1.t)).mean()

    if type_ola == 'DS_SIC':
        DATE_TS = pd.to_datetime(TSINN1N['t'].data-offset_time,unit='D')
        TSINN1N = TSINN1N.assign_coords({'date':('t', DATE_TS)})
        TSINN1S = TSINN1S.assign_coords({'date':('t', DATE_TS)})
    else:
        DATE_TS = pd.to_datetime(TSINN1['t'].data-offset_time,unit='D')
        TSINN1 = TSINN1.assign_coords({'date':('t', DATE_TS)})

    list1 = np.column_stack((LAT, LON, BIAS1))
    _, _, _, innovgrid1 = grid_data_NUMBA(list1,res,res,1)
 
    if do_comp:
        INNXR2 = xr.Dataset()
        INNXR2['BIAS'] = ('t', BIAS2)
        INNXR2 = INNXR2.assign_coords(t=TIME)
        if type_ola == 'DS_SIC':
            INNXR2N = INNXR2.where(LAT > 0)
            INNXR2S = INNXR2.where(LAT < 0)
            TSINN2N = INNXR2N.groupby(np.trunc(INNXR2N.t)).sum()
            TSINN2S = INNXR2S.groupby(np.trunc(INNXR2S.t)).sum()
            TSINN2N = TSINN2N.assign_coords({'date':('t', DATE_TS)})
            TSINN2S = TSINN2S.assign_coords({'date':('t', DATE_TS)})
        else:
            TSINN2 = INNXR2.groupby(np.trunc(INNXR2.t)).mean()
            TSINN2 = TSINN2.assign_coords({'date':('t', DATE_TS)})

        list2 = np.column_stack((LAT, LON, BIAS2))
        _, _, _, innovgrid2 = grid_data_NUMBA(list2,res,res,1)
    
    #FCST ERROR
    ERRXR1 = xr.Dataset()
    ERRXR1['ERROR'] = ('t', RMSE1)
    ERRXR1 = ERRXR1.assign_coords(t=TIME)
    if type_ola == 'DS_SIC':
        ERRXR1N = ERRXR1.where(LAT > 0)
        ERRXR1S = ERRXR1.where(LAT < 0)
        TSERR1N = ERRXR1N.groupby(np.trunc(ERRXR1N.t)).sum()
        TSERR1S = ERRXR1S.groupby(np.trunc(ERRXR1S.t)).sum()
        TSERR1N = TSERR1N.assign_coords({'date':('t', DATE_TS)})
        TSERR1S = TSERR1S.assign_coords({'date':('t', DATE_TS)})
    else:
        TSERR1 = ERRXR1.groupby(np.trunc(ERRXR1.t)).mean()
        TSERR1 = TSERR1.assign_coords({'date':('t', DATE_TS)})

    list1 = np.column_stack((LAT, LON, RMSE1))
    _, _, _, fcsterrorgrid1 = grid_data_NUMBA(list1,res,res,1)
    if do_comp:
        ERRXR2 = xr.Dataset()
        ERRXR2['ERROR'] = ('t', RMSE2)
        ERRXR2 = ERRXR2.assign_coords(t=TIME)
        if type_ola == 'DS_SIC':
            ERRXR2N = ERRXR2.where(LAT > 0)
            ERRXR2S = ERRXR2.where(LAT < 0)
            TSERR2N = ERRXR2N.groupby(np.trunc(ERRXR2N.t)).sum()
            TSERR2S = ERRXR2S.groupby(np.trunc(ERRXR2S.t)).sum()
            TSERR2N = TSERR2N.assign_coords({'date':('t', DATE_TS)})
            TSERR2S = TSERR2S.assign_coords({'date':('t', DATE_TS)})
        else:
            TSERR2 = ERRXR2.groupby(np.trunc(ERRXR2.t)).mean()
            TSERR2 = TSERR2.assign_coords({'date':('t', DATE_TS)})

        list2 = np.column_stack((LAT, LON, RMSE2))
        _, _, _, fcsterrorgrid2 = grid_data_NUMBA(list2,res,res,1)

    #NORM ERROR
    list1 = np.column_stack((LAT, LON, NORM_ERROR1))
    _, _, _, normerrorgrid1 = grid_data_NUMBA(list1,res,res,1)
    if do_comp:
        list2 = np.column_stack((LAT, LON, NORM_ERROR2))
        _, _, _, normerrorgrid2 = grid_data_NUMBA(list2,res,res,1)

    
    ## PLOT ##
    if type_ola == 'IS_SLA':
        param = 'SLA'
        label = 'sla obs [m]'
    elif type_ola == 'DS_SST':
        param = 'SST'
        label = 'sst obs [°C]'
    elif type_ola == 'DS_SSS':
        param = 'SSS'
        label = 'sss obs [psu]'
    elif type_ola == 'DS_SIC':
        param = 'SIC'
        label = 'sic obs'
    elif type_ola == 'VP_T':
        param = 'TEMP_level'+str(level)
        label = 'in situ T obs [°C]'
    elif type_ola == 'VP_S':
        param = 'PSAL_level'+str(level)
        label = 'in situ S obs [psu]'

    ###### OBS ######
    map2d.params['colormap'] = 'jet'
    map2d.params['cb_labelsize']=20
    map2dpolarn.params['colormap'] = 'jet'
    map2dpolars.params['colormap'] = 'jet'

    if type_ola == 'IS_SLA':
        map2d.params['cmin'] = -0.3
        map2d.params['cmax'] = 0.3
        map2d.params['subtitle'] = "SLA Observations [m]"
        map2d.params['colormap'] = 'seismic'
        ampdiff = 0.1
    elif type_ola == 'DS_SST':
        map2d.params['cmin'] = 8
        map2d.params['cmax'] = 25
        map2d.params['subtitle'] = "SST Observations [°C]"
        ampdiff = 1
    elif type_ola == 'DS_SSS':
        map2d.params['cmin'] = 32
        map2d.params['cmax'] = 37
        map2d.params['subtitle'] = "SSS Observations [psu]"
        ampdiff = 0.2
    elif type_ola == 'DS_SIC':
        map2dpolarn.params['cmin'] = 0
        map2dpolarn.params['cmax'] = 1
        map2dpolarn.params['subtitle'] = "SIC Observations"
        map2dpolars.params['cmin'] = 0
        map2dpolars.params['cmax'] = 1
        map2dpolars.params['subtitle'] = "SIC Observations"
        ampdiff = 0.1
    elif type_ola == 'VP_T':
        map2d.params['cmin'] = 8
        map2d.params['cmax'] = 25
        map2d.params['subtitle'] = "Temperature Observations at level "+str(int(level))+" [°C]"
        ampdiff = 1
    elif type_ola == 'VP_S':
        map2d.params['cmin'] = 34
        map2d.params['cmax'] = 36
        map2d.params['subtitle'] = "Salinity Observations at level "+str(int(level))+" [°C]"
        ampdiff = 0.2

    if type_ola == 'DS_SIC':
        figname1N = param+'_OBS_'+RUNid1+'-North.png'
        figname1S = param+'_OBS_'+RUNid1+'-South.png'
        figname2N = param+'_OBS_'+RUNid2+'-North.png'
        figname2S = param+'_OBS_'+RUNid2+'-South.png'
        fignamediffN = param+'_OBS_'+RUNid1+'_'+RUNid2+'-North.png'
        fignamediffS = param+'_OBS_'+RUNid1+'_'+RUNid2+'-South.png'
    else:
        figname1 = param+'_OBS_'+RUNid1+'.png'
        figname2 = param+'_OBS_'+RUNid2+'.png'
        fignamediff = param+'_OBS_'+RUNid1+'_'+RUNid2+'.png'

    if type_ola == 'DS_SIC':
        map2dpolarn.pcolor(xlon, ylat, obsgrid1, show=False, output=DIR_PLOT_EXP+figname1N)
        map2dpolars.pcolor(xlon, ylat, obsgrid1, show=False, output=DIR_PLOT_EXP+figname1S)
    else:
        map2d.pcolor(xlon, ylat, obsgrid1, show=False, output=DIR_PLOT_EXP+figname1)
    if do_comp:
        if type_ola == 'DS_SIC':
            map2dpolarn.pcolor(xlon, ylat, obsgrid2, show=False, output=DIR_PLOT_EXP+figname2N)
            map2dpolars.pcolor(xlon, ylat, obsgrid2, show=False, output=DIR_PLOT_EXP+figname2S)
            # diff
            map2dpolarn.params['colormap'] = 'MO_greyed70_middle_rev'
            map2dpolars.params['colormap'] = 'MO_greyed70_middle_rev'
            map2dpolarn.params['cmin'] = -ampdiff; map2dpolarn.params['cmax'] = ampdiff
            map2dpolarn.pcolor(xlon, ylat, obsgrid1-obsgrid2, show=False, output=DIR_PLOT_EXP+fignamediffN)
            map2dpolars.params['cmin'] = -ampdiff; map2dpolars.params['cmax'] = ampdiff
            map2dpolars.pcolor(xlon, ylat, obsgrid1-obsgrid2, show=False, output=DIR_PLOT_EXP+fignamediffS)
        else:
            map2d.pcolor(xlon, ylat, obsgrid2, show=False, output=DIR_PLOT_EXP+figname2)
            # diff
            map2d.params['colormap'] = 'MO_greyed70_middle_rev'
            map2d.params['cmin'] = -ampdiff; map2d.params['cmax'] = ampdiff
            map2d.pcolor(xlon, ylat, obsgrid1-obsgrid2, show=False, output=DIR_PLOT_EXP+fignamediff)

    ###### EQUIV MODEL ######
    map2d.params['colormap'] = 'jet'
    map2dpolarn.params['colormap'] = 'jet'
    map2dpolars.params['colormap'] = 'jet'
    if type_ola == 'IS_SLA':
        map2d.params['cmin'] = -0.3
        map2d.params['cmax'] = 0.3
        map2d.params['subtitle'] = "SLA model equivalent [m]"
        map2d.params['colormap'] = 'seismic'
        ampdiff = 0.1
    elif type_ola == 'DS_SST':
        map2d.params['cmin'] = 8
        map2d.params['cmax'] = 25
        map2d.params['subtitle'] = "SST model equivalent [°C]"
        ampdiff = 1
    elif type_ola == 'DS_SSS':
        map2d.params['cmin'] = 32
        map2d.params['cmax'] = 37
        map2d.params['subtitle'] = "SSS model equivalent [psu]"
        ampdiff = 0.2
    elif type_ola == 'DS_SIC':
        map2dpolarn.params['cmin'] = 0
        map2dpolarn.params['cmax'] = 1
        map2dpolarn.params['subtitle'] = "SIC model equivalent"
        map2dpolars.params['cmin'] = 0
        map2dpolars.params['cmax'] = 1
        map2dpolars.params['subtitle'] = "SIC model equivalent"
        ampdiff = 0.1
    elif type_ola == 'VP_T':
        map2d.params['cmin'] = 8
        map2d.params['cmax'] = 25
        map2d.params['subtitle'] = "Temperature model equivalent at level "+str(int(level))+" [°C]"
        ampdiff = 1
    elif type_ola == 'VP_S':
        map2d.params['cmin'] = 34
        map2d.params['cmax'] = 36
        map2d.params['subtitle'] = "Salinity model equivalent at level "+str(int(level))+" [°C]"
        ampdiff = 0.2

    if type_ola == 'DS_SIC':
        figname1N = param+'_EQUIV_'+RUNid1+'-North.png'
        figname1S = param+'_EQUIV_'+RUNid1+'-South.png'
        figname2N = param+'_EQUIV_'+RUNid2+'-North.png'
        figname2S = param+'_EQUIV_'+RUNid2+'-South.png'
        fignamediffN = param+'_EQUIV_'+RUNid1+'_'+RUNid2+'-North.png'
        fignamediffS = param+'_EQUIV_'+RUNid1+'_'+RUNid2+'-South.png'
    else:
        figname1 = param+'_EQUIV_'+RUNid1+'.png'
        figname2 = param+'_EQUIV_'+RUNid2+'.png'
        fignamediff = param+'_EQUIV_'+RUNid1+'_'+RUNid2+'.png'

    if type_ola == 'DS_SIC':
        map2dpolarn.pcolor(xlon, ylat, equivgrid1, show=False, output=DIR_PLOT_EXP+figname1N)
        map2dpolars.pcolor(xlon, ylat, equivgrid1, show=False, output=DIR_PLOT_EXP+figname1S)
    else:
        map2d.pcolor(xlon, ylat, equivgrid1, show=False, output=DIR_PLOT_EXP+figname1)
    if do_comp:
        if type_ola == 'DS_SIC':
            map2dpolarn.pcolor(xlon, ylat, equivgrid2, show=False, output=DIR_PLOT_EXP+figname2N)
            map2dpolars.pcolor(xlon, ylat, equivgrid2, show=False, output=DIR_PLOT_EXP+figname2S)
            # diff
            map2dpolarn.params['colormap'] = 'MO_greyed70_middle_rev'
            map2dpolars.params['colormap'] = 'MO_greyed70_middle_rev'
            map2dpolarn.params['cmin'] = -ampdiff; map2dpolarn.params['cmax'] = ampdiff
            map2dpolarn.pcolor(xlon, ylat, equivgrid1-equivgrid2, show=False, output=DIR_PLOT_EXP+fignamediffN)
            map2dpolars.params['cmin'] = -ampdiff; map2dpolars.params['cmax'] = ampdiff
            map2dpolars.pcolor(xlon, ylat, equivgrid1-equivgrid2, show=False, output=DIR_PLOT_EXP+fignamediffS)
        else:
            map2d.pcolor(xlon, ylat, equivgrid2, show=False, output=DIR_PLOT_EXP+figname2)
            # diff
            map2d.params['colormap'] = 'MO_greyed70_middle_rev'
            map2d.params['cmin'] = -ampdiff; map2d.params['cmax'] = ampdiff
            map2d.pcolor(xlon, ylat, equivgrid1-equivgrid2, show=False, output=DIR_PLOT_EXP+fignamediff)


    ###### BIAS ######
    map2d.params['colormap'] = 'MO_greyed70_middle_rev'
    map2dpolarn.params['colormap'] = 'MO_greyed70_middle_rev'
    map2dpolars.params['colormap'] = 'MO_greyed70_middle_rev'
    if type_ola == 'IS_SLA':
        map2d.params['cmin'] = -0.25
        map2d.params['cmax'] = 0.25
        map2d.params['subtitle'] = "SLA Difference with obs [m]"
        ampdiff = 0.1
    elif type_ola == 'DS_SST':
        map2d.params['cmin'] = -2
        map2d.params['cmax'] = 2
        map2d.params['subtitle'] = "SST Difference with obs [°C]"
        ampdiff = 0.5
    elif type_ola == 'DS_SSS':
        map2d.params['cmin'] = -0.5
        map2d.params['cmax'] = 0.5
        map2d.params['subtitle'] = "SSS Difference with obs [psu]"
        ampdiff = 0.1
    elif type_ola == 'DS_SIC':
        map2dpolarn.params['cmin'] = -0.5
        map2dpolarn.params['cmax'] = 0.5
        map2dpolarn.params['subtitle'] = "SIC Difference with obs"
        map2dpolars.params['cmin'] = -0.5
        map2dpolars.params['cmax'] = 0.5
        map2dpolars.params['subtitle'] = "SIC Difference with obs"
        ampdiff = 0.1
    elif type_ola == 'VP_T':
        map2d.params['cmin'] = -1
        map2d.params['cmax'] = 1
        map2d.params['subtitle'] = "Temperature Difference with obs at level "+str(int(level))+" [°C]"
        ampdiff = 0.5
    elif type_ola == 'VP_S':
        map2d.params['cmin'] = -0.25
        map2d.params['cmax'] = 0.25
        map2d.params['subtitle'] = "Salinity Difference with obs at level "+str(int(level))+" [psu]"
        ampdiff = 0.1

    if type_ola == 'DS_SIC':
        figname1N = param+'_BIAS_'+RUNid1+'-North.png'
        figname1S = param+'_BIAS_'+RUNid1+'-South.png'
        figname2N = param+'_BIAS_'+RUNid2+'-North.png'
        figname2S = param+'_BIAS_'+RUNid2+'-South.png'
        fignamediffN = param+'_BIAS_'+RUNid1+'_'+RUNid2+'-North.png'
        fignamediffS = param+'_BIAS_'+RUNid1+'_'+RUNid2+'-South.png'
        figtsN =  param+'_TS_BIAS-North.png'
        figtsS =  param+'_TS_BIAS-South.png'
    else:
        figname1 = param+'_BIAS_'+RUNid1+'.png'
        figname2 = param+'_BIAS_'+RUNid2+'.png'
        fignamediff = param+'_BIAS_'+RUNid1+'_'+RUNid2+'.png'
        figts = param+'_TS_BIAS.png'

    if type_ola == 'DS_SIC':
        map2dpolarn.pcolor(xlon, ylat, innovgrid1, show=False, output=DIR_PLOT_EXP+figname1N)
        map2dpolars.pcolor(xlon, ylat, innovgrid1, show=False, output=DIR_PLOT_EXP+figname1S)
    else:
        map2d.pcolor(xlon, ylat, innovgrid1, show=False, output=DIR_PLOT_EXP+figname1)
    if do_comp:
        if type_ola == 'DS_SIC':
            map2dpolarn.pcolor(xlon, ylat, innovgrid2, show=False, output=DIR_PLOT_EXP+figname2N)
            map2dpolars.pcolor(xlon, ylat, innovgrid2, show=False, output=DIR_PLOT_EXP+figname2S)
            # diff
            map2dpolarn.params['colormap'] = 'MO_greyed70_middle_rev'
            map2dpolars.params['colormap'] = 'MO_greyed70_middle_rev'
            map2dpolarn.params['cmin'] = -ampdiff; map2dpolarn.params['cmax'] = ampdiff
            map2dpolarn.pcolor(xlon, ylat, innovgrid1-innovgrid2, show=False, output=DIR_PLOT_EXP+fignamediffN)
            map2dpolars.params['cmin'] = -ampdiff; map2dpolars.params['cmax'] = ampdiff
            map2dpolars.pcolor(xlon, ylat, innovgrid1-innovgrid2, show=False, output=DIR_PLOT_EXP+fignamediffS)
        else:
            map2d.pcolor(xlon, ylat, innovgrid2, show=False, output=DIR_PLOT_EXP+figname2)
            # diff
            map2d.params['colormap'] = 'MO_greyed70_middle_rev'
            map2d.params['cmin'] = -ampdiff; map2d.params['cmax'] = ampdiff
            map2d.pcolor(xlon, ylat, innovgrid1-innovgrid2, show=False, output=DIR_PLOT_EXP+fignamediff)

    if type_ola != 'DS_SIC':
        # Time Series
        try:
            plt.figure()
            TSINN1.BIAS.plot(x='date', label=RUNid1, color='b')
            if do_comp:
                TSINN2.BIAS.plot(x='date', label=RUNid2,color='r')
            plt.legend()
            plt.xlabel('time')
            plt.ylabel('Mean Difference with obs')
            plt.grid(linewidth=0.5,linestyle='--')
            plt.savefig(DIR_PLOT_EXP+figts, format='png',bbox_inches='tight',dpi=200)
        except TypeError:
            print('Not enough date for TS plot! pass...')
            pass
    else:
        try:
            plt.figure()
            TSINN1N.BIAS.plot(x='date', label=RUNid1, color='b')
            if do_comp:
                TSINN2N.BIAS.plot(x='date', label=RUNid2,color='r')
            plt.legend()
            plt.xlabel('time')
            plt.ylabel('Summed Difference with obs')
            plt.grid(linewidth=0.5,linestyle='--')
            plt.savefig(DIR_PLOT_EXP+figtsN, format='png',bbox_inches='tight',dpi=200)
            #
            plt.figure()
            TSINN1S.BIAS.plot(x='date', label=RUNid1, color='b')
            if do_comp:
                TSINN2S.BIAS.plot(x='date', label=RUNid2, color='r')
            plt.legend()
            plt.xlabel('time')
            plt.ylabel('Summed Difference with obs')
            plt.grid(linewidth=0.5,linestyle='--')
            plt.savefig(DIR_PLOT_EXP+figtsS, format='png',bbox_inches='tight',dpi=200)
        except TypeError:
            print('Not enough date for TS plot! pass...')
            pass

    ###### ERROR ######
    map2d.params['colormap'] = 'MO_Blue-Red-greyed_minimum'
    map2dpolarn.params['colormap'] = 'MO_Blue-Red-greyed_minimum'
    map2dpolars.params['colormap'] = 'MO_Blue-Red-greyed_minimum'
    if type_ola == 'IS_SLA':
        map2d.params['cmin'] = 0
        map2d.params['cmax'] = 0.3
        map2d.params['subtitle'] = "SLA RMSE [m]"
        ampdiff = 0.1
    elif type_ola == 'DS_SST':
        map2d.params['cmin'] = 0
        map2d.params['cmax'] = 2
        map2d.params['subtitle'] = "SST RMSE [°C]"
        ampdiff = 0.5
    elif type_ola == 'DS_SSS':
        map2d.params['cmin'] = 0
        map2d.params['cmax'] = 0.5
        map2d.params['subtitle'] = "SSS RMSE [psu]"
        ampdiff = 0.1
    elif type_ola == 'DS_SIC':
        map2dpolarn.params['cmin'] = 0
        map2dpolarn.params['cmax'] = 0.5
        map2dpolarn.params['subtitle'] = "SIC RMSE"
        map2dpolars.params['cmin'] = 0
        map2dpolars.params['cmax'] = 0.5
        map2dpolars.params['subtitle'] = "SIC RMSE"
        ampdiff = 0.1
    elif type_ola == 'VP_T':
        map2d.params['cmin'] = 0
        map2d.params['cmax'] = 1
        map2d.params['subtitle'] = "Temperature RMSE at level "+str(int(level))+" [°C]"
        ampdiff = 0.5
    elif type_ola == 'VP_S':
        map2d.params['cmin'] = 0
        map2d.params['cmax'] = 0.25
        map2d.params['subtitle'] = "Salinity RMSE at level "+str(int(level))+" [psu]"
        ampdiff = 0.1

    if type_ola == 'DS_SIC':
        figname1N = param+'_RMSE_'+RUNid1+'-North.png'
        figname1S = param+'_RMSE_'+RUNid1+'-South.png'
        figname2N = param+'_RMSE_'+RUNid2+'-North.png'
        figname2S = param+'_RMSE_'+RUNid2+'-South.png'
        fignamediffN = param+'_RMSE_'+RUNid1+'_'+RUNid2+'-North.png'
        fignamediffS = param+'_RMSE_'+RUNid1+'_'+RUNid2+'-South.png'
        figtsN = param+'_TS_RMSE-North.png'
        figtsS = param+'_TS_RMSE-South.png'
    else:
        figname1 = param+'_RMSE_'+RUNid1+'.png'
        figname2 = param+'_RMSE_'+RUNid2+'.png'
        fignamediff = param+'_RMSE_'+RUNid1+'_'+RUNid2+'.png'
        figts = param+'_TS_RMSE.png'

    if type_ola == 'DS_SIC':
        map2dpolarn.pcolor(xlon, ylat, fcsterrorgrid1, show=False, output=DIR_PLOT_EXP+figname1N)
        map2dpolars.pcolor(xlon, ylat, fcsterrorgrid1, show=False, output=DIR_PLOT_EXP+figname1S)
    else:
        map2d.pcolor(xlon, ylat, fcsterrorgrid1, show=False, output=DIR_PLOT_EXP+figname1)
    if do_comp:
        if type_ola == 'DS_SIC':
            map2dpolarn.pcolor(xlon, ylat, fcsterrorgrid2, show=False, output=DIR_PLOT_EXP+figname2N)
            map2dpolars.pcolor(xlon, ylat, fcsterrorgrid2, show=False, output=DIR_PLOT_EXP+figname2S)
            # diff
            map2dpolarn.params['colormap'] = 'MO_greyed70_middle_rev'
            map2dpolars.params['colormap'] = 'MO_greyed70_middle_rev'
            map2dpolarn.params['cmin'] = -ampdiff; map2dpolarn.params['cmax'] = ampdiff
            map2dpolarn.pcolor(xlon, ylat, fcsterrorgrid1-fcsterrorgrid2, show=False, output=DIR_PLOT_EXP+fignamediffN)
            map2dpolars.params['cmin'] = -ampdiff; map2dpolars.params['cmax'] = ampdiff
            map2dpolars.pcolor(xlon, ylat, fcsterrorgrid1-fcsterrorgrid2, show=False, output=DIR_PLOT_EXP+fignamediffS)
        else:
            map2d.pcolor(xlon, ylat, fcsterrorgrid2, show=False, output=DIR_PLOT_EXP+figname2)
            # diff
            map2d.params['colormap'] = 'MO_greyed70_middle_rev'
            map2d.params['cmin'] = -ampdiff; map2d.params['cmax'] = ampdiff
            map2d.pcolor(xlon, ylat, fcsterrorgrid1-fcsterrorgrid2, show=False, output=DIR_PLOT_EXP+fignamediff)

    if type_ola != 'DS_SIC':
        # Time Series
        try:
            plt.figure()
            TSERR1.ERROR.plot(x='date', label=RUNid1, color='b')
            if do_comp:
                TSERR2.ERROR.plot(x='date', label=RUNid2, color='r')
            plt.legend()
            plt.xlabel('time')
            plt.ylabel('Mean RMSE')
            plt.grid(linewidth=0.5,linestyle='--')
            plt.savefig(DIR_PLOT_EXP+figts, format='png',bbox_inches='tight',dpi=200)
        except TypeError:
            print('Not enough date for TS plot! pass...')
            pass
    else:
        try:
            plt.figure()
            TSERR1N.ERROR.plot(x='date', label=RUNid1, color='b')
            if do_comp:
                TSERR2N.ERROR.plot(x='date', label=RUNid2, color='r')
            plt.legend()
            plt.xlabel('time')
            plt.ylabel('Summed RMSE')
            plt.grid(linewidth=0.5,linestyle='--')
            plt.savefig(DIR_PLOT_EXP+figtsN, format='png',bbox_inches='tight',dpi=200)
            #
            plt.figure()
            TSERR1S.ERROR.plot(x='date', label=RUNid1, color='b')
            if do_comp:
                TSERR2S.ERROR.plot(x='date', label=RUNid2, color='r')
            plt.legend()
            plt.xlabel('time')
            plt.ylabel('Summed RMSE')
            plt.grid(linewidth=0.5,linestyle='--')
            plt.savefig(DIR_PLOT_EXP+figtsS, format='png',bbox_inches='tight',dpi=200)
        except TypeError:
            print('Not enough date for TS plot! pass...')
            pass

    ###### NORM ERROR ######
    map2d.params['colormap'] = 'MO_Blue-Red-greyed_minimum'
    map2dpolarn.params['colormap'] = 'MO_Blue-Red-greyed_minimum'
    map2dpolars.params['colormap'] = 'MO_Blue-Red-greyed_minimum'
    if type_ola == 'IS_SLA':
        map2d.params['cmin'] = 0
        map2d.params['cmax'] = 300
        map2d.params['subtitle'] = "SLA norm. RMSE [%]"
        ampdiff = 100
    elif type_ola == 'DS_SST':
        map2d.params['cmin'] = 0
        map2d.params['cmax'] = 300
        map2d.params['subtitle'] = "SST norm. RMSE [%]"
        ampdiff = 100
    elif type_ola == 'DS_SSS':
        map2d.params['cmin'] = 0
        map2d.params['cmax'] = 300
        map2d.params['subtitle'] = "SSS norm. RMSE [%]"
        ampdiff = 100
    elif type_ola == 'DS_SIC':
        map2dpolarn.params['cmin'] = 0
        map2dpolarn.params['cmax'] = 300
        map2dpolarn.params['subtitle'] = "SIC norm. RMSE [%]"
        map2dpolars.params['cmin'] = 0
        map2dpolars.params['cmax'] = 300
        map2dpolars.params['subtitle'] = "SIC norm. RMSE [%]"
        ampdiff = 100
    elif type_ola == 'VP_T':
        map2d.params['cmin'] = 0
        map2d.params['cmax'] = 300
        map2d.params['subtitle'] = "Temperature norm. RMSE at level "+str(int(level))+" [%]"
        ampdiff = 100
    elif type_ola == 'VP_S':
        map2d.params['cmin'] = 0
        map2d.params['cmax'] = 300
        map2d.params['subtitle'] = "Salinity norm. RMSE at level "+str(int(level))+" [%]"
        ampdiff = 100

    if type_ola == 'DS_SIC':
        figname1N = param+'_NORM_RMSE_'+RUNid1+'-North.png'
        figname1S = param+'_NORM_RMSE_'+RUNid1+'-South.png'
        figname2N = param+'_NORM_RMSE_'+RUNid2+'-North.png'
        figname2S = param+'_NORM_RMSE_'+RUNid2+'-South.png'
        fignamediffN = param+'_NORM_RMSE_'+RUNid1+'_'+RUNid2+'-North.png'
        fignamediffS = param+'_NORM_RMSE_'+RUNid1+'_'+RUNid2+'-South.png'
    else:
        figname1 = param+'_NORM_RMSE_'+RUNid1+'.png'
        figname2 = param+'_NORM_RMSE_'+RUNid2+'.png'
        fignamediff = param+'_NORM_RMSE_'+RUNid1+'_'+RUNid2+'.png'

    if type_ola == 'DS_SIC':
        map2dpolarn.pcolor(xlon, ylat, normerrorgrid1, show=False, output=DIR_PLOT_EXP+figname1N)
        map2dpolars.pcolor(xlon, ylat, normerrorgrid1, show=False, output=DIR_PLOT_EXP+figname1S)
    else:
        map2d.pcolor(xlon, ylat, normerrorgrid1, show=False, output=DIR_PLOT_EXP+figname1)
    if do_comp:
        if type_ola == 'DS_SIC':
            map2dpolarn.pcolor(xlon, ylat, normerrorgrid2, show=False, output=DIR_PLOT_EXP+figname2N)
            map2dpolars.pcolor(xlon, ylat, normerrorgrid2, show=False, output=DIR_PLOT_EXP+figname2S)
            # diff
            map2dpolarn.params['colormap'] = 'MO_greyed70_middle_rev'

            map2dpolarn.params['cmin'] = -ampdiff; map2dpolarn.params['cmax'] = ampdiff
            map2dpolarn.pcolor(xlon, ylat, normerrorgrid1-normerrorgrid2, show=False, output=DIR_PLOT_EXP+fignamediffN)
            map2dpolars.params['cmin'] = -ampdiff; map2dpolars.params['cmax'] = ampdiff
            map2dpolars.pcolor(xlon, ylat, normerrorgrid1-normerrorgrid2, show=False, output=DIR_PLOT_EXP+fignamediffS)
        else:
            map2d.pcolor(xlon, ylat, normerrorgrid2, show=False, output=DIR_PLOT_EXP+figname2)
            # diff
            map2d.params['colormap'] = 'MO_greyed70_middle_rev'
            map2d.params['cmin'] = -ampdiff; map2d.params['cmax'] = ampdiff
            map2d.pcolor(xlon, ylat, normerrorgrid1-normerrorgrid2, show=False, output=DIR_PLOT_EXP+fignamediff)

