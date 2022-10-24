#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:41:00 2020

@author: mhamon
"""

import os
import configparser
from distutils.util import strtobool
import pandas as pd

nmlist = 'namelist_plotola.nml'
config = configparser.ConfigParser()
config.read(nmlist)


## READ NAMELIST
#[GENERAL]
SYSid = config['GENERAL']['SYSid']
CYCLE = config['GENERAL']['CYCLE'].split(',')

if len(CYCLE) == 2:
    CYCNAM = CYCLE[0]+'-'+CYCLE[1]
    CYClist = []
    cc = CYCLE[0]
    while int(cc) <= int(CYCLE[1]):
        CYClist.append(cc)
        cc = (pd.to_datetime(cc)+pd.Timedelta("7 days")).strftime('%Y%m%d')
    CYCLE = CYClist
else:
    CYCNAM = CYCLE[0]

do_IS = strtobool(config['GENERAL']['do_IS'])
setid_IS_SLA = config['GENERAL']['setid_IS'].split(',')
do_DS_SST = strtobool(config['GENERAL']['do_DS_SST'])
setid_DS_SST = config['GENERAL']['setid_DS_SST'].split(',')
do_DS_SSS = strtobool(config['GENERAL']['do_DS_SSS'])
setid_DS_SSS = config['GENERAL']['setid_DS_SSS'].split(',')
do_DS_SIC = strtobool(config['GENERAL']['do_DS_SIC'])
setid_DS_SIC = config['GENERAL']['setid_DS_SIC'].split(',')
do_VPT = strtobool(config['GENERAL']['do_VPT'])
setid_VP_T = config['GENERAL']['setid_VPT'].split(',')
do_VPS = strtobool(config['GENERAL']['do_VPS'])
setid_VP_S = config['GENERAL']['setid_VPS'].split(',')
level = int(config['GENERAL']['level'])
AUTOSHIFT_SLA= strtobool(config['GENERAL']['AUTOSHIFT_SLA']
#[EXP1]
RUNid1 = config['EXP1']['RUNid']
type_ccl1 = config['EXP1']['type_ccl']
DIR_OLA_OFF1 = config['EXP1']['DIR_OLA_OFF']
samout1 = strtobool(config['EXP1']['samout'])

#[EXP2]
RUNid2 = config['EXP2']['RUNid']
type_ccl2 = config['EXP2']['type_ccl']
DIR_OLA_OFF2 = config['EXP2']['DIR_OLA_OFF']
try:
    samout2 = strtobool(config['EXP2']['samout'])
except ValueError:
    samout2 = ''

#[PLOT]
DIR_PLOT = config['PLOT']['DIR_PLOT']
PARAM_PLOT = config['PLOT']['PARAM_PLOT']
PARAM_PLOT_POLARNorth = config['PLOT']['PARAM_PLOT_POLARNorth']
PARAM_PLOT_POLARSouth = config['PLOT']['PARAM_PLOT_POLARSouth']
res_IS_SLA = float(config['PLOT']['res_IS'])
res_DS_SST = float(config['PLOT']['res_DS_SST'])
res_DS_SSS = float(config['PLOT']['res_DS_SSS'])
res_DS_SIC = float(config['PLOT']['res_DS_SIC'])
res_VP_T = float(config['PLOT']['res_VPT'])
res_VP_S = float(config['PLOT']['res_VPS'])
#

TYPE2DO = []
if do_IS:
    TYPE2DO.append('IS_SLA')
if do_DS_SST:
    TYPE2DO.append('DS_SST')
if do_DS_SSS:
    TYPE2DO.append('DS_SSS')
if do_DS_SIC:
    TYPE2DO.append('DS_SIC')
if do_VPT:
    TYPE2DO.append('VP_T')
if do_VPS:
    TYPE2DO.append('VP_S')

do_comp = len(DIR_OLA_OFF2) > 0
if do_comp:
    RUNid_DIR = RUNid2+'-'+RUNid1
else:
    RUNid_DIR = RUNid1

RES = {}
RES['res_IS_SLA'] = res_IS_SLA
RES['res_DS_SST'] = res_DS_SST
RES['res_DS_SSS'] = res_DS_SSS
RES['res_DS_SIC'] = res_DS_SIC
RES['res_VP_T'] = res_VP_T
RES['res_VP_S'] = res_VP_S

SET_CORR_SAM = {}
SET_CORR_SAM['al'] = 'al.list' 
SET_CORR_SAM['c2'] = 'c2.list'
SET_CORR_SAM['en'] = 'en.list'
SET_CORR_SAM['h2'] = 'h2.list'
SET_CORR_SAM['j1g'] = 'j1g.list'
SET_CORR_SAM['j2'] = 'j2.list'
SET_CORR_SAM['j3'] = 'j3.list'
SET_CORR_SAM['alg'] = 'alg.list'
SET_CORR_SAM['HR_ASSIM'] = 'DS_GEN_SST_HR.list'
SET_CORR_SAM['OSTIA_ASSIM'] = 'DS_GEN_SST_OSTIA.list'
SET_CORR_SAM['ESACCI_SSS_ASSIM'] = 'ESACCI_SSS_ASSIM.list'
SET_CORR_SAM['SIC_OSISAF_Antarctic_ASSIM'] = 'DS_GEN_SIC_CERSAT_Antarctic'
SET_CORR_SAM['SIC_OSISAF_Arctic_ASSIM'] = 'DS_GEN_SIC_CERSAT_Arctic.list'
SET_CORR_SAM['INSITU_ARMOR'] = 'VP_GEN_INSITU_ARMOR.list'

SET1 = {}
if not samout1:
    SET1['setid_IS_SLA'] = setid_IS_SLA
    SET1['setid_DS_SST'] = setid_DS_SST
    SET1['setid_DS_SSS'] = setid_DS_SSS
    SET1['setid_DS_SIC'] = setid_DS_SIC
    SET1['setid_VP_T'] = setid_VP_T
    SET1['setid_VP_S'] = setid_VP_S
else:
    try:
        SET1['setid_IS_SLA'] = [SET_CORR_SAM[ii] for ii in setid_IS_SLA]
        SET1['setid_DS_SST'] = [SET_CORR_SAM[ii] for ii in setid_DS_SST]
        SET1['setid_DS_SSS'] = [SET_CORR_SAM[ii] for ii in setid_DS_SSS]
        SET1['setid_DS_SIC'] = [SET_CORR_SAM[ii] for ii in setid_DS_SIC]
        SET1['setid_VP_T'] = [SET_CORR_SAM[ii] for ii in setid_VP_T]
        SET1['setid_VP_S'] = [SET_CORR_SAM[ii] for ii in setid_VP_S]
    except KeyError:
        raise Exception('Wrong setid for SET1. Please add key in SET_CORR_SAM')

SET2 = {}
if not samout2:
    SET2['setid_IS_SLA'] = setid_IS_SLA
    SET2['setid_DS_SST'] = setid_DS_SST
    SET2['setid_DS_SSS'] = setid_DS_SSS
    SET2['setid_DS_SIC'] = setid_DS_SIC
    SET2['setid_VP_T'] = setid_VP_T
    SET2['setid_VP_S'] = setid_VP_S
else:
    try:
        SET2['setid_IS_SLA'] = [SET_CORR_SAM[ii] for ii in setid_IS_SLA]
        SET2['setid_DS_SST'] = [SET_CORR_SAM[ii] for ii in setid_DS_SST]
        SET2['setid_DS_SSS'] = [SET_CORR_SAM[ii] for ii in setid_DS_SSS]
        SET2['setid_DS_SIC'] = [SET_CORR_SAM[ii] for ii in setid_DS_SIC]
        SET2['setid_VP_T'] = [SET_CORR_SAM[ii] for ii in setid_VP_T]
        SET2['setid_VP_S'] = [SET_CORR_SAM[ii] for ii in setid_VP_S]
    except KeyError:
        raise Exception('Wrong setid for SET2. Please add key in SET_CORR_SAM')


PREC = {}
PREC['prec_IS_SLA'] = 3
PREC['prec_DS_SST'] = 3
PREC['prec_DS_SSS'] = 3
PREC['prec_DS_SIC'] = 1
PREC['prec_VP_T'] = 3
PREC['prec_VP_S'] = 3


DIR_PLOT_EXP = DIR_PLOT+SYSid+'/'+RUNid_DIR+'/R'+CYCNAM+'/'
if not os.path.exists(DIR_PLOT_EXP):
    os.makedirs(DIR_PLOT_EXP)
