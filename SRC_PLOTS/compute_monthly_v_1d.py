#!/usr/bin/env python3

#SBATCH -J PLOTS
#SBATCH -N 1
#SBATCH -p normal256
#SBATCH --no-requeue
#SBATCH --time=10:30:00

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:04:08 2019

@author: tbrivoal
"""
import os
import xarray as xr
import numpy as np
import pandas
import datetime
from datetime import date, timedelta

import netCDF4 as nc

import multiprocessing
import time
import glob
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "1"




str_start_date = 'DSTART'
str_end_date = 'DEND'
format = '%Y-%m-%d'
sdate = datetime.datetime.strptime(str_start_date, format)   # start date
edate = datetime.datetime.strptime(str_end_date, format)

list_dates=pandas.date_range(sdate,edate-timedelta(days=1),freq='d')

print(list_dates)
window=31
def rolling31j_KE(idate):
    ds = xr.open_dataset('FLDR/CFG_NM_1d_gridV25h.nc')
    format2='%Y%m%d'
    sdate2 = (idate - timedelta(days=np.round(window/2) + 15)).strftime(format2)
    edate2 = (idate + timedelta(days=np.round(window/2) + 15)).strftime(format2) 
    print(sdate2, edate2)
    U = ds.vomecrty.sel(time_counter=slice(sdate2, edate2))
    print('U loaded')
    U_roll = U.rolling(time_counter=31, center=True).mean()
#
    print('U_roll computed')
    idate_str=idate.strftime("%Y-%m-%d")
    U_roll_day = U_roll.sel(time_counter=idate_str).compute()
    print('SHP U_roll_day',U_roll_day.shape)
    print(U_roll_day)
    U_submon = (U.sel(time_counter=idate_str) - U_roll_day).compute()
    U_roll_day = U_roll_day.expand_dims(dim='time_counter')
    
    #U_roll_day = U_roll_day.assign_coords({"time_counter": ds.time_counter.sel(time_counter=idate_str)})
    U_submon = U_submon.expand_dims(dim='time_counter')
    #U_submon = U_submon.assign_coords({"time_counter": ds.time_counter.sel(time_counter=idate_str)})

    print('ok')
    U_roll_day.to_netcdf('FLDR/eNEATL36_1d_gridV25h_rolling31j_'+idate_str+'.nc')
    U_submon.to_netcdf('FLDR/eNEATL36_1d_gridV25h_submonthly_'+idate_str+'.nc')

    print('write FLDR/eNEATL36_1d_gridV25h_rolling31j_'+idate_str+'.nc    OK')
    print('write FLDR/eNEATL36_1d_gridV25h_submonthly_'+idate_str+'.nc    OK')

#
pool = multiprocessing.Pool(processes=32)
# a is a list of matrix
a=pool.map(rolling31j_KE,list_dates)



