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
import pandas as pd
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

list_dates=pd.date_range(sdate,edate-timedelta(days=1),freq='d')

print(list_dates)
window=744
def rolling31j_KE(idate):
    ds = xr.open_dataset('FLDR/CFG_NM_1h_gridU.nc') #, chunks={'x': 100, 'y':100})
    format2='%Y%m%d %H:%M:%S'
    sdate2 = (idate - timedelta(hours=np.round(window/2))).strftime(format2)
    edate2 = (idate + timedelta(hours=np.round(window/2))).strftime(format2) 
    #print(sdate2, edate2)
    U = ds.sozocrtx.sel(time_counter=slice(sdate2, edate2)) #.load()
    idate_str=idate.strftime("%Y-%m-%d")
    format_time_counter = '%Y-%m-%dT%H:%M:%S'
    U_day = U.sel(time_counter=idate_str)
    U_roll_day = U_day.copy()
    for i, times in enumerate(U_day.time_counter):
        print(i)
        times_fmt = str(times.values)[0:19]
        times_dt = datetime.datetime.strptime(times_fmt,format_time_counter)
        sdate3 = (times_dt - timedelta(hours=np.round(window/2))).strftime(format2)
        edate3 = (times_dt + timedelta(hours=np.round(window/2))).strftime(format2)
        print(times_dt,sdate3,edate3)   
        U_mean = ds.sozocrtx.sel(time_counter=slice(sdate3, edate3)).mean(dim='time_counter')
        U_roll_day.data[i,:,:] = U_mean.values


    U_roll_day.to_netcdf('FLDR/test2_eNEATL36_1h_gridU_rolling31j_'+idate_str+'.nc')
    print('write FLDR/test2_eNEATL36_1h_gridU_rolling31j_'+idate_str+'.nc    OK')
#
pool = multiprocessing.Pool(processes=128)
# a is a list of matrix
a=pool.map(rolling31j_KE,list_dates)



