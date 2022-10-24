#!/usr/bin/env python3

import xarray as xr
import numpy as np
import pandas
import datetime
from datetime import date, timedelta
import multiprocessing


str_start_date = 'DSTART'
str_end_date = 'DEND'
format = '%Y-%m-%d'
sdate = datetime.datetime.strptime(str_start_date, format)   # start date
edate = datetime.datetime.strptime(str_end_date, format) 

list_dates=pandas.date_range(sdate,edate-timedelta(days=1),freq='d')
print(list_dates)
def submonthly_KE(idate):
    idate_str=idate.strftime("%Y-%m-%d")
    print(idate_str)
    ds_V = xr.open_dataset('FLDR/eNEATL36_1d_gridV25h_submonthly.nc').sel(time_counter=idate_str).drop_dims('depthv')
    ds_U = xr.open_dataset('FLDR/eNEATL36_1d_gridU25h_submonthly.nc').sel(time_counter=idate_str).drop_dims('depthu')
    #idate_da = ds_U.time_counter.isel(time_counter=11)
    #print(idate_da)
    U = ds_U.sozocrtx.rename('KE')
    V = ds_V.somecrty.rename('KE')
    KE = 0.5 * (U**2 + V**2) #.mean(dim="time_counter")
    KE.to_netcdf('FLDR/eNEATL36_1d_KE_submonthly'+idate_str+'.nc')
    print('write FLDR/eNEATL36_1d_KE_submonthly'+idate_str+'.nc    OK')

pool = multiprocessing.Pool(processes=32)
# a is a list of matrix
a=pool.map(submonthly_KE,list_dates)


#def submonthly(list_dates):
#ds_V = xr.open_dataset('FLDR/eNEATL36_1d_gridV25h_submonthly.nc')
#ds_U = xr.open_dataset('FLDR/eNEATL36_1d_gridV25h_submonthly.nc')

