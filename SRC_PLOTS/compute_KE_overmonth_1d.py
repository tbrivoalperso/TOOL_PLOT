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
def rolling31j_KE(idate):
    idate_str=idate.strftime("%Y-%m-%d")
    print(idate_str)
    ds_V = xr.open_dataset('FLDR/eNEATL36_1h_gridV_rolling31j.nc').sel(time_counter=idate_str)
    ds_U = xr.open_dataset('FLDR/eNEATL36_1h_gridU_rolling31j.nc').sel(time_counter=idate_str)
    #idate_da = ds_U.time_counter.isel(time_counter=11)
    #print(idate_da)
    U = ds_U.sozocrtx.rename('KE')
    V = ds_V.somecrty.rename('KE')
    KE = 0.5 * (U**2 + V**2).mean(dim="time_counter")
    KE['time_counter'] = idate
    KE = KE.expand_dims(dim='time_counter')
    #KE['time_counter'] = idate
    KE.to_netcdf('FLDR/eNEATL36_1h_KE_rolling31j'+idate_str+'.nc')
    print('write FLDR/eNEATL36_1h_KE_rolling31j'+idate_str+'.nc    OK')

pool = multiprocessing.Pool(processes=32)
# a is a list of matrix
a=pool.map(rolling31j_KE,list_dates)


#def rolling31j(list_dates):
#ds_V = xr.open_dataset('FLDR/eNEATL36_1h_gridV_rolling31j.nc')
#ds_U = xr.open_dataset('FLDR/eNEATL36_1h_gridV_rolling31j.nc')

