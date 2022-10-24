#!/usr/bin/env python3

import xarray as xr
import numpy as np
import pandas
import datetime
from datetime import date, timedelta
import multiprocessing
from astropy.convolution import convolve         as astro_convolve
from astropy.convolution import Gaussian2DKernel as astro_gaussian
from pathlib import Path



def filter_gaussian(input_values,kernel2D):
    values=input_values

    input_filtered = astro_convolve(values, kernel2D, boundary='fill', fill_value=np.nan)

#    from scipy import signal # other way
#    input_filtered = signal.convolve2d(values, kernel,  boundary='symm', mode='same')
    # The filtered data is extrapolated over land from astropy, we have to remask
    #mask=np.isnan(input_values)
    #output=np.ma.array(input_filtered,mask=mask)
    output=input_filtered
    return output  #Filtered masked array

str_start_date = 'DSTART'
str_end_date = 'DEND'
format = '%Y-%m-%d'
sdate = datetime.datetime.strptime(str_start_date, format)   # start date
edate = datetime.datetime.strptime(str_end_date, format) 

list_dates=pandas.date_range(sdate,edate-timedelta(days=1),freq='d')
print(list_dates)
def filter_hourly_KE(idate):
    nsigma  = 20
    kernlen = 6 * nsigma + 1
    kernel2D_20  = astro_gaussian( x_stddev=nsigma, y_stddev=nsigma, theta=0., x_size=kernlen, y_size=kernlen)
    
    nsigma  = 8
    kernlen = 6 * nsigma + 1
    kernel2D_8  = astro_gaussian( x_stddev=nsigma, y_stddev=nsigma, theta=0., x_size=kernlen, y_size=kernlen)

    nsigma  = 36
    kernlen = 6 * nsigma + 1
    kernel2D_36  = astro_gaussian( x_stddev=nsigma, y_stddev=nsigma, theta=0., x_size=kernlen, y_size=kernlen)


    idate_str=idate.strftime("%Y-%m-%d")
    my_file = Path('FLDR/eNEATL36_1h_KE_below_8_'+idate_str+'_daymean.nc')
    if my_file.is_file():
        print('FLDR/eNEATL36_1h_KE_below_8_'+idate_str+'_daymean.nc already exists, next')
    else:
        print(idate_str)
        ds_V = xr.open_dataset('FLDR/eNEATL36_1h_gridV.nc').sel(time_counter=idate_str)
        ds_U = xr.open_dataset('FLDR/eNEATL36_1h_gridU.nc').sel(time_counter=idate_str)
        #idate_da = ds_U.time_counter.isel(time_counter=11)
        #print(idate_da)
        U = ds_U.sozocrtx.rename('KE')
        V = ds_V.somecrty.rename('KE')
        U_8 = np.zeros(U.shape)
        V_8 = np.zeros(V.shape)
        U_20 = np.zeros(U.shape)
        V_20 = np.zeros(V.shape)
        U_above_36 = np.zeros(U.shape)
        V_above_36 = np.zeros(V.shape)

        U_tofilt = U.values
        V_tofilt = V.values
        for i in range(len(U[:,0,0])):
            U_8[i,:,:] = filter_gaussian(U_tofilt[i,:,:],kernel2D_8)
            V_8[i,:,:] = filter_gaussian(V_tofilt[i,:,:],kernel2D_8)

            U_20[i,:,:] = filter_gaussian(U_tofilt[i,:,:],kernel2D_20)
            V_20[i,:,:] = filter_gaussian(V_tofilt[i,:,:],kernel2D_20)

            U_above_36[i,:,:] = filter_gaussian(U_tofilt[i,:,:],kernel2D_36)
            V_above_36[i,:,:] = filter_gaussian(V_tofilt[i,:,:],kernel2D_36)

        U_20_36 =  U_20 - U_above_36
        V_20_36 =  V_20 - V_above_36
        U_8_20 =  U_8 - U_20
        V_8_20 =  V_8 - V_20
        U_below_8 = U - U_8
        V_below_8 = V - V_8
 
        U_above_36 = xr.DataArray( U_above_36    , coords=U.coords, dims=U.dims, attrs=U.attrs, name='KE_above_36')
        V_above_36 = xr.DataArray( V_above_36    , coords=V.coords, dims=V.dims, attrs=V.attrs, name='KE_above_36')
        U_20_36 = xr.DataArray( U_20_36    , coords=U.coords, dims=U.dims, attrs=U.attrs, name='KE_20_36')
        V_20_36 = xr.DataArray( V_20_36    , coords=V.coords, dims=V.dims, attrs=V.attrs, name='KE_20_36')
        U_8_20 = xr.DataArray( U_8_20    , coords=U.coords, dims=U.dims, attrs=U.attrs, name='KE_8_20')
        V_8_20 = xr.DataArray( V_8_20    , coords=V.coords, dims=V.dims, attrs=V.attrs, name='KE_8_20')
        U_below_8 = xr.DataArray( U_below_8    , coords=U.coords, dims=U.dims, attrs=U.attrs, name='KE_below_8')
        V_below_8 = xr.DataArray( V_below_8    , coords=V.coords, dims=V.dims, attrs=V.attrs, name='KE_below_8')

        

        KE_above_36 = 0.5 * (U_above_36**2 + V_above_36**2).mean(dim="time_counter")
        KE_above_36['time_counter'] = idate
        KE_above_36 = KE_above_36.expand_dims(dim='time_counter')

        KE_20_36 = 0.5 * (U_20_36**2 + V_20_36**2).mean(dim="time_counter")
        KE_20_36['time_counter'] = idate
        KE_20_36 = KE_20_36.expand_dims(dim='time_counter')

        KE_8_20 = 0.5 * (U_8_20**2 + V_8_20**2).mean(dim="time_counter")
        KE_8_20['time_counter'] = idate
        KE_8_20 = KE_8_20.expand_dims(dim='time_counter')

        KE_below_8 = 0.5 * (U_below_8**2 + V_below_8**2).mean(dim="time_counter")
        KE_below_8['time_counter'] = idate
        KE_below_8 = KE_below_8.expand_dims(dim='time_counter')


        KE_above_36.to_netcdf('FLDR/eNEATL36_1h_KE_above_36_'+idate_str+'_daymean.nc')
        print('write FLDR/eNEATL36_1h_KE_above_36_'+idate_str+'_daymean.nc   OK')

        KE_20_36.to_netcdf('FLDR/eNEATL36_1h_KE_20_36_'+idate_str+'_daymean.nc')
        print('write FLDR/eNEATL36_1h_KE_20_36_'+idate_str+'_daymean.nc   OK')

        KE_8_20.to_netcdf('FLDR/eNEATL36_1h_KE_8_20_'+idate_str+'_daymean.nc')
        print('write FLDR/eNEATL36_1h_KE_8_20_'+idate_str+'_daymean.nc   OK')

        KE_below_8.to_netcdf('FLDR/eNEATL36_1h_KE_below_8_'+idate_str+'_daymean.nc')
        print('write FLDR/eNEATL36_1h_KE_below_8_'+idate_str+'_daymean.nc   OK')




pool = multiprocessing.Pool(processes=64)
# a is a list of matrix
a=pool.map(filter_hourly_KE,list_dates)


#def subdaily(list_dates):
#ds_V = xr.open_dataset('FLDR/eNEATL36_1h_gridV_subdaily.nc')
#ds_U = xr.open_dataset('FLDR/eNEATL36_1h_gridV_subdaily.nc')

