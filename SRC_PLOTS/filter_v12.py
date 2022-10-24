#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:04:08 2019

@author: tbrivoal
"""
import os
import xarray as xr
import numpy as np
import scipy.stats as st

import netCDF4 as nc
from scipy.signal import convolve as scipy_convolve
from scipy.signal import gaussian as scipy_gaussian

from astropy.convolution import convolve         as astro_convolve
from astropy.convolution import Gaussian2DKernel as astro_gaussian
import multiprocessing
import time
import glob
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "1"


def scipy_kernel( kernlen, std):

    """Returns a 2D Gaussian kernel array."""

    gkern1d = scipy_gaussian( kernlen, std=std ).reshape( kernlen, 1)
    gkern2d = np.outer( gkern1d, gkern1d)
    print(gkern2d.shape)

    return gkern2d

def filter_gaussian(input_values,kernel2D):
    values=input_values

    input_filtered = astro_convolve(values, kernel2D, boundary='fill', fill_value=np.nan)

#    from scipy import signal # other way
#    input_filtered = signal.convolve2d(values, kernel,  boundary='symm', mode='same')
    # The filtered data is extrapolated over land from astropy, we have to remask
    mask=np.isnan(input_values)
    output=np.ma.array(input_filtered,mask=mask)
    return output  #Filtered masked array
def filter_file(filepath):
    nsigma  = 12
    kernlen = 6 * nsigma + 1
    kernel2D  = astro_gaussian( x_stddev=nsigma, y_stddev=nsigma, theta=0., x_size=kernlen, y_size=kernlen)
    ind_file=1
    print(filepath)
    my_file = Path(filepath +  '.filt12')
    if my_file.is_file():
        print(filepath +  '.filt12' +' already exists, next')
    else:
        ds= xr.open_dataset(filepath).isel(depthv=1).squeeze()
        var_V = ds.vomecrty.squeeze().values #alues
        var_V_LS = np.zeros(var_V.shape)
        var_V_LS = filter_gaussian(var_V,kernel2D)
        var_V_filt=var_V-var_V_LS
        ds_tmp=ds.vomecrty
        print(str(filepath) + '.filt12')
        var_V_filt = xr.DataArray( var_V_filt    , coords=ds_tmp.coords, dims=ds_tmp.dims, attrs=ds_tmp.attrs, name='vomecrty_filt12')
        var_V_filt = var_V_filt.expand_dims(dim='time_counter')
        var_V_filt.to_netcdf(str(filepath) + '.filt12')


print("LETS GO FOR FILTERING!!!!")
filein = sorted(glob.glob(os.path.join('FLDR/', 'CFG_NAME_1d_gridV25h_????????-????????.nc')))
pool = multiprocessing.Pool(processes=12)
# a is a list of matrix
print(filein)
a=pool.map(filter_file,filein)
pool.close()


