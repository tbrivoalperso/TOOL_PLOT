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
def merge_file(filepath):
        ds= xr.open_dataset(filepath).isel(depthu=1).squeeze()
        var_U = ds.vozocrtx.values #alues
         

print("LETS GO FOR FILTERING!!!!")
filein = sorted(glob.glob(os.path.join('FLDR/', 'CFG_NAME_1d_gridU25h_????????-????????.nc')))
pool = multiprocessing.Pool(processes=64)
# a is a list of matrix
print(filein)
a=pool.map(merge_file,filein)
pool.close()


