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
import multiprocessing
import time
import glob
from pathlib import Path
from netCDF4 import Dataset


def select_coupe(fname):
#    WGIBR
#    aname='WGIBR' 
#    y_min,y_max = 300 , 467
#    x_coupe = 430

#GIBRALTAR
#    aname='gibraltar'
#    y_min,y_max = 375 , 475
#    x_coupe = 516
#    varin='vozocrtx'

    aname='gibraltar2'
    y_min,y_max = 375 , 475
    x_coupe = 520
    varin='vozocrtx'

    dataset = xr.open_dataset(fname).isel(y=np.arange(y_min,y_max)).isel(x=x_coupe)
    #dataset = xr.open_dataset(fname).isel(y=np.arange(y_min,y_max)).isel(x=np.arange(x_coupe-20, x_coupe+20))

    dataset.to_netcdf(fname + '.' + aname)


filein = sorted(glob.glob(os.path.join('FLDR/', 'CFG_NAME_1d_gridU25h_????????-????????.nc')))

pool = multiprocessing.Pool(processes=12)
# a is a list of matrix
a=pool.map(select_coupe,filein)
pool.close()





