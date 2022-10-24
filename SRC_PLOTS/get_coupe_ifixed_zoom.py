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

    imin_IBI_AGRIF=IMIN
    imax_IBI_AGRIF=IMAX
    jmin_IBI_AGRIF=JMIN
    jmax_IBI_AGRIF=JMAX

    y_min_agrif=(y_min - jmin_IBI_AGRIF)*3 + 5
    y_max_agrif=(y_max - jmin_IBI_AGRIF)*3 + 5
    x_coupe_agrif=(x_coupe - imin_IBI_AGRIF)*3 + 5
    varin='vozocrtx'
    dataset = xr.open_dataset(fname).isel(y=np.arange(y_min_agrif,y_max_agrif)).isel(x=x_coupe_agrif)
#   dataset = xr.open_dataset(fname).isel(y=np.arange(y_min_agrif,y_max_agrif)).isel(x=np.arange(x_coupe_agrif-3*20, x_coupe_agrif+3*20))
    dataset.to_netcdf(fname + '.' + aname)


filein = sorted(glob.glob(os.path.join('FLDR/', 'AGRIF_CFG_NAME_1d_gridU25h_????????-????????.nc')))

pool = multiprocessing.Pool(processes=12)
# a is a list of matrix
a=pool.map(select_coupe,filein)
pool.close()





