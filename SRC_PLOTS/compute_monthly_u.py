#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:04:08 2019

@author: tbrivoal
"""
import os
import xarray as xr
import numpy as np

import netCDF4 as nc

import multiprocessing
import time
import glob
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "1"



ds = xr.open_dataset('FLDR/CFG_NM_1h_gridU.nc', chunks={'time_counter': 50})

U = ds.sozocrtx.sel(time_counter=slice('DSTART', 'DEND'))
print(U.shape)
U.load()
print(U.shape)
U_roll = U.rolling(time_counter=744, center=True).mean()


print("let's go")
U_roll.to_netcdf('FLDR/CFG_NM_1h_gridU_rolling31j.nc')

