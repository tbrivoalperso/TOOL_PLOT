#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:10:00 2020

@author: mhamon

Contains:

"""


import os
import numpy as np

def get_args():
    """
    Get the input arguments

    """

    import argparse

    parser = argparse.ArgumentParser(
                   description='''Launch combine_ola''')
    # Obligatoire
    parser.add_argument('-ola1', type=str,
                        help='ola file whose grid information will be modified.')
    parser.add_argument('-ola2', type=str,
                        help='ola file in which containing grid information.')
    parser.add_argument('-suffix_out', type=str,
                        help='suffix for output files.')

    args = parser.parse_args()
    ola1n = args.ola1
    ola2n = args.ola2
    suff = args.suffix_out

    return ola1n, ola2n, suff


def main():

    import re

    import numpy as np
    import xarray as xr

    from noobs.utils.compare_list import FFcompare_list

    ola1n, ola2n, suff = get_args()

    ola1n_split = ola1n.split('.')
    ola1n_out = ".".join(ola1n_split[:-1])+"_"+suff+"."+ola1n_split[1]
    ola2n_split = ola2n.split('.')
    ola2n_out = ".".join(ola2n_split[:-1])+"_"+suff+"."+ola2n_split[1]
    ola1 = xr.open_dataset(ola1n)
    ola2 = xr.open_dataset(ola2n)

    dimdup1 = [dd for dd in ola1.dims if re.search('DUP', dd, re.I)]
    dimdup2 = [dd for dd in ola2.dims if re.search('DUP', dd, re.I)]

    if dimdup1 == dimdup2:
        dimdup = dimdup1[0]
    else:
        raise AttributeError("ola1 and ola2 have different dimension names.")

    #list1 = np.concatenate((ola1.LATITUDE.data[:, np.newaxis],
    #                        ola1.LONGITUDE.data[:, np.newaxis],
    #                        ola1.JULTIME.data[:, np.newaxis]), axis=1)

    #list2 = np.concatenate((ola2.LATITUDE.data[:, np.newaxis],
    #                        ola2.LONGITUDE.data[:, np.newaxis],
    #                        ola2.JULTIME.data[:, np.newaxis]), axis=1)

    #arg1ok, arg2ok = FFcompare_list(list1, list2)

    arg1ok, arg2ok = find_intersection(ola1, ola2)

    ola1 = ola1.isel({dimdup:arg1ok})
    ola2 = ola2.isel({dimdup:arg2ok})

    ola1['vertices'][:] = ola2['vertices']
    ola1['lambda'][:] = ola2['lambda']
    ola1['zone'][:] = ola2['zone']
    
    ola1.to_netcdf(ola1n_out)
    ola2.to_netcdf(ola2n_out)


def find_intersection(obj1, obj2, output_diff=False):
    """ find common point (x,y,t) between, two ola data sets """
    c_f = zip(obj1.LONGITUDE.values, obj1.LATITUDE.values, obj1.JULTIME.values)
    set_c_f = set(c_f)

    c_a = zip(obj2.LONGITUDE.values, obj2.LATITUDE.values, obj2.JULTIME.values)
    set_c_a = set(c_a)

    #Find intersection between two sets
    u_set = set_c_a ^ set_c_f

    # create zipped list with data position
    c_f = zip(obj1.LONGITUDE.values, obj1.LATITUDE.values, obj1.JULTIME.values, np.arange(obj1.JULTIME.values.size))
    ind1 = [pos[3] for pos in c_f if not pos[0:3] in u_set]
    if output_diff:
        c_f = zip(obj1.LONGITUDE.values, obj1.LATITUDE.values, obj1.JULTIME.values, np.arange(obj1.JULTIME.values.size))
        ind1notin2 = [pos[3] for pos in c_f if pos[0:3] in u_set]

    c_a = zip(obj2.LONGITUDE.values, obj2.LATITUDE.values, obj2.JULTIME.values,np.arange(obj2.JULTIME.values.size))
    ind2 = [pos[3] for pos in c_a if not pos[0:3] in u_set]
    if output_diff:
        c_f = zip(obj1.LONGITUDE.values, obj1.LATITUDE.values, obj1.JULTIME.values, np.arange(obj1.JULTIME.values.size))
        ind2notin1 = [pos[3] for pos in c_a if pos[0:3] in u_set]

    if not output_diff:
        return ind1, ind2
    else:
        return ind1, ind2, ind1notin2, ind2notin1


if __name__ == '__main__':

    import time

    T0 = time.time()
    main()
    print(time.time()-T0)
