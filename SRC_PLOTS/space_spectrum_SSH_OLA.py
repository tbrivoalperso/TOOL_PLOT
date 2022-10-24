#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:04:08 2019

@author: tbrivoal
"""
import os, sys, re
import matplotlib as mpl
mpl.use('Agg')
import cartopy.crs as ccrs
from lib_pyt.Griddata_custom import grid_data_NUMBA
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy import signal

import cmocean
import xarray as xr
import numpy as np
from netCDF4 import Dataset
from datetime import date, timedelta

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

######################################################################################################################
def tukeywin(window_length, alpha=0.5):
    '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.
 
    We use the same reference as MATLAB to provide the same results in case users compare a MATLAB output to this function
    output
 
    Reference
    ---------
    http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html
 
    '''
    # Special cases
    if alpha <= 0:
        return np.ones(window_length) #rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)

    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)

    # first condition 0 <= x < alpha/2
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))

    # second condition already taken care of

    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2)))
    return w


######################################################################################################################
def ar2haversine(lon1, lat1, lon2, lat2):
    from math import radians, cos, sin, asin, sqrt
    import numpy as np
    """
    Calculate the great circle distance between a point and an array field
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    l1=len(lon1)
    l2=len(lon2)
    lon1rad=np.zeros(l1)
    lat1rad=np.zeros(l1)
    lon2rad=np.zeros(l2)
    lat2rad=np.zeros(l2)
    for ll in np.arange(l1):
       llon1, llat1 = list(map(radians, [lon1[ll],lat1[ll]]))
       llon2, llat2 = list(map(radians, [lon2[ll],lat2[ll]]))
       lon1rad[ll]=llon1
       lat1rad[ll]=llat1
       lon2rad[ll]=llon2
       lat2rad[ll]=llat2
    # haversine formula
    dlon = lon2rad - lon1rad
    dlat = lat2rad - lat1rad
    a = np.sin(dlat/2)**2 + np.cos(lat1rad) * np.cos(lat2rad) * np.sin(dlon/2)**2
    c=np.zeros(l2)
    for ll in np.arange(l2):
       c[ll] = 2 * asin(sqrt(a[ll]))
    km = 6367 * c
    return km,c

def ar2haversine_1(lon1, lat1, lon2, lat2):
    from math import radians, cos, sin, asin, sqrt
    import numpy as np

    lon1rad, lat1rad, lon2rad,lat2rad =np.radians(lon1), np.radians(lat1),np.radians(lon2), np.radians(lat2)
    # haversine formula
    dlon = lon2rad - lon1rad
    dlat = lat2rad - lat1rad
    a = np.sin(dlat/2)**2 + np.cos(lat1rad) * np.cos(lat2rad) * np.sin(dlon/2)**2
    c=2*asin(sqrt(a))
    km = 6367 * c
    return km

######################################################################################################################
def loadfile(DIR_IN,satid,start_date,end_date,lonmin,lonmax,latmin,latmax):
    listOBS_out=np.array([[],[],[]]).T
    listMOD_out=np.array([[],[],[]]).T
    listdistance_out=np.array([])
    listtime_out=np.array([])

    delta = timedelta(days=7)
    while start_date <= end_date:
        tdate=start_date.strftime("%Y%m%d")
        print(tdate)
        start_date += delta
        for olafile in os.listdir(DIR_IN):
            if re.search('OLA_IS_SLA_R'+str(tdate)+'.nc',olafile) :
                existfile=False
                olafile=olafile[:]
                print("olafile: "+olafile)
                try :
                   nc=Dataset(DIR_IN+olafile,'r')
                except:
                   print('file %s is missing' %(DIR_IN+olafile))
                   sys.exit()
                existfile=True
                SETID=nc.variables['SETID'][:].astype('U13')
                LONGITUDE=np.array(nc.variables['LONGITUDE'][:])
                LATITUDE=np.array(nc.variables['LATITUDE'][:])
                JULTIME=nc.variables['JULTIME'][:]
                OBSERVATION=nc.variables['OBSERVATION'][:]
                EQUIVALENT_MODELE=nc.variables['EQUIVALENT_MODELE'][:]
                indexok={}
                testsat=(SETID[:,0]==satid[0]) & (SETID[:,1]==satid[1])
                testlat=(LATITUDE>=latmin) & (LATITUDE<=latmax)
                testlon=(LONGITUDE>=lonmin) & (LONGITUDE<=lonmax)
                indexok=np.where((testsat) & (testlat) & (testlon) )
                #
                LONGITUDE_all={}
                LATITUDE_all={}
                JULTIME_all={}
                OBSERVATION_all={}
                EQUIVALENT_MODELE_all={}
                argsort_all={}
                time_sort={}
                for sat in np.arange(len(indexok)):
                   testempty=indexok[0]
                   if testempty.any():
                      ###
                      LONGITUDE_all=LONGITUDE[indexok]
                      LATITUDE_all=LATITUDE[indexok]
                      JULTIME_all=JULTIME[indexok]
                      OBSERVATION_all=OBSERVATION[indexok]
                      EQUIVALENT_MODELE_all=EQUIVALENT_MODELE[indexok]
                      argsort_all=np.argsort(JULTIME_all)
                      time_sort=JULTIME_all[argsort_all]
                      LONGITUDE_all=LONGITUDE_all[argsort_all]
                      LATITUDE_all=LATITUDE_all[argsort_all]
                      OBSERVATION_all=OBSERVATION_all[argsort_all]
                      EQUIVALENT_MODELE_all=EQUIVALENT_MODELE_all[argsort_all]
    #                  LONGITUDE_all=LONGITUDE_all[::2]
    #                  LATITUDE_all=LATITUDE_all[::2]
    #                  OBSERVATION_all=OBSERVATION_all[::2]
    #                  EQUIVALENT_MODELE_all=EQUIVALENT_MODELE_all[::2]
                      LONGITUDE_all=LONGITUDE_all[:]
                      LATITUDE_all=LATITUDE_all[:]
                      OBSERVATION_all=OBSERVATION_all[:]
                      EQUIVALENT_MODELE_all=EQUIVALENT_MODELE_all[:]
    
                      ### MSE
                      lonnow=LONGITUDE_all[:-1]
                      lonafter=LONGITUDE_all[1:]
                      latnow=LATITUDE_all[:-1]
                      latafter=LATITUDE_all[1:]
                      dwbtw,angle=ar2haversine(lonnow,latnow,lonafter,latafter)
                      dwbtw_all=np.append(dwbtw[0:1],dwbtw)
                      dwbtw_all=np.append(dwbtw_all, dwbtw[-1:-2])
                      listOBS=np.column_stack((LONGITUDE_all,LATITUDE_all,OBSERVATION_all))
                      listMOD=np.column_stack((LONGITUDE_all,LATITUDE_all,EQUIVALENT_MODELE_all))
                      listOBS_out=np.concatenate((listOBS_out,listOBS),axis=0)
                      listMOD_out=np.concatenate((listMOD_out,listMOD),axis=0)
                      listdistance_out=np.concatenate((listdistance_out,dwbtw_all),axis=0)
                      listtime_out=np.concatenate((listtime_out,time_sort),axis=0)

    return listOBS_out, listMOD_out, listdistance_out , listtime_out


################################################################################################################################
def select_and_compute_spectrum(dataobs, datamod, distance,satid, dx):
   meandx=dx*2.03
   distmax=1200 # longueur des traces 
   trace={}
   tracemod={}

   nbtrace=0
   distsum=0
   minnbpoint=999999
   ii=iitr=0
   while (iitr<len(distance)) :
      ln_lgth=False
      ln_gap=False
      iitr=ii
      distnow=distsum
      distsum+=distance[iitr]
      while (not ln_lgth) and (not ln_gap) and (iitr<len(distance)):
         distnow=distsum
         distsum+=distance[iitr]
         if distance[iitr]>meandx or distnow>distmax :
            distsum=0
            if distnow>distmax and distance[iitr]<=meandx:
               minnbpoint=min(minnbpoint,iitr-ii)

               trace[nbtrace]=dataobs[ii:iitr,:]
               tracemod[nbtrace]=datamod[ii:iitr,:]
               nbtrace+=1
               ii+=int(200/dx)
               ln_lgth=True
            else:
               ii=iitr+1
               ln_gap=True
         iitr+=1
   # 
   print('Calcul du spectre pour %s' %(satid))
   nbpts=minnbpoint
   nbpts_sp=np.int(np.floor(minnbpoint/2))
   window = tukeywin(nbpts, alpha=0.1)
   freq=np.fft.fftfreq(nbpts,dx)
   freq = freq[0:len(freq)//2]
   nbtr=len(trace)
   sptot=np.zeros((nbpts_sp,nbtr))
   spmodtot=np.zeros((nbpts_sp,nbtr))
   for indextr in np.arange(nbtr-1):
      data=trace[indextr][0:nbpts,2]
      data_dtr=signal.detrend(data)
      data_filt=data_dtr*window
      datamod=tracemod[indextr][0:nbpts,2]
      datamod_dtr=signal.detrend(datamod)
      datamod_filt=datamod_dtr*window
      y = np.fft.fft(data_filt)
      sp =y*np.conj(y) / nbpts
      sp =sp[0:len(sp)//2]
      sp[1:-1]=sp[1:-1] * 2
      sptot[:,indextr]=np.real(sp)

      ymod = np.fft.fft(datamod_filt)
      spmod =ymod*np.conj(ymod) / nbpts
      spmod =spmod[0:len(spmod)//2]
      spmod[1:-1]=spmod[1:-1] * 2
      spmodtot[:,indextr]=np.real(spmod)
   #
   meansp=np.median(sptot,axis=1)
   meanspmod=np.median(spmodtot,axis=1)
   return freq, meansp, meanspmod


################################################################################################################################
def select_and_compute_slope(dataobs, datamod, distance, satid, dx, rangewn):
   meandx=dx*2.03
   distmax=1000 # longueur des traces 
   nbtrace=0
   distsum=0
   minnbpoint=999999
   ii=iitr=0
   tracemap={}
   tracemap_mod={}
   indextrmap=np.zeros((180,90))


   while (iitr<len(distance)) :
      ln_lgth=False
      ln_gap=False
      iitr=ii
      distnow=distsum
      distsum+=distance[iitr]
      while (not ln_lgth) and (not ln_gap) and (iitr<len(distance)):
         distnow=distsum
         distsum+=distance[iitr]
         if distance[iitr]>meandx or distnow>distmax :
            distsum=0
            if distnow>distmax and distance[iitr]<=meandx:
               minnbpoint=min(minnbpoint,iitr-ii)
               if not do_slope_map:
                  trace[nbtrace]=dataobs[ii:iitr,:]
                  tracemod[nbtrace]=datamod[ii:iitr,:]
                  nbtrace+=1
               else:
                  ltr=iitr-ii
                  indbox0=ltr/4
                  indbox1=ltr-ltr/4
                  loncenter_prev=-999
                  latcenter_prev=-999
                  for iibox in np.arange(indbox0,indbox1):
                     #loncenter=int((dataobs[int(ii+iibox),0]+180)/2)
                     #latcenter=int((dataobs[int(ii+iibox),1]+90)/2)
                     loncenter=int(dataobs[int(ii+iibox),0])
                     latcenter=int(dataobs[int(ii+iibox),1])
                     if loncenter!=loncenter_prev or latcenter!=latcenter_prev:
                        if indextrmap[loncenter,latcenter]==0:
                           tracemap[loncenter,latcenter]={}
                           tracemap_mod[loncenter,latcenter]={}

                        tracemap[loncenter,latcenter][indextrmap[loncenter,latcenter]]=dataobs[ii:iitr,:]
                        tracemap_mod[loncenter,latcenter][indextrmap[loncenter,latcenter]]=datamod[ii:iitr,:]
                        indextrmap[loncenter,latcenter]+=1
                     loncenter_prev=loncenter
                     latcenter_prev=latcenter
                  #
                  nbtrace+=1
               #
               ii+=int(200/dx)
               ln_lgth=True
            else:
               ii=iitr+1
               ln_gap=True
         iitr+=1
   # 
   print('Calcul du spectre pour %s' %(satid))
   nbpts=minnbpoint
   nbpts_sp=np.int(np.floor(minnbpoint/2))
   window = tukeywin(nbpts, alpha=0.1)
   freq=np.fft.fftfreq(nbpts,dx)
   freq = freq[0:len(freq)//2]
   sptot={}
   spmodtot={}
   for jj in np.arange(latmin,latmax):
      #print("latitude N %i" %jj)
      for ii in np.arange(lonmin,lonmax):
         try:
            nbtr=len(tracemap[int(ii),int(jj)])
            for indextr in np.arange(nbtr):
               data=tracemap[ii,jj][indextr][:nbpts,2]
               data_dtr=signal.detrend(data)
               data_filt=data_dtr*window
               y = np.fft.fft(data_filt)
               sp =y*np.conj(y) / nbpts
               sp =sp[0:len(sp)//2]
               sp[1:-1]=sp[1:-1] * 2

               datamod=tracemap_mod[ii,jj][indextr][:nbpts,2]
               datamod_dtr=signal.detrend(datamod)
               datamod_filt=datamod_dtr*window
               ymod = np.fft.fft(datamod_filt)
               spmod =ymod*np.conj(ymod) / nbpts
               spmod =spmod[0:len(spmod)//2]
               spmod[1:-1]=spmod[1:-1] * 2

               try:
                  sptot[ii,jj]=np.concatenate((sptot[ii,jj],np.expand_dims(np.real(sp),axis=0)),axis=0)
                  spmodtot[ii,jj]=np.concatenate((spmodtot[ii,jj],np.expand_dims(np.real(spmod),axis=0)),axis=0)
               except:
                  sptot[ii,jj]=np.expand_dims(np.real(sp),axis=0)
                  spmodtot[ii,jj]=np.expand_dims(np.real(spmod),axis=0)
         except:
            nbtr=0
   #

   SLOPE=np.zeros((int(abs(lonmin - lonmax)),int(abs(latmin - latmax))))
   SLOPEmod=np.zeros((int(abs(lonmin - lonmax)),int(abs(latmin - latmax))))
   for jj in np.arange(latmin,latmax):
      for ii in np.arange(lonmin,lonmax):
         try:
            specij=np.array([])
            spmodecij=np.array([])

            for ilat in np.arange(-1,1,1):
               for ilon in np.arange(-2,2,1):
                  try:
                     specij=np.concatenate(specij,sptot[ii+ilon,jj+ilat])
                     spmodecij=np.concatenate(spmodecij,spmodtot[ii+ilon,jj+ilat])

                  except TypeError:
                     specij=sptot[ii+ilon,jj+ilat]
                     spmodecij=spmodtot[ii+ilon,jj+ilat]

                  except KeyError :
                     specij=specij
                     spmodecij=spmodecij
            # Utilisation de la mediane

            meansp=np.median(specij,axis=0)
            spinterp=np.interp(1./rangewn[::-1],freq,np.real(meansp))
            a,b=np.polyfit(np.log10(1./rangewn[::-1]),np.log10(spinterp),1)
            SLOPE[ii-lonmin,jj-latmin]=abs(a)
            meanspmod=np.median(spmodecij,axis=0)
            spmodinterp=np.interp(1./rangewn[::-1],freq,np.real(meanspmod))
            amod,bmod=np.polyfit(np.log10(1./rangewn[::-1]),np.log10(spmodinterp),1)
            SLOPEmod[ii-lonmin,jj-latmin]=abs(amod)
         except KeyError and ValueError :
            SLOPE[ii-lonmin,jj-latmin]=np.nan
            SLOPEmod[ii-lonmin,jj-latmin]=np.nan
    
#   exec('freqsat'+str(satid)+'=freq')
#   exec('SLOPE'+str(satid)+'=SLOPE')
#   exec('SLOPEmod'+str(satid)+'=SLOPEmod')
################################################################################################################################
                                                   


DIR_parent='FLDROLA_2/'
DIR_zoom='FLDROLA_1_zoom/'
DIR_save="PLTDIR"
SYSid_ref='eNEATL36_AGRIF'
RUNid_ref=''
TYPEid_ref='TYPEid_NM'
RUNID_zoom='VER1'
RUNID_parent='VER2'

listsat=['alg','j3', 's3a']
start_date_init = date(DCYCST_yy, DCYCST_mm, DCYCST_dd)
end_date = date(DCYCEN_yy, DCYCEN_mm, DCYCEN_dd)

lonmin=-21
lonmax=15
latmin=15
latmax=70
loncenter=(lonmin+lonmax)/2
latcenter=(latmin+latmax)/2
maxsearch=0.001 # degrees
#lonmin=-5
#lonmax=5
#latmin=45
#latmax=55
dxsat={} # pas moyen des altimètres

#dxsat['c2']=7
#dxsat['alg']=7
#dxsat['h2g']=7
#dxsat['j3']=7
#dxsat['s3a']=7


colordict={}# couleur pour les plots
colordict['c2']='b'
colordict['alg']='r'
colordict['h2g']='orange'
colordict['j3']='green'
colordict['s3a']='purple'
rangewn=np.arange(70,250,5) 

plot_maps_sla=True
do_slope_map=False

for satid in listsat :
   print("selecting tracks for sat : " , satid)
   exec('listOBS_parent'+str(satid)+'=np.array([[],[],[]]).T')
   exec('listMOD_parent'+str(satid)+'=np.array([[],[],[]]).T')
   exec('listdistance_parent'+str(satid)+'=np.array([])')
   exec('listOBS_zoom'+str(satid)+'=np.array([[],[],[]]).T')
   exec('listMOD_zoom'+str(satid)+'=np.array([[],[],[]]).T')
   exec('listdistance_zoom'+str(satid)+'=np.array([])')
   exec('listdistance_parent_on_zoom'+satid+'=np.array([])')

   listOBS_out_p, listMOD_out_p, listdistance_out_p , listtime_out_p = loadfile(DIR_parent,satid,start_date_init,end_date,lonmin,lonmax,latmin,latmax)    
   listOBS_out_z_tmp, listMOD_out_z_tmp, listdistance_out_z_tmp, listtime_out_z_tmp = loadfile(DIR_zoom,satid,start_date_init,end_date,lonmin,lonmax,latmin,latmax)
   # first, check time on parent 
   # Then, check time then lat / lon on parent
   lon_identical_ind=np.arange(listOBS_out_p[:,0].shape[0])[np.in1d(listOBS_out_p[:,0],listOBS_out_z_tmp[:,0])]
   listOBS_out_p_tmp=listOBS_out_p[np.array(lon_identical_ind),:]
   listMOD_out_p_tmp=listMOD_out_p[np.array(lon_identical_ind),:]
   listtime_out_p_tmp=listtime_out_p[np.array(lon_identical_ind)]
   time_identical_ind=np.array(np.nonzero(np.in1d(listtime_out_p_tmp, listtime_out_z_tmp, assume_unique=False))).squeeze()
   listOBS_out_p_z=listOBS_out_p_tmp[np.array(time_identical_ind),:]
   listMOD_out_p_z=listMOD_out_p_tmp[np.array(time_identical_ind),:]
   listtime_out_p_z=listtime_out_p_tmp[np.array(time_identical_ind)]
   # Then, recheck time on child
   time_identical_ind2=np.array(np.nonzero(np.in1d(listtime_out_z_tmp,listtime_out_p_z, assume_unique=False))).squeeze()
   listOBS_out_z=listOBS_out_z_tmp[np.array(time_identical_ind2),:]
   listMOD_out_z=listMOD_out_z_tmp[np.array(time_identical_ind2),:]
   #listOBS_out_z=listOBS_out_z_tmp
   #listMOD_out_z=listMOD_out_z_tmp
   print("number of points in parent domain ",listOBS_out_p[:,0].shape)
   print("number of selected points in parent domain ",listOBS_out_p_z[:,0].shape)
   print("number of points in child domain ",listOBS_out_z_tmp[:,0].shape)
   print("number of points in child domain now",listOBS_out_z[:,0].shape)

   LONGITUDE_all={}
   LATITUDE_all={}
   LONGITUDE_all=listOBS_out_p_z[:,0]
   LATITUDE_all=listOBS_out_p_z[:,1]
   lonnow=LONGITUDE_all[:-1]
   lonafter=LONGITUDE_all[1:]
   latnow=LATITUDE_all[:-1]
   latafter=LATITUDE_all[1:]
   dwbtw,angle=ar2haversine(lonnow,latnow,lonafter,latafter)
   print(np.nanmean(dwbtw))
   exec('listdistance_parent_on_zoom'+str(satid)+'=np.concatenate((listdistance_parent_on_zoom'+str(satid)+',dwbtw),axis=0)')
   LONGITUDE_all={}
   LATITUDE_all={}

   LONGITUDE_all=listOBS_out_z[:,0]
   LATITUDE_all=listOBS_out_z[:,1]
   lonnow=LONGITUDE_all[:-1]
   lonafter=LONGITUDE_all[1:]
   latnow=LATITUDE_all[:-1]
   latafter=LATITUDE_all[1:]
   dwbtw,angle=ar2haversine(lonnow,latnow,lonafter,latafter)
   print(np.nanmean(dwbtw))
   exec('listdistance_zoom'+str(satid)+'=np.concatenate((listdistance_zoom'+str(satid)+',dwbtw),axis=0)')
   #exec('listdistance_parent_on_zoom'+str(satid)+'=listdistance_zoom'+str(satid))

   exec('listOBS_parent_on_zoom'+str(satid)+'=listOBS_out_p_z')
   exec('listMOD_parent_on_zoom'+str(satid)+'=listMOD_out_p_z')
   
   exec('listOBS_zoom'+str(satid)+'=listOBS_out_z')
   exec('listMOD_zoom'+str(satid)+'=listMOD_out_z')

   exec('listOBS_parent'+str(satid)+'=listOBS_out_p')
   exec('listMOD_parent'+str(satid)+'=listMOD_out_p')
   exec('listdistance_parent'+str(satid)+'=listdistance_out_p')

   dxsat[satid]=np.nanmean(np.where(dwbtw < 100, dwbtw, np.nan))

   print("pas moyen de l'altimètre ", satid, "=", dxsat[satid])


 #PLOT MAPS SLA ########################################

   if plot_maps_sla:
       proj=ccrs.Mercator()
       plt.figure(figsize=(8,8))
       LAT, LON, OBS1 = listOBS_out_z[:,1], listOBS_out_z[:,0] , listOBS_out_z[:,2]
       list1 = np.column_stack((LAT, LON, OBS1))
       nbdata1, xlon, ylat, obsgrid1 = grid_data_NUMBA(list1,0.5,0.5,1)
       lon, lat =np.meshgrid(xlon, ylat)

       ax = plt.subplot(111, projection=proj)
       
       ax.coastlines(resolution='50m')
       lon_formatter = LongitudeFormatter(degree_symbol='° ')
       lat_formatter = LatitudeFormatter(degree_symbol='° ')
       ax.xaxis.set_major_formatter(lon_formatter)
       ax.yaxis.set_major_formatter(lat_formatter)
       
       axes=ax.gridlines( draw_labels=False, linewidth=0)
       im1 = plt.contourf(lon, lat, obsgrid1 ,levels=np.linspace(-3,3,51), cmap='seismic',transform=ccrs.PlateCarree(),extend='both')
       cbar=plt.colorbar(ax=ax,ticks=np.linspace(-3,3,11),orientation="vertical",fraction=0.046, pad=0.04) #ticks=[-40,20,0,20,40]) #ticks=[0,1,2,3]) #
       cbar.set_label('SST (°C)', rotation=270, labelpad=12, fontsize=12,fontweight="bold")
       ax.set_extent((-21.0, 16.0, 25.0, 63.5))
       ax.set_title('VER')
       plt.tight_layout()
       plt.savefig('SLA_OBS_son_'+str(satid)+'.png')
       plt.close()


       proj=ccrs.Mercator()
       plt.figure(figsize=(8,8))
       LAT, LON, OBS1 = listOBS_out_p[:,1], listOBS_out_p[:,0] , listOBS_out_p[:,2]
       list1 = np.column_stack((LAT, LON, OBS1))
       nbdata1, xlon, ylat, obsgrid1 = grid_data_NUMBA(list1,0.5,0.5,1)
       lon, lat =np.meshgrid(xlon, ylat)

       ax = plt.subplot(111, projection=proj)

       ax.coastlines(resolution='50m')
       lon_formatter = LongitudeFormatter(degree_symbol='° ')
       lat_formatter = LatitudeFormatter(degree_symbol='° ')
       ax.xaxis.set_major_formatter(lon_formatter)
       ax.yaxis.set_major_formatter(lat_formatter)

       axes=ax.gridlines( draw_labels=False, linewidth=0)
       im1 = plt.contourf(lon, lat, obsgrid1 ,levels=np.linspace(-3,3,51), cmap='seismic',transform=ccrs.PlateCarree(),extend='both')
       cbar=plt.colorbar(ax=ax,ticks=np.linspace(-3,3,11),orientation="vertical",fraction=0.046, pad=0.04) #ticks=[-40,20,0,20,40]) #ticks=[0,1,2,3]) #
       cbar.set_label('SST (°C)', rotation=270, labelpad=12, fontsize=12,fontweight="bold")
       ax.set_extent((-21.0, 16.0, 25.0, 63.5))
       ax.set_title('VER')
       plt.tight_layout()
       plt.savefig('SLA_OBS_parent_'+str(satid)+'.png')
       plt.close()
       del(lon, lat, obsgrid1)
       proj=ccrs.Mercator()
       plt.figure(figsize=(8,8))
       del(LAT, LON, OBS1)
       LAT, LON, OBS1 = listOBS_out_p_z[:,1], listOBS_out_p_z[:,0] , listOBS_out_p_z[:,2]
       list1 = np.column_stack((LAT, LON, OBS1))
       nbdata1, xlon, ylat, obsgrid1 = grid_data_NUMBA(list1,0.5,0.5,1)
       lon, lat =np.meshgrid(xlon, ylat)
       plt.imshow(obsgrid1)
       plt.savefig('tmp.png')
       plt.close()
       ax = plt.subplot(111, projection=proj)

       ax.coastlines(resolution='50m')
       lon_formatter = LongitudeFormatter(degree_symbol='° ')
       lat_formatter = LatitudeFormatter(degree_symbol='° ')
       ax.xaxis.set_major_formatter(lon_formatter)
       ax.yaxis.set_major_formatter(lat_formatter)

       axes=ax.gridlines( draw_labels=False, linewidth=0)
       im1 = plt.contourf(lon, lat, obsgrid1 ,levels=np.linspace(-3,3,51), cmap='seismic',transform=ccrs.PlateCarree(),extend='both')
       cbar=plt.colorbar(ax=ax,ticks=np.linspace(-3,3,11),orientation="vertical",fraction=0.046, pad=0.04) #ticks=[-40,20,0,20,40]) #ticks=[0,1,2,3]) #
       cbar.set_label('SST (°C)', rotation=270, labelpad=12, fontsize=12,fontweight="bold")
       ax.set_extent((-21.0, 16.0, 25.0, 63.5))
       ax.set_title('VER')
       plt.tight_layout()
       plt.savefig('SLA_OBS_parent_sel_'+str(satid)+'.png')
       plt.close()

   del(listOBS_out_p, listMOD_out_p, listdistance_out_p)
   del(listOBS_out_p_z, listMOD_out_p_z)
   del(listOBS_out_z, listMOD_out_z)


############################################################################################
##
##    2) SELECTION DES TRACES ET CALCUL DES SPECTRES
##
############################################################################################

   print('Selection des morceaux de traces pour %s' %(satid))

   exec('dataobs_zoom=listOBS_zoom'+str(satid))
   exec('datamod_zoom=listMOD_zoom'+str(satid))
   exec('distance_zoom=listdistance_zoom'+str(satid))
   exec('dataobs_parent_on_zoom=listOBS_parent_on_zoom'+str(satid))
   exec('datamod_parent_on_zoom=listMOD_parent_on_zoom'+str(satid))
   exec('distance_parent_on_zoom=listdistance_parent_on_zoom'+str(satid))

   print(dataobs_zoom.shape, datamod_zoom.shape, distance_zoom.shape) 
   print(dataobs_parent_on_zoom.shape, datamod_parent_on_zoom.shape, distance_parent_on_zoom.shape)
   freq_zoom, meansp_zoom, meanspmod_zoom =select_and_compute_spectrum(dataobs_zoom, datamod_zoom, distance_zoom, satid, dxsat[satid])
   freq_parent_on_zoom, meansp_parent_on_zoom, meanspmod_parent_on_zoom = select_and_compute_spectrum(dataobs_parent_on_zoom,\
                                                                           datamod_parent_on_zoom, distance_parent_on_zoom, satid, dxsat[satid])
   



   exec('freqsat_zoom'+str(satid)+'=freq_zoom')
   exec('meansp_zoom'+str(satid)+'=meansp_zoom')
   exec('meanspmod_zoom'+str(satid)+'=meanspmod_zoom')
#
   exec('freqsat_parent_on_zoom'+str(satid)+'=freq_parent_on_zoom')
   exec('meansp_parent_on_zoom'+str(satid)+'=meansp_parent_on_zoom')
   exec('meanspmod_parent_on_zoom'+str(satid)+'=meanspmod_parent_on_zoom')
   del(freq_zoom, meansp_zoom, meanspmod_zoom, freq_parent_on_zoom, meansp_parent_on_zoom, meanspmod_parent_on_zoom )
for satid in listsat :
   if not do_slope_map:
      exec('xplot_zoom=freqsat_zoom'+str(satid))
      exec('varplot_zoom=meansp_zoom'+str(satid))
      exec('varplotmod_zoom=meanspmod_zoom'+str(satid))

      exec('xplot_parent_on_zoom=freqsat_parent_on_zoom'+str(satid))
      exec('varplot_parent_on_zoom=meansp_parent_on_zoom'+str(satid))
      exec('varplotmod_parent_on_zoom=meanspmod_parent_on_zoom'+str(satid))

      spinterp_zoom=np.interp(1./rangewn[::-1],xplot_zoom,np.real(varplot_zoom))
      spmodinterp_zoom=np.interp(1./rangewn[::-1],xplot_zoom,np.real(varplotmod_zoom))
      spmodinterp_parent_on_zoom=np.interp(1./rangewn[::-1],xplot_parent_on_zoom,np.real(varplotmod_parent_on_zoom))

      a,b=np.polyfit(np.log10(1./rangewn[::-1]),np.log10(spinterp_zoom),1)
      print("OBS Spectrum slope for %s : %f" %(satid, a))

      amod_zoom,bmod_zoom=np.polyfit(np.log10(1./rangewn[::-1]),np.log10(spmodinterp_zoom),1)
      print("MOD zoom Spectrum slope for %s : %f" %(satid, amod_zoom))

      amod_parent_on_zoom,bmod_parent_on_zoom=np.polyfit(np.log10(1./rangewn[::-1]),np.log10(spmodinterp_parent_on_zoom),1)
      print("MOD parent_on_zoom Spectrum slope for %s : %f" %(satid, amod_parent_on_zoom))

      plt.loglog(xplot_zoom,varplot_zoom,c=colordict[satid],ls='-',label=satid)
      plt.loglog(xplot_zoom,varplotmod_zoom,c=colordict[satid],ls='dashed',label=str(satid)+' ZOOM '+str(RUNID_zoom))
      #plt.loglog(xplot_parent_on_zoom,varplotmod_parent_on_zoom,c=colordict[satid],ls='dotted',label=str(satid)+' PARENT '+str(RUNID_parent))
      ax = plt.gca()
      plt.axis([1e-1,1e-4,1e-5,1e-0])
      plt.gca().invert_xaxis()
      ax.grid(True)
      ticklines = ax.get_xticklines() + ax.get_yticklines()
      gridlines = ax.get_xgridlines() + ax.get_ygridlines()
      ticklabels = ax.get_xticklabels() + ax.get_yticklabels()
      for line in ticklines:
          line.set_linewidth(1)
      for line in gridlines:
          line.set_linestyle(':')
      for label in ticklabels:
          label.set_color('k')
          label.set_fontsize('medium')
      #plt.yticks(np.hstack([np.arange(1e-2,1e-1+1e-2,1e-2),np.arange(1e-3,1e-2+1e-3,1e-3),np.arange(1e-4,1e-3+1e-4,1e-4),np.arange(1e-5,1e-4+1e-5,1e-5),np.arange(1e-6,1e-5+1e-6,1e-6),np.arange(1e-7,1e-6+1e-7,1e-7)]))
      plt.xticks(np.hstack([np.arange(1e-3,1e-2,1e-3),np.arange(1e-2,1e-1+1e-2,1e-2),]))
      plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=4, mode="expand", borderaxespad=0.)
      plt.xlabel('Wavenumber (1/km)',size=14.0)
      plt.ylabel('Spectral power density m2/(cy/km)',size=14.0)


    #PLOT MAP
     #---------------------------------
      if do_slope_map:
         exec('varplot=SLOPE'+str(satid))
         exec('varplotmod=SLOPEmod'+str(satid))
         fig=plt.figure(isat*2+1)
         fig,cb=plotmap_2deg(varplot.T,lonmin,lonmax,latmin,latmax,1,5,11,'Spectral_r')
         cb.ax.set_ylabel("Spectrum slope", fontsize = 8)
         fig_name3="slope_OBS_"+satid+SYSid_ref+RUNid_ref+"_"+TYPEid_ref+".png"
         plt.savefig(DIR_save+fig_name3, format='png',bbox_inches='tight',dpi=200)
         print("figure saved here : "+DIR_save+fig_name3)
         plt.close()
         fig2=plt.figure(isat*2+1)
         varplot_tmp=varplot[~np.isnan(varplot)]
         varplotmod_tmp=varplotmod[~np.isnan(varplotmod)]

         fig2,cb=plotmap_2deg(varplotmod.T,lonmin,lonmax,latmin,latmax,1,5,11,'Spectral_r')
         cb.ax.set_ylabel("Spectrum slope", fontsize = 8)
         fig_name3="slope_MOD_"+satid+SYSid_ref+RUNid_ref+"_"+TYPEid_ref+".png"
         plt.savefig(DIR_save+fig_name3, format='png',bbox_inches='tight',dpi=200)
         print("figure saved here : "+DIR_save+fig_name3)
         plt.close()

      #plt.close()  
      print("Mission Clear!")
      if not do_slope_map:
          fig_name1="SPD_"+SYSid_ref+RUNid_ref+"_"+TYPEid_ref+"_"+str(satid)+".png"
          plt.savefig(DIR_save+fig_name1, format='png',bbox_inches='tight',dpi=200)
          print("figure saved here : "+ DIR_save+fig_name1)
          plt.close()
