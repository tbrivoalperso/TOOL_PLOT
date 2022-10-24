#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:04:08 2019

@author: tbrivoal
"""
import matplotlib as mpl
mpl.use('Agg')

import os, sys, re
from datetime import date, timedelta
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy import signal


def extractfield(testbool):
   import numpy as np
   nbjextract=testbool.shape[0]
   nbiextract=testbool.shape[1]
   ind_min_lat=0
   ind_max_lat=nbjextract
   ind_min_lon=0
   ind_max_lon=nbiextract
   flaglonmin=0
   flaglonmax=0
   for i in np.arange(nbiextract) :
      line=testbool[:,i]
      if line.any() and flaglonmin==0:
         ind_min_lon=i
         flaglonmin=1
      if ~line.any() and flaglonmin==1 and flaglonmax==0:
         ind_max_lon=i-1
         flaglonmax=1
   flaglatmin=0
   flaglatmax=0
   for j in np.arange(nbjextract) :
      line=testbool[j,:]
      if line.any() and flaglatmin==0:
         ind_min_lat=j
         flaglatmin=1
      if ~line.any() and flaglatmin==1 and flaglatmax==0:
         ind_max_lat=j-1
         flaglatmax=1
   return ind_min_lat,ind_max_lat,ind_min_lon,ind_max_lon


def plotmap_2deg(var, minlon=-180, maxlon=180, minlat=-90, maxlat=90, minc=-10, maxc=10, nbcol=10,palette='seismic'):
    """
    M.Hamon 03/2016

    Plot des champs 2D sur grille 2*2deg
        - Choisir les bornes du plot (minlon,maxlon,minlat,maxlat)
        - Choisir l'étendue de la palette (minc, maxc)
        - Choisir la palette ( help('matplotlib.pyplot.cm') )

    """
    try:
        import os, sys, re
        import numpy as np
        from netCDF4 import Dataset
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        from reverse_clm import reverse_colourmap

        RdBu_rv=reverse_colourmap(plt.cm.RdBu)
        X=np.arange(minlon,maxlon,1)
        Y=np.arange(minlat,maxlat,1)
        longi,lati=np.meshgrid(X,Y)
        #
        nbtick=10
        lon_0=0
        lon_shift = np.asarray(longi)
        #lon_shift = np.where(lon_shift > lon_0+180, lon_shift-360 , lon_shift)
        #lon_shift = np.where(lon_shift < lon_0-180, lon_shift+360 , lon_shift)
        nparange = np.arange(lon_shift.shape[0])[:,None]
        npargsort = np.argsort(lon_shift)
        longi = lon_shift[nparange,npargsort]
        lati = lati[nparange,npargsort]
        plvar = var[nparange,npargsort]

        plvar_extract=plvar
        longi_extract=longi
        lati_extract=lati

        ####################################################
        # FIG INFO
        dim_coltext = 8
        gris_fill = "0.5"
        proj = "merc"
        res = "i" #c,l,i,f

        # GET MIN MAX
        if ('minc' in locals()) and ('maxc' in locals()):
                minvalue=minc
                maxvalue=maxc
        else:
                minvalue = np.floor(np.nanmin(plvar_extract)*10)/10
                maxvalue = np.ceil(np.nanmax(plvar_extract)*10)/10
        # SET COLOR LEVEL AND LABELS
        tickslevs = np.linspace(minvalue,maxvalue,nbtick)
        labellevs = [ ('%4.2f'%v).rjust(5) for v in tickslevs]
        #labellevs = [ ('%d'%v).rjust(5) for v in tickslevs]
        colorlevs = np.linspace(minvalue,maxvalue,nbcol)

        # For parallels and meridians
        ddlon = 4
        delta_lat = (maxlat - minlat)/4
        delta_lon = (maxlon - minlon)/ddlon
        xparallels = np.floor(delta_lat)
        ymeridians = np.floor(delta_lon)
        # OPEN FIGURE
        fig1 = plt.figure()
        # MAP PROJECTION
        map_proj = Basemap(projection = proj , llcrnrlon = minlon, llcrnrlat = minlat,\
                      urcrnrlon = maxlon, urcrnrlat = maxlat, resolution = res)
        x,y = map_proj(longi_extract,lati_extract)
        map_proj.fillcontinents(color = gris_fill)
        if palette=='RdBu_rv':
           colormap=palette
        else:
           colormap=palette
        cs = map_proj.contourf(x, y, plvar_extract, colorlevs, cmap = colormap,extend = 'both')
        # ADD PARALLELS and MERDIANS
        parallels = np.round(np.arange(minlat, maxlat+xparallels/2., xparallels))
        map_proj.drawparallels(parallels, labels = [True, False, False, False], \
                    fontsize = int(dim_coltext), linewidth = 0.2)
        meridians = np.round(np.arange(minlon, maxlon+ymeridians/2., ymeridians))
        map_proj.drawmeridians(meridians, labels = [False, False, False, True],\
                    fontsize = int(dim_coltext), linewidth = 0.2)
        map_proj.fillcontinents(color = gris_fill)

        # COLORBAR
        cb = map_proj.colorbar(cs, "right", format = '%4.2f', size=0.2 , pad = 0.1)
        #cb = map_proj.colorbar(cs, "right", format = '%d', size=0.2 , pad = 0.1)
        cb.set_ticks(tickslevs)
        cb.set_ticklabels(labellevs)
        cb.ax.tick_params(labelsize = int(dim_coltext))

        return fig1, cb
    except IndexError:
        print("IndexError : Please check input file (2D). ")





RUN1='VER1'
RUN2='VER2'
DIR_IN_1='FLDROLA_1'
DIR_IN_2='FLDROLA_2'
DIR_LIST=[DIR_IN_1] #, DIR_IN_2]
RUN_LIST=[RUN1] #, RUN2]
print("DIRECTORIES USED : ")
Linefmt_list=["--", "dotted"]
DIR_save="PLTDIR"
SYSid_ref='eNEATL36'
RUNid_ref=''
TYPEid_ref=''
lonmin=-21
lonmax=15
latmin=15
latmax=70
#int((dataobs[int(ii+iibox),0]+180)/2)
start_date_init = date(DCYCST_yy, DCYCST_mm, DCYCST_dd)
end_date = date(DCYCEN_yy, DCYCEN_mm, DCYCEN_dd)
delta = timedelta(days=7)
#listsat=['c2', 'alg', 'j3', 's3a', 'h2g'] # liste des satellite a prendre en compte
listsat=['j3', 's3a', 'h2g'] # liste des satellite a prendre en compte

distmax=1000 # longueur des traces 
do_output=True # Sauvegarde des figures
do_map=True # False: statistique 2D, False: statistique globale
do_ratio=False # MAH, pas forcément adapté à tous les jeux de données

rangewn=np.arange(70,250,5) # intervalle pour le calcul de la pente (do_map=True)

################################################################################################
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

###
######## Fin des routines indépendantes
###

############################################################################################
##
##    1) CHARGEMENT DES DONNEES
##
############################################################################################
for satid in listsat:
   exec('listOBS'+satid+'=np.array([[],[],[]]).T')
   exec('listMOD'+satid+'=np.array([[],[],[]]).T')
   exec('listdistance'+satid+'=np.array([])')

dxsat={} # pas moyen des altimètres
dxsat['c2']=12.4
dxsat['alg']=12.2
dxsat['h2g']=7.3
dxsat['j3']=6.7
dxsat['s3a']=11.5

colordict={}# couleur pour les plots
colordict['c2']='b'
colordict['alg']='r'
colordict['h2g']='orange'
colordict['j3']='green'
colordict['s3a']='purple'

dir_count=0
plt.figure(1)
for DIR_IN in DIR_LIST:
     start_date=start_date_init
     while start_date <= end_date:
         tdate=start_date.strftime("%Y%m%d")
         print(tdate)
         start_date += delta
     
         for olafile in os.listdir(DIR_IN):
             if re.search('OLA_IS_SLA_R'+str(tdate)+'.nc',olafile) :
                 existfile=False
                 olafile=olafile[:]
                 print("olafile: "+olafile)
                 print(DIR_IN+olafile)
                 try :
                    print(DIR_IN+olafile)
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
     #            for n in range(len(SETID[:,0])):
     #                print("n =",n, SETID[n,:],LONGITUDE[n],LATITUDE[n],OBSERVATION[n] ,JULTIME[n])
                 indexok={}
                 iisat=0
                 for sat in listsat:
                    testsat=(SETID[:,0]==sat[0]) & (SETID[:,1]==sat[1])
                    testlat=(LATITUDE>=latmin) & (LATITUDE<=latmax)
                    indexok[iisat]=np.where((testsat) & (testlat))
                    iisat+=1
                 #
                 LONGITUDE_all={}
                 LATITUDE_all={}
                 JULTIME_all={}
                 OBSERVATION_all={}
                 EQUIVALENT_MODELE_all={}
                 argsort_all={}
                 time_sort={}
                 for sat in np.arange(len(indexok)):
                    testempty=indexok[sat][0]
                    if testempty.any():
                       ###
                      
                       LONGITUDE_all[sat]=LONGITUDE[indexok[sat]]
                       LATITUDE_all[sat]=LATITUDE[indexok[sat]]
                       JULTIME_all[sat]=JULTIME[indexok[sat]]
                       OBSERVATION_all[sat]=OBSERVATION[indexok[sat]]
                       EQUIVALENT_MODELE_all[sat]=EQUIVALENT_MODELE[indexok[sat]]
                       argsort_all[sat]=np.argsort(JULTIME_all[sat])
                       time_sort[sat]=JULTIME_all[sat][argsort_all[sat]]
                       LONGITUDE_all[sat]=LONGITUDE_all[sat][argsort_all[sat]]
                       LATITUDE_all[sat]=LATITUDE_all[sat][argsort_all[sat]]
                       OBSERVATION_all[sat]=OBSERVATION_all[sat][argsort_all[sat]]
                       EQUIVALENT_MODELE_all[sat]=EQUIVALENT_MODELE_all[sat][argsort_all[sat]]
                       LONGITUDE_all[sat]=LONGITUDE_all[sat][::2]
                       LATITUDE_all[sat]=LATITUDE_all[sat][::2]
                       OBSERVATION_all[sat]=OBSERVATION_all[sat][::2]
                       EQUIVALENT_MODELE_all[sat]=EQUIVALENT_MODELE_all[sat][::2]
                       ### MSE
                       lonnow=LONGITUDE_all[sat][:-1]
                       lonafter=LONGITUDE_all[sat][1:]
                       latnow=LATITUDE_all[sat][:-1]
                       latafter=LATITUDE_all[sat][1:]
                       dwbtw,angle=ar2haversine(lonnow,latnow,lonafter,latafter)
     
                       listOBS=np.column_stack((LONGITUDE_all[sat],LATITUDE_all[sat],OBSERVATION_all[sat]))
                       listMOD=np.column_stack((LONGITUDE_all[sat],LATITUDE_all[sat],EQUIVALENT_MODELE_all[sat]))
     
                       #
                       exec('listOBS'+str(listsat[sat])+'=np.concatenate((listOBS'+str(listsat[sat])+',listOBS),axis=0)')
                       exec('listMOD'+str(listsat[sat])+'=np.concatenate((listMOD'+str(listsat[sat])+',listMOD),axis=0)')
                       exec('listdistance'+str(listsat[sat])+'=np.concatenate((listdistance'+str(listsat[sat])+',dwbtw),axis=0)')
     
     
     
     
     
     
     exec('print(listdistance'+str(listsat[sat])+'.shape)')
     #exec('print(listOBS'+str(listsat[sat])+'[0,:,0])')
     #exec('print(listOBS'+str(listsat[sat])+'[0,0,:])')
     
############################################################################################
##
##    2) SELECTION DES TRACES ET CALCUL DES SPECTRES
##
############################################################################################

for sat in np.arange(len(indexok)):
   print('Selection des morceaux de traces pour %s' %(listsat[sat]))
   nbtrace=0
   if do_map:
      tracemap={}
      tracemap_mod={}
      indextrmap=np.zeros((180,90))
   else:
      trace={}
      tracemod={}
   exec('dataobs=listOBS'+str(listsat[sat]))
   exec('datamod=listMOD'+str(listsat[sat]))
   exec('distance=listdistance'+str(listsat[sat]))
   meandx=dxsat[listsat[sat]]*2.03
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
               if not do_map:
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
               ii+=int(200/dxsat[listsat[sat]])
               ln_lgth=True
            else:
               ii=iitr+1
               ln_gap=True
         iitr+=1
   # 
   print('Calcul du spectre pour %s' %(listsat[sat]))
   nbpts=minnbpoint
   nbpts_sp=np.int(np.floor(minnbpoint/2))
   window = tukeywin(nbpts, alpha=0.1)
   freq=np.fft.fftfreq(nbpts,dxsat[listsat[sat]])
   freq = freq[0:len(freq)//2]
   if not do_map:
      nbtr=len(trace)
      sptot=np.zeros((nbpts_sp,nbtr))
      spmodtot=np.zeros((nbpts_sp,nbtr))

      for indextr in np.arange(nbtr):
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
      exec('freqsat'+str(listsat[sat])+'=freq')
      exec('meansp'+str(listsat[sat])+'=np.median(sptot,axis=1)')
      exec('meanspmod'+str(listsat[sat])+'=np.median(spmodtot,axis=1)')

   else:
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

      exec('freqsat'+str(listsat[sat])+'=freq')
      exec('SLOPE'+str(listsat[sat])+'=SLOPE')
      exec('SLOPEmod'+str(listsat[sat])+'=SLOPEmod')

############################################################################################
##    2) PLOTS
##
############################################################################################

if do_output:
   isat=0
   print("Save figures...")
   for sat in np.arange(len(indexok)):
      if not do_map:
         exec('xplot=freqsat'+str(listsat[sat]))
         #exec('xplot=1./freqsat'+str(listsat[sat]))
         exec('varplot=meansp'+str(listsat[sat]))
         exec('varplotmod=meanspmod'+str(listsat[sat]))

         spinterp=np.interp(1./rangewn[::-1],xplot,np.real(varplot))
         spmodinterp=np.interp(1./rangewn[::-1],xplot,np.real(varplotmod))

         a,b=np.polyfit(np.log10(1./rangewn[::-1]),np.log10(spinterp),1)
         print("OBS Spectrum slope for %s : %f" %(listsat[sat], a))
         amod,bmod=np.polyfit(np.log10(1./rangewn[::-1]),np.log10(spmodinterp),1)
         print("MOD Spectrum slope for %s : %f" %(listsat[sat], amod))
         if dir_count == 0:     
             plt.loglog(xplot,varplot,c=colordict[listsat[sat]],ls='-',label=listsat[sat])
         plt.loglog(xplot,varplotmod,c=colordict[listsat[sat]],ls=Linefmt_list[dir_count],label=str(listsat[sat])+' '+RUN_LIST[dir_count])
         ax = plt.gca()
         plt.axis([1e-0,1e-4,1e-7,1e-1])
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
         plt.xlabel('Frequency',size=14.0)
         plt.ylabel('Spectral power density m2/(cy/km)',size=14.0)
         #fig_name1="SPD_"+SYSid_ref+RUNid_ref+"_"+TYPEid_ref+RUN_LIST[dir_count]+".png"
         #plt.savefig(DIR_save+fig_name1, format='png',bbox_inches='tight',dpi=200)
         #print("figure saved here : "+ DIR_save+fig_name1)
               #---------------------------------
     #PLOT MAP
     #---------------------------------
      if do_map:
         exec('varplot=SLOPE'+str(listsat[sat]))
         exec('varplotmod=SLOPEmod'+str(listsat[sat]))
         print("varplot",np.nanmax(varplot))
         print("varplotmod",np.nanmax(varplotmod))
         fig=plt.figure(isat*2+1)
         fig,cb=plotmap_2deg(varplot.T,lonmin,lonmax,latmin,latmax,1,5,11,'Spectral_r')
         cb.ax.set_ylabel("Spectrum slope", fontsize = 8)
         fig_name3="slope_OBS_"+listsat[sat]+SYSid_ref+RUNid_ref+"_"+TYPEid_ref+".png"
         plt.savefig(DIR_save+fig_name3, format='png',bbox_inches='tight',dpi=200)
         print("figure saved here : "+DIR_save+fig_name3)            
         plt.close()
         fig2=plt.figure(isat*2+1)
         varplot_tmp=varplot[~np.isnan(varplot)]
         varplotmod_tmp=varplotmod[~np.isnan(varplotmod)]

         print("varplot",varplot_tmp)
         print("varplotmod",varplotmod_tmp)
         fig2,cb=plotmap_2deg(varplotmod.T,lonmin,lonmax,latmin,latmax,1,5,11,'Spectral_r')
         cb.ax.set_ylabel("Spectrum slope", fontsize = 8)
         fig_name3="slope_MOD_"+listsat[sat]+SYSid_ref+RUNid_ref+"_"+TYPEid_ref+".png"
         plt.savefig(DIR_save+fig_name3, format='png',bbox_inches='tight',dpi=200)
         print("figure saved here : "+DIR_save+fig_name3)
         plt.close()

dir_count=+1     
#plt.close()  
print("Mission Clear!")
if not do_map:
    fig_name1="SPD_"+SYSid_ref+RUNid_ref+"_"+TYPEid_ref+".png"
    plt.savefig(DIR_save+fig_name1, format='png',bbox_inches='tight',dpi=200)
    print("figure saved here : "+ DIR_save+fig_name1)

