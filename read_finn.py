#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:37:14 2020

@author: jone
"""

#Investigate explicit By effect on NOAA dataset from Finn where oval boundaries
#has ben identified

import glob
import scipy.io
import datetime as dt
import numpy as np
import dipole #Kalles dipole public library on github
import matplotlib.pyplot as plt
import pandas as pd

hemi = 'nord'

files = glob.glob('./finn_data/*'+hemi+'.mat')



for ff in files:
    f = scipy.io.loadmat(ff)
    minyear = f['year'].min()
    maxyear = f['year'].max()
    if minyear == maxyear:
        year = maxyear
    else:
        print('File contain data from multiple years. Aborting processing.')
        print(3/0)
    #Check for leap year
    if (year % 4) == 0:
       if (year % 100) == 0:
           if (year % 400) == 0:
               leap = 1
           else:
               leap = 0
       else:
           leap = 1
    else:
       leap = 0
   
    #Convert to datetime
    doy = f['doy'].flatten().astype(int)
    hr = (f['doy'].flatten() - doy)*24.
    dtimes = []
    cc = 0
    for d in doy:
        dtimes.append(dt.datetime.strptime(str(year) + ' ' + str(d) + ' ' + \
                    str(int(round((hr[cc])))), '%Y %j %H'))
        cc += 1
    dtimesold = dtimes[:]
    if ff != files[0]: #not first time for loop runs
        if dtimes != dtimesold:
            print('The files has different time axis. Aborting.')
            print(3/0)
        
        
###############
#Combine with omni data
y = dtimes[0].year
omnifile = '/home/jone/Documents/Dropbox/science/superdarn/lobe_circulation/omni_1min_1999-2017.hdf'
omni = pd.read_hdf(omnifile)
use = (omni.index >= dt.datetime(y,1,1)) & (omni.index <= dt.datetime(y+1,1,1))
omni = omni[use].copy()
omni = omni.interpolate(limit=5, limit_direction='both')
bypos = (omni.BY_GSM>0)# & (np.abs(omni.BZ_GSM)<np.abs(omni.BY_GSM))
omni.loc[:,'bypos'] = omni.BY_GSM[bypos]
byneg = (omni.BY_GSM<0)# & (np.abs(omni.BZ_GSM)<np.abs(omni.BY_GSM))
omni.loc[:,'byneg'] = omni.BY_GSM[byneg]    
omni.loc[:,'milan'] = 3.3e5 * (omni['flow_speed']*1000.)**(4./3)  * (np.sqrt(omni.BY_GSM**2 + omni.BZ_GSM**2)) * 1e-9 * \
                        np.sin(np.abs(np.arctan2(omni['BY_GSM'],omni['BZ_GSM']))/2.)**(4.5) * 0.001
milanpos = 3.3e5 * (omni['flow_speed']*1000.)**(4./3)  * (np.sqrt(omni.bypos**2 + omni.BZ_GSM**2)) * 1e-9 * \
                        np.sin(np.abs(np.arctan2(omni['bypos'],omni['BZ_GSM']))/2.)**(4.5) * 0.001                        
milanneg = 3.3e5 * (omni['flow_speed']*1000.)**(4./3)  * (np.sqrt(omni.byneg**2 + omni.BZ_GSM**2)) * 1e-9 * \
                        np.sin(np.abs(np.arctan2(omni['byneg'],omni['BZ_GSM']))/2.)**(4.5) * 0.001
window = 60    #minutes
nobsinwindow = omni.milan.rolling(window).count()
cumsumneg = milanneg.rolling(window,min_periods=1).sum()                                            
cumsumpos = milanpos.rolling(window,min_periods=1).sum()
omni.loc[:,'bxlong'] = omni.BX_GSE.rolling(window,min_periods=1).mean()
omni.loc[:,'bzlong'] = omni.BZ_GSM.rolling(window,min_periods=1).mean()
omni.loc[:,'bylong'] = omni.BY_GSM.rolling(window,min_periods=1).mean()
bxlim = 200
usepos = ((cumsumpos>2.*cumsumneg) & (nobsinwindow==window) & (np.abs(omni.bxlong)<bxlim)) | ((cumsumneg.isnull()) & (np.invert(cumsumpos.isnull())) & (nobsinwindow==window) & (np.abs(omni.bxlong)<bxlim))
useneg = ((cumsumneg>2.*cumsumpos) & (nobsinwindow==window) & (np.abs(omni.bxlong)<bxlim)) | ((cumsumpos.isnull()) & (np.invert(cumsumneg.isnull())) & (nobsinwindow==window) & (np.abs(omni.bxlong)<bxlim))                                     
omni.loc[:,'usepos'] = usepos
omni.loc[:,'useneg'] = useneg
omni.loc[:,'milanlong'] = omni.milan.rolling(window, min_periods=window, center=False).mean()    #average IMF data
#omni = omni.drop(['bxlong','bzlong','byneg','bypos','PC_N_INDEX','Beta','E','Mach_num','Mgs_mach_num','y'],axis=1) #need to drop PC index as it contain a lot of nans. Also other fields will exclude data when we later use dropna()
omnicopy = omni.copy()
omni2 = omni.reindex(index=dtimes, method='nearest',tolerance='30sec')    #new ocb
omni2.loc[:,'tilt'] = dipole.dipole_tilt(omni2.index)

#Sophie list statistics
sophie = pd.read_hdf('sophie75.h5')
use = (sophie.index>=dtimes[0]) & (sophie.index<=dtimes[-1])
sophie = sophie[use].copy()
exp = sophie.ssphase == 2
sophiexp = sophie[exp].copy()

#begin experimental
sophiexp.loc[:,'tilt'] = dipole.dipole_tilt(sophiexp.index)
omni3 = omni.reindex(index=sophiexp.index, method='nearest', tolerance='30sec')
use = (omni.index>=sophiexp.index[0]) & (omni.index<=sophiexp.index[-1])
omni4 = omni[use].copy()
chunk = 1e6
n = len(omni4)
idxs = omni4.index.values
chunked = np.array_split(idxs, n/chunk)
for chunk in chunked:
    omni4.loc[omni4.index.isin(chunk),'tilt'] = dipole.dipole_tilt(omni4.index[omni4.index.isin(chunk)])
sophiexp.loc[:,'bylong'] = omni3.bylong
sophiexp.loc[:,'milanlong'] = omni3.milanlong
sophiexp.loc[:,'substorm'] = sophiexp.ssphase==2
ssum = sophiexp.groupby([pd.cut(sophiexp.tilt, bins=np.array([-35,-10,10,35])), \
                pd.cut(sophiexp.bylong, bins=np.array([-50,-2,2,50]))]).substorm.sum()
ocount = omni4.groupby([pd.cut(omni4.tilt, bins=np.array([-35,-10,10,35])), \
                pd.cut(omni4.bylong, bins=np.array([-50,-2,2,50]))]).tilt.count()
ssum/ocount

#end experimental

sophie2 = sophiexp.reindex(index=dtimes, method='pad',tolerance='60min')
sophie2.loc[:,'substorm'] = sophie2.ssphase==2

if hemi == 'nord':
    p ='n'
if hemi == 'south':
    p='s'
        
for ff in files:
    f= scipy.io.loadmat(ff)
    mlteq = np.concatenate((f['ifmlt_eqa15e'+p].flatten(), f['ifmlt_eqa16e'+p].flatten(), \
                          f['ifmlt_eqa17e'+p].flatten(), f['ifmlt_eqa15m'+p].flatten(), \
                          f['ifmlt_eqa16m'+p].flatten(), f['ifmlt_eqa17m'+p].flatten()))
    mlteq = mlteq/15.
    mltpol = np.concatenate((f['ifmlt_pol15e'+p].flatten(), f['ifmlt_pol16e'+p].flatten(), \
                          f['ifmlt_pol17e'+p].flatten(), f['ifmlt_pol15m'+p].flatten(), \
                          f['ifmlt_pol16m'+p].flatten(), f['ifmlt_pol17m'+p].flatten()))  
    mltpol = mltpol/15.
    mlateq = np.concatenate((f['ifyeqa15e'+p].flatten(), f['ifyeqa16e'+p].flatten(), \
                          f['ifyeqa17e'+p].flatten(), f['ifyeqa15m'+p].flatten(), \
                          f['ifyeqa16m'+p].flatten(), f['ifyeqa17m'+p].flatten()))
    mlatpol = np.concatenate((f['ifypol15e'+p].flatten(), f['ifypol16e'+p].flatten(), \
                          f['ifypol17e'+p].flatten(), f['ifypol15m'+p].flatten(), \
                          f['ifypol16m'+p].flatten(), f['ifypol17m'+p].flatten()))  
    sat = np.concatenate((np.ones(len(f['ifypol15e'+p]))*15, np.ones(len(f['ifypol16e'+p]))*16, \
                          np.ones(len(f['ifypol17e'+p]))*17, np.ones(len(f['ifypol15m'+p]))*15, \
                          np.ones(len(f['ifypol16m'+p]))*16, np.ones(len(f['ifypol17m'+p]))*17))
    ovalflux = np.concatenate((f['ifftenergi15e'+p].flatten(), f['ifftenergi16e'+p].flatten(), \
                               f['ifftenergi17e'+p].flatten(), f['ifftenergi15m'+p].flatten(), \
                               f['ifftenergi16m'+p].flatten(), f['ifftenergi17m'+p].flatten()))
    By_GSM = np.concatenate((f['By_GSM'].flatten(), f['By_GSM'].flatten(), f['By_GSM'].flatten(), \
                             f['By_GSM'].flatten(), f['By_GSM'].flatten(), f['By_GSM'].flatten()))
    milanlong = np.concatenate((omni2.milanlong, omni2.milanlong, omni2.milanlong, \
                                omni2.milanlong, omni2.milanlong, omni2.milanlong))
    bylong = np.concatenate((omni2.bylong, omni2.bylong, omni2.bylong, \
                                omni2.bylong, omni2.bylong, omni2.bylong))    
    usepos = np.concatenate((omni2.usepos, omni2.usepos, omni2.usepos, \
                                omni2.usepos, omni2.usepos, omni2.usepos))    
    useneg = np.concatenate((omni2.useneg, omni2.useneg, omni2.useneg, \
                                omni2.useneg, omni2.useneg, omni2.useneg))            
    dta = np.concatenate((omni2.tilt, omni2.tilt, omni2.tilt, omni2.tilt, omni2.tilt, omni2.tilt))
    dates = np.concatenate((omni2.index, omni2.index, omni2.index, omni2.index, \
                            omni2.index, omni2.index))
    substorm = np.concatenate((sophie2.substorm, sophie2.substorm, sophie2.substorm, \
                               sophie2.substorm, sophie2.substorm, sophie2.substorm))
    
    data = {'dates':dates, 'mlteq':mlteq, 'mltpol':mltpol, 'mlateq':mlateq, 'mlatpol':mlatpol, \
            'sat':sat, 'ovalflux':ovalflux, 'By_GSM':By_GSM, 'milanlong':milanlong, \
            'bylong':bylong, 'usepos':usepos, 'useneg':useneg, 'tilt':dta, 'substorm':substorm}
    
    df = pd.DataFrame(data)
    #df.index = df.dates
    
    if f['det'].flatten()[0] == 1:
        channel = 'TED_electron_0.15-20kev'
    elif f['det'].flatten()[0] == 2:
        channel = 'MEPED_proton_0.15-20kev'        
    elif f['det'].flatten()[0] == 3:
        channel = 'MEPED_electron_>30kev'
    elif f['det'].flatten()[0] == 4:
        channel = 'MEPED_proton_20-80kev'                


    fname = './jone_data/'+str(year)+'_'+hemi+'_noaa'+str(int(df.sat.min()))+'-'+str(int(df.sat.max())) + \
            '_' + channel + '.h5'
    
    df.to_hdf(fname, mode='w', format = 'table', key='noaa')
#%  forklaring på data til Jone 12 mars 2020
#% save('jone_datadett4nord.mat','ifftenergi15en','ifftenergi15mn','ifftenergi16en','ifftenergi16mn','ifftenergi17en','ifftenergi17mn',...
#% 'ifmlt_eqa15en','ifmlt_eqa15mn','ifmlt_eqa16en','ifmlt_eqa16mn','ifmlt_eqa17en','ifmlt_eqa17mn',...
#% 'ifmlt_pol15en','ifmlt_pol15mn','ifmlt_pol16en','ifmlt_pol16mn','ifmlt_pol17en','ifmlt_pol17mn',...
#% 'ifyeqa15en','ifyeqa15mn','ifyeqa16en','ifyeqa16mn','ifyeqa17en','ifyeqa17mn','ifypol15en','ifypol15mn',...
#% 'ifypol16en','ifypol16mn','ifypol17en','ifypol17mn','Kp','Bx_GSM','By_GSM','By_GSM','Bz_GSM','Btot','Dst_nT','doy','year','hemi','det')
#%  Dataene er lagret med kommandoen ovenfor.
#%  Alle daaene er interpolert mot doy timeverdier.
#%  Data fra satellittene POES 15, 16 og17
#%  ifftenergi15en er energi summert over ovalen for POES 15 en (evening north)
#%  ifmlt_eqa15en er MLT for ekvatorgrensen i grader
#%  ifmlt_pol15en er MLT for polgrensen i grader
#%  mn angir (morning north)
#%  DET SAMME FOR POES 16 OG 17
#%  doy er desimal dag i året
#%  year er år
#%  hemi er NORTH eller SOUTH
#%  det lik 1 er elektroner 0.15 - 20 keV, 2 er protoner 0.15 - 20 keV, 3 elektroner > 30 keV
#%  4 er protoner 20 - 80 keV 