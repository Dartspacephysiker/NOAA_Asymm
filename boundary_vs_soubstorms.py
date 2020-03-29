#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:26:17 2020

@author: jone
"""

import pandas as pd
import numpy as np
import dipole
import matplotlib.pyplot as plt
import datetime as dt

#investigate the relationship between equatorward boundary in NOAA data and the 
#occurrence rate of substorms from sophie list

#Load omni data
omnifile = '/home/jone/Documents/Dropbox/science/superdarn/lobe_circulation/omni_1min_1999-2017.hdf'
omni = pd.read_hdf(omnifile)

#Load Sophie list
sophie = pd.read_hdf('./jone_data/sophie75.h5')
use = (sophie.index>=omni.index[0]) & (sophie.index<=omni.index[-1])
#use = (sophie.index>=dt.datetime(2003,1,1,0,0)) & (sophie.index<dt.datetime(2004,1,1,0,0))
sophie = sophie[use].copy()
exp = sophie.ssphase == 2   #expansion phase list
sophiexp = sophie[exp].copy()


#Process omni data
use = (omni.index >= sophiexp.index[0]) & (omni.index <= sophiexp.index[-1])
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




#Combine omni and sophie list
sophiexp.loc[:,'tilt'] = dipole.dipole_tilt(sophiexp.index)
omni.loc[:,'tilt'] = dipole.dipole_tilt(omni.index)
omni2 = omni.reindex(index=sophiexp.index, method='nearest', tolerance='30sec')
sophiexp.loc[:,'bylong'] = omni2.bylong
sophiexp.loc[:,'milanlong'] = omni2.milanlong
sophiexp.loc[:,'substorm'] = sophiexp.ssphase==2
bybins = np.append(np.append([-50],np.linspace(-9,9,10)),[50])
bybincenter = np.linspace(-10,10,11)
sgroup = sophiexp.groupby([pd.cut(sophiexp.tilt, bins=np.array([-35,-10,10,35])), \
                pd.cut(sophiexp.bylong, bins=bybins)])
ogroup = omni.groupby([pd.cut(omni.tilt, bins=np.array([-35,-10,10,35])), \
                pd.cut(omni.bylong, bins=bybins)])


#Plotting
fig = plt.figure(figsize=(15,15))

ax = fig.add_subplot(221)
res = ogroup.tilt.count() / sgroup.substorm.sum()
ax.plot(bybincenter, np.array(res[0:11]), label='tilt < -10')
ax.plot(bybincenter, np.array(res[11:22]), label='|tilt| < 10')
ax.plot(bybincenter, np.array(res[22:33]), label='tilt > 10')
ax.legend()
ax.set_xlabel('IMF By')
ax.set_ylabel('Average time between onsets [min]')
ax.set_title('Onsets from Sophie-75 list, 1999-2014')

ax = fig.add_subplot(222)
#milanstat, bins= pd.qcut(omni.milanlong, 10, retbins=True)
#milanbincenter = [(a + b) /2 for a,b in zip(bins[:-1], bins[1:])]
#sgroup = sophiexp.groupby([pd.cut(sophiexp.tilt, bins=np.array([-35,-10,10,35])), \
#                pd.qcut(sophiexp.milanlong, 10)])
#ogroup = omni.groupby([pd.cut(omni.tilt, bins=np.array([-35,-10,10,35])), \
#                pd.qcut(omni.milanlong, 10)])
res = ogroup.milanlong.median()
ax.plot(bybincenter, np.array(res[0:11]), label='tilt < -10')
ax.plot(bybincenter, np.array(res[11:22]), label='|tilt| < 10')
ax.plot(bybincenter, np.array(res[22:33]), label='tilt > 10')
ax.legend()
ax.set_xlabel('IMF By')
ax.set_ylabel('$\Phi_D$')
ax.set_title('Median $\Phi_D$, 1999-2014')

#
omni_2 = omni[(omni.milanlong >= 5) & (omni.milanlong <= 15)].copy()
sophiexp2 = sophiexp[(sophiexp.milanlong >= 5) & (sophiexp.milanlong <= 15)].copy()
sgroup2 = sophiexp2.groupby([pd.cut(sophiexp2.tilt, bins=np.array([-35,-10,10,35])), \
                pd.cut(sophiexp2.bylong, bins=bybins)])
ogroup2 = omni_2.groupby([pd.cut(omni_2.tilt, bins=np.array([-35,-10,10,35])), \
                pd.cut(omni_2.bylong, bins=bybins)])
##

ax = fig.add_subplot(223)
res = ogroup2.tilt.count() / sgroup2.substorm.sum()
ax.plot(bybincenter, np.array(res[0:11]), label='tilt < -10')
ax.plot(bybincenter, np.array(res[11:22]), label='|tilt| < 10')
ax.plot(bybincenter, np.array(res[22:33]), label='tilt > 10')
ax.legend()
ax.set_xlabel('IMF By')
ax.set_ylabel('Average time between onsets [min]')
ax.set_title('Onsets from Sophie-75 list, 1999-2014, $\Phi_D \in [5,15]$')

ax = fig.add_subplot(224)
#milanstat, bins= pd.qcut(omni.milanlong, 10, retbins=True)
#milanbincenter = [(a + b) /2 for a,b in zip(bins[:-1], bins[1:])]
#sgroup = sophiexp.groupby([pd.cut(sophiexp.tilt, bins=np.array([-35,-10,10,35])), \
#                pd.qcut(sophiexp.milanlong, 10)])
#ogroup = omni.groupby([pd.cut(omni.tilt, bins=np.array([-35,-10,10,35])), \
#                pd.qcut(omni.milanlong, 10)])
res = ogroup2.milanlong.median()
ax.plot(bybincenter, np.array(res[0:11]), label='tilt < -10')
ax.plot(bybincenter, np.array(res[11:22]), label='|tilt| < 10')
ax.plot(bybincenter, np.array(res[22:33]), label='tilt > 10')
ax.legend()
ax.set_xlabel('IMF By')
ax.set_ylabel('$\Phi_D$')
ax.set_title('Median $\Phi_D$, 1999-2014, $\Phi_D \in [5,15]$')

#####################333

fig = plt.figure(figsize=(10,15))
ax = fig.add_subplot(311) 
res = sgroup.substorm.sum()
ax.plot(bybincenter, np.array(res[0:11]), label='tilt < -10')
ax.plot(bybincenter, np.array(res[11:22]), label='|tilt| < 10')
ax.plot(bybincenter, np.array(res[22:33]), label='tilt > 10')
ax.legend()
ax.set_xlabel('IMF By')
ax.set_ylabel('\# onsets')
ax.set_title('# substorm onset, sophie-75, 1999-2014')

ax = fig.add_subplot(312) 
res = ogroup.tilt.count()
ax.plot(bybincenter, np.array(res[0:11]), label='tilt < -10')
ax.plot(bybincenter, np.array(res[11:22]), label='|tilt| < 10')
ax.plot(bybincenter, np.array(res[22:33]), label='tilt > 10')
ax.legend()
ax.set_xlabel('IMF By')
ax.set_ylabel('# data points')
ax.set_title('Amount of observations, 1999-2014, omni 1min')

ax = fig.add_subplot(313)
res = ogroup.tilt.count() / sgroup.substorm.sum()
ax.plot(bybincenter, np.array(res[0:11]), label='tilt < -10')
ax.plot(bybincenter, np.array(res[11:22]), label='|tilt| < 10')
ax.plot(bybincenter, np.array(res[22:33]), label='tilt > 10')
ax.legend()
ax.set_xlabel('IMF By')
ax.set_ylabel('Average time between onsets [min]')
ax.set_title('Middle panel / upper panel')

#######################
#substorm list from FUV and UVI
fuvlist = pd.read_csv('./jone_data/merged_substormlist.csv')
fuvlist.index = pd.to_datetime(fuvlist['Unnamed: 0'])
#Combine omni and sophie list
fuvlist.loc[:,'tilt'] = dipole.dipole_tilt(fuvlist.index)
omni2 = omni.reindex(index=fuvlist.index, method='nearest', tolerance='30sec')
fuvlist.loc[:,'bylong'] = omni2.bylong
fuvlist.loc[:,'milanlong'] = omni2.milanlong
fuvlist.loc[:,'substorm'] = True
bybins = np.append(np.append([-50],np.linspace(-9,9,10)),[50])
bybincenter = np.linspace(-10,10,11)
sgroup2 = fuvlist.groupby([pd.cut(fuvlist.tilt, bins=np.array([-35,-10,10,35])), \
                pd.cut(fuvlist.bylong, bins=bybins)])
#####################333

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(211) 
res = sgroup2.substorm.sum()
ax.plot(bybincenter, np.array(res[0:11]), label='tilt < -10')
ax.plot(bybincenter, np.array(res[11:22]), label='|tilt| < 10')
ax.plot(bybincenter, np.array(res[22:33]), label='tilt > 10')
ax.set_xlabel('IMF By')
ax.set_ylabel('# onsets')
ax.set_title('# substorm onset, combined FUV lists after Jan-1999')
sgroup3 = fuvlist.groupby([pd.cut(fuvlist.tilt, bins=np.array([-35,35])), \
                pd.cut(fuvlist.bylong, bins=bybins)])
res = sgroup3.substorm.sum()
ax.plot(bybincenter, np.array(res)/3., label='all tilt/3', linewidth=2)
ax.legend()

ax = fig.add_subplot(212) 
res = sgroup.substorm.sum()
ax.plot(bybincenter, np.array(res[0:11]), label='tilt < -10')
ax.plot(bybincenter, np.array(res[11:22]), label='|tilt| < 10')
ax.plot(bybincenter, np.array(res[22:33]), label='tilt > 10')
ax.set_xlabel('IMF By')
ax.set_ylabel('# onsets')
ax.set_title('# substorm onset, sophie-75, 1999-2014')
sgroup3 = sophiexp.groupby([pd.cut(sophiexp.tilt, bins=np.array([-35,35])), \
                pd.cut(sophiexp.bylong, bins=bybins)])
res = sgroup3.substorm.sum()
ax.plot(bybincenter, np.array(res)/3., label='all tilt/3')
ax.legend()