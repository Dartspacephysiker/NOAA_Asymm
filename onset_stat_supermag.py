#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:58:18 2020

@author: jone
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 06:52:59 2020

@author: jone
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dipole
import datetime as dt

#substorm onset statistics and IMF By

#Load omni data
fromDate = '"1996-01-01 00:00"' # from (and inluding)
toDate = '"2016-01-01 00:00"' # to (not including)
omnifile = '/home/jone/Documents/Dropbox/science/omni/omni_interp_1min_1996-2017.h5'
# columns = ['F', 'BX_GSE', 'BY_GSM', 'BZ_GSM',
#    'flow_speed', 'Vx', 'Vy', 'Vz',
#    'proton_density', 'T', 'Pressure', 'E', 'Beta', 'Mach_num',
#    'AL_INDEX', 'AU_INDEX', 'SYM_D', 'SYM_H', 'ASY_D', 'ASY_H',
#    'PC_N_INDEX']
#columns = ['Bx', 'By', 'Bz','V']
#omni = pd.read_hdf(omnifile,where='index>={}&index<{}'.format(fromDate,toDate),columns=columns)
#omni = omni.rename(columns={"F":"B","BX_GSE":"Bx","BY_GSM":"By","BZ_GSM":"Bz","flow_speed":"V"})

######################3
#Make plot to directly compare positive and negative IMF By
#First, sophie list
supermag = pd.read_hdf('./jone_data/sml_substorm_1995-2015.h5')
columns = ['Bx', 'By', 'Bz','V']
omni = pd.read_hdf(omnifile,where='index>="' + (supermag.index[0]-dt.timedelta(days=1)).strftime("%Y-%m-%d %H:%M") + \
                   '"&index<"' + (supermag.index[-1]+dt.timedelta(days=1)).strftime("%Y-%m-%d %H:%M") + '"',columns=columns)
bypos = (omni.By>0)# & (np.abs(omni.BZ_GSM)<np.abs(omni.BY_GSM))
omni.loc[:,'bypos'] = omni.By[bypos]
byneg = (omni.By<0)# & (np.abs(omni.BZ_GSM)<np.abs(omni.BY_GSM))
omni.loc[:,'byneg'] = omni.By[byneg]    
omni.loc[:,'milan'] = 3.3e5 * (omni['V']*1000.)**(4./3)  * (np.sqrt(omni.By**2 + omni.Bz**2)) * 1e-9 * \
                        np.sin(np.abs(np.arctan2(omni['By'],omni['Bz']))/2.)**(4.5) * 0.001
milanpos = 3.3e5 * (omni['V']*1000.)**(4./3)  * (np.sqrt(omni.bypos**2 + omni.Bz**2)) * 1e-9 * \
                        np.sin(np.abs(np.arctan2(omni['bypos'],omni['Bz']))/2.)**(4.5) * 0.001                        
milanneg = 3.3e5 * (omni['V']*1000.)**(4./3)  * (np.sqrt(omni.byneg**2 + omni.Bz**2)) * 1e-9 * \
                        np.sin(np.abs(np.arctan2(omni['byneg'],omni['Bz']))/2.)**(4.5) * 0.001
window = 60    #minutes
nobsinwindow = omni.milan.rolling(window).count()
cumsumneg = milanneg.rolling(window,min_periods=1).sum()                                            
cumsumpos = milanpos.rolling(window,min_periods=1).sum()
omni.loc[:,'bxlong'] = omni.Bx.rolling(window,min_periods=1).mean()
omni.loc[:,'bzlong'] = omni.Bz.rolling(window,min_periods=1).mean()
omni.loc[:,'bylong'] = omni.By.rolling(window,min_periods=1).mean()
bxlim = 200
usepos = ((cumsumpos>2.*cumsumneg) & (nobsinwindow==window) & (np.abs(omni.bxlong)<bxlim)) | ((cumsumneg.isnull()) & (np.invert(cumsumpos.isnull())) & (nobsinwindow==window) & (np.abs(omni.bxlong)<bxlim))
useneg = ((cumsumneg>2.*cumsumpos) & (nobsinwindow==window) & (np.abs(omni.bxlong)<bxlim)) | ((cumsumpos.isnull()) & (np.invert(cumsumneg.isnull())) & (nobsinwindow==window) & (np.abs(omni.bxlong)<bxlim))                                     
omni.loc[:,'usepos'] = usepos
omni.loc[:,'useneg'] = useneg
omni.loc[:,'milanlong'] = omni.milan.rolling(window, min_periods=window, center=False).mean()    #average IMF data
supermag.loc[:,'tilt'] = dipole.dipole_tilt(supermag.index)
omni.loc[:,'tilt'] = dipole.dipole_tilt(omni.index)
omni_supermag = omni.reindex(index=supermag.index, method='nearest', tolerance='30sec')
supermag.loc[:,'bylong'] = omni_supermag.bylong
supermag.loc[:,'milanlong'] = omni_supermag.milanlong
supermag.loc[:,'substorm'] = True
#bybins = np.append(np.append([-50],np.linspace(-9,9,10)),[50])
#bybincenter = np.linspace(-10,10,11)

#plot substorm occurrence from the sophie-75 list
fig = plt.figure(figsize=(10,15))

#Negative dipole tilt
ax = fig.add_subplot(321)
tiltmin = -35
tiltmax = -10
usepos = (supermag.tilt>=tiltmin) & (supermag.tilt<=tiltmax) & (supermag.bylong>0)
useneg = (supermag.tilt>=tiltmin) & (supermag.tilt<=tiltmax) & (supermag.bylong<0)
supermagpos = supermag[usepos].copy()
supermagneg = supermag[useneg].copy()
omnipos = (omni.tilt>=tiltmin) & (omni.tilt<=tiltmax) & (omni.bylong>0)
omnineg = (omni.tilt>=tiltmin) & (omni.tilt<=tiltmax) & (omni.bylong<0)
bybins = np.linspace(0,10,6)
bybincenter = np.linspace(1,9,5)
sgrouppos = supermagpos.groupby(pd.cut(supermagpos.bylong, bins=bybins))
ogrouppos = omni[omnipos].groupby(pd.cut(omni[omnipos].bylong, bins=bybins))
sgroupneg = supermagneg.groupby(pd.cut(np.abs(supermagneg.bylong), bins=bybins))
ogroupneg = omni[omnineg].groupby(pd.cut(np.abs(omni[omnineg].bylong), bins=bybins))
respos = sgrouppos.substorm.sum()
resneg = sgroupneg.substorm.sum()
ax.plot(bybincenter, np.array(respos), color='blue',alpha=0.4, label='IMF By positive')
ax.plot(bybincenter, np.array(resneg), color='orange',alpha=0.4, label='IMF By nagative')
ax.legend()
ax.set_xlabel('|IMF By| [nT]')
ax.set_ylabel('# onsets')
ax.set_title('%3i < tilt < %3i' % (tiltmin, tiltmax))
ax = fig.add_subplot(322)
respos = ogrouppos.tilt.count() / sgrouppos.substorm.sum()
resneg = ogroupneg.tilt.count() / sgroupneg.substorm.sum()
ax.plot(bybincenter, np.array(respos), color='blue',alpha=0.4, label='IMF By positive')
ax.plot(bybincenter, np.array(resneg), color='orange',alpha=0.4, label='IMF By nagative')
ax.legend()
ax.set_xlabel('|IMF By| [nT]')
ax.set_ylabel('Avg. time between onsets [min]')
ax.set_title('%3i < tilt < %3i' % (tiltmin, tiltmax))

#Positive dipole tilt
ax = fig.add_subplot(323)
tiltmin = 10
tiltmax = 35
usepos = (supermag.tilt>=tiltmin) & (supermag.tilt<=tiltmax) & (supermag.bylong>0)
useneg = (supermag.tilt>=tiltmin) & (supermag.tilt<=tiltmax) & (supermag.bylong<0)
supermagpos = supermag[usepos].copy()
supermagneg = supermag[useneg].copy()
omnipos = (omni.tilt>=tiltmin) & (omni.tilt<=tiltmax) & (omni.bylong>0)
omnineg = (omni.tilt>=tiltmin) & (omni.tilt<=tiltmax) & (omni.bylong<0)
bybins = np.linspace(0,10,6)
bybincenter = np.linspace(1,9,5)
sgrouppos = supermagpos.groupby(pd.cut(supermagpos.bylong, bins=bybins))
ogrouppos = omni[omnipos].groupby(pd.cut(omni[omnipos].bylong, bins=bybins))
sgroupneg = supermagneg.groupby(pd.cut(np.abs(supermagneg.bylong), bins=bybins))
ogroupneg = omni[omnineg].groupby(pd.cut(np.abs(omni[omnineg].bylong), bins=bybins))
respos = sgrouppos.substorm.sum()
resneg = sgroupneg.substorm.sum()
ax.plot(bybincenter, np.array(respos), color='blue',alpha=0.4, label='IMF By positive')
ax.plot(bybincenter, np.array(resneg), color='orange',alpha=0.4, label='IMF By nagative')
ax.legend()
ax.set_xlabel('|IMF By| [nT]')
ax.set_ylabel('# onsets')
ax.set_title('%3i < tilt < %3i' % (tiltmin, tiltmax))
ax = fig.add_subplot(324)
respos = ogrouppos.tilt.count() / sgrouppos.substorm.sum()
resneg = ogroupneg.tilt.count() / sgroupneg.substorm.sum()
ax.plot(bybincenter, np.array(respos), color='blue',alpha=0.4, label='IMF By positive')
ax.plot(bybincenter, np.array(resneg), color='orange',alpha=0.4, label='IMF By nagative')
ax.legend()
ax.set_xlabel('|IMF By| [nT]')
ax.set_ylabel('Avg. time between onsets [min]')
ax.set_title('%3i < tilt < %3i' % (tiltmin, tiltmax))

#All dipole tilt
ax = fig.add_subplot(325)
tiltmin = -35
tiltmax = 35
usepos = (supermag.tilt>=tiltmin) & (supermag.tilt<=tiltmax) & (supermag.bylong>0)
useneg = (supermag.tilt>=tiltmin) & (supermag.tilt<=tiltmax) & (supermag.bylong<0)
supermagpos = supermag[usepos].copy()
supermagneg = supermag[useneg].copy()
omnipos = (omni.tilt>=tiltmin) & (omni.tilt<=tiltmax) & (omni.bylong>0)
omnineg = (omni.tilt>=tiltmin) & (omni.tilt<=tiltmax) & (omni.bylong<0)
bybins = np.linspace(0,10,6)
bybincenter = np.linspace(1,9,5)
sgrouppos = supermagpos.groupby(pd.cut(supermagpos.bylong, bins=bybins))
ogrouppos = omni[omnipos].groupby(pd.cut(omni[omnipos].bylong, bins=bybins))
sgroupneg = supermagneg.groupby(pd.cut(np.abs(supermagneg.bylong), bins=bybins))
ogroupneg = omni[omnineg].groupby(pd.cut(np.abs(omni[omnineg].bylong), bins=bybins))
respos = sgrouppos.substorm.sum()
resneg = sgroupneg.substorm.sum()
ax.plot(bybincenter, np.array(respos), color='blue',alpha=0.4, label='IMF By positive')
ax.plot(bybincenter, np.array(resneg), color='orange',alpha=0.4, label='IMF By nagative')
ax.legend()
ax.set_xlabel('|IMF By| [nT]')
ax.set_ylabel('# onsets')
ax.set_title('%3i < tilt < %3i' % (tiltmin, tiltmax))
ax = fig.add_subplot(326)
respos = ogrouppos.tilt.count() / sgrouppos.substorm.sum()
resneg = ogroupneg.tilt.count() / sgroupneg.substorm.sum()
ax.plot(bybincenter, np.array(respos), color='blue',alpha=0.4, label='IMF By positive')
ax.plot(bybincenter, np.array(resneg), color='orange',alpha=0.4, label='IMF By nagative')
ax.legend()
ax.set_xlabel('|IMF By| [nT]')
ax.set_ylabel('Avg. time between onsets [min]')
ax.set_title('%3i < tilt < %3i' % (tiltmin, tiltmax))
ax.text(-3, 1060, 'SuperMAG list (old)', fontsize=20)
