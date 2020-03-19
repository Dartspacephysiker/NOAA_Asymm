#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 07:04:54 2020

@author: jone
"""

#Do analysis on the data from finn that has been put into pandas dataframe in read_finn.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

det = 3
hemi = 'nord' #south eller nord

if det == 1:
    channel = 'TED_electron_0.15-20kev'
    logrange = (0.3,2.8)
elif det == 2:
    channel = 'MEPED_proton_0.15-20kev'        
elif det == 3:
    channel = 'MEPED_electron_>30kev'
    logrange = (4,9) # (0.3,2.8)
elif det == 4:
    channel = 'MEPED_proton_20-80kev' 
    logrange = (2,7) # (0.3,2.8)
            
year = 2003
satmin = 15
satmax = 17
fname = './jone_data/'+str(year)+'_'+hemi+'_noaa'+str(satmin)+'-'+str(satmax) + '_' + channel + '.h5'
noaa = pd.read_hdf(fname)
noaa.loc[:,'mlt'] = (noaa.mlteq + noaa.mltpol)/(2.)
noaa.loc[:,'mlat'] = (noaa.mlateq + noaa.mlatpol)/(2.)

tiltmin = -35
tiltmax = -10
bylim = 2
median_milanlong = noaa.milanlong.median()
mltmin = 0
mltmax = 12
histscale =200


############################################3
#Plotting, negative dipole tilts
fig = plt.figure(figsize=(11,11))  #summary figure
gs = gridspec.GridSpec(2, 2)
fig2 = plt.figure(figsize=(11,11))  #summary figure

#Negative By, high driving
ax1 = fig.add_subplot(gs[0,0])
if mltmin > mltmax:
    use = ((noaa.milanlong > median_milanlong) & (noaa.bylong < -bylim) & (noaa.tilt > tiltmin) & \
        (noaa.tilt < tiltmax) & (noaa.mlt > mltmin)) | \
        ((noaa.milanlong > median_milanlong) & (noaa.useneg) & (noaa.tilt > tiltmin) & \
        (noaa.tilt < tiltmax) & (noaa.mlt < mltmax))
else:
    use = (noaa.milanlong > median_milanlong) & (noaa.bylong < -bylim) & (noaa.tilt > tiltmin) & \
        (noaa.tilt < tiltmax) & (noaa.mlt > mltmin) & (noaa.mlt < mltmax)
ax1.hist(np.log10(noaa.ovalflux[use]), bins=30, range=logrange)
ax1.vlines(np.log10(noaa.ovalflux[use]).median(),0,histscale)
ax1.set_xlabel='log10 numberflux inside oval'
ax1.text(logrange[0]+0.1, histscale-20, 'Median flux: %4.2f' % np.log10(noaa.ovalflux[use]).median())
ax1.text(logrange[0]+0.1, histscale-30, 'Median coupling: %4.2f' % noaa.milanlong[use].median())
ax1.set_title('Negative IMF By, high driving, NH')
ax1.set_ylim((0,histscale))
ax1.set_ylabel('#')
ax21 = fig2.add_subplot(gs[0,0])
ax21.hist(np.abs(noaa.mlatpol[use]), bins=30, range=(60,80), alpha=0.5, color='blue')
ax21.hist(np.abs(noaa.mlateq[use]), bins=30, range=(60,80), alpha=0.5, color='orange')
medianpol = np.abs(noaa.mlatpol[use]).median()
medianeq = np.abs(noaa.mlateq[use]).median()
ax21.vlines([medianpol,medianeq], 0, histscale)
ax21.text(61, histscale-20, 'Median poleward:       %4.2f' % medianpol, color='blue')
ax21.text(61, histscale-40, 'Median equatorward: %4.2f' % medianeq, color='orange')
ax21.set_title('Negative IMF By, high driving, NH')
ax21.set_ylim((0,histscale))
ax21.set_ylabel('#')
                
#Positive By, high driving
ax2 = fig.add_subplot(gs[0,1])
if mltmin > mltmax:
    use = ((noaa.milanlong > median_milanlong) & (noaa.bylong > bylim) & (noaa.tilt > tiltmin) & \
        (noaa.tilt < tiltmax) & (noaa.mlt > mltmin)) | \
        ((noaa.milanlong > median_milanlong) & (noaa.useneg) & (noaa.tilt > tiltmin) & \
        (noaa.tilt < tiltmax) & (noaa.mlt < mltmax))
else:
    use = (noaa.milanlong > median_milanlong) & (noaa.bylong > bylim) & (noaa.tilt > tiltmin) & \
        (noaa.tilt < tiltmax) & (noaa.mlt > mltmin) & (noaa.mlt < mltmax)
ax2.hist(np.log10(noaa.ovalflux[use]), bins=30, range=logrange)
ax2.vlines(np.log10(noaa.ovalflux[use]).median(),0,histscale)
ax2.set_xlabel='log10 numberflux inside oval'
ax2.text(logrange[0]+0.1, histscale-20, 'Median flux: %4.2f' % np.log10(noaa.ovalflux[use]).median())
ax2.text(logrange[0]+0.1, histscale-30, 'Median coupling: %4.2f' % noaa.milanlong[use].median())
ax2.set_title('Positive IMF By, high driving, NH')
ax2.set_ylim((0,histscale))
ax2.set_ylabel('#')
ax22 = fig2.add_subplot(gs[0,1])
ax22.hist(np.abs(noaa.mlatpol[use]), bins=30, range=(60,80), alpha=0.5, color='blue')
ax22.hist(np.abs(noaa.mlateq[use]), bins=30, range=(60,80), alpha=0.5, color='orange')
medianpol = np.abs(noaa.mlatpol[use]).median()
medianeq = np.abs(noaa.mlateq[use]).median()
ax22.vlines([medianpol,medianeq], 0, histscale)
ax22.text(61, histscale-20, 'Median poleward:       %4.2f' % medianpol, color='blue')
ax22.text(61, histscale-40, 'Median equatorward: %4.2f' % medianeq, color='orange')
ax22.set_title('Positive IMF By, high driving, NH')
ax22.set_ylim((0,histscale))
ax22.set_ylabel('#')
               
#Negative By, low driving
ax3 = fig.add_subplot(gs[1,0])
#ax3.set_xlabel('log flux inside oval')
if mltmin > mltmax:
    use = ((noaa.milanlong < median_milanlong) & (noaa.bylong < -bylim) & (noaa.tilt > tiltmin) & \
        (noaa.tilt < tiltmax) & (noaa.mlt > mltmin)) | \
        ((noaa.milanlong > median_milanlong) & (noaa.useneg) & (noaa.tilt > tiltmin) & \
        (noaa.tilt < tiltmax) & (noaa.mlt < mltmax))
else:
    use = (noaa.milanlong < median_milanlong) & (noaa.bylong < -bylim) & (noaa.tilt > tiltmin) & \
        (noaa.tilt < tiltmax) & (noaa.mlt > mltmin) & (noaa.mlt < mltmax)
ax3.hist(np.log10(noaa.ovalflux[use]), bins=30, range=logrange)
ax3.vlines(np.log10(noaa.ovalflux[use]).median(),0,histscale)
ax3.set_xlabel('$\\log10 ( \\sum$ numberflux inside oval$)$')
ax3.text(logrange[0]+0.1, histscale-20, 'Median flux: %4.2f' % np.log10(noaa.ovalflux[use]).median())
ax3.text(logrange[0]+0.1, histscale-30, 'Median coupling: %4.2f' % noaa.milanlong[use].median())
ax3.set_title('Negative IMF By, low driving, NH')
ax3.set_ylim((0,histscale))
ax3.set_ylabel('#')
ax23 = fig2.add_subplot(gs[1,0])
ax23.set_xlabel('|Oval latitude| [degrees]')
ax23.hist(np.abs(noaa.mlatpol[use]), bins=30, range=(60,80), alpha=0.5, color='blue')
ax23.hist(np.abs(noaa.mlateq[use]), bins=30, range=(60,80), alpha=0.5, color='orange')
medianpol = np.abs(noaa.mlatpol[use]).median()
medianeq = np.abs(noaa.mlateq[use]).median()
ax23.vlines([medianpol,medianeq], 0, histscale)
ax23.text(61, histscale-20, 'Median poleward:       %4.2f' % medianpol, color='blue')
ax23.text(61, histscale-40, 'Median equatorward: %4.2f' % medianeq, color='orange')
ax23.set_title('Negative IMF By, low driving, NH')
ax23.set_ylim((0,histscale))
ax23.set_ylabel('#')

#Positive By, low driving
ax4 = fig.add_subplot(gs[1,1])
#ax4.set_xlabel('log flux inside oval')
if mltmin > mltmax:
    use = ((noaa.milanlong < median_milanlong) & (noaa.bylong > bylim) & (noaa.tilt > tiltmin) & \
        (noaa.tilt < tiltmax) & (noaa.mlt > mltmin)) | \
        ((noaa.milanlong > median_milanlong) & (noaa.useneg) & (noaa.tilt > tiltmin) & \
        (noaa.tilt < tiltmax) & (noaa.mlt < mltmax))
else:
    use = (noaa.milanlong < median_milanlong) & (noaa.bylong > bylim) & (noaa.tilt > tiltmin) & \
        (noaa.tilt < tiltmax) & (noaa.mlt > mltmin) & (noaa.mlt < mltmax)
ax4.hist(np.log10(noaa.ovalflux[use]), bins=30, range=logrange)
ax4.vlines(np.log10(noaa.ovalflux[use]).median(),0,500)
ax4.set_xlabel('$\\log10 ( \\sum$ numberflux inside oval$)$')
ax4.text(logrange[0]+0.1, histscale-20, 'Median flux: %4.2f' % np.log10(noaa.ovalflux[use]).median())
ax4.text(logrange[0]+0.1, histscale-30, 'Median coupling: %4.2f' % noaa.milanlong[use].median())
ax4.set_title('Positive IMF By, low driving, NH')
ax4.set_ylim((0,histscale))
ax4.set_ylabel('#')
ax24 = fig2.add_subplot(gs[1,1])
ax24.set_xlabel('|Oval latitude| [degrees]')
ax24.hist(np.abs(noaa.mlatpol[use]), bins=30, range=(60,80), alpha=0.5, color='blue')
ax24.hist(np.abs(noaa.mlateq[use]), bins=30, range=(60,80), alpha=0.5, color='orange')
medianpol = np.abs(noaa.mlatpol[use]).median()
medianeq = np.abs(noaa.mlateq[use]).median()
ax24.vlines([medianpol,medianeq], 0, histscale)
ax24.text(61, histscale-20, 'Median poleward:       %4.2f' % medianpol, color='blue')
ax24.text(61, histscale-40, 'Median equatorward: %4.2f' % medianeq, color='orange')
ax24.set_title('Positive IMF By, low driving, NH')
ax24.set_ylim((0,histscale))
ax24.set_ylabel('#')

ax = fig.add_axes([0, 0, 1, 1])  
ax.set_axis_off()
ax.set_xlim([0,1])
ax.set_ylim([0,1])    
ax.text(0.3,0.96, hemi+', %2i $<$ MLT $<$ %2i, %2i < tilt < %2i' %(mltmin, mltmax, tiltmin, tiltmax), fontsize= 20)
ax.text(0.35,0.92,channel, fontsize= 20)

savename = fname[12:-3]+'_%2i<MLT<%2i_%2i<tilt<%2i' % (mltmin, mltmax, tiltmin, tiltmax)
fig.savefig('./plots/'+savename+'.pdf', bbox_inches='tight', dpi = 250)

ax = fig2.add_axes([0, 0, 1, 1])  
ax.set_axis_off()
ax.set_xlim([0,1])
ax.set_ylim([0,1])    
ax.text(0.3,0.96, hemi+', %2i $<$ MLT $<$ %2i, %2i < tilt < %2i' %(mltmin, mltmax, tiltmin, tiltmax), fontsize= 20)
ax.text(0.35,0.92,channel, fontsize= 20)

savename = fname[12:-3]+'_%2i<MLT<%2i_%2i<tilt<%2i_latitudes' % (mltmin, mltmax, tiltmin, tiltmax)
fig2.savefig('./plots/'+savename+'.pdf', bbox_inches='tight', dpi = 250)