#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 08:06:30 2018

@author: jone
"""
import datetime as dt
import pandas as pd
import os
os.sys.path.append('/home/jone/Documents/Dropbox/')
from pyrkeland.earth import synth_values
from pyrkeland.earth import subsol
import numpy as np
#os.sys.path.append('/Home/stud8/jre081/Documents/Dropbox/')

#Calculate dipole tilt angle baseed on IGRF coefficients (Obtained from Kalle)

def tilt(dates):
    #dates is a python datetime object, or a list/array of such
    PYRKELANDDATAPATH = os.sys.path[-1]+'pyrkeland/datafiles/magnetic_field_models/'
    g, h = synth_values.read_shc(PYRKELANDDATAPATH + 'IGRF12.shc')

    #interpolate to desired times
    index = list(g.index) + list(dates)   
    g = g.reindex(index).sort_index().interpolate(method = 'time').loc[dates]
    g = g.groupby(g.index).first() # remove duplicate entries when dates has an entry at the same time as IGRF coef is updated
    h = h.reindex(index).sort_index().interpolate(method = 'time').loc[dates]
    h = h.groupby(h.index).first() # remove duplicate entries when dates has an entry at the same time as IGRF coef is updated

    #The coeficients defining the orientation of the centered dipole
    g10 = g[1][0]
    g11 = g[1][1]
    h11 = h[1][1]
    dipoledir = np.array([g11,h11,g10])*-1. #reverse to point opposite to the direction of the field-lines (out of northern hemisphere)
    dipolemags = np.linalg.norm(dipoledir,axis=0)    
    #Now we relate this direction to the subsolar point to get the dipole tilt angle
    sslat, sslon = subsol.subsol_vector(dates)
    #Transform subsolar locations into to cartesian coordinates
    xss = np.sin(np.pi/2.-np.deg2rad(sslat))*np.cos(np.deg2rad(sslon))
    yss = np.sin(np.pi/2.-np.deg2rad(sslat))*np.sin(np.deg2rad(sslon))
    zss = np.cos(np.pi/2.-np.deg2rad(sslat))
    ssdir = np.array([xss,yss,zss])

    cos_cotilt = np.sum(np.multiply(dipoledir,ssdir),axis=0)/dipolemags
    return 90. - np.degrees(np.arccos(cos_cotilt))
