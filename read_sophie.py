#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:39:28 2020

@author: jone
"""

#Read sophie
import pandas as pd

s=pd.read_csv('./data_jone/sophie75.txt',delimiter=' ', header=10)
s.loc[:,'datetime'] = pd.to_datetime(s.DATE, format='%Y/%m/%d-%H:%M:%S')
s.index = s.datetime
s.loc[:,'ssphase'] = s['UTC']
s.loc[:,'SMUflag'] = s['-']
s = s.drop(['DATE','UTC','-','PHASE','-.1','FLAG','datetime'], axis=1)
s.to_hdf('./data_jone/sophie75.h5',key='data')
