#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:59:04 2020

@author: jone
"""


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:39:28 2020

@author: jone
"""

#Read sophie
import pandas as pd

s=pd.read_csv('./jone_data/20200327-18-39-substorms.csv')
s.loc[:,'datetime'] = pd.to_datetime(s.Date_UTC, format='%Y-%m-%d %H:%M:%S')
s.index = s.datetime
s = s.drop(['Date_UTC','datetime'], axis=1)
s.to_hdf('./jone_data/sml_substorm_1995-2015.h5',key='data')
