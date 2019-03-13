# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 23:55:39 2018

@author: Greg
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

ss_file = "20181219-00-53-substorms.csv"
ind_file  = "20181219-00-57-supermag.csv"

ss = pd.read_csv(ss_file)
ind = pd.read_csv(ind_file)

sstime = pd.to_datetime(ss.Date_UTC)
ss.index = sstime
ss.drop(columns=['Date_UTC'])
ss['ph'] = 0

indtime = pd.to_datetime(ind.Date_UTC)
ind.index = indtime
ind.drop(columns=['Date_UTC'])

epoch = np.datetime64('1970-01-01T00:00:00')
dt = np.timedelta64(1, 's')

plt.figure()
plt.plot(ind.SML)
plt.plot(ss.ph, 'r.')

"""
This should probably be a notebook
"""