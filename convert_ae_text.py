# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 21:57:57 2019

@author: Greg
"""
import pandas as pd
import datetime
import os

os.chdir('data')
ae_text = "ae_index.txt"
data = []
times = []

with open(ae_text) as f:
    while True:
        line = f.readline()
        
        if not line:
            break
        
        if '|' in line:
            continue
        
        date, time, doy, ae, au, al, ao = line.split()
        
        date = datetime.datetime.strptime(date+"T"+time, "%Y-%m-%dT%H:%M:%S.%f")
        
        data.append({'ae':ae, 'au':au, 'al':al, 'ao':ao})
        times.append(date)
        
dset = pd.DataFrame(data = data, index = times)
dset.to_hdf('ae_index_2000_2018.hdf', 'data')