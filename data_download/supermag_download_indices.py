# -*- coding: utf-8 -*-
"""
Download data from the SuperMAG website. This is way easier than I thought! :)

I think the script will look something like this:
    - input start and end date
    - input stations / regions ???
    - go through days one at a time, checking on available stations and 
      downloading data from all available stations
    - save the whole thing to a big hdf5 file, oh yeah

Created on Wed Dec 26 10:03:48 2018

@author: Greg
"""

import requests
import xarray as xr
import pandas as pd
from io import StringIO
import time
import os

#%% CONSTANTS

# download directory
DOWNLOAD_DIR = "data"
NINTERVALS = 10
START_YEAR = 2000
END_YEAR = 2018

MAX_TRIES = 5
SLEEP_TIME = 5

GET_DATA = "http://supermag.jhuapl.edu/indices/lib/services/?service=indices" \
            "&user=gregstarr&start={}-01-01T00:00&end={}-12-31T23:59&ind" \
            "ices=smu,sml,sme,mlt,mlat,num,smu,sml&imf=bgsm,bgse,vgsm,vgse,p" \
            "dyn,density,newell,epsilon,clockgsm,clockgse&format=csv"

def getData(data_params):
    tries = 0
    while tries < MAX_TRIES:
        try:
            data_rq = requests.get(GET_DATA, data_params, timeout=10)
            break
        except:
            tries += 1
            time.sleep(SLEEP_TIME)
    if tries == MAX_TRIES:
        print("couldn't get data for {}".format(data_params['start']))
        return
    buffer = StringIO(data_rq.text)
    df = pd.read_csv(buffer)
    times = pd.to_datetime(df.Date_UTC)
    df.index = times
    return df.drop(columns=['Date_UTC', 'IAGA'])

        
if __name__ == "__main__":
    os.chdir(DOWNLOAD_DIR)
    for i in range(START_YEAR, END_YEAR):
        url = GET_DATA.format(i, i+1)
        print(i)
        req = requests.get(url)
        buffer = StringIO(req.text)
        df = pd.read_csv(buffer)
        times = pd.to_datetime(df.Date_UTC)
        df.index = times
        df = df.drop(columns=['Date_UTC'])
        df.to_hdf("supermag_indices_{}.hdf".format(i), 'data')
        