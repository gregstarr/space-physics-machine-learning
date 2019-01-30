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
from datetime import datetime
import xarray as xr
import pandas as pd
from io import StringIO
import time
import os

#%% CONSTANTS

# download directory
DOWNLOAD_DIR = "data"
NINTERVALS = 10
START_YEAR = 2001
END_YEAR = 2018

MAX_TRIES = 5
SLEEP_TIME = 5
DATEFORMAT = "%Y-%m-%dT%H:%M:%S.000Z"

# request string constants
GET_STATION_LIST = "http://supermag.jhuapl.edu/mag/lib/services/inventory.php"
STATION_PARAMS = {"service"     : "inventory",
                  "start"       : None,
                  "interval"    : None}

GET_DATA = "http://supermag.jhuapl.edu/mag/lib/services/"
DATA_PARAMS = {"?user"      : "gregstarr",
               "start"      : None,
               "interval"   : None,
               "service"    : "mag",
               "stations"   : None,
               "delta"      : "none",
               "baseline"   : "none",
               "options"    : "mlt sza decl",
               "fmt"        : "csv"}

def date_range(start, end, intv):
    diff = (end  - start ) / intv
    for i in range(intv):
        yield start + diff * i
    yield end
    
    
def getDataForStation(data_params):
    tries = 0
    while tries < MAX_TRIES:
        try:
            data_rq = requests.get(GET_DATA, data_params, timeout=10)
            break
        except:
            tries += 1
            time.sleep(SLEEP_TIME)
    if tries == MAX_TRIES:
        print("couldn't get data for {} {}".format(data_params['stations'], data_params['start']))
        return
    buffer = StringIO(data_rq.text)
    df = pd.read_csv(buffer)
    times = pd.to_datetime(df.Date_UTC)
    df.index = times
    return df.drop(columns=['Date_UTC', 'IAGA'])

def getAvailableStations(station_params):
    tries = 0
    while tries < MAX_TRIES:
        try:
            station_rq = requests.get(GET_STATION_LIST, station_params, timeout=10)
            break
        except:
            tries += 1
            time.sleep(SLEEP_TIME)
    if tries == MAX_TRIES:
        print("couldn't get stations for {}".format(station_params['start']))
        return
    station_list = station_rq.json()['stations']
    return station_list
    
def getDataForInterval(start_date, hours, minutes):
    data = xr.Dataset()
    
    station_params = STATION_PARAMS.copy()
    station_params['start'] = start_date.strftime(DATEFORMAT)
    station_params['interval'] = "{}:{}".format(hours, minutes)
    
    # get stations for interval
    station_list = getAvailableStations(station_params)
    print("{} stations for interval starting {}".format(len(station_list),
          start_date.strftime("%Y-%m-%d")))
    
    for station in station_list:
        data_params = DATA_PARAMS.copy()
        data_params['start'] = start_date.strftime(DATEFORMAT)
        data_params['interval'] = "{}:{}".format(hours, minutes)
        data_params['stations'] = station
        
        print("{} / {}: {}".format(station_list.index(station)+1,
              len(station_list), station))
        
        df = getDataForStation(data_params)
        data[station] = df
    
    return data

def downloadDataToFile(fn, start_date, end_date, nintervals):
    dataset = xr.Dataset()
    
    dates = list(date_range(start_date, end_date, nintervals))
    for i in range(nintervals):
        print("Interval {} of {} for {}".format(i+1, nintervals, start_date.year))
        tdelt = dates[i+1] - dates[i]
        hours = int(tdelt.total_seconds()//3600)
        minutes = int((tdelt.total_seconds() - 3600*hours)//60)
        data = getDataForInterval(dates[i], hours, minutes)
        dataset = xr.merge([dataset, data])
        
    print("saving {}".format(fn))
    dataset.to_netcdf(fn)

def downloadData():
    os.chdir(DOWNLOAD_DIR)
    for yr in range(START_YEAR, END_YEAR):
        start_date = datetime(yr, 1, 1, 0, 0, 0)
        end_date = datetime(yr+1, 1, 1, 0, 0, 0)
        fn = "mag_data_{}".format(yr)
        downloadDataToFile(fn, start_date, end_date, NINTERVALS)
        
if __name__ == "__main__":
    downloadData()