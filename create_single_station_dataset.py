"""
single station CNN, just to see if a nearby station could be used to classify whether a substorm is coming in the
next 30 minutes or so

create dataset:
    - Positive: for dataset_size/2 random substorms, grab closest station, substorm occurs on a random time index between
                0 and Tfinal minutes, the input data is the mag time series from t = T0 to 0
    - Negative: find a Tfinal minute time interval without any substorm in it, then grab a mag time series from a random
                magnetometer from t = T0 to 0, do this dataset_size/2 times
"""

import numpy as np
import pandas as pd
import datetime
import xarray as xr

dataset_size = 500
T0 = 96
Tfinal = 30
year = '2010'

substorms = pd.read_csv("./data/substorms_2000_2018.csv")
substorms.index = pd.to_datetime(substorms.Date_UTC)

ss = substorms[year]
mag_file = "./data/mag_data_{}.nc".format(year)
mag_data = xr.open_dataset(mag_file)
stations_to_drop = [station for station in mag_data if np.any(np.all(np.isnan(mag_data[station]), axis=0))]
mag_data = mag_data.drop(stations_to_drop)

# initialize training data arrays
X = np.empty((dataset_size, T0, 3))
y = np.empty(dataset_size)

# find positive exampels
idx = np.arange(ss.shape[0])
np.random.shuffle(idx)
nstorms = 0
i = 0
while nstorms < dataset_size//2:
    storm_index = idx[i]
    i += 1
    ss_info = ss.iloc[storm_index]
    date = datetime.datetime.strptime(ss_info.Date_UTC, '%Y-%m-%d %H:%M:%S')
    ss_time_index = np.random.randint(0, Tfinal)
    input_beginning = date - datetime.timedelta(minutes=ss_time_index) - datetime.timedelta(minutes=T0)
    input_end = date - datetime.timedelta(minutes=ss_time_index+1)
    storm_data = mag_data.sel(Date_UTC=slice(input_beginning, input_end))
    if len(storm_data.Date_UTC) != T0:
        print("Skip 1")
        continue
    min_dist = 180
    min_st = "Station"
    for st in storm_data:
        if np.any(np.isnan(storm_data[st])):
            continue
            print("Skip 2")
        dist = np.mean(np.sum((storm_data[st].sel(dim_1=['MLT', 'MLAT']) - ss_info[['MLT', 'MLAT']].values)**2, axis=1)).values
        if dist < min_dist:
            min_dist = dist
            min_st = st
    if min_dist > 20:
        continue
        print("Skip 3")
    X[nstorms] = storm_data[min_st].sel(dim_1=['N', 'E', 'Z'])
    y[nstorms] = 1
    nstorms += 1
    print(ss_info.Date_UTC, min_st, min_dist)

# find negative examples
initial_date = datetime.datetime(int(year), 1, 1) + datetime.timedelta(minutes=T0)
stations = [st for st in mag_data]
nsamples = 0
while nsamples < dataset_size//2:
    random_date = initial_date + datetime.timedelta(minutes=np.random.randint(0, 60*24*365 - Tfinal))
    if len(ss[slice(random_date, random_date+datetime.timedelta(minutes=Tfinal))]) != 0:
        continue
    st = np.random.choice(stations)
    data = mag_data[st].sel(dim_1=['N', 'E', 'Z']).sel(Date_UTC=slice(random_date-datetime.timedelta(minutes=T0-1), random_date))
    if data.shape != (T0, 3):
        continue
    if np.any(np.isnan(data)):
        continue
    X[nstorms+nsamples] = data.values
    y[nstorms+nsamples] = 0
    nsamples += 1
    print(random_date)

np.savez("./data/single_station_data.npz", X=X, y=y)