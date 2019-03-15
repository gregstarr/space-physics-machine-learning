"""
single station CNN, just to see if a nearby station could be used to classify whether a substorm is coming in the
next 30 minutes or so

create dataset:
    - Positive: for dataset_size/2 random substorms, grab closest station, substorm occurs on a random time index between
                0 and Tfinal minutes, the input data is the mag time series from t = T0 to 0
    - Negative: find a Tfinal minute time interval without any substorm in it, then grab a mag time series from a random
                magnetometer from t = T0 to 0, do this dataset_size/2 times
    - Other possibilities:
        - time output [0,1]: 0 -> ss occurred at the beginning of the interval, 1 -> end of interval
        - one-hot time labels: Tfinal+1 outputs, the first one means no SS, the rest of them are the minute that
            the SS occurs in
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import xarray as xr
import os
os.chdir("C:\\Users\\Greg\\code\\space-physics-machine-learning")


def find_closest_stations(mag_data, loc):
    # find closest N_STATIONS stations with finite data
    distances = np.mean(np.sum((mag_data[:, :, :2] - loc) ** 2, axis=2), axis=1)
    if np.all(np.isnan(distances)):
        return
    distances[np.isnan(distances)] = np.nanmax(distances)
    sort_idx = np.argsort(distances)
    return sort_idx


T0 = 128
Tfinal = 60
N_STATIONS = 5

substorms = pd.read_csv("./data/substorms_2000_2018.csv")
substorms.index = pd.to_datetime(substorms.Date_UTC)

kde = KernelDensity(bandwidth=.5)
kde.fit(substorms[["MLT", "MLAT"]].values)

X = []
y = []
time = []
one_hot_time = []
for yr in range(2000, 2019):
    print(yr)
    X_yr = []
    y_yr = []
    time_yr = []
    one_hot_time_yr = []
    year = str(yr)

    ss = substorms[year]
    mag_file = "./data/mag_data_{}.nc".format(year)
    dataset = xr.open_dataset(mag_file).sel(dim_1=['MLT', 'MLAT', 'N', 'E', 'Z'])
    dates = dataset.Date_UTC.values
    data = dataset.to_array().values  # (stations x time x component)

    # find substorms
    for i in range(ss.shape[0]):
        ss_loc = ss.iloc[i][["MLT", "MLAT"]].astype(float)
        date = np.datetime64(ss.index[i])
        ss_interval_index = np.random.randint(0, Tfinal)
        ss_date_index = np.argmax(np.datetime64(date) == dates)
        mag_data = data[:, ss_date_index - ss_interval_index - T0:ss_date_index - ss_interval_index]

        if mag_data.shape[1] != T0:
            print("Not enough mag data", ss.index[i], ss_loc.ravel())
            continue

        # components = MLT - MLAT - N - E - Z
        sort_idx = find_closest_stations(mag_data, ss_loc.values)
        if sort_idx is None:
            continue
        sorted_mag_data = mag_data[sort_idx[:N_STATIONS], :, 2:]
        X_yr.append(sorted_mag_data)
        y_yr.append(1)
        time_yr.append(ss_interval_index / Tfinal)
        one_hot = np.zeros(Tfinal + 1)
        one_hot[ss_interval_index+1] = 1
        one_hot_time_yr.append(one_hot)
        i += 1

    n_positive_examples = len(X_yr)
    print("{} substorms from {}".format(n_positive_examples, yr))
    i = 0
    while i < n_positive_examples:
        random_date_index = np.random.randint(T0 + Tfinal, dates.shape[0] - Tfinal)
        if len(ss.iloc[random_date_index: random_date_index+Tfinal]) != 0:
            continue
        mag_data = data[:, random_date_index - T0:random_date_index]
        random_location = kde.sample()[0]
        random_location[0] = np.mod(random_location[0], 24)
        sort_idx = find_closest_stations(mag_data, random_location)
        if sort_idx is None:
            continue
        sorted_mag_data = mag_data[sort_idx[:N_STATIONS], :, 2:]
        if sorted_mag_data.shape != (N_STATIONS, T0, 3):
            continue
        X_yr.append(sorted_mag_data)
        y_yr.append(0)
        time_yr.append(0)
        one_hot = np.zeros(Tfinal + 1)
        one_hot[0] = 1
        one_hot_time_yr.append(one_hot)
        i += 1

    X.append(np.stack(X_yr, axis=0))
    y.append(np.array(y_yr))
    time.append(np.array(time_yr))
    one_hot_time.append(np.stack(one_hot_time_yr, axis=0))

X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)
time = np.concatenate(time, axis=0)
one_hot_time = np.concatenate(one_hot_time, axis=0)

np.savez("./data/closest_station_data.npz", X=X, y=y, time=time, one_hot_time=one_hot_time)
