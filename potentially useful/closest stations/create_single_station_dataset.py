"""
This script is used to create a dataset where the input is  a time series from a few of the closest stations. It goes
through every year of the overall dataset and creates training examples for both the positive and negative classes.

For the positive class, it selects a substorm and chooses a random "interval index" for the substorm. This number
is the time index (0-60) within the hour long prediction interval at which the substorm takes place. This is so the
network has examples of predicting substorms scattered throughout the hour long prediction interval. The script then
sorts the stations by their distance from the substorm location and gathers the 5 closest stations. It selects the
2 hours of magnetometer data before the prediction interval and creates an input example:
x = (5 stations x 128 minutes of mag data x 3 magnetic field components), y = 1.

The script also creates 2 time related labels, the first is a scalar and represents the portion of the way through
the hour long interval the substorm occurs (0 - 1), and the other treats each minute of the prediction interval as
a separate class and one-hot encodes the substorm time.

For the negative classes, the only thing to be careful of is that the locations come from the same distribution as the
substorm locations. This distribution is estimated using the sklearn kernel density estimator. The process of creating
the input data is the same as before except the output label is 0 and one of the one-hot classes is reserved for the
negative class.

NOTE: the find_closest_stations function basically disqualifies any station with any NaNs in its time interval by
assigning the distance to be equal to the maximum distance in the list.

NOTE: The distance the script uses is L2^2 of [lon, lat], so there is obviously a problem with the 0 -> 360 border.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import xarray as xr
from pymap3d.vincenty import vdist
import os
os.chdir("C:\\Users\\Greg\\code\\space-physics-machine-learning")


def find_closest_stations(mag_loc, loc):
    # calculate distance between 'loc' and the location in mag_data
    lat1 = mag_loc[:, 1]
    lon1 = mag_loc[:, 0] * 360 / 24
    lat2 = np.ones_like(lat1) * loc[1]
    lon2 = np.ones_like(lon1) * loc[0] * 360 / 24
    s, a12, a21 = vdist(lat1, lon1, lat2, lon2)
    # if for some reason everything is NaNs, exit
    if np.all(np.isnan(s)):
        return
    # if there is a single NaN in the time series for a station, the np.mean will cause that entry in 'distances' to
    # be NaN, here I make all NaNs equal to a maximum value, therefor disqualifying the station from being used.
    s[np.isnan(s)] = np.nanmax(s)
    sort_idx = np.argsort(s)
    # return the indexes of the stations, sorted by distance
    return sort_idx


T0 = 128  # length of interval to use as input data (~2 hours)
Tfinal = 30  # length of prediction interval
N_STATIONS = 5  # how many of the closest stations to use as input
MAX_NAN_RATIO = .1

# substorm file, make it datetime indexable
substorms = pd.read_csv("data/substorms_2000_2018.csv")
substorms.index = pd.to_datetime(substorms.Date_UTC)

region_corners = [[-130, 45], [-60, 70]]
all_stations = pd.read_csv("data/supermag_stations.csv", index_col=0, usecols=[0, 1, 2, 5])

# estimate the distribution of substorm locations
kde = KernelDensity(bandwidth=.5)
kde.fit(substorms[["MLT", "MLAT"]].values)

X = []
y = []
time = []
one_hot_time = []
for yr in range(2000, 2019):
    print(yr)
    # buffer for this year's data, to be concatenated into a numpy array later
    X_yr = []
    y_yr = []
    time_yr = []
    one_hot_time_yr = []
    year = str(yr)

    # gather substorms for the year
    ss = substorms[year]
    # gather magnetometer data for the year
    mag_file = "./data/mag_data_{}.nc".format(year)
    # get rid of extra columns / put the columns in the desired order
    dataset = xr.open_dataset(mag_file).sel(dim_1=['MLT', 'MLAT', 'N', 'E', 'Z'])
    # gather stations, restrict to the region
    stations = all_stations.loc[[st for st in dataset]].values[:, :2]
    stations[stations > 180] -= 360
    region_mask = ((stations[:, 0] > region_corners[0][0]) * (stations[:, 0] < region_corners[1][0]) *
                   (stations[:, 1] > region_corners[0][1]) * (stations[:, 1] < region_corners[1][1]))
    # grab the dates before turning it into a numpy array
    dates = dataset.Date_UTC.values
    # turn the data into a big numpy array
    data = dataset.to_array().values  # (stations x time x component)
    # filter out all stations outside of the region
    data = data[region_mask]

    # find substorms
    for i in range(ss.shape[0]):
        # substorm location
        ss_loc = ss.iloc[i][["MLT", "MLAT"]].astype(float)
        # substorm date
        date = np.datetime64(ss.index[i])
        # minute within the prediction interval at which the substorm takes place
        ss_interval_index = np.random.randint(0, Tfinal)
        # index within the entire year's worth of data that the substorm takes place
        ss_date_index = np.argmax(date == dates)
        # if the substorm occurs too early in the year (before 2 hours + substorm interval index), skip this substorm
        if ss_date_index - ss_interval_index - T0 < 0:
            print("Not enough mag data", ss.index[i], ss_date_index, ss_loc.ravel())
            continue
        # gather up the magnetometer data for the input interval
        mag_data = data[:, ss_date_index - ss_interval_index - T0:ss_date_index - ss_interval_index]

        # check if substorm is within region
        region_check = data[:, ss_date_index, :2] - ss_loc[None, :]
        region_check = region_check[np.all(np.isfinite(region_check), axis=1)]
        if (np.all(region_check[:, 0] > 0) or np.all(region_check[:, 1] > 0) or
                np.all(region_check[:, 0] < 0) or np.all(region_check[:, 1] < 0)):
            continue

        # find the closest stations
        # exclude stations that have more than MAX_NAN_RATIO average NaNs during input interval
        nan_mask = np.mean(np.any(np.isnan(mag_data), axis=2), axis=1) > MAX_NAN_RATIO
        # locations of the stations from 5 minutes before up to the substorm
        mag_locs = data[:, ss_date_index-5:ss_date_index, :2]
        # which time index has the least NaNs? the actual substorm creates many NaNs
        best_idx = np.argmin(np.mean(np.any(np.isnan(mag_locs), axis=2), axis=0))
        # grab the best time index
        mag_locs = mag_locs[:, best_idx, :]
        # discard the stations who's input data has lots of NaNs
        mag_locs[nan_mask] = np.nan
        # find the closest stations
        sort_idx = find_closest_stations(mag_locs, ss_loc.values)
        # skip this interval if anything has gone wrong
        if sort_idx is None:
            continue
        # select the magnetometer data for the closest stations (only N, E, Z components, not location)
        sorted_mag_data = mag_data[sort_idx[:N_STATIONS], :, 2:]
        if np.any(np.isnan(sorted_mag_data)):
            nan_ratio = np.isnan(sorted_mag_data).sum() / np.prod(sorted_mag_data.shape)
            print(year, "{} NaN ratio".format(nan_ratio))
            sorted_mag_data[np.isnan(sorted_mag_data)] = 0
        # add this example to this years data buffer
        X_yr.append(sorted_mag_data)
        y_yr.append(1)
        time_yr.append(ss_interval_index / Tfinal)
        one_hot = np.zeros(Tfinal + 1)
        one_hot[ss_interval_index+1] = 1
        one_hot_time_yr.append(one_hot)

    # make sure to create equal number of positive and negative examples
    n_positive_examples = len(X_yr)
    print("{} substorms from {}".format(n_positive_examples, yr))
    for i in range(4 * n_positive_examples):
        # choose a random data during the year
        random_date_index = np.random.randint(T0 + Tfinal, dates.shape[0] - Tfinal)
        # skip this one if there is a substorm occurring (we are looking for negative examples here)
        if len(ss.iloc[random_date_index: random_date_index+Tfinal]) != 0:
            continue
        # collect the magnetometer data for this interval
        mag_data = data[:, random_date_index - T0:random_date_index]
        # sample a random location from the estimated distribution of substorm locations
        random_location = kde.sample()[0]
        # make sure that the MLT is in the interval [0, 24]
        random_location[0] = np.mod(random_location[0], 24)
        # check if substorm is within region
        region_check = data[:, random_date_index, :2] - random_location[None, :]
        region_check = region_check[np.all(np.isfinite(region_check), axis=1)]
        if np.all(region_check[:, 0] > 0) or np.all(region_check[:, 1] > 0) or np.all(region_check[:, 0] < 0) or \
                np.all(region_check[:, 1] < 0):
            continue
        # find the closest stations
        # exclude stations that have more than MAX_NAN_RATIO average NaNs during input interval
        nan_mask = np.mean(np.any(np.isnan(mag_data), axis=2), axis=1) > MAX_NAN_RATIO
        # locations of the stations from 5 minutes before up to the substorm
        mag_locs = data[:, random_date_index - 5:random_date_index, :2]
        # which time index has the least NaNs? the actual substorm creates many NaNs
        best_idx = np.argmin(np.mean(np.any(np.isnan(mag_locs), axis=2), axis=0))
        # grab the best time index
        mag_locs = mag_locs[:, best_idx, :]
        # discard the stations who's input data has lots of NaNs
        mag_locs[nan_mask] = np.nan
        # find the closest stations
        sort_idx = find_closest_stations(mag_locs, random_location)
        # skip this interval if anything has gone wrong
        if sort_idx is None:
            continue
        # grab the data for the closest stations
        sorted_mag_data = mag_data[sort_idx[:N_STATIONS], :, 2:]
        # double check that the data is of the correct shape
        if sorted_mag_data.shape != (N_STATIONS, T0, 3):
            continue
        if np.any(np.isnan(sorted_mag_data)):
            nan_ratio = np.isnan(sorted_mag_data).sum() / np.prod(sorted_mag_data.shape)
            print(year, "{} NaN ratio".format(nan_ratio))
            sorted_mag_data[np.isnan(sorted_mag_data)] = 0
        # add the negative examples to this years data buffer
        X_yr.append(sorted_mag_data)
        y_yr.append(0)
        time_yr.append(0)
        one_hot = np.zeros(Tfinal + 1)
        one_hot[0] = 1
        one_hot_time_yr.append(one_hot)

    # add this years data buffer to the overall data buffers
    X.append(np.stack(X_yr, axis=0))
    y.append(np.array(y_yr))
    time.append(np.array(time_yr))
    one_hot_time.append(np.stack(one_hot_time_yr, axis=0))

# concatenate all of the data buffers into one big numpy array
X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)
time = np.concatenate(time, axis=0)
one_hot_time = np.concatenate(one_hot_time, axis=0)

# save the dataset
np.savez("./data/closest_station_data.npz", X=X, y=y, time=time, one_hot_time=one_hot_time)
