import numpy as np
import pandas as pd
import datetime
import xarray as xr
import matplotlib.pyplot as plt

plt.style.use('ggplot')
before = 200
after = 10
n_ss = 100
substorms = pd.read_csv("./data/substorms_2000_2018.csv")
substorms.index = pd.to_datetime(substorms.Date_UTC)

year = '2010'

ss = substorms[year]
mag_file = "./data/mag_data_{}.nc".format(year)
mag_data = xr.open_dataset(mag_file)
stations_to_drop = [station for station in mag_data if np.any(np.all(np.isnan(mag_data[station]), axis=0))]
mag_data = mag_data.drop(stations_to_drop)

th = 1
data = []
idx = np.arange(ss.shape[0])
np.random.shuffle(idx)
for storm_index in idx[:n_ss]:
    ss_info = ss.iloc[storm_index]
    print(ss_info.Date_UTC)
    date = datetime.datetime.strptime(ss_info.Date_UTC, '%Y-%m-%d %H:%M:%S')
    storm_data = mag_data.sel(Date_UTC=slice(date - datetime.timedelta(minutes=before),
                                             date+datetime.timedelta(minutes=after)))
    cond = np.logical_and(np.abs(storm_data.sel(dim_1='MLT') - ss_info.MLT) < th,
                          np.abs(storm_data.sel(dim_1='MLAT') - ss_info.MLAT) < th)
    if np.any(cond.to_array().shape == 0) or len(storm_data.indexes['Date_UTC']) != before+after+1:
        continue
    cond = cond.mean(dim='Date_UTC') > .5
    close_stations = [st for st in cond if cond[st]]
    if not close_stations:
        continue
    storm_data = storm_data[close_stations].to_array().values
    storm_data = storm_data[np.all(np.isfinite(storm_data.reshape((storm_data.shape[0], -1))), axis=1)]
    if np.any(storm_data.shape == 0):
        continue
    data.append(storm_data[:, :, -3:] - np.mean(storm_data[:, :, -3:], axis=1, keepdims=True))

data = np.concatenate(data, axis=0)
avg = np.mean(data, axis=0)
std = np.std(data, axis=0)
time = np.arange(-before, after+1)
fig = plt.figure(figsize=(20, 12))
for i in range(data.shape[0]):
    plt.plot(time, data[i, :, 0], 'r', lw=.1)
fig = plt.figure(figsize=(20, 12))
for i in range(data.shape[0]):
    plt.plot(time, data[i, :, 1], 'b', lw=.1)
fig = plt.figure(figsize=(20, 12))
for i in range(data.shape[0]):
    plt.plot(time, data[i, :, 2], 'g', lw=.1)
plt.show()
