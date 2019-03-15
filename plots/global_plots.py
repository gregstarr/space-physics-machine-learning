import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import keras


class Converter:

    def __init__(self, mlt, mlat, lon, lat):
        corrs = np.empty((240, 2))
        for i, hr in enumerate(np.arange(0, 24, .1)):
            smlt = mlt + hr
            mask = np.isfinite(smlt)
            smlt = smlt[mask]
            smlt[smlt > 24] -= 24
            corrs[i] = (hr, np.corrcoef(smlt, lon[mask])[0, 1])

        self.mlt_shift = corrs[np.argmax(corrs[:, 1]), 0]
        smlt = mlt + self.mlt_shift
        mask = np.isfinite(smlt)
        smlt = smlt[mask]
        smlt[smlt > 24] -= 24
        self.mlt_coeff = np.mean((lon[mask] - lon[mask].mean()) / (smlt - smlt.mean()))

        mask = np.isfinite(mlat)
        self.mlat_coeff = np.mean((lat[mask]-lat[mask].mean()) / (mlat[mask]-mlat[mask].mean()))

    def mag_to_geo(self, mlt, mlat):
        mlt = mlt + self.mlt_shift - 12
        if isinstance(mlt, np.ndarray):
            mask = np.isfinite(mlt)
            mask[mask] *= mlt[mask] > 12
            mlt[mask] -= 24
        elif mlt > 12:
            mlt -= 24
        mlt *= self.mlt_coeff
        return mlt, mlat


model_name = "closest_n_stations_model"

with open("../models/{}_architecture.json".format(model_name)) as f:
    model_architecture = f.read()
model = keras.models.model_from_json(model_architecture)
model.load_weights("../models/{}_weights.h5".format(model_name))

T0 = 128
Tfinal = 60
N_STATIONS = 5
year = '2017'

resolution = 4

stations = pd.read_csv("../data/supermag_stations.csv", index_col=0, usecols=[0, 1, 2, 5])
substorms = pd.read_csv("../data/substorms_2000_2018.csv", index_col=0, parse_dates=True)
substorms = substorms[year]

mag_file = "../data/mag_data_{}.nc".format(year)
dataset = xr.open_dataset(mag_file).sel(dim_1=['MLT', 'MLAT', 'N', 'E', 'Z'])

X, Y = np.meshgrid(np.arange(-180, 180, resolution), np.arange(-90, 90, resolution))
I, J = np.meshgrid(np.arange(X.shape[1]), np.arange(X.shape[0]))
ss_map = np.zeros_like(X, dtype=np.float32)
points = np.stack((X, Y), axis=2).reshape((-1, 2))
map_indices = np.stack((I, J), axis=2).reshape((-1, 2))

stations = stations.loc[[st for st in dataset]].values[:, :2]
stations[stations[:, 0] > 180, 0] -= 360

distances = np.sum((points[:, None, :] - stations[None, :, :])**2, axis=2)
close_indices = np.argsort(distances, axis=1)

dates = dataset.Date_UTC.values
dataset = dataset.to_array().values  # station x time x component

for t in range(T0, dataset.shape[1], 15):
    print(t)
    fig = plt.figure(figsize=(10, 10))
    for pt in range(points.shape[0]):
        data = dataset[:, t-T0:t, :]

        top_k = []
        for i in range(close_indices.shape[1]):
            if len(top_k) == N_STATIONS:
                break
            if np.any(np.isnan(data[close_indices[pt, i]])):
                continue
            top_k.append(close_indices[pt, i])

        data = data[top_k]
        confidences = model.predict(data[None, :, :, 2:])
        ss_map[map_indices[pt, 1], map_indices[pt, 0]] = confidences[0, 0]

    bm_obj = Basemap(projection='npaeqd', boundinglat=10, lon_0=270, resolution='l')
    bm_obj.drawparallels(np.arange(-80., 81., 20.))
    bm_obj.drawmeridians(np.arange(-180., 181., 20.))
    bm_obj.drawcoastlines()
    lon, lat = bm_obj(X, Y)
    bm_obj.pcolormesh(lon, lat, ss_map, cmap="Blues", vmin=0, vmax=1)

    present_substorms = np.in1d(np.array(substorms.index), dates[t:t + 15])
    if present_substorms.sum() > 0:
        ss_index = np.argmax(np.in1d(np.array(substorms.index), dates[t:t + Tfinal]))
        date_index = np.argmax(np.datetime64(substorms.index[ss_index]) == dates)
        converter = Converter(dataset[:, date_index, 0], dataset[:, date_index, 1], stations[:,0].astype(float), stations[:,1].astype(float))
        lon, lat = converter.mag_to_geo(substorms.iloc[ss_index]["MLT"], substorms.iloc[ss_index]["MLAT"])
        print(lon, lat)
        lon, lat = bm_obj(lon, lat)
        bm_obj.plot(lon, lat, 'rx')

    cb = plt.colorbar()
    cb.set_label("Substorm Probability")
    plt.title("{} - {}".format(pd.to_datetime(dates[t]).strftime('%Y-%m-%d %H:%M:%S'),
                               pd.to_datetime(dates[t+15]).strftime('%Y-%m-%d %H:%M:%S')))
    plt.tight_layout()
    plt.savefig("world/{}.png".format(t))
    plt.clf()
    plt.close(fig)
    del fig
