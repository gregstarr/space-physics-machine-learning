import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
os.chdir("C:\\Users\\Greg\\code\\space-physics-machine-learning")


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


year = '2017'
print("YEAR", year)

stations = pd.read_csv("data/supermag_stations.csv", index_col=0, usecols=[0, 1, 2, 5])
substorms = pd.read_csv("data/substorms_2000_2018.csv", index_col=0, parse_dates=True)
substorms = substorms[year]

print(len(substorms))

mag_file = "./data/mag_data_{}.nc".format(year)
dataset = xr.open_dataset(mag_file)

stations = stations.loc[[st for st in dataset]].values[:, :2]
stations[stations[:, 0] > 180, 0] -= 360

dates = dataset.Date_UTC.values
dataset = dataset.to_array()  # station x time x component

print(dataset.shape)

ss_lon = []
ss_lat = []
for i in range(len(substorms)):
    if i % (len(substorms)//10) == 0:
        print("{:5.2f} %".format(i*100/len(substorms)))
    if np.datetime64(substorms.index[i]) not in dates:
        continue
    mlt = dataset.sel(Date_UTC=substorms.index[i]).values[:, 0].astype(float)
    lon = stations[:, 0].astype(float)
    mlat = dataset.sel(Date_UTC=substorms.index[i]).values[:, 1].astype(float)
    lat = stations[:, 1].astype(float)
    converter = Converter(mlt, mlat, lon, lat)
    lon_h, lat_h = converter.mag_to_geo(substorms["MLT"][i], substorms["MLAT"][i])
    if abs(lon_h) > 360 or abs(lat_h) > 360:
        continue
    ss_lon.append(lon_h)
    ss_lat.append(lat_h)

ss_locations = np.stack((ss_lon, ss_lat), axis=1)
ss_locations = ss_locations[~np.any(np.abs(ss_locations) > 180, axis=1)]
kde = KernelDensity(bandwidth=4)
kde.fit(ss_locations)
#
# X, Y = np.meshgrid(np.arange(-180, 180, .25), np.arange(-90, 90, .25))
# xy = np.stack((X.ravel(), Y.ravel()), axis=1)
# Z = np.exp(kde.score_samples(xy))
# Z = Z.reshape(X.shape)
#
# bm_obj = Basemap(projection='npaeqd', boundinglat=10, lon_0=270, resolution='l')
# bm_obj.drawparallels(np.arange(-80., 81., 20.))
# bm_obj.drawmeridians(np.arange(-180., 181., 20.))
# bm_obj.drawcoastlines()
# lon, lat = bm_obj(X, Y)
# bm_obj.pcolormesh(lon, lat, Z, cmap="Blues")
# lat, lon = bm_obj(ss_lon, ss_lat)
# bm_obj.plot(lat, lon, '.')
# plt.show()
