import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import keras
import apexpy

model_name = "closest_n_station_model"

with open("./models/{}_architecture.json".format(model_name)) as f:
    model_architecture = f.read()
model = keras.models.model_from_json(model_architecture)
model.load_weights("./models/{}_weights.h5".format(model_name))

T0 = 128
Tfinal = 60
N_STATIONS = 5
year = '2017'

resolution = 4

stations = pd.read_csv("data/supermag_stations.csv", index_col=0, usecols=[0, 1, 2, 5])
substorms = pd.read_csv("./data/substorms_2000_2018.csv", index_col=0, parse_dates=True)
substorms = substorms[year]

bm_obj = Basemap(projection='npaeqd', boundinglat=10, lon_0=270, resolution='l')
bm_obj.drawparallels(np.arange(-80., 81., 20.))
bm_obj.drawmeridians(np.arange(-180., 181., 20.))
bm_obj.drawcoastlines()
for i in range(len(substorms)):
    apex = apexpy.Apex(date=substorms.index[i])
    lat, lon = apex.convert(substorms.iloc[i]['MLAT'], substorms.iloc[i]['MLT'], 'mlt', 'geo', substorms.index[i])
    print(lon, lat)
    lon, lat = bm_obj(lon, lat)
    bm_obj.plot(lon, lat, '.')
plt.show()

# mag_file = "./data/mag_data_{}.nc".format(year)
# dataset = xr.open_dataset(mag_file).sel(dim_1=['N', 'E', 'Z'])
#
# X, Y = np.meshgrid(np.arange(-180, 180, resolution), np.arange(-90, 90, resolution))
# I, J = np.meshgrid(np.arange(X.shape[1]), np.arange(X.shape[0]))
# ss_map = np.zeros_like(X, dtype=np.float32)
# points = np.stack((X, Y), axis=2).reshape((-1, 2))
# map_indices = np.stack((I, J), axis=2).reshape((-1, 2))
#
# stations = stations.loc[[st for st in dataset]].values[:, :2]
# stations[stations[:, 0] > 180, 0] -= 360
#
# distances = np.sum((points[:, None, :] - stations[None, :, :])**2, axis=2)
# close_indices = np.argsort(distances, axis=1)
#
# dates = dataset.Date_UTC.values
# dataset = dataset.to_array().values  # station x time x component
#
# for t in range(T0, dataset.shape[1], 15):
#     print(t)
#     fig = plt.figure(figsize=(10, 10))
#     for pt in range(points.shape[0]):
#         data = dataset[:, t-T0:t, :]
#
#         top_k = []
#         for i in range(close_indices.shape[1]):
#             if len(top_k) == N_STATIONS:
#                 break
#             if np.any(np.isnan(data[close_indices[pt, i]])):
#                 continue
#             top_k.append(close_indices[pt, i])
#
#         data = data[top_k]
#         confidences = model.predict(data[None, :])
#         # avg_distance = np.sqrt(np.mean(distances[pt, top_k]))
#         ss_map[map_indices[pt, 1], map_indices[pt, 0]] = confidences[0, 0] #/ avg_distance
#
#     bm_obj = Basemap(projection='npaeqd', boundinglat=10, lon_0=270, resolution='l')
#     bm_obj.drawparallels(np.arange(-80., 81., 20.))
#     bm_obj.drawmeridians(np.arange(-180., 181., 20.))
#     bm_obj.drawcoastlines()
#     # lat, lon = bm_obj(stations[:, 0], stations[:, 1])
#     # bm_obj.plot(lat, lon, '.')
#     lon, lat = bm_obj(X, Y)
#     bm_obj.pcolormesh(lon, lat, ss_map, cmap="Blues", vmin=0, vmax=1)
#
#     present_substorms = np.in1d(np.array(substorms.index), dates[t:t + Tfinal])
#     if present_substorms.sum() > 0:
#         ss_index = np.argmax(np.in1d(np.array(substorms.index), dates[t:t + Tfinal]))
#         lat, lon, _ = aacgmv2.convert_latlon_arr(substorms.iloc[ss_index]['MLAT'],
#                                                  substorms.iloc[ss_index]['MLT'], 0,
#                                                  substorms.index[ss_index])
#         print(lon, lat)
#         lon, lat = bm_obj(lon, lat)
#         bm_obj.plot(lon, lat, 'rx')
#
#     plt.colorbar()
#     plt.tight_layout()
#     plt.savefig("plots/world/{}.png".format(t))
#     plt.clf()
#     plt.close(fig)
#     del fig
