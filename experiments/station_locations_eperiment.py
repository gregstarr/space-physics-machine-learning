import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

mag_file = "../data/mag_data_2017.nc"
stations = pd.read_csv("../data/supermag_stations.csv", index_col=0, usecols=[0, 1, 2, 5])
dataset = xr.open_dataset(mag_file)

stations = stations.loc[[st for st in dataset]].values[:, :2]
stations[stations > 180] -= 360

corners = [[-130, 45], [-60, 70]]
border_pts = np.array([[corners[0][0], corners[0][1]],
                       [corners[0][0], corners[1][1]],
                       [corners[1][0], corners[1][1]],
                       [corners[1][0], corners[0][1]],
                       [corners[0][0], corners[0][1]]])
mask = ((stations[:, 0] > corners[0][0]) * (stations[:, 0] < corners[1][0]) *
        (stations[:, 1] > corners[0][1]) * (stations[:, 1] < corners[1][1]))
border = []
for i in range(border_pts.shape[0]):
    lon1, lat1 = border_pts[i]
    lon2, lat2 = border_pts[(i+1) % (border_pts.shape[0] - 1)]
    if lat1 == lat2:
        lon = np.arange(lon1, lon2, np.sign(lon2-lon1))
        lat = np.interp(lon, [lon1, lon2], [lat1, lat2])
    else:
        lat = np.arange(lat1, lat2, np.sign(lat2 - lat1))
        lon = np.interp(lat, [lat1, lat2], [lon1, lon2])
    border.append(np.stack((lon, lat), axis=1))
border = np.concatenate(border, axis=0)

# setup stereographic basemap.
# lat_ts is latitude of true scale.
# lon_0,lat_0 is central point.
m = Basemap(width=8000000,height=4000000, resolution='l', projection='stere', lat_ts=70, lat_0=60, lon_0=-100.)
m.drawmapboundary(fill_color='aqua')
m.drawcoastlines(linewidth=0.5)
m.fillcontinents(color='coral', lake_color='aqua')
# draw parallels and meridians.
m.drawparallels(np.arange(-80., 81., 20.))
m.drawmeridians(np.arange(-180., 181., 20.))

# draw stations
lon, lat = m(stations[:, 0], stations[:, 1])
m.plot(lon[mask], lat[mask], 'b.')
m.plot(lon[~mask], lat[~mask], 'r.')

# draw region
lon, lat = m(border[:, 0], border[:, 1])
m.plot(lon, lat, 'k-')
x, y = m(corners[0][0], corners[0][1])
plt.annotate('({}, {})'.format(corners[0][0], corners[0][1]), xy=(x-500000, y-200000))
x, y = m(corners[1][0], corners[1][1])
plt.annotate('({}, {})'.format(corners[1][0], corners[1][1]), xy=(x-250000, y+50000))

plt.title("{} stations in region".format(mask.sum()))
plt.show()
