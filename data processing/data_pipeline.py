import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf


def large_dset_gen(mag_files, ss_file, data_interval, prediction_interval):
    """My data generator feeding function
    
    Inputs:
        mag_files: list of magnetometer data file names
        ss_file: string file containing substorm data
        stats_file: string file containing mean and std
        data_interval: number of minutes for data interval
        prediction_interval: number of minutes for prediction interval
        
    Outputs:
        gen: the generator function
    
    What this needs to do:
        - load the i-th data file
        - strip away the unused variables
        - break the data into batches
            - time series for each station
            - sin(MLT), cos(MLT) and MLAT for each station
            - planet scale parameters
            - labels:
                - yes/no
                - time [0,1] for interval                           ------| for first substorm
                - location [sin(MLT), cos(MLT), MLAT in [-1,1]]     ------|
        - for the station time series':
            - remove stations with too many nans
            - outliers and remaining NaNs
            - remove mean, scale by STD
            - shuffle up stations? other augmentations?
        - yield (station x time x component), MLT, MLAT, planet params, targets
        - when current file is finished, load up next file
    """
    
    ss = pd.read_csv(ss_file)
    ss.index = pd.to_datetime(ss.Date_UTC)
    ss = ss.drop(columns=['Date_UTC'])     
    
    def gen():

        for file in mag_files:

            print(file)

            dataset = xr.open_dataset(file).sel(dim_1=['MLT', 'MLAT', 'N', 'E', 'Z'])
            
            dates = dataset.Date_UTC.values
            
            # STATION x TIME x COMPONENT
            da = dataset.to_array().values
            sinmlt = np.sin(da[:, :, 0])
            cosmlt = np.cos(da[:, :, 0])
            mlat = da[:, :, 1] / 90
            nez = da[:, :, 2:]

            data = np.concatenate((sinmlt[:, :, None], cosmlt[:, :, None], mlat[:, :, None], nez), axis=2)
            
            del dataset
            
            time_idx = 0
            
            while True:

                if time_idx + data_interval + prediction_interval > data.shape[1]:
                    break

                # select time slice
                chunk = data[:, time_idx:time_idx + data_interval]

                # planet scale parameters

                # increment time index
                time_idx += data_interval
                # create target
                int_beg = dates[time_idx]
                int_end = dates[time_idx + prediction_interval]
                target_chunk = ss[int_beg:int_end]
                ss_occurred = len(target_chunk) > 0
                time = 0
                location = (0, 0, 0)
                if ss_occurred:
                    first = target_chunk.iloc[0]
                    time = (first.name - int_beg).total_seconds() / (prediction_interval * 60)
                    location = (np.sin(first['MLT'] * 2 * np.pi / 24),
                                np.cos(first['MLT'] * 2 * np.pi / 24),
                                first['MLAT'] / 90)

                # filter out stations with NaNs
                cond = chunk.reshape([-1, data_interval * 6])
                cond = np.all(np.isfinite(cond), axis=1)
                chunk = chunk[cond]

                # mag data
                mag_chunk = chunk[:, :, 3:]

                # MLT / MLAT data
                mlat = chunk[:, 0, 0]
                sinmlt = chunk[:, data_interval // 2, 1]
                cosmlt = chunk[:, data_interval // 2, 2]
                st_loc = np.stack((sinmlt, cosmlt, mlat), axis=1)

                yield mag_chunk, st_loc, ss_occurred, time, location

    return gen


def small_dset_gen(mag_file, ss_file, data_interval, prediction_interval, n_pts=None):
    
    ss = pd.read_csv(ss_file)
    ss.index = pd.to_datetime(ss.Date_UTC)
    ss = ss.drop(columns=['Date_UTC'])
    
    dataset = xr.open_dataset(mag_file).sel(dim_1=['MLT', 'MLAT', 'N', 'E', 'Z'])

    da = dataset.to_array().values
    if n_pts is not None:
        idx = np.arange(da.shape[1] - da.shape[1] % data_interval)
        idx = np.random.permutation(idx.reshape((-1, data_interval)))[:n_pts]
        da = da[:, idx.ravel()]
    sinmlt = np.sin(da[:, :, 0])
    cosmlt = np.cos(da[:, :, 0])
    mlat = da[:, :, 1] / 90
    nez = da[:, :, 2:]

    data = np.concatenate((sinmlt[:, :, None], cosmlt[:, :, None], mlat[:, :, None], nez), axis=2)

    dates = dataset.Date_UTC.values
    
    def gen():

        time_idx = 0

        while True:

            if time_idx + data_interval + prediction_interval > data.shape[1]:
                break

            # select time slice
            chunk = data[:, time_idx:time_idx + data_interval]

            # planet scale parameters

            # increment time index
            time_idx += data_interval
            # create target
            int_beg = dates[time_idx]
            int_end = dates[time_idx + prediction_interval]
            target_chunk = ss[int_beg:int_end]
            ss_occurred = len(target_chunk) > 0
            time = 0
            location = (0, 0, 0)
            if ss_occurred:
                first = target_chunk.iloc[0]
                time = (first.name - int_beg).total_seconds() / (prediction_interval * 60)
                location = (np.sin(first['MLT'] * 2 * np.pi / 24),
                            np.cos(first['MLT'] * 2 * np.pi / 24),
                            first['MLAT'] / 90)

            # filter out stations with NaNs
            cond = chunk.reshape([-1, data_interval * 6])
            cond = np.all(np.isfinite(cond), axis=1)
            chunk = chunk[cond]

            # mag data
            mag_chunk = chunk[:, :, 3:]

            # MLT / MLAT data
            mlat = chunk[:, 0, 0]
            sinmlt = chunk[:, data_interval // 2, 1]
            cosmlt = chunk[:, data_interval // 2, 2]
            st_loc = np.stack((sinmlt, cosmlt, mlat), axis=1)

            yield mag_chunk, st_loc, ss_occurred, time, location
                
    return gen


def toy_set_gen(n, c):
    
    """ ? x N x C input and the target is sum(input[:,0,:])
    """
    inputs = []
    for i in range(10):
        n_inputs = np.random.randint(5, 10)
        random_input = np.random.randn(n_inputs, n, c)
        target = np.sum(random_input[:, 0, :])
        inputs.append((random_input, target))
    
    def gen():
        for ri, t in inputs:
            yield ri, [t]

    return gen


# this actually needs to be passed tensors, this means I will have to pass it chunk.to_array and make sure I know
def parallel_data_prep(chunk, occ, t, loc):
    # filter out stations with NaNs
    cond = tf.reshape(chunk, [-1, data_interval * 6])
    cond = tf.reduce_all(tf.is_finite(cond), axis=1)
    chunk = tf.boolean_mask(chunk, cond)

    # mag data
    mag_chunk = chunk[:, :, 3:]

    # MLT / MLAT data
    mlat = chunk[:, 0, 0]
    sinmlt = chunk[:, data_interval//2, 1]
    cosmlt = chunk[:, data_interval//2, 2]
    st_loc = tf.stack((sinmlt, cosmlt, mlat), axis=1)

    return mag_chunk, st_loc, occ, t, loc