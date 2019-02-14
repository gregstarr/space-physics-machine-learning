import numpy as np
import pandas as pd
import xarray as xr

def make_generator(mag_files, ss_file, stats_file, data_interval, prediction_interval):
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
    
    stats = np.load(stats_file)
    mean = 0#stats['mean']
    std = 1#stats['std']
    
    ss = pd.read_csv(ss_file)
    ss.index = pd.to_datetime(ss.Date_UTC)
    ss = ss.drop(columns=['Date_UTC'])
    
    data_dt = np.timedelta64(data_interval, 'm')
    predict_dt = np.timedelta64(prediction_interval, 'm')        
    
    def gen():
        
        for file in mag_files:
            print(file)
            
            dataset = xr.open_dataset(file).sel(dim_1=['MLT','MLAT','N','E','Z']).sel(Date_UTC=slice("2000-01-01","2000-02-01"))
            
            da = dataset.to_array().values
            sinmlt = np.sin(da[:,:,0])
            cosmlt = np.cos(da[:,:,0])
            mlat = da[:,:,1] / 90
            nez = (da[:,:,2:] - mean) / std

            data = np.concatenate((sinmlt[:,:,None], cosmlt[:,:,None], mlat[:,:,None], nez), axis=2)

            mag_data = xr.Dataset({st: (['Date_UTC', 'vals'], data[i]) for i,st in enumerate(dataset)},
                                  coords={'Date_UTC': dataset.Date_UTC, 'vals': ['SINMLT','COSMLT','MLAT','N','E','Z']})
            
            del dataset
            
            time_idx = mag_data.Date_UTC[0].values
            
            while True:
                
                if time_idx + data_dt > mag_data.Date_UTC[-1].values:
                    break
                
                # select time slice
                chunk = mag_data.sel(Date_UTC=slice(time_idx, time_idx + data_dt))
                
                # skip if there is missing data
                if chunk.Date_UTC.shape[0] != data_interval + 1:
                    time_idx += data_dt
                    continue
                    
                # filter out stations with NaNs
                cond = np.isnan(chunk).any(dim=['Date_UTC', 'vals'])
                drop = [st for st in cond if cond[st]]
                chunk = chunk.drop(drop)
                
                # mag data
                mag_chunk = chunk.sel(vals=['N','E','Z']).to_array()
                
                # MLT / MLAT data
                mlat = chunk.sel(vals='MLAT').to_array()[:,0]
                sinmlt = chunk.sel(vals='SINMLT').to_array()[:,data_interval//2]
                cosmlt = chunk.sel(vals='COSMLT').to_array()[:,data_interval//2]
                st_loc = np.stack((sinmlt, cosmlt, mlat), axis=1)
                
                # planet scale parameters
                
                # increment time index
                time_idx += data_dt
                # create target
                target_chunk = ss[time_idx:time_idx+predict_dt]
                ss_occurred = len(target_chunk) > 0
                time = 0
                location = (0, 0, 0)
                if ss_occurred:
                    first = target_chunk.iloc[0]
                    time = (first.name - (time_idx)).total_seconds() / (prediction_interval * 60)
                    location = (np.sin(first['MLT'] * 2 * np.pi / 24), 
                                np.cos(first['MLT'] * 2 * np.pi / 24), 
                                first['MLAT']/90)
                    
                yield mag_chunk, st_loc, [ss_occurred], [time], location
                
            del mag_data
                
    return gen


def small_dset_gen(mag_file, ss_file, stats_file, data_interval, prediction_interval):
    
    stats = np.load(stats_file)
    mean = stats['mean']
    std = stats['std']
    
    ss = pd.read_csv(ss_file)
    ss.index = pd.to_datetime(ss.Date_UTC)
    ss = ss.drop(columns=['Date_UTC'])
    
    data_dt = np.timedelta64(data_interval, 'm')
    predict_dt = np.timedelta64(prediction_interval, 'm')
    
    dataset = xr.open_dataset(mag_file).sel(dim_1=['MLT','MLAT','N','E','Z']).sel(Date_UTC=slice("2000-01-01","2000-02-01"))
    
    da = dataset.to_array().values
    sinmlt = np.sin(da[:,:,0])
    cosmlt = np.cos(da[:,:,0])
    mlat = da[:,:,1] / 90
    nez = (da[:,:,2:] - mean) / std

    data = np.concatenate((sinmlt[:,:,None], cosmlt[:,:,None], mlat[:,:,None], nez), axis=2)
    
    def gen():

        mag_data = xr.Dataset({st: (['Date_UTC', 'vals'], data[i]) for i,st in enumerate(dataset)},
                              coords={'Date_UTC': dataset.Date_UTC, 'vals': ['SINMLT','COSMLT','MLAT','N','E','Z']})

        time_idx = mag_data.Date_UTC[0].values

        while True:

            if time_idx + data_dt > mag_data.Date_UTC[-1].values:
                break

            # select time slice
            chunk = mag_data.sel(Date_UTC=slice(time_idx, time_idx + data_dt))

            # skip if there is missing data
            if chunk.Date_UTC.shape[0] != data_interval + 1:
                time_idx += data_dt
                continue

            # filter out stations with NaNs
            cond = np.isnan(chunk).any(dim=['Date_UTC', 'vals'])
            drop = [st for st in cond if cond[st]]
            chunk = chunk.drop(drop)

            # mag data
            mag_chunk = chunk.sel(vals=['N','E','Z']).to_array()

            # MLT / MLAT data
            mlat = chunk.sel(vals='MLAT').to_array()[:,0]
            sinmlt = chunk.sel(vals='SINMLT').to_array()[:,data_interval//2]
            cosmlt = chunk.sel(vals='COSMLT').to_array()[:,data_interval//2]
            st_loc = np.stack((sinmlt, cosmlt, mlat), axis=1)

            # planet scale parameters

            # increment time index
            time_idx += data_dt
            # create target
            target_chunk = ss[time_idx:time_idx+predict_dt]
            ss_occurred = len(target_chunk) > 0
            time = 0
            location = (0, 0, 0)
            if ss_occurred:
                first = target_chunk.iloc[0]
                time = (first.name - (time_idx)).total_seconds() / (prediction_interval * 60)
                location = (np.sin(first['MLT'] * 2 * np.pi / 24), 
                            np.cos(first['MLT'] * 2 * np.pi / 24), 
                            first['MLAT']/90)

            yield mag_chunk, st_loc, [ss_occurred], [time], location
                
    return gen