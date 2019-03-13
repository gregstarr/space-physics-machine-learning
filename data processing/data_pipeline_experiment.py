import tensorflow as tf
from dataGenerator import small_dset_gen
import numpy as np
import glob
import os
import time

os.chdir("data")

mag_files = glob.glob("mag_data_2000.nc")
ss_file = "substorms_2000_2018.csv"
stats_file = "statistics.npz"
data_interval = 128
prediction_interval = 64
channels = 3

BATCHSIZE = 100

gen = small_dset_gen(mag_files[0], ss_file, stats_file, data_interval, prediction_interval, 100)

dataset = tf.data.Dataset.from_generator(gen,
                                         output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                                         output_shapes=(tf.TensorShape([None, data_interval, channels]),
                                                        tf.TensorShape([None, 3]),
                                                        tf.TensorShape([1]),
                                                        tf.TensorShape([1]),
                                                        tf.TensorShape([3]))
                                         )

dataset = dataset.padded_batch(BATCHSIZE, (tf.TensorShape([None, data_interval, channels]),
                                           tf.TensorShape([None, 3]),
                                           tf.TensorShape([1]),
                                           tf.TensorShape([1]),
                                           tf.TensorShape([3])))

data_iter = dataset.make_initializable_iterator()

mag, st_loc, occ, ss_t, ss_loc = data_iter.get_next()

sess = tf.Session()
sess.run(data_iter.initializer)

n_iters = 0
t0 = time.time()
while True:
    try:
        print(sess.run(mag))
        n_iters += BATCHSIZE
    except tf.errors.OutOfRangeError:
        break
tf = time.time()

print(n_iters, tf - t0, (tf-t0) / n_iters)
