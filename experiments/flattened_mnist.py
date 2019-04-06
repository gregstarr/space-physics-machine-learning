from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt

mndata = MNIST('mnist')

images, labels = mndata.load_training()
images = np.array(images)
labels = np.array(labels)

new_images = np.empty_like(images)

batch = 100

import tensorflow as tf
freqs = np.fft.fftfreq(images.shape[1]).astype(np.complex128)
X = tf.placeholder(tf.complex128, shape=(None, images.shape[1]))
locs = tf.placeholder(dtype=tf.complex128, shape=(None, images.shape[1]))
FT = tf.signal.fft(X)
freqs = np.fft.fftfreq(images.shape[1])
sample_locations = freqs[:, None, None] * locs[None, :, :]
complex_argument = 2 * np.pi * 1j * sample_locations
complex_sinusoid = tf.exp(complex_argument)
newim = tf.reduce_sum(tf.transpose(FT)[:, :, None] * complex_sinusoid, axis=0) / images.shape[1]
newim = tf.cast(tf.round(tf.real(newim)), tf.int32)

session = tf.Session()
for i in range(images.shape[0] // batch):
    if not i % 60:
        print(i * batch * 100 / images.shape[0], '%')
    newlocs = np.sort(np.random.rand(batch, images.shape[1]) * images.shape[1])
    im = session.run(newim, feed_dict={X: images[i*batch:(i+1)*batch], locs: newlocs})
    new_images[i * batch:(i + 1) * batch] = im

np.save("grid_images.npy", images)
np.save("random_images.npy", new_images)
np.save("labels.npy", labels)
