import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt
from data_pipeline import small_dset_gen, large_dset_gen
plt.style.use('ggplot')

mag_files = glob.glob("./data/mag_data_*.nc")[:2]
ss_file = "./data/substorms_2000_2018.csv"
data_interval = 128
prediction_interval = 64
channels = 3
val_size = 300

BATCH_SIZE = 64

tf.reset_default_graph()

train_gen = large_dset_gen(mag_files[:-1], ss_file, data_interval, prediction_interval)
val_gen = small_dset_gen(mag_files[-1], ss_file, data_interval, prediction_interval, val_size)

train_dataset = tf.data.Dataset.from_generator(train_gen,
                                               output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                                               output_shapes=(tf.TensorShape([None, data_interval, channels]),
                                                              tf.TensorShape([None, 3]),
                                                              tf.TensorShape([]),
                                                              tf.TensorShape([]),
                                                              tf.TensorShape([3]))
                                               )

train_dataset = train_dataset.padded_batch(BATCH_SIZE, (tf.TensorShape([None, data_interval, channels]),
                                                        tf.TensorShape([None, 3]),
                                                        tf.TensorShape([]),
                                                        tf.TensorShape([]),
                                                        tf.TensorShape([3])))
train_dataset.shuffle(BATCH_SIZE*4).prefetch(2)

val_dataset = tf.data.Dataset.from_generator(val_gen,
                                             output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                                             output_shapes=(tf.TensorShape([None, data_interval, channels]),
                                                            tf.TensorShape([None, 3]),
                                                            tf.TensorShape([]),
                                                            tf.TensorShape([]),
                                                            tf.TensorShape([3]))
                                             )

val_dataset = val_dataset.padded_batch(val_size, (tf.TensorShape([None, data_interval, channels]),
                                                  tf.TensorShape([None, 3]),
                                                  tf.TensorShape([]),
                                                  tf.TensorShape([]),
                                                  tf.TensorShape([3])))

data_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
train_init = data_iterator.make_initializer(train_dataset)
val_init = data_iterator.make_initializer(val_dataset)

mag, station_loc, occ, occ_t, occ_loc = data_iterator.get_next()

learning_rate = tf.placeholder(tf.float32)

# conv - conv - pool
c1_kernel = tf.get_variable("c1_kernel", shape=(1, 3, channels, 16))
C1 = tf.nn.leaky_relu(tf.nn.conv2d(mag, c1_kernel, [1, 1, 1, 1], padding="SAME"))
c2_kernel = tf.get_variable("c2_kernel", shape=(1, 3, 16, 16))
C2 = tf.nn.leaky_relu(tf.nn.conv2d(C1, c2_kernel, [1, 1, 1, 1], padding="SAME"))
P1 = tf.nn.max_pool(C2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")
# conv - conv - pool
c3_kernel = tf.get_variable("c3_kernel", shape=(1, 3, 16, 32))
C3 = tf.nn.leaky_relu(tf.nn.conv2d(P1, c3_kernel, [1, 1, 1, 1], padding="SAME"))
c4_kernel = tf.get_variable("c4_kernel", shape=(1, 3, 32, 32))
C4 = tf.nn.leaky_relu(tf.nn.conv2d(C3, c4_kernel, [1, 1, 1, 1], padding="SAME"))
P2 = tf.nn.max_pool(C4, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")
# conv - conv - pool
c5_kernel = tf.get_variable("c5_kernel", shape=(1, 3, 32, 64))
C5 = tf.nn.leaky_relu(tf.nn.conv2d(P2, c5_kernel, [1, 1, 1, 1], padding="SAME"))
c6_kernel = tf.get_variable("c6_kernel", shape=(1, 3, 64, 64))
C6 = tf.nn.leaky_relu(tf.nn.conv2d(C5, c6_kernel, [1, 1, 1, 1], padding="SAME"))
P3 = tf.nn.max_pool(C6, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")
# batch x ? x N/4 x 32

# FC layers
sh = tf.shape(P3)
vectors = tf.reshape(P3, [sh[0], sh[1], 1024])
vectors = tf.concat((vectors, station_loc), axis=2)
FC1 = tf.nn.leaky_relu(tf.layers.dense(vectors, 1024))
FC2 = tf.nn.leaky_relu(tf.layers.dense(FC1, 1024))

# sum - FC layers
inp_sum = tf.reduce_mean(FC2, axis=1)
FC3 = tf.nn.leaky_relu(tf.layers.dense(inp_sum, 512))
FC4 = tf.nn.leaky_relu(tf.layers.dense(FC3, 512))
model_output = tf.layers.dense(FC3, 5)
model_output = {
    'occurrence': model_output[:, 0],
    'time': model_output[:, 1],
    'location': model_output[:, 2:]
}

conf = tf.nn.sigmoid(model_output['occurrence'])

occurrence_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=occ, logits=model_output['occurrence']))
time_loss = tf.abs(occ_t - model_output['time'])
location_loss = tf.reduce_mean(tf.abs(occ_loc - model_output['location']), axis=1)
loss = occurrence_loss + tf.reduce_mean(occ * (time_loss + location_loss))

opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_step = opt.minimize(loss)

lr = .001

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(train_init)

EPOCHS = 10
epoch = 0
train_loss_hist = []
train_o_acc_hist = []
train_o_t_acc_hist = []
train_o_l_acc_hist = []
train_loss = 0
train_o_acc = 0
train_o_t_acc = 0
train_o_l_acc = np.zeros((3,))
val_loss_hist = []
val_acc_hist = []
total_pts = 0

while epoch < EPOCHS:
    try:
        _, c, o, o_t, o_l, L, m_t, m_l = sess.run([train_step, conf, occ, occ_t, occ_loc, loss, model_output['time'],
                                                   model_output['location']], feed_dict={learning_rate: lr})
        n_pts = c.shape[0]
        train_loss += n_pts * L
        train_o_acc += np.sum(np.round(c) == o)
        train_o_t_acc += np.sum(np.abs(o_t - m_t))
        train_o_l_acc += np.sum(np.abs(o_l - m_l), axis=0)
        total_pts += n_pts
    except tf.errors.OutOfRangeError:
        # print out epoch results
        print("Epoch {:2d}, Training loss: {:6.4f}, Training occurance accuracy: {:5.3f}, Training time error: " \
                "{:5.3f}, Training location error: {}".format(epoch + 1, train_loss / total_pts, train_o_acc / total_pts,
                                                                   train_o_t_acc / total_pts, train_o_l_acc / total_pts), end='; ')
        # save training epoch results
        train_loss_hist.append(train_loss / total_pts)
        train_o_acc_hist.append(train_o_acc / total_pts)
        train_o_t_acc_hist.append(train_o_t_acc / total_pts)
        train_o_l_acc_hist.append(train_o_l_acc / total_pts)
        train_loss = 0
        train_o_acc = 0
        train_o_t_acc = 0
        train_o_l_acc = np.zeros((3,))
        total_pts = 0
        epoch += 1
        # run validation set
        sess.run(val_init)
        c, t, L = sess.run([conf, occ, loss])
        a = np.mean(np.round(c) == t)
        # print val results
        print("Validation loss: {:6.4f}, Validation accuracy: {:5.3f}".format(L, a))
        # collect val stats
        val_loss_hist.append(L)
        val_acc_hist.append(a)
        # re initialize training dataset
        sess.run(train_init)

checkpoint = saver.save(sess, "./model/model.ckpt")

plt.figure()
plt.subplot(211)
plt.plot(train_loss_hist)
plt.plot(val_loss_hist)
plt.xlabel("epoch number")
plt.ylabel("Loss")
plt.subplot(212)
plt.plot(train_o_acc_hist)
plt.plot(val_acc_hist)
plt.xlabel("epoch number")
plt.ylabel("Accuracy")

plt.figure()
plt.plot(train_o_t_acc_hist)

plt.figure()
plt.plot([l[0] for l in train_o_l_acc_hist])
plt.plot([l[1] for l in train_o_l_acc_hist])
plt.plot([l[2] for l in train_o_l_acc_hist])
plt.show()
