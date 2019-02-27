import tensorflow as tf
import numpy as np

from data_pipeline import small_dset_gen, large_dset_gen


class MagNN:

    def __init__(self, params):

        self.params = params

        tf.reset_default_graph()

        self.data_iterator, self.train_init, self.val_init = self.create_data_pipeline()
        self.mag, self.station_loc, self.occ, self.occ_t, self.occ_loc = self.data_iterator.get_next()
        self.output = self.model_definition()
        self.loss = self.get_loss()

        self.session = tf.Session()

    def train(self, total_epochs, learning_rate=.001):
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_step = opt.minimize(self.loss)
        conf = tf.nn.sigmoid(self.output['occurrence'])

        saver = tf.train.Saver()

        self.session.run(tf.global_variables_initializer())
        self.session.run(self.train_init)

        epoch = 0
        stats = {'train_loss_hist': np.zeros(total_epochs),
                 'train_accuracy_hist': np.zeros(total_epochs),
                 'train_time_error_hist': np.zeros(total_epochs),
                 'train_loc_error_hist': np.zeros((total_epochs, 3)),
                 'val_loss_hist': np.zeros(total_epochs),
                 'val_accuracy_hist': np.zeros(total_epochs),
                 'val_time_error_hist': np.zeros(total_epochs),
                 'val_loc_error_hist': np.zeros((total_epochs, 3)),
                 'val_loc_scatter': np.empty((total_epochs, self.params['val_size'], 3)),
                 'total_pts': 0}

        run_variables = [train_step, self.loss, conf, self.occ, self.occ_t, self.occ_loc,
                         self.output['time'], self.output['location']]

        while epoch < total_epochs:
            try:
                collect_variables = self.session.run([run_variables])
                stats = MagNN.process_train_batch_results(collect_variables, stats)
            except tf.errors.OutOfRangeError:
                stats = MagNN.process_train_epoch_results(stats, epoch)
                # run validation set
                self.session.run(self.val_init)
                collect_variables = self.session.run([run_variables])
                stats = MagNN.process_val_epoch_results(collect_variables, stats, epoch)
                # re initialize training dataset
                self.session.run(self.train_init)
                epoch += 1

        print(saver.save(self.session, "./model/model.ckpt"))

    @staticmethod
    def process_train_batch_results(collect, stats, epoch):
        _, loss, conf, occ, occ_t, occ_loc, time, location = collect
        n_pts = conf.shape[0]
        stats['total_pts'] += n_pts
        stats['train_accuracy_hist'][epoch] += np.sum(np.round(conf) == occ)
        stats['train_loss_hist'][epoch] += n_pts * loss
        stats['train_time_error_hist'][epoch] += np.sum(np.abs(occ_t - time))
        stats['train_loc_error_hist'][epoch] += np.sum(np.abs(occ_loc - location), axis=0)
        return stats

    @staticmethod
    def process_train_epoch_results(stats, epoch):
        stats['train_loss_hist'][epoch] /= stats['total_pts']
        stats['train_accuracy_hist'][epoch] /= stats['total_pts']
        stats['train_time_error_hist'][epoch] /= stats['total_pts']
        stats['train_loc_error_hist'][epoch] /= stats['total_pts']
        stats['total_pts'] = 0
        return stats

    @staticmethod
    def process_val_epoch_results(collect, stats, epoch):
        _, loss, conf, occ, occ_t, occ_loc, time, location = collect
        stats['val_accuracy_hist'][epoch] = np.mean(np.round(conf) == occ)
        stats['val_loss_hist'][epoch] = loss
        stats['val_time_error_hist'][epoch] = np.mean(np.abs(occ_t - time))
        stats['val_loc_error_hist'][epoch] = np.mean(np.abs(occ_loc - location), axis=0)
        stats['val_loc_scatter'][epoch] = occ_loc - location
        return stats

    def model_definition(self):
        # conv - conv - pool
        c1_kernel = tf.get_variable("c1_kernel", shape=(1, 3, 3, 16))
        C1 = tf.nn.leaky_relu(tf.nn.conv2d(self.mag, c1_kernel, [1, 1, 1, 1], padding="SAME"))
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
        # batch x ? x N/8 x 64

        # FC layers
        sh = tf.shape(P3)
        vectors = tf.reshape(P3, [sh[0], sh[1], 1024])
        vectors = tf.concat((vectors, self.station_loc), axis=2)
        FC1 = tf.nn.leaky_relu(tf.layers.dense(vectors, 1024))
        FC2 = tf.nn.leaky_relu(tf.layers.dense(FC1, 1024))

        # sum - FC layers
        inp_sum = tf.reduce_mean(FC2, axis=1)
        FC3 = tf.nn.leaky_relu(tf.layers.dense(inp_sum, 512))
        FC4 = tf.nn.leaky_relu(tf.layers.dense(FC3, 512))
        model_output = tf.layers.dense(FC4, 5)

        return {'occurrence': model_output[:, 0], 'time': model_output[:, 1], 'location': model_output[:, 2:]}

    def get_loss(self):
        occurrence_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.occ, logits=self.model_output['occurrence']))
        time_loss = tf.abs(self.occ_t - self.model_output['time'])
        location_loss = tf.reduce_mean(tf.abs(self.occ_loc - self.model_output['location']), axis=1)
        loss = occurrence_loss + tf.reduce_mean(self.occ * (time_loss + location_loss))

        return loss

    def predict(self):
        pass

    def create_data_pipeline(self):
        mag_files = self.params['mag_files']
        ss_file = self.params['ss_file']
        val_size = self.params['val_size']
        batch_size = self.params['batch_size']
        train_gen = large_dset_gen(mag_files, ss_file, self.data_interval, self.prediction_interval)
        val_gen = small_dset_gen(mag_files[-1], ss_file, self.data_interval, self.prediction_interval, val_size)

        train_dataset = self._create_dataset(train_gen, batch_size)
        train_dataset.shuffle(batch_size * 4).prefetch(2)

        val_dataset = self._create_dataset(val_gen, val_size)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

        train_init = iterator.make_initializer(train_dataset)
        val_init = iterator.make_initializer(val_dataset)

        return iterator, train_init, val_init

    def _create_dataset(self, gen, batch_size):

        dataset = tf.data.Dataset.from_generator(gen,
                                                 output_types=(
                                                     tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                                                 output_shapes=(
                                                     tf.TensorShape([None, self.data_interval, 3]),
                                                     tf.TensorShape([None, 3]),
                                                     tf.TensorShape([]),
                                                     tf.TensorShape([]),
                                                     tf.TensorShape([3]))
                                                 )

        dataset = dataset.padded_batch(batch_size, dataset.output_shapes)

        return dataset

