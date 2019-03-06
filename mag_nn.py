import tensorflow as tf
import numpy as np
import modules
from data_pipeline import small_dset_gen, large_dset_gen
import json


class MagNN:

    def __init__(self, params):

        self.params = params

        self.check = []

        tf.reset_default_graph()

        self.data_iterator, self.train_init, self.val_init = self.create_data_pipeline()
        self.mag, self.station_loc, self.occ, self.occ_t, self.occ_loc = self.data_iterator.get_next()
        self.output = self.model_definition()
        self.loss = self.get_loss()
        self.conf = tf.nn.sigmoid(self.output['occurrence'])

        self.session = tf.Session()

    def train(self, total_epochs, learning_rate=.001):
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_step = opt.minimize(self.loss)

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
                 'total_pts': 0}

        run_variables = [train_step, self.loss, self.conf, self.occ, self.occ_t, self.occ_loc,
                         self.output['time'], self.output['location']] + self.check

        while epoch < total_epochs:
            try:
                collect_variables = self.session.run(run_variables)
                stats = MagNN.process_train_batch_results(collect_variables, stats, epoch)
            except tf.errors.OutOfRangeError:
                stats = MagNN.process_train_epoch_results(stats, epoch)
                # run validation set
                self.session.run(self.val_init)
                while True:
                    try:
                        collect_variables = self.session.run(run_variables[1:])
                        stats = MagNN.process_val_batch_results(collect_variables, stats, epoch)
                    except tf.errors.OutOfRangeError:
                        stats = MagNN.process_val_epoch_results(stats, epoch)
                        break
                # re initialize training dataset
                self.session.run(self.train_init)
                print("Epoch: {}, Train loss: {}, Val loss: {}".format(epoch+1, stats['train_loss_hist'][epoch],
                                                                       stats['val_loss_hist'][epoch]))
                epoch += 1

        name = self.params['model_name']
        print(saver.save(self.session, "./model/{}.ckpt".format(name)))
        np.savez("./model/{}.npz".format(name), **stats)
        with open("./model/{}_params.json".format(name), 'w') as f:
            f.write(json.dumps(self.params))

        return stats

    def run_validation(self):
        stats = {'loss': 0,
                 'accuracy': 0,
                 'time_error': 0,
                 'location_error': np.zeros(3),
                 'total_pts': 0,
                 'scatter': []}
        run_variables = [self.loss, self.conf, self.occ, self.occ_t, self.occ_loc, self.output['time'],
                         self.output['location']]

        self.session.run(self.val_init)
        while True:
            try:
                (loss, confidences, occurrences, times,
                 locations, time_output, location_output) = self.session.run(run_variables)
                n_pts = confidences.shape[0]
                stats['loss'] += loss * n_pts
                stats['accuracy'] += np.sum(np.round(confidences) == occurrences)
                stats['time_error'] += np.sum(np.abs(times-time_output))
                stats['location_error'] += np.sum(np.abs(times - time_output), axis=0)
                stats['scatter'].append(location_output - locations)
                stats['total_pts'] += n_pts
            except tf.errors.OutOfRangeError:
                stats['loss'] /= stats['total_pts']
                stats['accuracy'] /= stats['total_pts']
                stats['time_error'] /= stats['total_pts']
                stats['location_error'] /= stats['total_pts']
                stats['scatter'] = np.concatenate(stats['scatter'], axis=0)
                return stats

    def create_data_pipeline(self):
        mag_files = self.params['mag_files']
        ss_file = self.params['ss_file']
        val_size = self.params['val_size']
        batch_size = self.params['batch_size']
        data_interval = self.params['data_interval']
        prediction_interval = self.params['prediction_interval']
        train_gen = large_dset_gen(mag_files[:-1], ss_file, data_interval, prediction_interval)
        val_gen = small_dset_gen(mag_files[-1], ss_file, data_interval, prediction_interval, val_size)

        train_dataset = MagNN._create_dataset(train_gen, data_interval, batch_size)
        train_dataset.shuffle(batch_size * 4).prefetch(2)

        val_dataset = MagNN._create_dataset(val_gen, data_interval, batch_size)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

        train_init = iterator.make_initializer(train_dataset)
        val_init = iterator.make_initializer(val_dataset)

        return iterator, train_init, val_init

    def load_checkpoint(self, checkpoint):
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint)

    # STATIC METHODS
    @staticmethod
    def process_train_batch_results(collect, stats, epoch):
        _, loss, conf, occ, occ_t, occ_loc, time, location, *args = collect
        for a in args:
            print(a.shape)
        n_pts = conf.shape[0]
        stats['total_pts'] += n_pts
        stats['train_accuracy_hist'][epoch] += np.sum(np.round(conf) == occ)
        stats['train_loss_hist'][epoch] += loss * n_pts
        stats['train_time_error_hist'][epoch] += np.sum(np.abs(occ_t - time))
        stats['train_loc_error_hist'][epoch] += np.sum(np.abs(occ_loc - location), axis=0)
        return stats

    @staticmethod
    def process_val_batch_results(collect, stats, epoch):
        loss, conf, occ, occ_t, occ_loc, time, location, *args = collect
        n_pts = conf.shape[0]
        stats['total_pts'] += n_pts
        stats['val_accuracy_hist'][epoch] += np.sum(np.round(conf) == occ)
        stats['val_loss_hist'][epoch] += loss * n_pts
        stats['val_time_error_hist'][epoch] += np.sum(np.abs(occ_t - time))
        stats['val_loc_error_hist'][epoch] += np.sum(np.abs(occ_loc - location), axis=0)
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
    def process_val_epoch_results(stats, epoch):
        stats['val_loss_hist'][epoch] /= stats['total_pts']
        stats['val_accuracy_hist'][epoch] /= stats['total_pts']
        stats['val_time_error_hist'][epoch] /= stats['total_pts']
        stats['val_loc_error_hist'][epoch] /= stats['total_pts']
        stats['total_pts'] = 0
        return stats

    @staticmethod
    def _create_dataset(gen, data_interval, batch_size):

        dataset = tf.data.Dataset.from_generator(gen,
                                                 output_types=(
                                                     tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                                                 output_shapes=(
                                                     tf.TensorShape([None, data_interval, 3]),
                                                     tf.TensorShape([None, 3]),
                                                     tf.TensorShape([]),
                                                     tf.TensorShape([]),
                                                     tf.TensorShape([3]))
                                                 )

        dataset = dataset.padded_batch(batch_size, dataset.output_shapes)

        return dataset

    # FUNCTIONS TO OVERLOAD
    def model_definition(self):
        pass

    def get_loss(self):
        pass


class SimpleMagNN(MagNN):

    def model_definition(self):
        # batch x stations x time x feature
        # conv - conv - pool
        C1 = tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu)(self.mag)
        C2 = tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu)(C1)
        P1 = tf.nn.max_pool(C2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")
        C3 = tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu)(P1)
        C4 = tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu)(C3)
        P2 = tf.nn.max_pool(C4, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")
        """
        c1_kernel = tf.get_variable("c1_kernel", shape=(1, 3, 3, 32))
        b1 = tf.get_variable("b1", shape=(1, 1, 1, 32))
        C1 = tf.nn.leaky_relu(tf.add(tf.nn.conv2d(self.mag, c1_kernel, [1, 1, 1, 1], padding="SAME"), b1))
        c2_kernel = tf.get_variable("c2_kernel", shape=(1, 3, 32, 32))
        b2 = tf.get_variable("b2", shape=(1, 1, 1, 32))
        C2 = tf.nn.leaky_relu(tf.add(tf.nn.conv2d(C1, c2_kernel, [1, 1, 1, 1], padding="SAME"), b2))
        P1 = tf.nn.max_pool(C2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")
        # conv - conv - pool
        c3_kernel = tf.get_variable("c3_kernel", shape=(1, 3, 32, 64))
        b3 = tf.get_variable("b3", shape=(1, 1, 1, 64))
        C3 = tf.nn.leaky_relu(tf.add(tf.nn.conv2d(P1, c3_kernel, [1, 1, 1, 1], padding="SAME"), b3))
        c4_kernel = tf.get_variable("c4_kernel", shape=(1, 3, 64, 64))
        b4 = tf.get_variable("b4", shape=(1, 1, 1, 64))
        C4 = tf.nn.leaky_relu(tf.add(tf.nn.conv2d(C3, c4_kernel, [1, 1, 1, 1], padding="SAME"), b4))
        P2 = tf.nn.max_pool(C4, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")
        """

        # FC layers
        sh = tf.shape(P2)
        vectors = tf.reshape(P2, [sh[0], sh[1], 128 * self.params['data_interval'] // 4])
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
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.occ, logits=self.output['occurrence']))
        time_loss = tf.abs(self.occ_t - self.output['time'])
        location_loss = tf.reduce_mean(tf.abs(self.occ_loc - self.output['location']), axis=1)
        loss = occurrence_loss + tf.reduce_mean(self.occ * (time_loss + location_loss))

        return loss


class ResMagNN(MagNN):

    def model_definition(self):
        # batch x stations x time x feature
        C1 = tf.layers.Conv2D(32, (3, 1), strides=(1, 1), padding='same', activation=tf.nn.relu)(self.mag)
        R1 = modules.residual_layer(C1, 32)
        R2 = modules.residual_layer(R1, 32)
        R3 = modules.residual_layer(R2, 32)
        P1 = tf.nn.max_pool(R3, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")
        R4 = modules.residual_layer(P1, 64)
        R5 = modules.residual_layer(R4, 64)
        R6 = modules.residual_layer(R5, 64)
        P2 = tf.nn.max_pool(R6, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")
        R7 = modules.residual_layer(P2, 128)
        R8 = modules.residual_layer(R7, 128)
        R9 = modules.residual_layer(R8, 128)
        P3 = tf.nn.max_pool(R9, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")
        R10 = modules.residual_layer(P3, 256)
        R11 = modules.residual_layer(R10, 256)
        R12 = modules.residual_layer(R11, 256)
        P4 = tf.nn.max_pool(R12, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")

        # FC layers
        sh = tf.shape(P4)
        vectors = tf.reshape(P3, [sh[0], sh[1], 128 * self.params['data_interval'] // 8])
        vectors = tf.concat((vectors, self.station_loc), axis=2)
        FC1 = tf.nn.relu(tf.layers.dense(vectors, 1024))
        FC2 = tf.nn.relu(tf.layers.dense(FC1, 1024))

        # sum - FC layers
        inp_sum = tf.reduce_mean(FC2, axis=1)
        FC3 = tf.nn.relu(tf.layers.dense(inp_sum, 1024))
        FC4 = tf.nn.relu(tf.layers.dense(FC3, 1024))
        model_output = tf.layers.dense(FC4, 5)

        return {'occurrence': model_output[:, 0], 'time': model_output[:, 1], 'location': model_output[:, 2:]}

    def get_loss(self):
        occurrence_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.occ, logits=self.output['occurrence']))
        time_loss = tf.abs(self.occ_t - self.output['time'])
        location_loss = tf.reduce_mean(tf.abs(self.occ_loc - self.output['location']), axis=1)
        loss = occurrence_loss + tf.reduce_mean(self.occ * (time_loss + location_loss))

        return loss


class ResMax(MagNN):

    def model_definition(self):
        # batch x stations x time x feature
        C1 = tf.layers.Conv2D(32, (5, 1), strides=(1, 1), padding='same', activation=tf.nn.relu)(self.mag)
        R1 = modules.residual_layer(C1, 64, 5)
        R2 = modules.residual_layer(R1, 64, 5)
        R3 = modules.residual_layer(R2, 64, 5)
        P1 = tf.nn.max_pool(R3, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")
        R4 = modules.residual_layer(P1, 128, 3)
        R5 = modules.residual_layer(R4, 128, 3)
        R6 = modules.residual_layer(R5, 128, 3)
        P2 = tf.nn.max_pool(R6, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")

        # FC layers
        sh = tf.shape(P2)
        vectors = tf.reshape(P2, [sh[0], sh[1], 128 * self.params['data_interval'] // 4])
        vectors = tf.concat((vectors, self.station_loc), axis=2)
        FC1 = tf.nn.relu(tf.layers.dense(vectors, 256))
        FC2 = tf.nn.relu(tf.layers.dense(FC1, 256))
        # batch x station x blasdafdad

        # sum - FC layers
        norm = tf.reduce_sum(FC2 ** 2, axis=2)
        idx = tf.argmax(norm, axis=1)
        inp_sum = tf.gather_nd(FC2, tf.stack((tf.range(self.params['batch_size'], dtype=tf.int64), idx), axis=1))
        # self.check = [FC2, norm, idx, inp_sum]
        FC3 = tf.nn.relu(tf.layers.dense(inp_sum, 256))
        FC4 = tf.nn.relu(tf.layers.dense(FC3, 256))
        model_output = tf.layers.dense(FC4, 5)

        return {'occurrence': model_output[:, 0], 'time': model_output[:, 1], 'location': model_output[:, 2:]}

    def get_loss(self):
        occurrence_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.occ, logits=self.output['occurrence']))
        time_loss = tf.abs(self.occ_t - self.output['time'])
        location_loss = tf.reduce_mean(tf.abs(self.occ_loc - self.output['location']), axis=1)
        loss = occurrence_loss + tf.reduce_mean(self.occ * (time_loss + location_loss))

        return loss
