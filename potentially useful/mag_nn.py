"""
This file defines neural networks using tensorflow and the data pipeline defined in "data_pipeline.py"
"""

import tensorflow as tf
import numpy as np
from . import modules
from .data_pipeline import small_dset_gen, large_dset_gen
import json


class MagNN:
    """
    This is a base class for other variable-number-of-stations networks, it contains functions that every network
    should have such as the training loop, stats collection for the training, model saving/loading, and data input
    pipeline. In order to define an architecture, you have to override model_definition and get_loss. model_definition
    can use the data pipeline outputs: self.mag, self.station_loc, self.occ, self.occ_t, self.occ_loc
    and should create the model_output dictionary {'occurrence': , 'time': , 'location': }, whereas get_loss just
    defines what the optimizer should minimize.


    params example:
    {
        'mag_files': glob.glob("./data/mag_data_*.nc")[:2],
        'ss_file': "./data/substorms_2000_2018.csv",
        'data_interval': 96,
        'prediction_interval': 96,
        'val_size': 512,
        'batch_size': 64,
        'model_name': "Wider_Net"
    }
    """

    def __init__(self, params):

        self.params = params

        # extra tf.placeholders to run
        self.check = []

        tf.reset_default_graph()

        # data pipeline iterators / initializers
        self.data_iterator, self.train_init, self.val_init = self.create_data_pipeline()
        # data pipeline tensors
        self.mag, self.station_loc, self.occ, self.occ_t, self.occ_loc = self.data_iterator.get_next()
        # output of the model
        self.output = self.model_definition()
        # loss
        self.loss = self.get_loss()
        # confidences
        self.conf = tf.nn.sigmoid(self.output['occurrence'])
        # session
        self.session = tf.Session()

    def train(self, total_epochs, learning_rate=.001):
        # define optimizer and training step
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_step = opt.minimize(self.loss)
        # model saver
        saver = tf.train.Saver()
        # initialize global variables
        self.session.run(tf.global_variables_initializer())
        # initialize the training dataset
        self.session.run(self.train_init)

        epoch = 0
        # these are all the stats to keep track of
        stats = {'train_loss_hist': np.zeros(total_epochs),
                 'train_accuracy_hist': np.zeros(total_epochs),
                 'train_time_error_hist': np.zeros(total_epochs),
                 'train_loc_error_hist': np.zeros((total_epochs, 3)),
                 'val_loss_hist': np.zeros(total_epochs),
                 'val_accuracy_hist': np.zeros(total_epochs),
                 'val_time_error_hist': np.zeros(total_epochs),
                 'val_loc_error_hist': np.zeros((total_epochs, 3)),
                 'total_pts': 0}
        # these are all the variables to run in session.run
        run_variables = [train_step, self.loss, self.conf, self.occ, self.occ_t, self.occ_loc,
                         self.output['time'], self.output['location']] + self.check

        while epoch < total_epochs:
            try:
                # run the variables (including the training step) and collect the stats for this batch
                collect_variables = self.session.run(run_variables)
                stats = MagNN.process_train_batch_results(collect_variables, stats, epoch)
            except tf.errors.OutOfRangeError:
                # if you get an out of range error, that means the epoch is over
                # collect all of the stats together for the epoch
                stats = MagNN.process_train_epoch_results(stats, epoch)
                # repeat above process but for the validation set
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
                # print out some results
                print("Epoch: {}, Train loss: {}, Val loss: {}".format(epoch+1, stats['train_loss_hist'][epoch],
                                                                       stats['val_loss_hist'][epoch]))
                epoch += 1
        # save the model architecture, weights, and training stats
        name = self.params['model_name']
        print(saver.save(self.session, "./model/{}.ckpt".format(name)))
        np.savez("./model/{}.npz".format(name), **stats)
        with open("./model/{}_params.json".format(name), 'w') as f:
            f.write(json.dumps(self.params))

        return stats

    def run_validation(self):
        """
        This just runs some data through the model and collects stats but doesn't perform any weight updates
        """
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
        """
        this creates the data pipeline using the generator defined in data_pipeline.py
        """
        # unpack the params dictionary
        mag_files = self.params['mag_files']
        ss_file = self.params['ss_file']
        val_size = self.params['val_size']
        batch_size = self.params['batch_size']
        data_interval = self.params['data_interval']
        prediction_interval = self.params['prediction_interval']
        # create the generators
        train_gen = large_dset_gen(mag_files[:-1], ss_file, data_interval, prediction_interval)
        # the validation generator is created from the final file in the input file list
        val_gen = small_dset_gen(mag_files[-1], ss_file, data_interval, prediction_interval, val_size)

        # creates the dataset using the generator, puts the data into padded batches (to accomodate variable n stations)
        train_dataset = MagNN._create_dataset(train_gen, data_interval, batch_size)
        # shuffles up the training data
        train_dataset.shuffle(batch_size * 4).prefetch(2)

        val_dataset = MagNN._create_dataset(val_gen, data_interval, batch_size)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

        # these iterators need to be run in order to select which dataset the model is using
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



# These are just some models I tried, none of them worked well


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
