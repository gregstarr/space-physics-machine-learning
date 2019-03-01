import tensorflow as tf


def residual_layer(input, filters):
    channels = input.shape[-1]
    if channels != filters:
        padded = tf.pad(input, [[0, 0], [0, 0], [0, 0], [channels//2, channels//2]])
    else:
        padded = input
    C1 = tf.layers.Conv2D(filters, (3, 1), strides=(1, 1), padding='same')(input)
    B1 = tf.layers.BatchNormalization()(C1)
    A1 = tf.nn.relu(B1)
    C2 = tf.layers.Conv2D(filters, (3, 1), strides=(1, 1), padding='same', activation=None)(A1)
    S = tf.add(C2, padded)
    return tf.nn.relu(S)
