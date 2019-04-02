"""
This script creates and trains a network with the "closest N stations" architecture. It also splits up the previously
created dataset into train, validation and test sets. After training, it saves the model.
"""

import keras
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("C:\\Users\\Greg\\code\\space-physics-machine-learning")

plt.style.use('ggplot')

# name of the saved model
model_name = "closest_n_stations_model"

# training parameters
EPOCHS = 10
BATCH_SIZE = 64
train_val_split = .1
train_test_split = .1


def split_data(list_of_data, split, random=True):
    """this function splits a list of equal length (first dimension) data arrays into two lists. The length of the data
    put into the second list is determined by the 'split' argument. This can be used for slitting [X, y] into
    [X_train, y_train] and [X_val, y_val]
    """

    split_idx = int((1 - split) * list_of_data[0].shape[0])

    idx = np.arange(list_of_data[0].shape[0])
    if random:
        np.random.shuffle(idx)

    split_a = []
    split_b = []

    for data in list_of_data:
        split_a.append(data[idx[:split_idx]])
        split_b.append(data[idx[split_idx:]])

    return split_a, split_b


# load in the data created by "create_single_station_dataset.py"
data = np.load("./data/closest_station_data.npz")
X = data['X']
y = data['y']
time = data['time']
one_hot_time = data['one_hot_time']

# create train, val and test sets
train, test = split_data([X, y, time, one_hot_time], train_test_split, random=False)
train, val = split_data(train, train_val_split, random=False)
X_train, y_train, time_train, one_hot_time_train = train
X_val, y_val, time_val, one_hot_time_val = val
X_test, y_test, time_test, one_hot_time_test = test

# save the test set for later evaluation (example x station x time x component)
np.savez("./data/test_set.npz", X_test=X_test, y_test=y_test, time_test=time_test, one_hot_time_test=one_hot_time_test)

print("X train shape:", X_train.shape)
print("X val shape:", X_val.shape)
print("X test shape:", X_test.shape)
print("proportion of substorms: ", np.mean(y_val))

# model input (N_STAIONS, T0, 3)

layers = [
    keras.layers.Conv2D(32, [1, 9], strides=[1, 1], padding='same', activation='relu'),
    keras.layers.Dropout(.1),
    keras.layers.Conv2D(32, [1, 5], strides=[1, 1], padding='same', activation='relu'),
    keras.layers.MaxPool2D([1, 2], strides=[1, 2]),
    keras.layers.Conv2D(64, [1, 5], strides=[1, 1], padding='same', activation='relu'),
    keras.layers.Dropout(.1),
    keras.layers.Conv2D(64, [1, 5], strides=[1, 1], padding='same', activation='relu'),
    keras.layers.MaxPool2D([1, 2], strides=[1, 2]),
    keras.layers.Conv2D(128, [1, 5], strides=[1, 1], padding='same', activation='relu'),
    keras.layers.Dropout(.1),
    keras.layers.Conv2D(128, [1, 5], strides=[1, 1], padding='same', activation='relu'),
    keras.layers.MaxPool2D([1, 2], strides=[1, 2]),
    keras.layers.Conv2D(256, [X_train.shape[1], 1], strides=[1, 1], padding='valid', activation='relu'),
    keras.layers.Dropout(.1),
    keras.layers.Conv2D(256, [1, 5], strides=[1, 1], padding='same', activation='relu'),
    keras.layers.Dropout(.1),
    keras.layers.Conv2D(256, [1, 5], strides=[1, 1], padding='same', activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dropout(.1),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(.5),
    keras.layers.Dense(1, activation='sigmoid')
]

model = keras.models.Sequential(layers)

# model_input = keras.layers.Input(shape=X_train.shape[1:])
# # some conv - relu - max pool modules, all on individual stations, note [1 station x 5 minutes] kernel
# net = keras.layers.Conv2D(32, [1, 9], strides=[1, 1], padding='same', activation='relu')(model_input)
# net = keras.layers.Dropout(.1)(net)
# net = keras.layers.Conv2D(32, [1, 5], strides=[1, 1], padding='same', activation='relu')(net)
# net = keras.layers.MaxPool2D([1, 2], strides=[1, 2])(net)
# net = keras.layers.Conv2D(64, [1, 5], strides=[1, 1], padding='same', activation='relu')(net)
# net = keras.layers.Dropout(.1)(net)
# net = keras.layers.Conv2D(64, [1, 5], strides=[1, 1], padding='same', activation='relu')(net)
# net = keras.layers.MaxPool2D([1, 2], strides=[1, 2])(net)
# net = keras.layers.Conv2D(128, [1, 5], strides=[1, 1], padding='same', activation='relu')(net)
# net = keras.layers.Dropout(.1)(net)
# net = keras.layers.Conv2D(128, [1, 5], strides=[1, 1], padding='same', activation='relu')(net)
# net = keras.layers.MaxPool2D([1, 2], strides=[1, 2])(net)
# # this layer combines data from the 5 stations with a [5 stations x 1 'time'] kernel
# net = keras.layers.Conv2D(256, [X_train.shape[1], 1], strides=[1, 1], padding='valid', activation='relu')(net)
# net = keras.layers.Dropout(.1)(net)
# # these layers process the combined 5 stations more
# net = keras.layers.Conv2D(256, [1, 5], strides=[1, 1], padding='same', activation='relu')(net)
# net = keras.layers.Dropout(.1)(net)
# net = keras.layers.Conv2D(256, [1, 5], strides=[1, 1], padding='same', activation='relu')(net)
# net = keras.layers.Flatten()(net)
# net = keras.layers.Dropout(.1)(net)
# net = keras.layers.Dense(1024, activation='relu')(net)
# net = keras.layers.Dropout(.5)(net)
# # predict single binary output (sigmoid)
# model_output = keras.layers.Dense(1, activation='sigmoid')(net)

# model = keras.models.Model(inputs=model_input, outputs=model_output)
opt = keras.optimizers.Adam(lr=1.0e-3)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))

# save the model architecture / weights
model_json = model.to_json()
with open("potentially useful/closest stations/models/{}_architecture.json".format(model_name), 'w') as f:
    f.write(model_json)
model.save_weights("potentially useful/closest stations/models/{}_weights.h5".format(model_name))


# plot the training curves
plt.figure()
plt.title("Accuracy")
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Val Accuracy')
plt.legend()

plt.figure()
plt.title("Loss")
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()

plt.show()

test_predictions = model.predict(X_test)
print("Test accuracy: {}".format(np.mean(np.round(test_predictions[:, 0]) == y_test)))

from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations

# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = -1

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

# This is the output node we want to maximize.
filter_idx = 0
img = visualize_activation(model, layer_idx, filter_indices=filter_idx)
plt.imshow(img[..., 0])
