"""
This script creates and trains a network with the "closest N stations" architecture. It also splits up the previously
created dataset into train, validation and test sets. After training, it saves the model.
"""

import keras
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# name of the saved model
model_name = "closest_n_stations_model"

# training parameters
EPOCHS = 25
BATCH_SIZE = 32
train_val_split = .2
train_test_split = .1


def randomly_split_data(list_of_data, split):
    """this function splits a list of equal length (first dimension) data arrays into two lists. The length of the data
    put into the second list is determined by the 'split' argument. This can be used for slitting [X, y] into
    [X_train, y_train] and [X_val, y_val]
    """

    split_idx = int((1 - split) * list_of_data[0].shape[0])

    idx = np.arange(list_of_data[0].shape[0])
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
train, test = randomly_split_data([X, y, time, one_hot_time], train_test_split)
train, val = randomly_split_data(train, train_val_split)
X_train, y_train, time_train, one_hot_time_train = train
X_val, y_val, time_val, one_hot_time_val = val
X_test, y_test, time_test, one_hot_time_test = test

# save the test set for later evaluation (example x station x time x component)
np.savez("./data/test_set.npz", X_test=X_test, y_test=y_test, time_test=time_test, one_hot_time_test=one_hot_time_test)

print("X train shape:", X_train.shape)
print("X val shape:", X_val.shape)
print("X test shape:", X_test.shape)
print("proportion of substorms: ", np.mean(y_val))

# weight decay
reg = keras.regularizers.l2(.0005)

# model input (N_STAIONS, T0, 3)
model_input = keras.layers.Input(shape=(5, 128, 3))
# some conv - relu - max pool modules, all on individual stations, note [1 station x 5 minutes] kernel
net = keras.layers.Conv2D(32, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(model_input)
net = keras.layers.Conv2D(32, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv2D(32, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.MaxPool2D([1, 2], strides=[1, 2])(net)
net = keras.layers.Conv2D(64, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv2D(64, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.MaxPool2D([1, 2], strides=[1, 2])(net)
net = keras.layers.Conv2D(128, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv2D(128, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.MaxPool2D([1, 2], strides=[1, 2])(net)
# this layer combines data from the 5 stations with a [5 stations x 1 'time'] kernel
net = keras.layers.Conv2D(256, [5, 1], strides=[1, 1], padding='valid', activation='relu', kernel_regularizer=reg)(net)
# these layers process the combined 5 stations more
net = keras.layers.Conv2D(256, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv2D(256, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(net)
# these layers are basically dense layers, I was trying the thing that they suggested in the stanford site
# also mentioned here https://piazza.com/class/jqyhgi70q3o2wj?cid=124
net = keras.layers.Conv2D(1024, [1, 16], strides=[1, 1], padding='valid', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv2D(1024, [1, 1], strides=[1, 1], padding='valid', activation='relu', kernel_regularizer=reg)(net)
# I think these two layers probably mess the previous comment up though
net = keras.layers.Flatten()(net)
# predict single binary output (sigmoid)
model_output = keras.layers.Dense(1, kernel_regularizer=reg, activation='sigmoid')(net)

model = keras.models.Model(inputs=model_input, outputs=model_output)
opt = keras.optimizers.Adam(lr=1.0e-3)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))

# save the model architecture / weights
model_json = model.to_json()
with open("./models/{}_architecture.json".format(model_name), 'w') as f:
    f.write(model_json)
model.save_weights("./models/{}_weights.h5".format(model_name))


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
