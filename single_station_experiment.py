import keras
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

model_name = "closest_n_stations_model"

EPOCHS = 15
BATCH_SIZE = 64
train_val_split = .2
train_test_split = .1


def randomly_split_data(list_of_data, split):

    split_idx = int((1 - split) * list_of_data[0].shape[0])

    idx = np.arange(list_of_data[0].shape[0])
    np.random.shuffle(idx)

    split_a = []
    split_b = []

    for data in list_of_data:
        split_a.append(data[idx[:split_idx]])
        split_b.append(data[idx[split_idx:]])

    return split_a, split_b


data = np.load("./data/closest_station_data.npz")
X = data['X']
y = data['y']
time = data['time']
one_hot_time = data['one_hot_time']

train, test = randomly_split_data([X, y, time, one_hot_time], train_test_split)
train, val = randomly_split_data(train, train_val_split)

X_train, y_train, time_train, one_hot_time_train = train
X_val, y_val, time_val, one_hot_time_val = val
X_test, y_test, time_test, one_hot_time_test = test

np.savez("./data/test_set.npz", X_test=X_test, y_test=y_test, time_test=time_test, one_hot_time_test=one_hot_time_test)
# instance x station x time x component

print("X train shape:", X_train.shape)
print("X val shape:", X_val.shape)
print("X test shape:", X_test.shape)
print("proportion of substorms: ", np.mean(y_val))

reg = keras.regularizers.l2(.015)

model_input = keras.layers.Input(shape=(5, 128, 3))
net = keras.layers.Conv2D(32, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(model_input)
net = keras.layers.Conv2D(32, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv2D(32, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.MaxPool2D([1, 2], strides=[1, 2])(net)
net = keras.layers.Conv2D(64, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv2D(64, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv2D(64, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.MaxPool2D([1, 2], strides=[1, 2])(net)
net = keras.layers.Conv2D(128, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv2D(128, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv2D(128, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.MaxPool2D([1, 2], strides=[1, 2])(net)
net = keras.layers.Conv2D(128, [5, 1], strides=[1, 1], padding='valid', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv2D(128, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv2D(128, [1, 5], strides=[1, 1], padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv2D(1024, [1, 16], strides=[1, 1], padding='valid', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv2D(1024, [1, 1], strides=[1, 1], padding='valid', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Flatten()(net)
model_output = keras.layers.Dense(1, kernel_regularizer=reg, activation='sigmoid')(net)

model = keras.models.Model(inputs=model_input, outputs=model_output)
opt = keras.optimizers.Adam(lr=1.0e-3)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))

model_json = model.to_json()
with open("./models/{}_architecture.json".format(model_name), 'w') as f:
    f.write(model_json)
model.save_weights("./models/{}_weights.h5".format(model_name))


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


