import keras
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

EPOCHS = 25
BATCH_SIZE = 64

data = np.load("./data/single_station_data.npz")
X = data['X']
y = data['y']
idx = np.arange(X.shape[0])
np.random.shuffle(idx)
X_train = X[idx[:400]]
y_train = y[idx[:400]]
X_val = X[idx[400:]]
y_val = y[idx[400:]]


print("X train shape:", X_train.shape)
print("y train shape: ", y_train.shape)
print("X val shape:", X_val.shape)
print("y val shape: ", y_val.shape)
print("proportion of substorms: ", np.mean(y_val))

reg = keras.regularizers.l2(.011)

model_input = keras.layers.Input(shape=(96, 3))
net = keras.layers.Conv1D(32, 5, strides=1, padding='same', activation='relu', kernel_regularizer=reg)(model_input)
net = keras.layers.Conv1D(32, 5, strides=1, padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv1D(32, 5, strides=1, padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.MaxPool1D(2, strides=2)(net)
net = keras.layers.Conv1D(64, 5, strides=1, padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv1D(64, 5, strides=1, padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv1D(64, 5, strides=1, padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.MaxPool1D(2, strides=2)(net)
net = keras.layers.Conv1D(128, 5, strides=1, padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv1D(128, 5, strides=1, padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv1D(128, 5, strides=1, padding='same', activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Reshape((1, 3072))(net)
net = keras.layers.Conv1D(1024, 1, activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Conv1D(1024, 1, activation='relu', kernel_regularizer=reg)(net)
net = keras.layers.Reshape((1024,))(net)
model_output = keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=reg)(net)

model = keras.models.Model(inputs=model_input, outputs=model_output)
opt = keras.optimizers.Adam(lr=1.0e-3)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))

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

