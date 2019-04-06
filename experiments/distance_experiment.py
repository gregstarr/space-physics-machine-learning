import matplotlib.pyplot as plt
import keras
import numpy as np
from pymap3d.vincenty import vdist
import talos as ta

X = np.random.rand(1000, 4) * [180, 360, 180, 360] - [90, 180, 90, 180]
y, *_ = vdist(X[:, 0], X[:, 1], X[:, 2], X[:, 3])
y /= 1000

def distance_model(X, y, X_val, y_val, params):

    layers = [keras.layers.Dense(params['first_layer_units'], input_dim=4, activation=params['activation'])]
    for i in range(params['n_layers']):
        layers.append(keras.layers.Dense(params['n_units'], activation=params['activation']))
    layers.append(keras.layers.Dense(1))

    model = keras.models.Sequential(layers)

    model.compile(optimizer='adam', loss=params['loss'], metrics=['mae'])
    hist = model.fit(X, y, epochs=params['epochs'], batch_size=params['batch_size'])

    return hist, model


# params = {
#     'first_layer_units': np.logspace(6, 11, num=10, base=2).astype(int),
#     'activation': [keras.activations.tanh, keras.activations.elu, keras.activations.sigmoid, keras.activations.relu],
#     'n_layers': [1, 2, 3, 4, 5],
#     'n_units': np.logspace(6, 11, num=10, base=2).astype(int),
#     'loss': ['mse', 'mae'],
#     'batch_size': 2**np.arange(4, 8),
#     'epochs': (100, 1000, 100)
# }
# t = ta.Scan(X, y, params, distance_model, grid_downsample=.001 )

params = {
    'first_layer_units': 64,
    'activation': keras.activations.elu,
    'n_layers': 5,
    'n_units': 1024,
    'loss': 'mse',
    'batch_size': 128,
    'epochs': 900
}

hist, model = distance_model(X, y, None, None, params)

plt.figure()
plt.plot(hist.history['loss'])

print(model.summary())

X_test = np.random.rand(10000, 4) * [180, 360, 180, 360] - [90, 180, 90, 180]
y_gt, *_ = vdist(X_test[:, 0], X_test[:, 1], X_test[:, 2], X_test[:, 3])
y /= 1000
y_model = model.predict(X_test)

plt.figure()
plt.scatter(X_test[:, 0]-X_test[:, 2], X_test[:, 1]-X_test[:, 3], s=10, c=y_gt.ravel())
plt.figure()
plt.scatter(X_test[:, 0]-X_test[:, 2], X_test[:, 1]-X_test[:, 3], s=10, c=y_model.ravel())
plt.show()
#
# plt.figure()
# plt.hist(y_gt.ravel() - y_model.ravel(), 200)
#
# scatter_max = np.max(np.concatenate((y_gt.ravel(), y_model.ravel())))
# scatter_min = np.min(np.concatenate((y_gt.ravel(), y_model.ravel())))
# plt.figure()
# plt.subplot(121)
# plt.scatter(X_test[:, 0] - X_test[:, 2], X_test[:, 1] - X_test[:, 3], s=5, c=y_model.ravel(), vmin=scatter_min, vmax=scatter_max)
# plt.subplot(122)
# plt.scatter(X_test[:, 0] - X_test[:, 2], X_test[:, 1] - X_test[:, 3], s=5, c=y_gt.ravel(), vmin=scatter_min, vmax=scatter_max)
# plt.show()
