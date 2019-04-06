"""
This is a toy example to compare CNNs on grid-structured data vs CNNs on randomly (spatially) sampled data.
The input data are functions on a circle using 10 randomly sampled frequency components. I will create two classes
which generate 10-d random vectors drawn from different multivariate gaussian distributions. The actual inputs will
be samples of the functions at 100 locations around the circle. The task will be to classify which distribution the
function came from. The gaussian distributions will be zero mean so I will just have to generate a few random PSD
matrices.
"""
import numpy as np
import keras
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# parameters
params = {
    'kernel_size': 5,
    'activation': keras.activations.relu,
    'loss': 'sparse_categorical_crossentropy',
    'batch_size': 32,
    'epochs': 5
}

n_coeffs = 20
n_classes = 10
samples_per_class = 1000
samples_in_circle = 10
R = 10


# model definition
def circle_model(X, y, X_val, y_val, params):

    layers = [keras.layers.Conv1D(32, params['kernel_size'], padding='same', activation='relu', input_shape=(784, 1)),
              keras.layers.Conv1D(32, params['kernel_size'], padding='same', activation='relu'),
              keras.layers.MaxPool1D(),
              keras.layers.Conv1D(64, params['kernel_size'], padding='same', activation='relu'),
              keras.layers.Conv1D(64, params['kernel_size'], padding='same', activation='relu'),
              keras.layers.MaxPool1D(),
              keras.layers.Flatten(),
              keras.layers.Dropout(.5),
              keras.layers.Dense(n_classes, activation='softmax')
              ]

    model = keras.models.Sequential(layers)

    model.compile(optimizer='adam', loss=params['loss'], metrics=['accuracy'])
    hist = model.fit(X, y, epochs=params['epochs'], batch_size=params['batch_size'], validation_data=(X_val, y_val))

    return hist, model


# Create Dataset
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

"""
X_grid = []
X_rand = []
y = []
theta_grid = np.linspace(0, 2*np.pi, samples_in_circle)
theta_rand = np.random.rand(samples_in_circle) * 2 * np.pi
covs = np.empty((n_classes, n_coeffs, n_coeffs))
for i in range(n_classes):
    rand_mat = np.random.randn(n_coeffs, n_coeffs)
    covariance_matrix = np.dot(rand_mat.T, rand_mat) / (.5 * n_coeffs**2)
    coeffs = np.random.multivariate_normal(np.zeros(n_coeffs), covariance_matrix, samples_per_class)
    X_grid.append(np.sum(coeffs[:, :, None] * np.cos(np.arange(1, n_coeffs+1)[:, None] * theta_grid[None, :])[None, :, :], axis=1))
    X_rand.append(np.sum(coeffs[:, :, None] * np.cos(np.arange(1, n_coeffs + 1)[:, None] * theta_rand[None, :])[None, :, :], axis=1))
    labels = np.zeros((samples_per_class, n_classes))
    labels[:, i] = 1
    y.append(labels)
    covs[i] = covariance_matrix
X_grid = np.concatenate(X_grid, axis=0)
X_rand = np.concatenate(X_rand, axis=0)
y = np.concatenate(y, axis=0)
idx = np.arange(n_classes * samples_per_class)
np.random.shuffle(idx)
X_grid = X_grid[idx, :, None]
X_rand = X_rand[idx, :, None]
y = y[idx]
[[Xgt, ygt], [Xgv, ygv]] = split_data([X_grid, y], .2)
[[Xrt, yrt], [Xrv, yrv]] = split_data([X_rand, y], .2)
"""

Xg = np.load("grid_images.npy")[:, :, None]
Xr = np.load("random_images.npy")[:, :, None]
y = np.load("labels.npy")[:, None]
[[Xgt, ygt], [Xgv, ygv]] = split_data([Xg, y], .2)
[[Xrt, yrt], [Xrv, yrv]] = split_data([Xr, y], .2)

# create model
hist_grid, model = circle_model(Xgt, ygt, Xgv, ygv, params)
hist_rand, model = circle_model(Xrt, yrt, Xrv, yrv, params)

plt.figure()
plt.plot(hist_grid.history['loss'], 'r-', label='grid train')
plt.plot(hist_grid.history['val_loss'], 'r--', label='grid val')
plt.plot(hist_rand.history['loss'], 'b-', label='rand train')
plt.plot(hist_rand.history['val_loss'], 'b--', label='rand val')
plt.legend()

plt.figure()
plt.plot(hist_grid.history['acc'], 'r-', label='grid train')
plt.plot(hist_grid.history['val_acc'], 'r--', label='grid val')
plt.plot(hist_rand.history['acc'], 'b-', label='rand train')
plt.plot(hist_rand.history['val_acc'], 'b--', label='rand val')
plt.legend()

plt.show()

"""
# show example functions
theta = np.linspace(0, 2*np.pi, 1000)
fig, ax = plt.subplots(5, n_classes, tight_layout=True)
for i in range(n_classes):
    for j in range(5):
        if j == 0:
            ax[j, i].imshow(covs[i])
            continue
        ax[j, i].plot(R * np.cos(theta_grid), R * np.sin(theta_grid), '-')
        ax[j, i].plot(R * np.cos(theta_rand), R * np.sin(theta_rand), 'k.')
        coeffs = np.random.multivariate_normal(np.zeros(n_coeffs), covs[i])
        fxn = np.sum(coeffs[:, None] * np.cos(np.arange(1, n_coeffs + 1)[:, None] * theta[None, :]), axis=0)

        ax[j, i].plot((R + fxn) * np.cos(theta), (R + fxn) * np.sin(theta), '-')
plt.show()
"""
