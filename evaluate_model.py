import keras
import numpy as np
import matplotlib.pyplot as plt


def display_activation(activations, row_size, station, act_index):
    activation = activations[act_index]
    activation_index = 0
    fig, ax = plt.subplots(row_size, 1, figsize=(row_size * 2.5, 1.5))
    for row in range(0, row_size):
        ax[row].plot(activation[0, station, :, activation_index])
        activation_index += 1


model_name = "closest_n_stations_model"

with open("potentially useful/closest stations/models/{}_architecture.json".format(model_name)) as f:
    model_architecture = f.read()
model = keras.models.model_from_json(model_architecture)
model.load_weights("potentially useful/closest stations/models/{}_weights.h5".format(model_name))

model.summary()

data = np.load("./data/test_set.npz")
X_test = data['X_test']
y_test = data['y_test']
time_test = data['time_test']
one_hot_time_test = data['one_hot_time_test']

confidences = model.predict(X_test)
predictions = np.round(confidences[:,0])
print("Accuracy: {}".format(np.mean(predictions==y_test)))

example = -1
print(y_test)

print("Class: ", y_test[example])

layer_outputs = [layer.output for layer in model.layers]
activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs[1:])
activations = activation_model.predict(X_test[example].reshape(1, 5, 128, 3))

# display_activation(activations, 8, 0, 4)
plt.figure()
plt.plot(X_test[example, 0, :, 0], label='N')
plt.plot(X_test[example, 0, :, 1], label='E')
plt.plot(X_test[example, 0, :, 2], label='Z')
plt.legend()

from vis.visualization import visualize_activation, visualize_saliency, visualize_cam
from vis.utils import utils
from keras import activations

# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = -1

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

# This is the output node we want to maximize.
# plt.figure()
# for i, tv_weight in enumerate([1e-3, 1e-2, 5e-2, 1e-1, 1, 10]):
#     # Lets turn off verbose output this time to avoid clutter and just see the output.
#     img = visualize_activation(model, layer_idx, tv_weight=tv_weight, lp_norm_weight=0., input_range=(-1000., 1000.))
#     plt.subplot(6, 1, i+1)
#     plt.plot(img[1, :, 0], 'r')
#     plt.plot(img[1, :, 1], 'b')
#     plt.plot(img[1, :, 2], 'g')


plt.figure()
grads = visualize_saliency(model, layer_idx, filter_indices=0, seed_input=X_test[example], backprop_modifier='guided')
plt.subplot(211)
plt.imshow(grads, cmap='jet')
grads = visualize_cam(model, layer_idx, filter_indices=0, seed_input=X_test[example], backprop_modifier='guided')
plt.subplot(212)
plt.imshow(grads, cmap='jet')
plt.show()
