import keras
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

model_name = "closest_n_station_model_one_hot"

with open("./models/{}_architecture.json".format(model_name)) as f:
    model_architecture = f.read()
model = keras.models.model_from_json(model_architecture)
model.load_weights("./models/{}_weights.h5".format(model_name))

data = np.load("./data/test_set.npz")
X_test = data['X_test']
y_test = data['y_test']
time_test = data['time_test']
one_hot_time_test = data['one_hot_time_test']

confidences = model.predict(X_test)
predictions = np.round(confidences[:,0])
print("Accuracy: {}".format(np.mean(predictions==y_test)))

plt.figure()
plt.ylabel("N")
for i in range(predictions.shape[0]):
    if predictions[i] != y_test[i]:
        if predictions[i] == 1:
            plt.plot(X_test[i, 0, :, 0] - np.mean(X_test[i, 0, :, 0]), 'r-', alpha=.2, label="false positive")
        elif predictions[i] == 0:
            plt.plot(X_test[i, 0, :, 0] - np.mean(X_test[i, 0, :, 0]), 'b-', alpha=.2, label="false negative")
plt.figure()
plt.ylabel("E")
for i in range(predictions.shape[0]):
    if predictions[i] != y_test[i]:
        if predictions[i] == 1:
            plt.plot(X_test[i, 0, :, 1] - np.mean(X_test[i, 0, :, 1]), 'r-', alpha=.2, label="false positive")
        elif predictions[i] == 0:
            plt.plot(X_test[i, 0, :, 1] - np.mean(X_test[i, 0, :, 1]), 'b-', alpha=.2, label="false negative")
plt.figure()
plt.ylabel("Z")
for i in range(predictions.shape[0]):
    if predictions[i] != y_test[i]:
        if predictions[i] == 1:
            plt.plot(X_test[i, 0, :, 2] - np.mean(X_test[i, 0, :, 2]), 'r-', alpha=.2, label="false positive")
        elif predictions[i] == 0:
            plt.plot(X_test[i, 0, :, 2] - np.mean(X_test[i, 0, :, 2]), 'b-', alpha=.2, label="false negative")

plt.show()
