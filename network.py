import layers
import math
import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred,2))
def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, data, labels, batch_size = 1, epochs = 100, learning_rate = 0.05, verbose = True):
        for e in range(epochs):    
            mean_squared_error = 0
            for i in range(0, len(data), batch_size):
                limit = min(i + batch_size, len(data))
                cur_data = data[:,i:limit]
                cur_labels = labels[:,i:limit]
                predictions = self.predict(cur_data)
                mean_squared_error += mse(cur_labels,predictions)
                gradients = mse_prime(cur_labels,predictions)
                for layer in reversed(self.layers):
                    gradients = layer.backward(gradients, learning_rate)
            if verbose:
                print(f"{e + 1}/{epochs}, error={mean_squared_error}")
    def predict(self, data):
        predictions = data
        for layer in self.layers:
            predictions = layer.forward(predictions)
        return predictions
    
    def save(self, path="models/custom_model.h5"):
        #TODO
        with open(path, 'w') as f:
            f.write(str(len(self.layers)) + "\n")
            for layer in self.layers:
                layer.save(path)
    def load_model(path='models/custom_model.h5'):
        #TODO
        model = Network()
        with open(path, 'r') as f:
            model.layers = []
            num_layers = int(f.readline())
            for i in range(num_layers):
                new_layer = layers.Layer()
                new_layer.name = f.readline()
                if new_layer.name == "Dense":
                    model.layers.append(layers.Dense())
                    for line in f:
                        model.layers[-1].weights.append([float(x) for x in line.split()])
                    for line in f:
                        model.layers[-1].bias.append(float(line))
        return model
