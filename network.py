import layers
import math

class Network:
    def __init__(self, learning_rate=0.7, batch_size=1) -> None:
        self.layers = []
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, data, labels):
        for i in range(0, len(data), self.batch_size):
            limit = min(i + self.batch_size, len(data))
            predictions = self.predict(data[i:limit])
            mse = 0
            gradients = []
            for prediction, label in zip(predictions, labels[i:limit]):
                for i in range(10):
                    mse += (prediction[i] - label[i])**2
                    gradients.append((prediction[i] - label[i]) * 2)
            mse /= len(data)
            print(mse)
            for layer in reversed(self.layers):
                gradients = layer.backpropagate(gradients)
            for layer in self.layers:
                layer.update(learning_rate=self.learning_rate)
    def predict(self, data):
        predictions = []
        for element in data:
            for layer in self.layers:
                element = layer.compute_output(input=element,activation=self.sigmoid)
            predictions.append(element)
        return predictions
    def relu(self, x):
        return max(0, x)
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    def save(self):
        with open('models/custom_model.h5', 'w') as f:
            f.write(str(self.learning_rate) + "\n")
            f.write(str(self.batch_size) + "\n")
            f.write(str(len(self.layers)) + "\n")
            for layer in self.layers:
                f.write(str(layer.input_size) + "\n")
                f.write(str(layer.output_size) + "\n")
                for line in layer.weights:
                    f.write(" ".join([str(x) for x in line]) + "\n")
                for line in layer.bias:
                    f.write(str(line) + "\n")
    def load_model(path='models/custom_model.h5'):
        model = Network()
        with open(path, 'r') as f:
            model.learning_rate = float(f.readline())
            model.batch_size = int(f.readline())
            model.layers = []
            num_layers = int(f.readline())
            for i in range(num_layers):
                model.layers.append(layers.Dense(int(f.readline()), int(f.readline())))
                for line in f:
                    model.layers[-1].weights.append([float(x) for x in line.split()])
                for line in f:
                    model.layers[-1].bias.append(float(line))
        return model
