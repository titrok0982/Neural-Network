import numpy as np
#Implement our own Neural Network
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.name = None
        pass
    def forward(self, input):
        #return output
        pass
    def backward(self, output_gradient, learning_rate):
        #return input_error
        pass


class Dense(Layer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size,1)
        self.name = "Dense"

    def forward(self, input):
        self.input = input
        self.output = self.weights.dot(input) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        m = output_gradient.shape[1]
        input_gradient = self.weights.T.dot(output_gradient)
        self.weights-= learning_rate * (output_gradient.dot(self.input.T) / m)
        self.bias -= learning_rate * (np.sum(output_gradient,axis=1,keepdims=True)/ m)
        return input_gradient
    
class Activation(Layer):

    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        self.name = None

    def forward(self, input):
        self.input = input
        return np.vectorize(self.activation)(self.input)

    def backward(self, output_gradient, learning_rate):
        res = np.vectorize(self.activation_prime)(self.input)
        return np.multiply(output_gradient, np.vectorize(self.activation_prime)(self.input))
    
class Relu(Activation):
    def __init__(self):
        relu = lambda x : np.maximum(0, x)
        relu_prime = lambda x : x>0
        super().__init__(relu, relu_prime)
        self.name = "Relu"

class Softmax(Layer):
    def __init__(self):
        self.name = "Softmax"
    def forward(self, input):
        e_input = np.exp(input)
        self.output = e_input / np.sum(e_input)
        return self.output
    def backward(self, output_gradient, learning_rate):
        I = np.eye(self.output.shape[0])
        return output_gradient.dot((I - self.output.T).T)