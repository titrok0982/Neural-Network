import numpy as np
#Implement our own Neural Network
def relu_derivative(x):
        if x > 0:
            return 1
        else:
            return 0
def sigmoid_derivative(x):
    return x * (1 - x)
class Dense():

    def __init__(self, input_size, output_size):

        self.input_size = input_size
        self.output_size = output_size
        self.weights = [[round(np.random.randn()*0.01,10) for _ in range(input_size)] for _ in range(output_size)]
        self.bias = [0 for _ in range(output_size)]
        self.delta_weights = [[0 for _ in range(input_size)] for _ in range(output_size)]
        self.delta_bias = [0 for _ in range(output_size)]
        self.output = [0 for _ in range(output_size)]
        self.input = [0 for _ in range(input_size)]

    def compute_output(self, input,activation=lambda x : x):
        self.input = input
        output = [0 for _ in range(self.output_size)]
        for i in range(self.output_size):
            for j in range(self.input_size):
                self.output[i] += input[j] * self.weights[i][j]
            self.output[i] += self.bias[i]
            self.output[i] = round(self.output[i],10)
            output[i] = activation(self.output[i])
        return output
    
    def backpropagate(self, cost):
        new_cost = [0 for _ in range(self.output_size)]
        for i in range(self.output_size):
            z = self.output[i]
            daz = sigmoid_derivative(z)
            dca = cost[i]
            for j in range(self.input_size):
                self.delta_weights[i][j] += dca * self.input[j] * daz
                new_cost[i] += dca * daz * self.weights[i][j]
            self.delta_bias[i] += dca * daz
        return new_cost
    def update(self, learning_rate):
        for i in range(self.output_size):
            for j in range(self.input_size):
                self.weights[i][j] -= learning_rate * self.delta_weights[i][j]
            self.bias[i] -= learning_rate * self.delta_bias[i]